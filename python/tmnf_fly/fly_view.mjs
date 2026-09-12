#!/usr/bin/env node
// Headless frame sequencer for the TMNF-C viewer, driven by tmnf_fly.fly_view.
//
//   node python/tmnf_fly/fly_view.mjs --chrome /usr/bin/chromium --url http://127.0.0.1:PORT/?scene=...
//        --sink http://127.0.0.1:PORT/fly_view/frame --width 1920 --height 1080
//        --supersample 2 --gl gl|swiftshader --frames frames.json [--visual collision]
//
// frames.json is a JSON array of viewer render-parameter query strings
// ("cam=px,py,pz,tx,ty,tz&tick=N&fov=F"). The page is loaded once; per frame
// the parameters go through window.__TMNF_APPLY_RENDER_PARAMETERS, the
// sequencer waits for the viewer's animate loop to render, downscales the WebGL
// canvas into a 2D canvas in that same task and POSTs the raw RGBA pixels
// (width*height*4 bytes, top-down rows) from the page to <sink>/<frame index>,
// which must answer 204.
//
// Anti-aliasing is supersampling: the viewport is width*supersample by
// height*supersample and the WebGL context is created without MSAA. With MSAA
// (ANGLE on NVIDIA GL) about one frame in 400 came back with a draw missing
// (sky sphere, a block batch); without it none did in 2500 frames.
//
// stdout: one JSON line {"renderer": <WebGL renderer string>, "scene": ..., "loadMs": ...}
// once the page is ready. stderr: diagnostics. Standard library only (Node >= 22).

import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = process.argv.slice(2);
function option(name) {
	const index = args.indexOf(name);
	if (index < 0 || index + 1 >= args.length)
		throw new Error(`missing ${name}`);
	return args[index + 1];
}
function optionalOption(name, fallback) {
	const index = args.indexOf(name);
	return index >= 0 ? args[index + 1] : fallback;
}

const CHROME = option("--chrome");
const URL_ = option("--url");
const SINK = option("--sink");
const WIDTH = Number(option("--width"));
const HEIGHT = Number(option("--height"));
const SUPERSAMPLE = Number(option("--supersample"));
const GL = option("--gl");
const FRAMES = JSON.parse(readFileSync(option("--frames"), "utf8"));
const VISUAL = optionalOption("--visual", "game");
const LOAD_TIMEOUT_MS = 180000;

const GL_FLAGS = {
	gl: ["--use-angle=gl", "--ignore-gpu-blocklist"],
	swiftshader: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
};
if (!(GL in GL_FLAGS))
	throw new Error(`--gl must be one of ${Object.keys(GL_FLAGS).join("|")}`);
if (!["game", "collision"].includes(VISUAL))
	throw new Error("--visual must be game|collision");
if (!(Number.isInteger(SUPERSAMPLE) && SUPERSAMPLE >= 1))
	throw new Error("--supersample must be a positive integer");
const RENDER_WIDTH = WIDTH * SUPERSAMPLE;
const RENDER_HEIGHT = HEIGHT * SUPERSAMPLE;

// The viewer asks for an antialiased context; the wrapper installed before the
// page runs turns MSAA off (see the header).
const CONTEXT_ATTRIBUTES_SCRIPT = `(() => {
	const getContext = HTMLCanvasElement.prototype.getContext;
	HTMLCanvasElement.prototype.getContext = function (type, attributes) {
		if (type === "webgl2" || type === "webgl")
			attributes = { ...(attributes || {}), antialias: false };
		return getContext.call(this, type, attributes);
	};
})()`;

function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

function launchChrome() {
	const profile = mkdtempSync(join(tmpdir(), "fly-view-chrome-"));
	const child = spawn(CHROME, [
		"--headless=new",
		"--no-sandbox",
		"--disable-dev-shm-usage",
		...GL_FLAGS[GL],
		"--remote-debugging-port=0",
		`--user-data-dir=${profile}`,
		`--window-size=${RENDER_WIDTH},${RENDER_HEIGHT}`,
		"--force-device-scale-factor=1",
		"--hide-scrollbars",
		"--disable-frame-rate-limit",
		"--disable-gpu-vsync",
		"--disable-background-timer-throttling",
		"--disable-renderer-backgrounding",
		"--mute-audio",
		"--no-first-run",
		"about:blank",
	], { stdio: ["ignore", "ignore", "pipe"] });
	const url = new Promise((resolve, reject) => {
		let buffer = "";
		child.stderr.on("data", chunk => {
			buffer += chunk.toString();
			const match = buffer.match(/DevTools listening on (ws:\/\/\S+)/);
			if (match)
				resolve(match[1]);
		});
		child.on("exit", code => reject(new Error(`chromium exited early (${code})\n${buffer}`)));
		setTimeout(() => reject(new Error(`chromium did not publish a DevTools URL\n${buffer}`)), 20000).unref();
	});
	const exited = new Promise(resolve => child.on("exit", resolve));
	return {
		url,
		// Chromium takes ~15 s to wind down after SIGTERM; give Browser.close a
		// moment, then kill it so the sequencer's exit is not delayed.
		async close(cdp) {
			if (cdp)
				cdp.send("Browser.close").catch(() => {});  // the reply never comes back once the browser is gone
			await Promise.race([exited, sleep(2000)]);
			if (child.exitCode === null)
				child.kill("SIGKILL");
			await exited;
			rmSync(profile, { recursive: true, force: true });
		},
	};
}

class Cdp {
	constructor(url) {
		this.socket = new WebSocket(url);
		this.nextId = 1;
		this.pending = new Map();
		this.events = [];
		this.ready = new Promise((resolve, reject) => {
			this.socket.addEventListener("open", resolve, { once: true });
			this.socket.addEventListener("error", reject, { once: true });
		});
		this.socket.addEventListener("message", event => {
			const message = JSON.parse(event.data);
			if (message.id !== undefined) {
				const entry = this.pending.get(message.id);
				this.pending.delete(message.id);
				if (message.error)
					entry.reject(new Error(`${entry.method}: ${message.error.message}`));
				else
					entry.resolve(message.result);
				return;
			}
			if (message.method === "Runtime.exceptionThrown") {
				const details = message.params.exceptionDetails;
				this.events.push(`exception: ${details.exception?.description || details.text}`);
			} else if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
				this.events.push(`console.error: ${message.params.args.map(arg => arg.description || arg.value).join(" ")}`);
			}
		});
	}

	send(method, params = {}, sessionId = undefined) {
		const id = this.nextId++;
		this.socket.send(JSON.stringify({ id, method, params, sessionId }));
		return new Promise((resolve, reject) => {
			this.pending.set(id, { method, resolve, reject });
		});
	}

	close() {
		this.socket.close();
	}
}

class Page {
	constructor(cdp, sessionId) {
		this.cdp = cdp;
		this.sessionId = sessionId;
	}

	async evaluate(expression) {
		const result = await this.cdp.send("Runtime.evaluate", {
			expression,
			returnByValue: true,
			awaitPromise: true,
		}, this.sessionId);
		if (result.exceptionDetails)
			throw new Error(`evaluate failed: ${result.exceptionDetails.text} ${result.exceptionDetails.exception?.description || ""}`);
		return result.result.value;
	}

	stats() {
		return this.evaluate("JSON.parse(JSON.stringify(window.__TMNF_VIEWER_STATS || null))");
	}
}

const RENDERER_PROBE = `(() => {
	const canvas = document.querySelector("#viewport canvas");
	const gl = canvas.getContext("webgl2");
	const info = gl.getExtension("WEBGL_debug_renderer_info");
	return gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
})()`;

// The viewer does not preserve the drawing buffer across compositing, so the
// pixels must be read in the same task as its render: wrap requestAnimationFrame
// so window.__flyAfterRender runs right after the viewer's animate callback
// returns. Resolves once the viewer's loop has been re-registered through the
// wrapper (its animate re-arms itself at the start of every frame).
const HOOK_EXPRESSION = `new Promise((resolve) => {
	const raf = window.requestAnimationFrame.bind(window);
	window.requestAnimationFrame = callback => raf((timestamp) => {
		callback(timestamp);
		if (window.__flyAfterRender)
			window.__flyAfterRender();
	});
	raf(() => raf(() => resolve(true)));
})`;

// Apply the render parameters, wait until the viewer's animate loop has
// rendered them, downscale the drawing buffer in that same task and POST the
// raw top-down RGBA pixels to the frame sink. Resolves the sink's HTTP status.
function frameExpression(index, params) {
	return `new Promise((resolve, reject) => {
		try {
			window.__TMNF_APPLY_RENDER_PARAMETERS(new URLSearchParams(${JSON.stringify(params)}));
		} catch (error) {
			reject(error);
			return;
		}
		const before = window.__TMNF_VIEWER_STATS.renderedFrames || 0;
		window.__flyAfterRender = () => {
			if ((window.__TMNF_VIEWER_STATS.renderedFrames || 0) <= before)
				return;
			window.__flyAfterRender = null;
			try {
				const canvas = document.querySelector("#viewport canvas");
				if (canvas.width !== ${RENDER_WIDTH} || canvas.height !== ${RENDER_HEIGHT})
					throw new Error("drawing buffer is " + canvas.width + "x" + canvas.height);
				if (!window.__flyScaler) {
					window.__flyScaler = document.createElement("canvas");
					window.__flyScaler.width = ${WIDTH};
					window.__flyScaler.height = ${HEIGHT};
				}
				const context = window.__flyScaler.getContext("2d");
				context.imageSmoothingEnabled = true;
				context.imageSmoothingQuality = "high";
				context.drawImage(canvas, 0, 0, ${WIDTH}, ${HEIGHT});
				const pixels = context.getImageData(0, 0, ${WIDTH}, ${HEIGHT}).data;
				fetch(${JSON.stringify(SINK)} + "/" + ${index}, { method: "POST", body: pixels })
					.then(response => resolve(response.status), reject);
			} catch (error) {
				reject(error);
			}
		};
	})`;
}

async function main() {
	const chrome = launchChrome();
	let cdp = null;
	try {
		cdp = new Cdp(await chrome.url);
		await cdp.ready;
		const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
		const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
		const page = new Page(cdp, sessionId);
		await cdp.send("Runtime.enable", {}, sessionId);
		await cdp.send("Page.enable", {}, sessionId);
		await cdp.send("Page.addScriptToEvaluateOnNewDocument", { source: CONTEXT_ATTRIBUTES_SCRIPT }, sessionId);
		await cdp.send("Emulation.setDeviceMetricsOverride", {
			width: RENDER_WIDTH, height: RENDER_HEIGHT, deviceScaleFactor: 1, mobile: false,
		}, sessionId);
		await cdp.send("Page.navigate", { url: URL_ }, sessionId);

		const started = Date.now();
		let stats = null;
		for (;;) {
			await sleep(200);
			stats = await page.stats();
			if (stats && stats.notice)
				throw new Error(`viewer notice: ${stats.notice}\n${cdp.events.join("\n")}`);
			if (stats && stats.ready)
				break;
			if (Date.now() - started > LOAD_TIMEOUT_MS)
				throw new Error(`viewer did not become ready in ${LOAD_TIMEOUT_MS} ms\n${cdp.events.join("\n")}`);
		}
		const size = await page.evaluate("[window.innerWidth, window.innerHeight]");
		if (size[0] !== RENDER_WIDTH || size[1] !== RENDER_HEIGHT)
			throw new Error(`viewport is ${size[0]}x${size[1]}, wanted ${RENDER_WIDTH}x${RENDER_HEIGHT}`);
		if (VISUAL === "collision") {
			await page.evaluate(`(() => {
				const select = document.getElementById("visual-mode");
				select.value = "collision";
				select.dispatchEvent(new Event("change", { bubbles: true }));
				return select.value;
			})()`);
		}
		const renderer = await page.evaluate(RENDERER_PROBE);
		await page.evaluate(HOOK_EXPRESSION);
		process.stderr.write(`fly_view.mjs: scene ${JSON.stringify(stats.scene)} loaded in ${Date.now() - started} ms, ` +
			`triangles ${stats.triangles}, renderer ${renderer}\n`);
		process.stdout.write(JSON.stringify({ renderer, scene: stats.scene, loadMs: Date.now() - started }) + "\n");

		const renderStarted = Date.now();
		for (let index = 0; index < FRAMES.length; ++index) {
			const status = await page.evaluate(frameExpression(index, FRAMES[index]));
			if (status !== 204)
				throw new Error(`frame sink answered HTTP ${status} for frame ${index}`);
			if (cdp.events.length > 0)
				throw new Error(`page errors during frame ${index}:\n${cdp.events.join("\n")}`);
		}
		const elapsed = (Date.now() - renderStarted) / 1000;
		process.stderr.write(`fly_view.mjs: ${FRAMES.length} frames rendered, read back and posted in ${elapsed.toFixed(1)} s ` +
			`(${(FRAMES.length / elapsed).toFixed(2)} fps)\n`);
	} finally {
		await chrome.close(cdp);
		if (cdp)
			cdp.close();
	}
}

main().catch(error => {
	process.stderr.write(`fly_view.mjs: ${error.stack || error}\n`);
	process.exit(1);
});
