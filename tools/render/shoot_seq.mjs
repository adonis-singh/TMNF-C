#!/usr/bin/env node
// Sequential frame renderer: load the viewer once, then re-apply camera/tick
// per frame through window.__TMNF_APPLY_RENDER_PARAMETERS and screenshot.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function findChrome() {
	const root = join(homedir(), ".cache", "ms-playwright");
	const shells = readdirSync(root).filter(name => name.startsWith("chromium_headless_shell-")).sort();
	return join(root, shells[shells.length - 1], "chrome-headless-shell-linux64", "chrome-headless-shell");
}

class Cdp {
	constructor(url) {
		this.socket = new WebSocket(url);
		this.nextId = 1;
		this.pending = new Map();
		this.ready = new Promise((resolve, reject) => {
			this.socket.addEventListener("open", resolve, { once: true });
			this.socket.addEventListener("error", reject, { once: true });
		});
		this.socket.addEventListener("message", (event) => {
			const message = JSON.parse(event.data);
			if (message.id === undefined) return;
			const entry = this.pending.get(message.id);
			this.pending.delete(message.id);
			if (message.error) entry.reject(new Error(`${entry.method}: ${message.error.message}`));
			else entry.resolve(message.result);
		});
	}
	send(method, params = {}, sessionId = undefined) {
		const id = this.nextId++;
		this.socket.send(JSON.stringify({ id, method, params, sessionId }));
		return new Promise((resolve, reject) => this.pending.set(id, { method, resolve, reject }));
	}
}

async function launchChrome(gpu) {
	const profile = mkdtempSync(join(tmpdir(), "tmnf-seq-"));
	const flags = gpu
		? ["--use-gl=angle", "--use-angle=gl", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
		: ["--disable-gpu", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"];
	const child = spawn(findChrome(), [
		"--headless", ...flags, "--no-sandbox", "--remote-debugging-port=0",
		`--user-data-dir=${profile}`, "--window-size=1920,1080", "about:blank",
	], { stdio: ["ignore", "ignore", "pipe"] });
	const url = await new Promise((resolve, reject) => {
		let buffer = "";
		child.stderr.on("data", (chunk) => {
			buffer += chunk.toString();
			const match = buffer.match(/DevTools listening on (ws:\/\/\S+)/);
			if (match) resolve(match[1]);
		});
		child.on("exit", code => reject(new Error(`chrome exited early (${code})\n${buffer}`)));
		setTimeout(() => reject(new Error("no DevTools URL")), 15000);
	});
	return { url, close() { child.kill("SIGTERM"); rmSync(profile, { recursive: true, force: true }); } };
}

async function evaluate(cdp, sessionId, expression) {
	const result = await cdp.send("Runtime.evaluate", {
		expression: `JSON.stringify((() => { return ${expression}; })() ?? null)`,
		returnByValue: true, awaitPromise: true,
	}, sessionId);
	if (result.exceptionDetails)
		throw new Error(`evaluate failed: ${result.exceptionDetails.text} ${result.exceptionDetails.exception?.description || ""}`);
	return JSON.parse(result.result.value);
}

const perFrameKeys = new Set(["tick", "cam", "fov", "camera"]);
function splitUrl(url) {
	const u = new URL(url);
	const base = new URLSearchParams();
	const frame = new URLSearchParams();
	for (const [key, value] of u.searchParams)
		(perFrameKeys.has(key) ? frame : base).set(key, value);
	return { base: `${u.origin}${u.pathname}?${base.toString()}`, frame: frame.toString() };
}

const jobs = JSON.parse(readFileSync(process.argv[2], "utf8"))
	.filter(job => !existsSync(job.out))
	.sort((a, b) => a.out.localeCompare(b.out));
const gpu = process.argv[3] !== "cpu";
console.error(`${jobs.length} frames to render, gpu=${gpu}`);
if (jobs.length === 0) process.exit(0);

const chrome = await launchChrome(gpu);
const cdp = new Cdp(chrome.url);
await cdp.ready;
const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
await cdp.send("Runtime.enable", {}, sessionId);
await cdp.send("Page.enable", {}, sessionId);
await cdp.send("Emulation.setDeviceMetricsOverride", {
	width: jobs[0].width, height: jobs[0].height, deviceScaleFactor: jobs[0].dsf || 1, mobile: false,
}, sessionId);

const first = splitUrl(jobs[0].url);
await cdp.send("Page.navigate", { url: jobs[0].url }, sessionId);
const started = Date.now();
let stats = null;
while (Date.now() - started < 120000) {
	await sleep(250);
	stats = await evaluate(cdp, sessionId, "window.__TMNF_VIEWER_STATS || null");
	if (stats && (stats.ready || stats.notice)) break;
}
if (!stats || !stats.ready) throw new Error(`viewer not ready: ${JSON.stringify(stats)}`);
const hasApply = await evaluate(cdp, sessionId, "typeof window.__TMNF_APPLY_RENDER_PARAMETERS");
if (hasApply !== "function") throw new Error("viewer lacks __TMNF_APPLY_RENDER_PARAMETERS (stale viewer.js?)");
const renderer = await evaluate(cdp, sessionId, "(() => { const c = document.querySelector('canvas'); const gl = c && (c.getContext('webgl2') || c.getContext('webgl')); const d = gl && gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'unknown'; })()");
console.error(`scene ready in ${Date.now() - started} ms, renderer: ${renderer}`);

let done = 0;
const t0 = Date.now();
for (const job of jobs) {
	const { base, frame } = splitUrl(job.url);
	if (base !== first.base) throw new Error(`job ${job.out} has a different base URL`);
	await evaluate(cdp, sessionId, `(window.__TMNF_APPLY_RENDER_PARAMETERS(new URLSearchParams(${JSON.stringify(frame)})), true)`);
	await evaluate(cdp, sessionId, `new Promise(resolve => {
		let n = ${job.frames || 3};
		const step = () => (--n <= 0 ? resolve(true) : requestAnimationFrame(step));
		requestAnimationFrame(step);
	})`);
	const shot = await cdp.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }, sessionId);
	writeFileSync(job.out, Buffer.from(shot.data, "base64"));
	done++;
	if (done % 50 === 0 || done === jobs.length)
		console.error(`${done}/${jobs.length} frames, ${((Date.now() - t0) / done).toFixed(0)} ms/frame`);
}
chrome.close();
