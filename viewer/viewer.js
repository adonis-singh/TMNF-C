(() => {
	"use strict";

	const COLORS = Object.freeze({
		carbon: 0x0b1018,
		blueSlate: 0x263440,
		asphalt: 0x59636d,
		bone: 0xc3b7a4,
		signalBlue: 0x52c7ff,
		velocityOrange: 0xff5a36,
		paper: 0xedf1f2,
	});

	const EXPECTED_FIELDS = Object.freeze([
		"raceTimeMs", "x", "y", "z", "qx", "qy", "qz", "qw",
		"speedMps", "rpm", "gear", "contactMask", "slidingMask",
		"inputSteer",
		"wheelSteerFL", "wheelSteerFR", "wheelSteerRL", "wheelSteerRR",
		"wheelDamperFL", "wheelDamperFR", "wheelDamperRL", "wheelDamperRR",
		"wheelSpeedFL", "wheelSpeedFR", "wheelSpeedRL", "wheelSpeedRR",
	]);

	const FIELD = Object.freeze({
		time: 0,
		x: 1,
		y: 2,
		z: 3,
		qx: 4,
		qy: 5,
		qz: 6,
		qw: 7,
		speed: 8,
		rpm: 9,
		gear: 10,
		contact: 11,
		sliding: 12,
		inputSteer: 13,
		wheelSteer: 14,
		wheelDamper: 18,
		wheelSpeed: 22,
	});

	// Committed base scene per track. Live streams and run replays both key
	// their track assets off this table; an unknown track is a hard error.
	const TRACK_SCENES = Object.freeze({
		a01: "scenes/policy_lap.json",
		a10: "scenes/a10_turbo.json",
		a08: "scenes/a08_mixed.json",
		b04: "scenes/b04_mixed.json",
		e01: "scenes/e01_mixed.json",
		c03: "scenes/c03_mixed.json",
	});
	const RUNS_POLL_MS = 5000;
	const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
	const REPLAY_SCENE_PATTERN = /^replays\/[A-Za-z0-9_.-]+\.json(\.gz)?$/;
	const GHOST_PATH_PATTERN = /^(scenes|runs)\/[A-Za-z0-9_-][A-Za-z0-9_./-]*\.json(\.gz)?$/;
	const SCENE_PATH_PATTERN = /^scenes\/[A-Za-z0-9_.-]+\.json(\.gz)?$/;
	// Live: consecutive failed spectate polls before the viewer surfaces the
	// run list as the fallback (250 ms cadence, so 4 is about one second).
	const LIVE_FAILURES_BEFORE_FALLBACK = 4;
	const FOLLOW_BEST_HYSTERESIS_M = 5;
	// Live tail playback. The playhead runs at 1x in trainer-decision space
	// and sits LIVE_TAIL_FRAMES behind the newest frame so a jittery poll
	// never starves it. It only jumps when the buffer overflows (trainer
	// faster than real time: LIVE_MAX_LAG_FRAMES behind) or runs dry
	// (trainer slower, or stalled); both are said in the HUD.
	const LIVE_TAIL_FRAMES = 6;
	const LIVE_MAX_LAG_FRAMES = 60;
	const LIVE_BUFFER_FRAMES = 512;
	const LIVE_STALL_MS = 2000;
	const ISO_OFFSET_PATTERN = /(Z|[+-]\d\d:?\d\d)$/;

	const viewport = document.getElementById("viewport");
	const status = document.getElementById("scene-status");
	const errorBox = document.getElementById("error");
	const viewerModeSelect = document.getElementById("viewer-mode");
	const sceneSelect = document.getElementById("scene-select");
	const ghostSelect = document.getElementById("ghost-select");
	const visualModeSelect = document.getElementById("visual-mode");
	const overlayToggle = document.getElementById("overlay-toggle");
	const liveUrlInput = document.getElementById("live-url");
	const liveConnectButton = document.getElementById("live-connect");
	const liveState = document.getElementById("live-state");
	const liveTrack = document.getElementById("live-track");
	const liveUpdate = document.getElementById("live-update");
	const liveFinishes = document.getElementById("live-finishes");
	const liveBest = document.getElementById("live-best");
	const liveDistance = document.getElementById("live-distance");
	const liveFollowing = document.getElementById("live-following");
	const followBestButton = document.getElementById("follow-best");
	const terminationFeed = document.getElementById("termination-feed");
	const runsState = document.getElementById("runs-state");
	const runsToggle = document.getElementById("runs-toggle");
	const runsSort = document.getElementById("runs-sort");
	const liveRunsList = document.getElementById("live-runs");
	const liveCount = document.getElementById("live-count");
	const pastRunsList = document.getElementById("past-runs");
	const compareStrip = document.getElementById("compare-strip");
	const compareName = document.getElementById("compare-name");
	const compareClear = document.getElementById("compare-clear");
	const compareCells = document.getElementById("compare-cells");
	const fileInput = document.getElementById("file-input");
	const playButton = document.getElementById("play");
	const speedSelect = document.getElementById("speed-select");
	const cameraSelect = document.getElementById("camera-select");
	const scrubber = document.getElementById("scrubber");
	const timelineMarkers = document.getElementById("timeline-markers");
	const timeReadout = document.getElementById("time-readout");
	const speedValue = document.getElementById("speed");
	const rpmValue = document.getElementById("rpm");
	const rpmFill = document.getElementById("rpm-fill");
	const gearValue = document.getElementById("gear");
	const raceTimeValue = document.getElementById("race-time");
	const tickValue = document.getElementById("tick");
	const wheelNodes = [...document.querySelectorAll(".wheel-node")];
	const minimap = document.getElementById("minimap");
	const minimapContext = minimap.getContext("2d");
	const mapDistance = document.getElementById("map-distance");
	const fpsValue = document.getElementById("fps");
	const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
	const gameTextureLoader = new THREE.TextureLoader();
	const gameTexturePromises = new Map();

	const scene = new THREE.Scene();
	scene.background = new THREE.Color(0x8ba9bd);
	scene.fog = new THREE.Fog(0x8ba9bd, 900, 2600);

	const camera = new THREE.PerspectiveCamera(54, 1, 0.1, 8000);
	camera.position.set(0, 5, -12);

	const renderer = new THREE.WebGLRenderer({
		antialias: true,
		powerPreference: "default",
	});
	renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
	renderer.setSize(window.innerWidth, window.innerHeight);
	renderer.info.autoReset = false;
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.toneMapping = THREE.NoToneMapping;
	renderer.shadowMap.enabled = true;
	renderer.shadowMap.type = THREE.PCFSoftShadowMap;
	viewport.appendChild(renderer.domElement);

	const sky = new THREE.Mesh(
		new THREE.SphereGeometry(3800, 32, 16),
		new THREE.ShaderMaterial({
			side: THREE.BackSide,
			depthWrite: false,
			fog: false,
			uniforms: {
				topColor: { value: new THREE.Color(0x739bb7) },
				horizonColor: { value: new THREE.Color(0xb9cbd4) },
				bottomColor: { value: new THREE.Color(0x81919a) },
			},
			vertexShader: `
				varying vec3 vDirection;
				void main() {
					vDirection = normalize(position);
					gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
				}
			`,
			fragmentShader: `
				uniform vec3 topColor;
				uniform vec3 horizonColor;
				uniform vec3 bottomColor;
				varying vec3 vDirection;
				void main() {
					float up = clamp(vDirection.y * 0.5 + 0.5, 0.0, 1.0);
					vec3 low = mix(bottomColor, horizonColor, smoothstep(0.30, 0.52, up));
					vec3 color = mix(low, topColor, smoothstep(0.52, 0.88, up));
					gl_FragColor = vec4(color, 1.0);
				}
			`,
		}),
	);
	sky.renderOrder = -1000;
	scene.add(sky);

	const hemisphere = new THREE.HemisphereLight(0xffffff, 0x75806f, 1.25);
	hemisphere.layers.enable(1);
	scene.add(hemisphere);
	const sun = new THREE.DirectionalLight(0xfff4df, 1.1);
	sun.layers.enable(1);
	const sunOffset = new THREE.Vector3(-42, 68, 28);
	sun.position.copy(sunOffset);
	sun.castShadow = true;
	sun.shadow.mapSize.set(1536, 1536);
	sun.shadow.camera.left = -22;
	sun.shadow.camera.right = 22;
	sun.shadow.camera.top = 22;
	sun.shadow.camera.bottom = -22;
	sun.shadow.camera.near = 4;
	sun.shadow.camera.far = 150;
	sun.shadow.bias = -0.00035;
	scene.add(sun, sun.target);

	let worldGroup = null;
	let carRoot = null;
	let collisionVisualGroup = null;
	let gameVisualGroup = null;
	let overlayVisualGroup = null;
	let stylizedCarVisual = null;
	let gameCarVisual = null;
	let stylizedWheelVisuals = [];
	let gameWheelVisuals = [];
	let wheelVisuals = [];
	let replay = null;
	let sceneData = null;
	let trailMaterial = null;
	let wheelPhases = [];
	let wheelDamperMid = [0, 0, 0, 0];
	let wheelDamperScale = [0, 0, 0, 0];
	let maximumRpm = 1;
	let playhead = 0;
	let playing = false;
	let playbackSpeed = 1;
	let currentSpeedMps = 0;
	let previousFrameTime = performance.now();
	let cameraJustChanged = true;
	let fixedFov = null;
	let beautyAngle = -0.8;
	let lastHudIndex = -1;
	let fpsFrameCount = 0;
	let fpsWindowStart = performance.now();
	let renderDirty = true;
	const renderedCameraPosition = new THREE.Vector3(Infinity, Infinity, Infinity);
	const renderedCameraQuaternion = new THREE.Quaternion();
	let renderedFov = NaN;
	let mapStatic = null;
	let mapMinimumX = 0;
	let mapMinimumZ = 0;
	let mapScale = 1;
	let mapOffsetX = 0;
	let mapOffsetY = 0;
	let viewerMode = "replay";
	let liveGeneration = 0;
	let livePollTimer = null;
	let liveStreamId = null;
	let liveMeta = null;
	// Ascending, unique-sequence frame buffer; `livePlayhead` is a float
	// sequence number into it (see LIVE_TAIL_FRAMES).
	let liveFrames = [];
	let livePlayhead = -1;
	let liveNewestSequence = -1;
	let liveNewestChangedAt = 0;
	let liveRateSamples = [];
	let liveLastPollAt = 0;
	let livePollGapMs = 0;
	let liveTailNote = "";
	let liveTailNoteUntil = 0;
	let liveDetached = false;
	let liveStopped = false;
	let liveRenderedFrame = null;
	let liveGhosts = [];
	let liveWheelPhases = [0, 0, 0, 0];
	let liveFailures = 0;
	let liveFollowedEnvId = null;
	let followBest = true;
	let terminationKeys = new Set();
	let terminationItems = [];
	// Run registry (served by tools/serve_viewer.py at /api/runs).
	let runsPayload = null;
	let runsTimer = null;
	let runsFirstPoll = null;
	let expandedRunId = null;
	let currentRun = null;
	let currentReplayScene = null;
	// Ghost comparison: a second replay drawn on the same track, synchronised
	// by race time. `ghostData` survives scene reloads on the same track;
	// `ghostVisual` is rebuilt with each world.
	let ghostData = null;
	let ghostVisual = null;
	let ghostSplits = [];
	let primaryProgress = null;
	let ghostProgress = null;
	let lastGapIndex = -1;

	const firstPosition = new THREE.Vector3();
	const secondPosition = new THREE.Vector3();
	const interpolatedPosition = new THREE.Vector3();
	const firstQuaternion = new THREE.Quaternion();
	const secondQuaternion = new THREE.Quaternion();
	const interpolatedQuaternion = new THREE.Quaternion();
	const localRight = new THREE.Vector3();
	const localUp = new THREE.Vector3();
	const basePosition = new THREE.Vector3();
	const desiredCamera = new THREE.Vector3();
	const desiredTarget = new THREE.Vector3();
	const chaseTarget = new THREE.Vector3();
	const carForward = new THREE.Vector3();
	const chaseHeading = new THREE.Vector3(0, 0, 1);
	const CAR_LAYER = 1;
	const orbitPosition = new THREE.Vector3();
	const temporaryColor = new THREE.Color();
	const coolColor = new THREE.Color(COLORS.signalBlue);
	const middleColor = new THREE.Color(COLORS.bone);
	const hotColor = new THREE.Color(COLORS.velocityOrange);
	const liveMapRecord = [];

	const orbit = {
		target: new THREE.Vector3(),
		radius: 300,
		azimuth: -0.65,
		polar: 0.96,
		dragging: false,
		x: 0,
		y: 0,
	};

	window.__TMNF_VIEWER_STATS = {
		ready: false,
		fps: 0,
		triangles: 0,
		scene: "",
		ghost: "",
		compare: null,
		notice: "",
		runs: { state: "connecting", live: 0, past: 0, errors: 0 },
		live: {
			state: "", following: null, url: "", sequence: null, raceTimeMs: null,
			lagFrames: null, resyncs: 0, waits: 0, rewinds: 0, trainerRate: null,
		},
	};

	function fail(message) {
		throw new Error(message);
	}

	function showError(error) {
		console.error(error);
		errorBox.textContent = error instanceof Error ? error.message : String(error);
		errorBox.style.display = "block";
		status.textContent = "Scene failed to load.";
		playing = false;
		playButton.textContent = "Play";
		window.__TMNF_VIEWER_STATS.ready = false;
		window.__TMNF_VIEWER_STATS.notice = errorBox.textContent;
	}

	// A notice is an error the user must read that does not invalidate the
	// loaded canvas (refused ghost, bad run.json, closed spectate port).
	function showNotice(error) {
		console.warn(error);
		errorBox.textContent = error instanceof Error ? error.message : String(error);
		errorBox.style.display = "block";
		window.__TMNF_VIEWER_STATS.notice = errorBox.textContent;
	}

	function clearNotice() {
		errorBox.style.display = "none";
		window.__TMNF_VIEWER_STATS.notice = "";
	}

	function requireArray(value, name) {
		if (!Array.isArray(value))
			fail(`${name} must be an array`);
		return value;
	}

	function validateScene(data) {
		if (data?.format !== "tmnf-c-viewer-scene")
			fail("This is not a TMNF-C viewer scene.");
		if (data.version !== 3) {
			fail(
				`Scene version ${String(data.version)} is obsolete. ` +
				"Regenerate it with the current export_viewer_scene tool. Version 3 is required.",
			);
		}
		const track = data.track;
		const route = data.route;
		const car = data.car;
		const lap = data.lap;
		if (!track || !route || !car || !lap || !data.gameVisuals)
			fail("Scene is missing track, route, car, lap, or game visuals data.");
		if (data.gameVisuals.manifest !== "assets/game/manifest.json" ||
			!/^[a-z][a-z0-9_-]*$/.test(data.gameVisuals.scene))
			fail("Scene has an invalid game visuals reference.");
		requireArray(track.positions, "track.positions");
		requireArray(track.indices, "track.indices");
		requireArray(track.groups, "track.groups");
		requireArray(track.materials, "track.materials");
		requireArray(route.centerline, "route.centerline");
		requireArray(route.checkpoints, "route.checkpoints");
		requireArray(car.wheelOffsets, "car.wheelOffsets");
		requireArray(lap.fields, "lap.fields");
		requireArray(lap.ticks, "lap.ticks");
		requireArray(lap.checkpointTicks, "lap.checkpointTicks");
		if (track.positions.length !== track.vertexCount * 3 ||
			track.indices.length !== track.triangleCount * 3)
			fail("Track array lengths do not match their declared counts.");
		if (lap.ticks.length !== lap.tickCount || lap.tickCount === 0)
			fail("Lap tick count is invalid.");
		if (lap.fields.length !== EXPECTED_FIELDS.length ||
			!lap.fields.every((field, index) => field === EXPECTED_FIELDS[index]))
			fail("Scene version 3 telemetry fields do not match the required schema.");
		if (lap.ticks.some((record) =>
			!Array.isArray(record) || record.length !== EXPECTED_FIELDS.length))
			fail("A replay tick has an invalid telemetry field count.");
		if (route.checkpoints.length !== lap.checkpointTicks.length)
			fail("Checkpoint geometry and crossing ticks do not match.");
		if (car.wheelOffsets.length !== 4)
			fail("Viewer scenes require exactly four wheel offsets.");
	}

	function disposeWorld() {
		if (!worldGroup)
			return;
		const geometries = new Set();
		const materials = new Set();
		worldGroup.traverse((object) => {
			if (object.isInstancedMesh)
				object.dispose();
			if (object.geometry)
				geometries.add(object.geometry);
			if (Array.isArray(object.material))
				object.material.forEach((material) => materials.add(material));
			else if (object.material)
				materials.add(object.material);
		});
		geometries.forEach((geometry) => geometry.dispose());
		materials.forEach((material) => material.dispose());
		scene.remove(worldGroup);
		worldGroup = null;
		carRoot = null;
		collisionVisualGroup = null;
		gameVisualGroup = null;
		overlayVisualGroup = null;
		stylizedCarVisual = null;
		gameCarVisual = null;
		stylizedWheelVisuals = [];
		gameWheelVisuals = [];
		wheelVisuals = [];
		liveGhosts = [];
		ghostVisual = null;
		trailMaterial = null;
	}

	function trackMaterialStyle(material) {
		const styles = {
			2: { color: 0x717b82, roughness: 0.7, metalness: 0.08 },
			4: { color: 0x56616a, roughness: 0.9, metalness: 0.01 },
			7: { color: 0x45525d, roughness: 0.76, metalness: 0.05 },
			9: { color: 0xb4a993, roughness: 0.82, metalness: 0.02 },
			16: { color: 0x35434e, roughness: 0.93, metalness: 0.01 },
		};
		if (styles[material.id])
			return styles[material.id];
		const lightness = 0.17 + ((material.id * 7) % 5) * 0.018;
		temporaryColor.setHSL(0.56, 0.14, lightness);
		return {
			color: temporaryColor.getHex(),
			roughness: 0.79 + ((material.id * 11) % 9) * 0.015,
			metalness: material.friction < 0.2 ? 0.15 : 0.025,
		};
	}

	function buildTrack(track) {
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute(
			"position",
			new THREE.Float32BufferAttribute(track.positions, 3),
		);
		geometry.setIndex(new THREE.Uint32BufferAttribute(track.indices, 1));
		geometry.clearGroups();
		for (const group of track.groups) {
			if (!Array.isArray(group) || group.length !== 3)
				fail("Track material group is malformed.");
			geometry.addGroup(group[0], group[1], group[2]);
		}
		geometry.computeVertexNormals();
		geometry.computeBoundingSphere();

		const materials = new Array(track.materials.length);
		for (const material of track.materials) {
			const style = trackMaterialStyle(material);
			materials[material.id] = new THREE.MeshStandardMaterial({
				name: `Collision material ${material.id}`,
				color: style.color,
				roughness: style.roughness,
				metalness: style.metalness,
				side: THREE.DoubleSide,
			});
		}
		for (let id = 0; id < materials.length; ++id) {
			if (!materials[id])
				fail("Track material IDs must form a dense zero-based array.");
		}

		const mesh = new THREE.Mesh(geometry, materials);
		mesh.name = "Merged collision world";
		mesh.receiveShadow = true;
		mesh.castShadow = false;
		return mesh;
	}

	function buildGround(bounds) {
		const minimum = bounds[0];
		const maximum = bounds[1];
		const span = Math.max(maximum[0] - minimum[0], maximum[2] - minimum[2]);
		const size = Math.max(2200, span * 4.2);
		const ground = new THREE.Mesh(
			new THREE.PlaneGeometry(size, size),
			new THREE.MeshStandardMaterial({
				color: 0x171e24,
				roughness: 1,
				metalness: 0,
			}),
		);
		ground.name = "Fog ground";
		ground.rotation.x = -Math.PI / 2;
		ground.position.set(
			(minimum[0] + maximum[0]) * 0.5,
			minimum[1] - 2.5,
			(minimum[2] + maximum[2]) * 0.5,
		);
		ground.receiveShadow = true;
		return ground;
	}

	function buildRouteReference(points) {
		if (points.length < 2)
			fail("Route centerline needs at least two points.");
		const positions = new Float32Array(points.length * 3);
		for (let i = 0; i < points.length; ++i) {
			positions[i * 3] = points[i][0];
			positions[i * 3 + 1] = points[i][1] + 0.06;
			positions[i * 3 + 2] = points[i][2];
		}
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
		const line = new THREE.Line(
			geometry,
			new THREE.LineBasicMaterial({
				color: COLORS.bone,
				transparent: true,
				opacity: 0.18,
				depthWrite: false,
			}),
		);
		line.name = "Route reference";
		line.renderOrder = 1;
		return line;
	}

	function gameQuaternion(record, output) {
		return output.set(
			record[FIELD.qy],
			record[FIELD.qz],
			record[FIELD.qw],
			record[FIELD.qx],
		).normalize();
	}

	function gameQuaternionValues(values, output) {
		return output.set(
			values[1],
			values[2],
			values[3],
			values[0],
		).normalize();
	}

	function setSpeedColor(speed, maximumSpeed, output) {
		const normalized = THREE.MathUtils.clamp(speed / Math.max(maximumSpeed, 1), 0, 1);
		if (normalized < 0.56)
			return output.copy(coolColor).lerp(middleColor, normalized / 0.56);
		return output.copy(middleColor).lerp(hotColor, (normalized - 0.56) / 0.44);
	}

	function buildRacingRibbon(lap) {
		const count = lap.tickCount;
		const positions = new Float32Array(count * 6);
		const colors = new Float32Array(count * 6);
		const progress = new Float32Array(count * 2);
		const indices = new Uint32Array((count - 1) * 6);
		let maximumSpeed = 1;
		for (const record of lap.ticks)
			maximumSpeed = Math.max(maximumSpeed, record[FIELD.speed]);

		for (let i = 0; i < count; ++i) {
			const record = lap.ticks[i];
			gameQuaternion(record, firstQuaternion);
			localRight.set(1, 0, 0).applyQuaternion(firstQuaternion).normalize();
			localUp.set(0, 1, 0).applyQuaternion(firstQuaternion).normalize();
			basePosition.set(record[FIELD.x], record[FIELD.y], record[FIELD.z]);
			basePosition.addScaledVector(localUp, -0.17);
			const width = 0.48 + 0.18 * THREE.MathUtils.clamp(
				record[FIELD.speed] / maximumSpeed,
				0,
				1,
			);
			const offset = i * 6;
			positions[offset] = basePosition.x + localRight.x * width;
			positions[offset + 1] = basePosition.y + localRight.y * width;
			positions[offset + 2] = basePosition.z + localRight.z * width;
			positions[offset + 3] = basePosition.x - localRight.x * width;
			positions[offset + 4] = basePosition.y - localRight.y * width;
			positions[offset + 5] = basePosition.z - localRight.z * width;
			setSpeedColor(record[FIELD.speed], maximumSpeed, temporaryColor);
			for (let side = 0; side < 2; ++side) {
				const colorOffset = offset + side * 3;
				colors[colorOffset] = temporaryColor.r;
				colors[colorOffset + 1] = temporaryColor.g;
				colors[colorOffset + 2] = temporaryColor.b;
				progress[i * 2 + side] = i / Math.max(count - 1, 1);
			}
			if (i + 1 < count) {
				const indexOffset = i * 6;
				const vertex = i * 2;
				indices[indexOffset] = vertex;
				indices[indexOffset + 1] = vertex + 2;
				indices[indexOffset + 2] = vertex + 1;
				indices[indexOffset + 3] = vertex + 2;
				indices[indexOffset + 4] = vertex + 3;
				indices[indexOffset + 5] = vertex + 1;
			}
		}

		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
		geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
		geometry.setAttribute("progress", new THREE.BufferAttribute(progress, 1));
		geometry.setIndex(new THREE.BufferAttribute(indices, 1));
		geometry.computeBoundingSphere();

		const group = new THREE.Group();
		group.name = "Measured speed trajectory";
		const ribbon = new THREE.Mesh(
			geometry,
			new THREE.MeshBasicMaterial({
				vertexColors: true,
				transparent: true,
				opacity: 0.82,
				side: THREE.DoubleSide,
				depthWrite: false,
				// The racing line is an instrument overlay: it must never be
				// buried by road geometry whose exact surface height varies
				// per block, so it ignores the depth buffer like the HUD does.
				depthTest: false,
			}),
		);
		ribbon.renderOrder = 3;
		group.add(ribbon);

		trailMaterial = new THREE.ShaderMaterial({
			transparent: true,
			depthWrite: false,
			depthTest: false,
			side: THREE.DoubleSide,
			blending: THREE.AdditiveBlending,
			vertexColors: true,
			uniforms: {
				uProgress: { value: 0 },
				uTrailLength: { value: Math.min(0.16, 240 / count) },
				uHot: { value: hotColor.clone() },
			},
			vertexShader: `
				attribute float progress;
				varying float vProgress;
				varying vec3 vColor;
				void main() {
					vProgress = progress;
					vColor = color;
					gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
				}
			`,
			fragmentShader: `
				uniform float uProgress;
				uniform float uTrailLength;
				uniform vec3 uHot;
				varying float vProgress;
				varying vec3 vColor;
				void main() {
					float distanceFromHead = uProgress - vProgress;
					if (distanceFromHead < -0.001 || distanceFromHead > uTrailLength)
						discard;
					float tail = 1.0 - smoothstep(0.0, uTrailLength, distanceFromHead);
					float head = 1.0 - smoothstep(0.0, 0.018, distanceFromHead);
					vec3 color = mix(vColor, uHot, head * 0.78);
					gl_FragColor = vec4(color, 0.16 + tail * 0.78);
				}
			`,
		});
		const liveTrail = new THREE.Mesh(geometry, trailMaterial);
		liveTrail.renderOrder = 4;
		group.add(liveTrail);
		return group;
	}

	function transformFromIso(transform) {
		const r = transform.rotation;
		const t = transform.translation;
		const matrix = new THREE.Matrix4();
		matrix.set(
			r[0], r[1], r[2], t[0],
			r[3], r[4], r[5], t[1],
			r[6], r[7], r[8], t[2],
			0, 0, 0, 1,
		);
		return matrix;
	}

	function addGateFrame(root, center, half, material, depthOffset, doubled) {
		const barThickness = Math.max(0.075, Math.min(half[0], half[1]) * 0.035);
		const depth = Math.max(0.08, Math.min(half[2] * 0.22, 0.2));
		const z = center[2] + depthOffset;
		const verticalGeometry = new THREE.BoxGeometry(
			barThickness,
			half[1] * 2 + barThickness,
			depth,
		);
		const horizontalGeometry = new THREE.BoxGeometry(
			half[0] * 2 + barThickness,
			barThickness,
			depth,
		);
		for (const x of [center[0] - half[0], center[0] + half[0]]) {
			const beam = new THREE.Mesh(verticalGeometry, material);
			beam.position.set(x, center[1], z);
			root.add(beam);
		}
		for (const y of [center[1] - half[1], center[1] + half[1]]) {
			const beam = new THREE.Mesh(horizontalGeometry, material);
			beam.position.set(center[0], y, z);
			root.add(beam);
		}
		if (doubled)
			return;
		const fill = new THREE.Mesh(
			new THREE.PlaneGeometry(half[0] * 2, half[1] * 2),
			new THREE.MeshBasicMaterial({
				color: material.color,
				transparent: true,
				opacity: 0.045,
				side: THREE.DoubleSide,
				depthWrite: false,
			}),
		);
		fill.position.set(center[0], center[1], z);
		root.add(fill);
	}

	function buildTrigger(trigger, finish, name) {
		const root = new THREE.Group();
		root.name = name;
		const transform = transformFromIso(trigger.transform);
		root.position.setFromMatrixPosition(transform);
		root.quaternion.setFromRotationMatrix(transform);
		const material = new THREE.MeshStandardMaterial({
			color: finish ? COLORS.velocityOrange : COLORS.bone,
			emissive: finish ? 0x4a1008 : 0x171819,
			emissiveIntensity: finish ? 1.5 : 0.45,
			roughness: 0.34,
			metalness: 0.58,
			transparent: true,
			opacity: finish ? 0.92 : 0.64,
		});
		addGateFrame(root, trigger.box.center, trigger.box.halfExtent, material, 0, finish);
		if (finish) {
			const separation = Math.min(0.35, trigger.box.halfExtent[2] * 0.34);
			addGateFrame(
				root,
				trigger.box.center,
				trigger.box.halfExtent,
				material,
				-separation,
				true,
			);
			addGateFrame(
				root,
				trigger.box.center,
				trigger.box.halfExtent,
				material,
				separation,
				true,
			);
		}
		return root;
	}

	function createLoftGeometry(sections, radialSegments) {
		const ringVertexCount = sections.length * radialSegments;
		const positions = new Float32Array((ringVertexCount + 2) * 3);
		const indices = [];
		for (let sectionIndex = 0; sectionIndex < sections.length; ++sectionIndex) {
			const section = sections[sectionIndex];
			for (let segment = 0; segment < radialSegments; ++segment) {
				const angle = segment / radialSegments * Math.PI * 2;
				const offset = (sectionIndex * radialSegments + segment) * 3;
				positions[offset] = Math.cos(angle) * section.width;
				positions[offset + 1] = section.centerY + Math.sin(angle) * section.height;
				positions[offset + 2] = section.z;
			}
		}
		for (let section = 0; section + 1 < sections.length; ++section) {
			for (let segment = 0; segment < radialSegments; ++segment) {
				const nextSegment = (segment + 1) % radialSegments;
				const a = section * radialSegments + segment;
				const b = section * radialSegments + nextSegment;
				const c = (section + 1) * radialSegments + segment;
				const d = (section + 1) * radialSegments + nextSegment;
				indices.push(a, c, b, b, c, d);
			}
		}
		const rearCenter = ringVertexCount;
		const frontCenter = ringVertexCount + 1;
		const rear = sections[0];
		const front = sections[sections.length - 1];
		positions[rearCenter * 3 + 1] = rear.centerY;
		positions[rearCenter * 3 + 2] = rear.z;
		positions[frontCenter * 3 + 1] = front.centerY;
		positions[frontCenter * 3 + 2] = front.z;
		for (let segment = 0; segment < radialSegments; ++segment) {
			const next = (segment + 1) % radialSegments;
			indices.push(rearCenter, next, segment);
			const frontBase = (sections.length - 1) * radialSegments;
			indices.push(frontCenter, frontBase + segment, frontBase + next);
		}
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
		geometry.setIndex(indices);
		geometry.computeVertexNormals();
		return geometry;
	}

	function buildCar(car) {
		const root = new THREE.Group();
		root.name = "Stadium silhouette";

		const paint = new THREE.MeshPhysicalMaterial({
			color: 0xaab2b5,
			roughness: 0.28,
			metalness: 0.42,
			clearcoat: 0.92,
			clearcoatRoughness: 0.17,
		});
		const darkMetal = new THREE.MeshStandardMaterial({
			color: 0x151b20,
			roughness: 0.42,
			metalness: 0.62,
		});
		const glass = new THREE.MeshPhysicalMaterial({
			color: 0x172a37,
			roughness: 0.12,
			metalness: 0.22,
			transparent: true,
			opacity: 0.82,
			clearcoat: 1,
			clearcoatRoughness: 0.08,
		});
		const signal = new THREE.MeshStandardMaterial({
			color: COLORS.velocityOrange,
			emissive: 0x7a1808,
			emissiveIntensity: 1.8,
			roughness: 0.3,
			metalness: 0.2,
		});

		const body = new THREE.Mesh(
			createLoftGeometry([
				{ z: -1.78, width: 0.7, centerY: 0.5, height: 0.31 },
				{ z: -1.42, width: 0.95, centerY: 0.5, height: 0.37 },
				{ z: -0.72, width: 1.02, centerY: 0.5, height: 0.38 },
				{ z: 0.2, width: 1.01, centerY: 0.48, height: 0.35 },
				{ z: 1.15, width: 0.91, centerY: 0.43, height: 0.3 },
				{ z: 1.82, width: 0.66, centerY: 0.38, height: 0.23 },
				{ z: 2.16, width: 0.22, centerY: 0.37, height: 0.13 },
			], 16),
			paint,
		);
		body.name = "Clearcoat body shell";
		root.add(body);

		const canopy = new THREE.Mesh(
			createLoftGeometry([
				{ z: -0.78, width: 0.48, centerY: 0.81, height: 0.12 },
				{ z: -0.43, width: 0.65, centerY: 0.91, height: 0.28 },
				{ z: 0.18, width: 0.64, centerY: 0.98, height: 0.31 },
				{ z: 0.72, width: 0.53, centerY: 0.9, height: 0.24 },
				{ z: 1.03, width: 0.34, centerY: 0.76, height: 0.09 },
			], 16),
			glass,
		);
		canopy.name = "Dark glass canopy";
		root.add(canopy);

		const undertray = new THREE.Mesh(
			new THREE.BoxGeometry(1.58, 0.09, 3.15),
			darkMetal,
		);
		undertray.position.set(0, 0.13, 0.02);
		root.add(undertray);

		const frontWing = new THREE.Mesh(
			new THREE.BoxGeometry(1.82, 0.07, 0.34),
			darkMetal,
		);
		frontWing.position.set(0, 0.22, 1.98);
		root.add(frontWing);
		const frontSignal = new THREE.Mesh(
			new THREE.BoxGeometry(0.78, 0.035, 0.07),
			signal,
		);
		frontSignal.position.set(0, 0.43, 2.13);
		root.add(frontSignal);

		const rearWing = new THREE.Mesh(
			new THREE.BoxGeometry(1.92, 0.09, 0.38),
			paint,
		);
		rearWing.position.set(0, 1.13, -1.57);
		rearWing.rotation.x = -0.12;
		root.add(rearWing);
		for (const x of [-0.58, 0.58]) {
			const support = new THREE.Mesh(
				new THREE.BoxGeometry(0.055, 0.58, 0.08),
				darkMetal,
			);
			support.position.set(x, 0.84, -1.48);
			root.add(support);
		}

		const tireMaterial = new THREE.MeshStandardMaterial({
			color: 0x111316,
			roughness: 0.86,
			metalness: 0.04,
		});
		const rimMaterial = new THREE.MeshStandardMaterial({
			color: 0x737e84,
			roughness: 0.28,
			metalness: 0.82,
		});
		const glowMaterial = new THREE.MeshBasicMaterial({
			color: COLORS.velocityOrange,
			transparent: true,
			opacity: 0,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
		});

		const wheelRadius = car.wheelRadii[1];
		const wheelWidth = car.wheelRadii[0] * 2;
		const tireGeometry = new THREE.CylinderGeometry(
			wheelRadius,
			wheelRadius,
			wheelWidth,
			24,
			1,
			false,
		);
		tireGeometry.rotateZ(Math.PI / 2);
		const rimGeometry = new THREE.CylinderGeometry(
			wheelRadius * 0.55,
			wheelRadius * 0.55,
			wheelWidth * 1.03,
			18,
		);
		rimGeometry.rotateZ(Math.PI / 2);
		const glowGeometry = new THREE.TorusGeometry(
			wheelRadius * 1.015,
			0.028,
			8,
			28,
		);
		glowGeometry.rotateY(Math.PI / 2);

		const visuals = car.wheelOffsets.map((offset, index) => {
			const suspensionRoot = new THREE.Group();
			suspensionRoot.position.set(offset[0], offset[1], offset[2]);
			const steerRoot = new THREE.Group();
			suspensionRoot.add(steerRoot);
			const tire = new THREE.Mesh(tireGeometry, tireMaterial);
			tire.name = `Wheel ${index + 1} tire`;
			tire.castShadow = true;
			steerRoot.add(tire);
			const rim = new THREE.Mesh(rimGeometry, rimMaterial);
			rim.castShadow = true;
			steerRoot.add(rim);
			const glow = new THREE.Mesh(glowGeometry, glowMaterial.clone());
			glow.visible = false;
			glow.renderOrder = 6;
			steerRoot.add(glow);
			root.add(suspensionRoot);
			return {
				baseY: offset[1],
				suspensionRoot,
				steerRoot,
				tire,
				rim,
				glow,
			};
		});

		const archGeometry = new THREE.TorusGeometry(
			wheelRadius + 0.065,
			0.055,
			10,
			30,
		);
		archGeometry.rotateY(Math.PI / 2);
		car.wheelOffsets.forEach((offset) => {
			const arch = new THREE.Mesh(archGeometry, paint);
			arch.position.set(
				offset[0] + Math.sign(offset[0]) * 0.025,
				offset[1] + 0.04,
				offset[2],
			);
			root.add(arch);
		});

		root.traverse((object) => {
			if (object.isMesh && object.material.blending !== THREE.AdditiveBlending) {
				object.castShadow = true;
				object.receiveShadow = true;
			}
		});
		return { root, wheelVisuals: visuals };
	}

	async function fetchJson(url, label) {
		const response = await fetch(url, { cache: "no-store" });
		if (!response.ok)
			fail(`${label} request failed with HTTP ${response.status}.`);
		return response.json();
	}

	function accessorAttribute(document, binary, accessorIndex) {
		const accessor = document.accessors[accessorIndex];
		const view = document.bufferViews[accessor.bufferView];
		if (!accessor || !view || view.buffer !== 0 || view.byteStride)
			fail("Game asset has an unsupported glTF accessor.");
		const itemSizes = { SCALAR: 1, VEC2: 2, VEC3: 3 };
		const itemSize = itemSizes[accessor.type];
		const constructors = {
			5125: Uint32Array,
			5126: Float32Array,
		};
		const Constructor = constructors[accessor.componentType];
		if (!itemSize || !Constructor)
			fail("Game asset uses an unsupported glTF component type.");
		const byteOffset = (view.byteOffset || 0) + (accessor.byteOffset || 0);
		const values = new Constructor(binary, byteOffset, accessor.count * itemSize);
		return new THREE.BufferAttribute(values, itemSize);
	}

	async function loadGameAsset(url, { lit = false } = {}) {
		const document = await fetchJson(url, "Game asset");
		if (document?.asset?.version !== "2.0" ||
			document.buffers?.length !== 1 ||
			document.scenes?.length !== 1)
			fail("Game asset is not a supported glTF 2.0 file.");
		const bufferUrl = new URL(document.buffers[0].uri, url);
		const bufferResponse = await fetch(bufferUrl, { cache: "no-store" });
		if (!bufferResponse.ok)
			fail(`Game asset buffer request failed with HTTP ${bufferResponse.status}.`);
		const binary = await bufferResponse.arrayBuffer();
		if (binary.byteLength !== document.buffers[0].byteLength)
			fail("Game asset buffer length does not match glTF.");

		const textures = await Promise.all((document.textures || []).map(async (entry) => {
			const image = document.images[entry.source];
			if (!image?.uri)
				fail("Game asset texture has no image URI.");
			const imageUrl = new URL(image.uri, url).href;
			if (!gameTexturePromises.has(imageUrl)) {
				gameTexturePromises.set(
					imageUrl,
					gameTextureLoader.loadAsync(imageUrl).then((texture) => {
						texture.colorSpace = THREE.SRGBColorSpace;
						texture.flipY = false;
						texture.wrapS = THREE.RepeatWrapping;
						texture.wrapT = THREE.RepeatWrapping;
						return texture;
					}),
				);
			}
			const texture = await gameTexturePromises.get(imageUrl);
			return texture;
		}));
		// Metallic-roughness maps are data, not colour: load them linear under
		// a separate cache key so they never share an sRGB texture object.
		const linearTexture = async (index) => {
			const entry = document.textures[index];
			const image = document.images[entry.source];
			if (!image?.uri)
				fail("Game asset texture has no image URI.");
			const key = `${new URL(image.uri, url).href}#linear`;
			if (!gameTexturePromises.has(key)) {
				gameTexturePromises.set(
					key,
					gameTextureLoader.loadAsync(key.slice(0, -7)).then((texture) => {
						texture.colorSpace = THREE.NoColorSpace;
						texture.flipY = false;
						texture.wrapS = THREE.RepeatWrapping;
						texture.wrapT = THREE.RepeatWrapping;
						return texture;
					}),
				);
			}
			return gameTexturePromises.get(key);
		};
		const materialEntries = document.materials || [];
		const metallicRoughnessMaps = await Promise.all(materialEntries.map((entry) => {
			const index = entry.pbrMetallicRoughness?.metallicRoughnessTexture?.index;
			return index === undefined ? Promise.resolve(null) : linearTexture(index);
		}));
		const materials = materialEntries.map((entry, materialIndex) => {
			const pbr = entry.pbrMetallicRoughness;
			if (!pbr)
				fail("Game asset material has no PBR description.");
			const factor = pbr.baseColorFactor || [1, 1, 1, 1];
			const common = {
				name: entry.name,
				color: new THREE.Color(factor[0], factor[1], factor[2]),
				opacity: factor[3],
				transparent: entry.alphaMode === "BLEND" || factor[3] < 1,
				alphaTest: entry.alphaMode === "MASK" ? entry.alphaCutoff || 0.5 : 0,
				side: entry.doubleSided ? THREE.DoubleSide : THREE.FrontSide,
			};
			// Only the car is lit (paint, chrome, glass). Track blocks keep
			// the game's flat lightmapped look: every block glTF carries
			// explicit default PBR factors, so lighting cannot be inferred
			// from the material and is decided by the caller instead.
			const material = lit
				? new THREE.MeshStandardMaterial({
					...common,
					metalness: pbr.metallicFactor ?? 1,
					roughness: pbr.roughnessFactor ?? 1,
				})
				: new THREE.MeshBasicMaterial(common);
			if (pbr.baseColorTexture)
				material.map = textures[pbr.baseColorTexture.index];
			const metallicRoughness = metallicRoughnessMaps[materialIndex];
			if (lit && metallicRoughness) {
				material.roughnessMap = metallicRoughness;
				material.metalnessMap = metallicRoughness;
			}
			material.depthWrite = !material.transparent;
			return material;
		});
		const meshes = document.meshes.map((entry) => {
			if (entry.primitives.length !== 1 || entry.primitives[0].mode !== 4)
				fail("Game asset mesh must contain one triangle primitive.");
			const primitive = entry.primitives[0];
			const geometry = new THREE.BufferGeometry();
			geometry.setAttribute(
				"position",
				accessorAttribute(document, binary, primitive.attributes.POSITION),
			);
			if (primitive.attributes.NORMAL !== undefined) {
				geometry.setAttribute(
					"normal",
					accessorAttribute(document, binary, primitive.attributes.NORMAL),
				);
			} else {
				geometry.computeVertexNormals();
			}
			if (primitive.attributes.TEXCOORD_0 !== undefined) {
				geometry.setAttribute(
					"uv",
					accessorAttribute(document, binary, primitive.attributes.TEXCOORD_0),
				);
			}
			geometry.setIndex(accessorAttribute(document, binary, primitive.indices));
			geometry.computeBoundingSphere();
			const mesh = new THREE.Mesh(geometry, materials[primitive.material]);
			mesh.name = entry.name;
			return mesh;
		});

		const active = new Set();
		function buildNode(index) {
			if (active.has(index))
				fail("Game asset node hierarchy contains a cycle.");
			const entry = document.nodes[index];
			if (!entry)
				fail("Game asset references a missing node.");
			active.add(index);
			const node = entry.mesh === undefined
				? new THREE.Group()
				: meshes[entry.mesh];
			node.name = entry.name || "";
			if (entry.matrix) {
				node.matrix.fromArray(entry.matrix);
				node.matrix.decompose(node.position, node.quaternion, node.scale);
			}
			for (const child of entry.children || [])
				node.add(buildNode(child));
			active.delete(index);
			return node;
		}

		const root = new THREE.Group();
		root.name = url.pathname.split("/").pop();
		for (const index of document.scenes[0].nodes)
			root.add(buildNode(index));
		return root;
	}

	function findNamedNode(root, name) {
		let result = null;
		root.traverse((node) => {
			if (node.name === name) {
				if (result)
					fail(`Game car contains duplicate node ${name}.`);
				result = node;
			}
		});
		if (!result)
			fail(`Game car is missing node ${name}.`);
		return result;
	}

	async function buildGameCar(manifest, manifestUrl) {
		const root = await loadGameAsset(new URL(manifest.car.visual, manifestUrl), { lit: true });
		root.name = "StadiumCar game visual";
		const names = ["1FLWheel", "1FRWheel", "1RLWheel", "1RRWheel"];
		const visuals = names.map((name) => {
			const wheel = findNamedNode(root, name);
			const parent = wheel.parent;
			if (!parent)
				fail(`${name} has no parent.`);
			const pivot = wheel.position.clone();
			parent.remove(wheel);
			const suspensionRoot = new THREE.Group();
			suspensionRoot.name = `${name} suspension animation`;
			suspensionRoot.position.copy(pivot);
			const steerRoot = new THREE.Group();
			steerRoot.name = `${name} steering animation`;
			const rollingRoot = new THREE.Group();
			rollingRoot.name = `${name} rolling animation`;
			wheel.position.set(0, 0, 0);
			rollingRoot.add(wheel);
			steerRoot.add(rollingRoot);
			suspensionRoot.add(steerRoot);
			parent.add(suspensionRoot);
			return {
				baseY: pivot.y,
				suspensionRoot,
				steerRoot,
				rollingRoot,
			};
		});
		return { root, wheelVisuals: visuals };
	}

	async function buildGameWorld(reference) {
		const manifestUrl = new URL(reference.manifest, window.location.href);
		const manifest = await fetchJson(manifestUrl, "Game visual manifest");
		if (manifest?.format !== "tmnf-game-visual-manifest" ||
			manifest.version !== 2)
			fail("Game visual manifest version 2 is required.");
		const sceneEntry = manifest.scenes?.[reference.scene];
		if (!sceneEntry)
			fail(`Game visual manifest has no ${reference.scene} scene.`);
		const placements = await fetchJson(
			new URL(sceneEntry.url, manifestUrl),
			"Game visual placement",
		);
		if (placements?.format !== "tmnf-game-visual-scene" ||
			placements.version !== 2 ||
			!Array.isArray(placements.placements))
			fail("Game visual placement version 2 is required.");

		const keys = [...new Set(placements.placements.map(
			placement => placement.asset,
		))];
		const templates = new Map(await Promise.all(keys.map(async (key) => {
			const asset = manifest.assets?.[key];
			if (!asset?.visual)
				fail(`Game visual asset ${key} is missing.`);
			const model = await loadGameAsset(new URL(asset.visual, manifestUrl));
			return [key, model];
		})));
		const track = buildTrackInstances(templates, placements.placements);
		window.__TMNF_VIEWER_STATS.batching = track.userData.batching;
		const gameCar = await buildGameCar(manifest, manifestUrl);
		return { track, car: gameCar };
	}

	function prepareWheelTelemetry(lap) {
		wheelPhases = Array.from({ length: 4 }, () => new Float32Array(lap.tickCount));
		const minimum = [Infinity, Infinity, Infinity, Infinity];
		const maximum = [-Infinity, -Infinity, -Infinity, -Infinity];
		const dt = lap.tickMs / 1000;
		for (let tick = 0; tick < lap.tickCount; ++tick) {
			const record = lap.ticks[tick];
			for (let wheel = 0; wheel < 4; ++wheel) {
				const damper = record[FIELD.wheelDamper + wheel];
				minimum[wheel] = Math.min(minimum[wheel], damper);
				maximum[wheel] = Math.max(maximum[wheel], damper);
				if (tick > 0) {
					const previousSpeed = lap.ticks[tick - 1][FIELD.wheelSpeed + wheel];
					const speed = record[FIELD.wheelSpeed + wheel];
					wheelPhases[wheel][tick] =
						wheelPhases[wheel][tick - 1] + (previousSpeed + speed) * 0.5 * dt;
				}
			}
		}
		for (let wheel = 0; wheel < 4; ++wheel) {
			const range = maximum[wheel] - minimum[wheel];
			wheelDamperMid[wheel] = (minimum[wheel] + maximum[wheel]) * 0.5;
			wheelDamperScale[wheel] = range > 0.00001 ? 0.2 / range : 0;
		}
		maximumRpm = 1;
		for (const record of lap.ticks)
			maximumRpm = Math.max(maximumRpm, record[FIELD.rpm]);
	}

	function resetOrbit(bounds) {
		const minimum = bounds[0];
		const maximum = bounds[1];
		orbit.target.set(
			(minimum[0] + maximum[0]) * 0.5,
			(minimum[1] + maximum[1]) * 0.5,
			(minimum[2] + maximum[2]) * 0.5,
		);
		orbit.radius = Math.max(maximum[0] - minimum[0], maximum[2] - minimum[2]) * 0.72;
		orbit.radius = Math.max(orbit.radius, 50);
		orbit.azimuth = -0.65;
		orbit.polar = 0.96;
	}

	function buildTimeline(lap) {
		timelineMarkers.replaceChildren();
		const denominator = Math.max(lap.tickCount - 1, 1);
		for (const tick of lap.checkpointTicks) {
			if (tick === null)
				continue;
			const marker = document.createElement("span");
			marker.className = "timeline-marker";
			marker.style.left = `${tick / denominator * 100}%`;
			timelineMarkers.appendChild(marker);
		}
		if (lap.finishTick !== null) {
			const marker = document.createElement("span");
			marker.className = "timeline-marker finish";
			marker.style.left = `${lap.finishTick / denominator * 100}%`;
			timelineMarkers.appendChild(marker);
		}
	}

	function minimapX(worldX) {
		return mapOffsetX + (worldX - mapMinimumX) * mapScale;
	}

	function minimapY(worldZ) {
		return mapOffsetY - (worldZ - mapMinimumZ) * mapScale;
	}

	function triggerWorldCenter(trigger, output) {
		output.set(...trigger.box.center);
		return output.applyMatrix4(transformFromIso(trigger.transform));
	}

	function buildMinimap(data) {
		const width = minimap.width;
		const height = minimap.height;
		const padding = 28;
		let minimumX = Infinity;
		let maximumX = -Infinity;
		let minimumZ = Infinity;
		let maximumZ = -Infinity;
		for (const point of data.route.centerline) {
			minimumX = Math.min(minimumX, point[0]);
			maximumX = Math.max(maximumX, point[0]);
			minimumZ = Math.min(minimumZ, point[2]);
			maximumZ = Math.max(maximumZ, point[2]);
		}
		const spanX = Math.max(maximumX - minimumX, 1);
		const spanZ = Math.max(maximumZ - minimumZ, 1);
		mapScale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanZ);
		mapMinimumX = minimumX;
		mapMinimumZ = minimumZ;
		mapOffsetX = (width - spanX * mapScale) * 0.5;
		mapOffsetY = height - (height - spanZ * mapScale) * 0.5;

		mapStatic = document.createElement("canvas");
		mapStatic.width = width;
		mapStatic.height = height;
		const context = mapStatic.getContext("2d");
		context.clearRect(0, 0, width, height);
		context.lineJoin = "round";
		context.lineCap = "round";

		context.beginPath();
		data.route.centerline.forEach((point, index) => {
			const x = minimapX(point[0]);
			const y = minimapY(point[2]);
			if (index === 0)
				context.moveTo(x, y);
			else
				context.lineTo(x, y);
		});
		context.strokeStyle = "rgba(195, 183, 164, 0.16)";
		context.lineWidth = 11;
		context.stroke();

		let maximumSpeed = 1;
		for (const record of data.lap.ticks)
			maximumSpeed = Math.max(maximumSpeed, record[FIELD.speed]);
		context.lineWidth = 4;
		for (let tick = 1; tick < data.lap.tickCount; ++tick) {
			const previous = data.lap.ticks[tick - 1];
			const record = data.lap.ticks[tick];
			setSpeedColor(record[FIELD.speed], maximumSpeed, temporaryColor);
			context.beginPath();
			context.moveTo(minimapX(previous[FIELD.x]), minimapY(previous[FIELD.z]));
			context.lineTo(minimapX(record[FIELD.x]), minimapY(record[FIELD.z]));
			context.strokeStyle = `#${temporaryColor.getHexString()}`;
			context.stroke();
		}

		if (ghostData && ghostData.data.gameVisuals.scene === data.gameVisuals.scene) {
			const ghostLap = ghostData.data.lap;
			context.beginPath();
			for (let tick = 0; tick < ghostLap.tickCount; ++tick) {
				const record = ghostLap.ticks[tick];
				const x = minimapX(record[FIELD.x]);
				const y = minimapY(record[FIELD.z]);
				if (tick === 0)
					context.moveTo(x, y);
				else
					context.lineTo(x, y);
			}
			context.strokeStyle = "rgba(82, 199, 255, 0.85)";
			context.lineWidth = 1.5;
			context.stroke();
		}

		for (const checkpoint of data.route.checkpoints) {
			triggerWorldCenter(checkpoint, basePosition);
			context.fillStyle = "#c3b7a4";
			context.fillRect(minimapX(basePosition.x) - 2, minimapY(basePosition.z) - 2, 4, 4);
		}
		triggerWorldCenter(data.route.finish, basePosition);
		context.fillStyle = "#ff5a36";
		context.fillRect(minimapX(basePosition.x) - 3, minimapY(basePosition.z) - 3, 6, 6);
		mapDistance.textContent = `${Math.round(data.route.length).toLocaleString()} M`;
	}

	function drawMinimap(record, quaternion) {
		if (!mapStatic)
			return;
		minimapContext.clearRect(0, 0, minimap.width, minimap.height);
		minimapContext.drawImage(mapStatic, 0, 0);
		carForward.set(0, 0, 1).applyQuaternion(quaternion);
		const angle = Math.atan2(carForward.x, carForward.z);
		const x = minimapX(record[FIELD.x]);
		const y = minimapY(record[FIELD.z]);
		minimapContext.save();
		minimapContext.translate(x, y);
		minimapContext.rotate(angle);
		minimapContext.beginPath();
		minimapContext.moveTo(0, -8);
		minimapContext.lineTo(5, 6);
		minimapContext.lineTo(-5, 6);
		minimapContext.closePath();
		minimapContext.fillStyle = "#edf1f2";
		minimapContext.shadowColor = "#52c7ff";
		minimapContext.shadowBlur = 11;
		minimapContext.fill();
		minimapContext.restore();
	}

	function applyVisualMode() {
		if (!collisionVisualGroup || !gameVisualGroup ||
			!stylizedCarVisual || !gameCarVisual)
			return;
		const game = visualModeSelect.value === "game";
		gameVisualGroup.visible = game;
		collisionVisualGroup.visible = !game;
		gameCarVisual.visible = game;
		stylizedCarVisual.visible = !game;
		wheelVisuals = game ? gameWheelVisuals : stylizedWheelVisuals;
		lastHudIndex = -1;
		if (viewerMode === "live")
			updateLiveVisuals(0);
		else
			updateReplayVisuals();
	}

	function applyOverlayVisibility() {
		renderDirty = true;
		const hidden = document.body.classList.contains("overlays-hidden");
		if (overlayVisualGroup)
			overlayVisualGroup.visible = !hidden;
		overlayToggle.textContent = hidden ? "Show overlays" : "Hide overlays";
	}

	async function installScene(data, sourceName) {
		validateScene(data);
		status.textContent = `Loading game visuals for ${sourceName}…`;
		const gameWorld = await buildGameWorld(data.gameVisuals);
		disposeWorld();
		sceneData = data;
		worldGroup = new THREE.Group();
		worldGroup.name = "TMNF-C replay scene";
		collisionVisualGroup = new THREE.Group();
		collisionVisualGroup.name = "Stylized collision geometry";
		collisionVisualGroup.add(buildGround(data.track.bounds));
		collisionVisualGroup.add(buildTrack(data.track));
		gameVisualGroup = gameWorld.track;
		overlayVisualGroup = new THREE.Group();
		overlayVisualGroup.name = "Telemetry overlays";
		overlayVisualGroup.add(buildRouteReference(data.route.centerline));
		overlayVisualGroup.add(buildRacingRibbon(data.lap));
		data.route.checkpoints.forEach((trigger, index) => {
			overlayVisualGroup.add(
				buildTrigger(trigger, false, `Checkpoint ${index + 1}`),
			);
		});
		overlayVisualGroup.add(buildTrigger(data.route.finish, true, "Finish"));
		const stylizedCar = buildCar(data.car);
		stylizedCarVisual = stylizedCar.root;
		stylizedWheelVisuals = stylizedCar.wheelVisuals;
		gameCarVisual = gameWorld.car.root;
		gameWheelVisuals = gameWorld.car.wheelVisuals;
		carRoot = new THREE.Group();
		carRoot.name = "Animated replay car";
		carRoot.add(stylizedCarVisual, gameCarVisual);
		// The racing-line overlay skips the depth test, so the car must be
		// drawn after it to occlude it; the car itself still depth-tests
		// against the world.
		// The car lives on render layer 1 and is drawn in a second pass over
		// the world (depth buffer kept), so it always covers the racing-line
		// overlay while the world still occludes it normally.
		carRoot.traverse((node) => {
			node.layers.set(CAR_LAYER);
		});
		worldGroup.add(
			collisionVisualGroup,
			gameVisualGroup,
			overlayVisualGroup,
			carRoot,
		);
		scene.add(worldGroup);

		replay = data.lap;
		prepareWheelTelemetry(replay);
		buildTimeline(replay);
		buildMinimap(data);
		resetOrbit(data.track.bounds);
		playhead = 0;
		playing = false;
		playButton.textContent = "Play";
		scrubber.max = String(replay.tickCount - 1);
		scrubber.value = "0";
		scrubber.style.setProperty("--progress", "0%");
		cameraJustChanged = true;
		lastHudIndex = -1;
		clearNotice();
		attachGhost();
		applyVisualMode();
		applyOverlayVisibility();
		const finish = replay.finishTimeMs > 0
			? `finish ${formatTime(replay.finishTimeMs)}`
			: "finish not reached";
		status.textContent =
			`${sourceName} · ${data.track.triangleCount.toLocaleString()} triangles · ` +
			`${replay.tickCount.toLocaleString()} ticks · ${finish}`;
		updateReplayVisuals();
		window.__TMNF_VIEWER_STATS.ready = true;
		window.__TMNF_VIEWER_STATS.scene = sourceName;
	}

	// --- Ghost comparison -------------------------------------------------

	// Cumulative arc length at every centerline point (metres, continuous).
	function centerlineArcs(centerline) {
		const arcs = new Float64Array(centerline.length);
		for (let i = 1; i < centerline.length; ++i) {
			const a = centerline[i - 1];
			const b = centerline[i];
			arcs[i] = arcs[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
		}
		return arcs;
	}

	// Route progress per tick as continuous arc length, computed the way the
	// race layer does (src/race.c project_dense): project onto centerline
	// segments in a window of +/-PROJECTION_WINDOW_SEGMENTS around the
	// previous segment, and only re-acquire with a full search when the car
	// moved more than TELEPORT_DISTANCE_M in one tick (respawn). No running
	// max and no distance-based full scan: a car 40 m off the road keeps
	// following its own branch instead of latching onto a far part of a
	// route that passes back nearby.
	const PROJECTION_WINDOW_SEGMENTS = 32;
	const TELEPORT_DISTANCE_M = 32;
	function computeProgress(lap, route) {
		const centerline = route.centerline;
		const arcs = centerlineArcs(centerline);
		const segmentCount = centerline.length - 1;
		const progress = new Float64Array(lap.tickCount);
		let cursor = 0;
		let previousX = 0;
		let previousY = 0;
		let previousZ = 0;
		for (let tick = 0; tick < lap.tickCount; ++tick) {
			const record = lap.ticks[tick];
			const x = record[FIELD.x];
			const y = record[FIELD.y];
			const z = record[FIELD.z];
			const moved = tick === 0
				? Infinity
				: (x - previousX) ** 2 + (y - previousY) ** 2 + (z - previousZ) ** 2;
			const full = moved > TELEPORT_DISTANCE_M * TELEPORT_DISTANCE_M;
			const lower = full ? 0 : Math.max(0, cursor - PROJECTION_WINDOW_SEGMENTS);
			const upper = full
				? segmentCount - 1
				: Math.min(segmentCount - 1, cursor + PROJECTION_WINDOW_SEGMENTS);
			let bestDistance = Infinity;
			let bestArc = 0;
			for (let i = lower; i <= upper; ++i) {
				const a = centerline[i];
				const b = centerline[i + 1];
				const dx = b[0] - a[0];
				const dy = b[1] - a[1];
				const dz = b[2] - a[2];
				const px = x - a[0];
				const py = y - a[1];
				const pz = z - a[2];
				const lengthSquared = dx * dx + dy * dy + dz * dz;
				const t = lengthSquared > 0
					? THREE.MathUtils.clamp((px * dx + py * dy + pz * dz) / lengthSquared, 0, 1)
					: 0;
				const ox = px - t * dx;
				const oy = py - t * dy;
				const oz = pz - t * dz;
				const distance = ox * ox + oy * oy + oz * oz;
				if (distance < bestDistance) {
					bestDistance = distance;
					bestArc = arcs[i] + t * (arcs[i + 1] - arcs[i]);
					cursor = i;
				}
			}
			progress[tick] = bestArc;
			previousX = x;
			previousY = y;
			previousZ = z;
		}
		return progress;
	}

	// Race time of a tick with the exporter's post-finish freeze undone: from
	// the finish tick on, raceTimeMs repeats finishTimeMs while the car
	// coasts, so count ticks instead (the finish tick itself already repeats
	// the previous tick's time).
	function unfrozenTime(lap, tick) {
		const frozen = lap.finishTick !== null && tick >= lap.finishTick
			? tick - lap.finishTick + 1
			: 0;
		return lap.ticks[tick][FIELD.time] + frozen * lap.tickMs;
	}

	// Race time at which the ghost was at arc length `target`, interpolated
	// along its arc-vs-time curve. The curve is not monotonic (spins,
	// respawns, off-road excursions), so when the ghost crossed `target`
	// more than once the crossing nearest to `nearTime` is used: that is
	// the passage a car at `nearTime` is actually racing against, and it
	// makes a lap compared with itself read exactly zero everywhere.
	function ghostTimeAtArc(progress, lap, target, nearTime) {
		let best = null;
		for (let i = 0; i + 1 < progress.length; ++i) {
			const a = progress[i];
			const b = progress[i + 1];
			if ((target < a && target < b) || (target > a && target > b))
				continue;
			const timeA = unfrozenTime(lap, i);
			const timeB = unfrozenTime(lap, i + 1);
			// On a plateau (stationary car, or past the end of the centerline
			// where the projection saturates) the ghost held this arc for the
			// whole interval; the instant nearest the car's time is the one
			// to race against.
			const time = a === b
				? THREE.MathUtils.clamp(nearTime, timeA, timeB)
				: timeA + (target - a) / (b - a) * (timeB - timeA);
			if (best === null || Math.abs(time - nearTime) < Math.abs(best - nearTime))
				best = time;
		}
		return best;
	}

	// Largest per-tick change of arc length: a latch onto the wrong part of
	// the route shows up here as a jump of hundreds of metres.
	function largestProgressStep(progress) {
		let largest = 0;
		for (let i = 1; i < progress.length; ++i)
			largest = Math.max(largest, Math.abs(progress[i] - progress[i - 1]));
		return largest;
	}

	// Route identity beyond the track id: two scenes of the same track can
	// carry different route files (re-ordered or dropped checkpoints), and
	// pairing their splits by index would be silently wrong.
	function routeSignature(data) {
		const route = data.route;
		const gate = entry => (entry?.transform?.translation || []).map(v => Number(v).toFixed(2)).join(",");
		return [
			`points=${route.centerline.length}`,
			`length=${Number(route.length).toFixed(1)}`,
			`checkpoints=${route.checkpoints.map(gate).join(";")}`,
			`finish=${gate(route.finish)}`,
		].join(" ");
	}

	function describeRouteDifference(a, b) {
		if (a.route.checkpoints.length !== b.route.checkpoints.length)
			return `${a.route.checkpoints.length} vs ${b.route.checkpoints.length} checkpoints`;
		if (a.route.centerline.length !== b.route.centerline.length ||
			Math.abs(a.route.length - b.route.length) > 0.05)
			return `route length ${a.route.length.toFixed(1)} m vs ${b.route.length.toFixed(1)} m`;
		for (let i = 0; i < a.route.checkpoints.length; ++i) {
			const ta = a.route.checkpoints[i].transform?.translation || [];
			const tb = b.route.checkpoints[i].transform?.translation || [];
			if (ta.some((v, k) => Math.abs(v - tb[k]) > 0.01))
				return `checkpoint ${i + 1} sits at (${ta.join(", ")}) vs (${tb.join(", ")})`;
		}
		return "finish gate position differs";
	}

	function tickTime(lap, tick) {
		return tick === null || tick === undefined || tick < 0
			? null
			: lap.ticks[Math.min(tick, lap.tickCount - 1)][FIELD.time];
	}

	function buildGhostPath(lap) {
		const positions = new Float32Array(lap.tickCount * 3);
		for (let i = 0; i < lap.tickCount; ++i) {
			const record = lap.ticks[i];
			positions[i * 3] = record[FIELD.x];
			positions[i * 3 + 1] = record[FIELD.y] + 0.05;
			positions[i * 3 + 2] = record[FIELD.z];
		}
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
		const line = new THREE.Line(
			geometry,
			new THREE.LineBasicMaterial({
				color: COLORS.signalBlue,
				transparent: true,
				opacity: 0.55,
				depthWrite: false,
				depthTest: false,
			}),
		);
		line.name = "Ghost trajectory";
		line.renderOrder = 2;
		return line;
	}

	function formatDelta(milliseconds) {
		const sign = milliseconds > 0 ? "+" : milliseconds < 0 ? "−" : "±";
		return `${sign}${(Math.abs(milliseconds) / 1000).toFixed(3)}`;
	}

	function deltaClass(milliseconds) {
		if (milliseconds === null)
			return "none";
		return milliseconds > 0 ? "loss" : milliseconds < 0 ? "gain" : "none";
	}

	function buildSplits() {
		const primary = sceneData.lap;
		const ghost = ghostData.data.lap;
		const rows = [];
		primary.checkpointTicks.forEach((tick, index) => {
			rows.push({
				label: `CP ${index + 1}`,
				primaryTick: tick,
				primaryMs: tickTime(primary, tick),
				ghostMs: tickTime(ghost, ghost.checkpointTicks[index] ?? null),
			});
		});
		rows.push({
			label: "Finish",
			primaryTick: primary.finishTick,
			primaryMs: tickTime(primary, primary.finishTick),
			ghostMs: tickTime(ghost, ghost.finishTick),
		});
		for (const row of rows) {
			row.deltaMs = row.primaryMs !== null && row.ghostMs !== null
				? row.primaryMs - row.ghostMs
				: null;
		}
		return rows;
	}

	function renderCompareStrip() {
		if (!ghostData) {
			compareStrip.hidden = true;
			compareCells.replaceChildren();
			return;
		}
		compareStrip.hidden = false;
		compareName.textContent = ghostData.name;
		const gapCell = document.createElement("div");
		gapCell.className = "compare-cell gap";
		gapCell.id = "compare-gap";
		gapCell.append(
			labelled("cell-label", "Gap now (car − ghost)"),
			labelled("cell-delta none", "--"),
			labelled("cell-times", "same route position"),
		);
		const cells = [gapCell];
		for (const row of ghostSplits) {
			const cell = document.createElement("div");
			cell.className = "compare-cell";
			cell.dataset.primaryTick = row.primaryTick === null ? "" : String(row.primaryTick);
			const delta = row.deltaMs === null ? "--" : formatDelta(row.deltaMs);
			cell.append(
				labelled("cell-label", row.label),
				labelled(`cell-delta ${deltaClass(row.deltaMs)}`, delta),
				labelled(
					"cell-times",
					`${row.primaryMs === null ? "--" : formatTime(row.primaryMs)} · ` +
					`${row.ghostMs === null ? "--" : formatTime(row.ghostMs)}`,
				),
			);
			cells.push(cell);
		}
		compareCells.replaceChildren(...cells);
	}

	function labelled(className, text) {
		const element = document.createElement("div");
		element.className = className;
		element.textContent = text;
		return element;
	}

	// Why a ghost cannot be compared with the loaded replay, or null.
	function ghostRefusal(data, name) {
		if (data.gameVisuals.scene !== sceneData.gameVisuals.scene) {
			return `Ghost "${name}" is on track ${data.gameVisuals.scene} but the loaded ` +
				`replay is on track ${sceneData.gameVisuals.scene}. Comparison refused.`;
		}
		if (routeSignature(data) !== routeSignature(sceneData)) {
			return `Ghost "${name}" is on track ${data.gameVisuals.scene} like the loaded ` +
				`replay, but its route differs (${describeRouteDifference(data, sceneData)}); ` +
				"its splits would pair the wrong gates. Comparison refused.";
		}
		return null;
	}

	function attachGhost() {
		if (!ghostData)
			return;
		const refusal = ghostRefusal(ghostData.data, ghostData.name);
		if (refusal) {
			ghostData = null;
			ghostSelect.value = "";
			renderCompareStrip();
			setQuery({ ghost: null });
			window.__TMNF_VIEWER_STATS.ghost = "";
			window.__TMNF_VIEWER_STATS.compare = null;
			showNotice(`${refusal} The ghost was cleared.`);
			return;
		}
		const root = stylizedCarVisual.clone(true);
		root.name = "Ghost comparison car";
		root.visible = true;
		root.traverse((object) => {
			if (!object.material)
				return;
			object.material = object.material.clone();
			object.material.transparent = true;
			object.material.opacity = 0.38;
			object.material.depthWrite = false;
			if (object.material.color)
				object.material.color.lerp(coolColor, 0.7);
			if (object.material.emissive)
				object.material.emissive.set(0x0a2a3a);
		});
		root.traverse((node) => node.layers.set(CAR_LAYER));
		worldGroup.add(root);
		const path = buildGhostPath(ghostData.data.lap);
		overlayVisualGroup.add(path);
		ghostVisual = { root, path };
		primaryProgress = computeProgress(sceneData.lap, sceneData.route);
		ghostProgress = computeProgress(ghostData.data.lap, sceneData.route);
		ghostSplits = buildSplits();
		lastGapIndex = -1;
		renderCompareStrip();
		window.__TMNF_VIEWER_STATS.ghost = ghostData.name;
		window.__TMNF_VIEWER_STATS.compare = {
			tick: -1,
			gapMs: null,
			arcM: null,
			ghostDistanceM: null,
			primaryLargestStepM: largestProgressStep(primaryProgress),
			primaryArcEndM: primaryProgress[primaryProgress.length - 1],
			ghostLargestStepM: largestProgressStep(ghostProgress),
		};
	}

	function detachGhost() {
		renderDirty = true;
		if (ghostVisual) {
			worldGroup.remove(ghostVisual.root);
			overlayVisualGroup.remove(ghostVisual.path);
			ghostVisual.path.geometry.dispose();
			ghostVisual.path.material.dispose();
			ghostVisual.root.traverse((object) => {
				if (object.material)
					object.material.dispose();
			});
		}
		ghostVisual = null;
		ghostData = null;
		ghostSplits = [];
		primaryProgress = null;
		ghostProgress = null;
		renderCompareStrip();
		window.__TMNF_VIEWER_STATS.ghost = "";
		window.__TMNF_VIEWER_STATS.compare = null;
		if (sceneData)
			buildMinimap(sceneData);
	}

	async function loadGhost(url, name, query) {
		try {
			if (!sceneData)
				fail("Load a replay before adding a ghost.");
			if (viewerMode === "live")
				fail("Ghost comparison works on replays, not on live streams.");
			const previousStatus = status.textContent;
			status.textContent = `Loading ghost ${name}…`;
			// A run replay must be one the run actually lists, like replay=.
			const runMatch = url.match(/^runs\/([^/]+)\/(replays\/[^/]+)$/);
			if (runMatch) {
				const run = await fetchRun(runMatch[1]);
				if (!run.replays.some(entry => entry.scene === runMatch[2]))
					fail(`Run ${run.run_id} lists no replay ${runMatch[2]}; ghost refused.`);
			}
			const data = await fetchScene(url);
			validateScene(data);
			const refusal = ghostRefusal(data, name);
			if (refusal)
				fail(refusal);
			detachGhost();
			ghostData = { data, name, url };
			attachGhost();
			// Keep the ghost menu truthful: select the matching committed
			// entry, or add one for a run replay.
			ghostSelect.querySelector("option[data-dynamic]")?.remove();
			if (![...ghostSelect.options].some(entry => entry.value === url)) {
				const entry = document.createElement("option");
				entry.value = url;
				entry.textContent = `Ghost · ${name}`;
				entry.dataset.dynamic = "1";
				ghostSelect.append(entry);
			}
			ghostSelect.value = ghostData ? url : "";
			buildMinimap(sceneData);
			lastHudIndex = -1;
			updateReplayVisuals();
			status.textContent = previousStatus;
			setQuery({ ghost: query });
			renderRuns();
		} catch (error) {
			ghostSelect.value = "";
			status.textContent = "Ghost not loaded.";
			showNotice(error);
		}
	}

	function updateGhost(firstIndex) {
		if (!ghostVisual)
			return;
		const ghostLap = ghostData.data.lap;
		const ghostStart = ghostLap.ticks[0][FIELD.time];
		// Synchronise by race time with the post-finish freeze undone so the
		// ghost keeps coasting alongside the car after the line.
		const carTime = unfrozenTime(sceneData.lap, firstIndex);
		const ghostPlayhead = THREE.MathUtils.clamp(
			(carTime - ghostStart) / ghostLap.tickMs + (playhead - firstIndex),
			0,
			ghostLap.tickCount - 1,
		);
		const gFirst = Math.floor(ghostPlayhead);
		const gSecond = Math.min(gFirst + 1, ghostLap.tickCount - 1);
		const gAlpha = ghostPlayhead - gFirst;
		const ga = ghostLap.ticks[gFirst];
		const gb = ghostLap.ticks[gSecond];
		firstPosition.set(ga[FIELD.x], ga[FIELD.y], ga[FIELD.z]);
		secondPosition.set(gb[FIELD.x], gb[FIELD.y], gb[FIELD.z]);
		ghostVisual.root.position.lerpVectors(firstPosition, secondPosition, gAlpha);
		gameQuaternion(ga, firstQuaternion);
		gameQuaternion(gb, secondQuaternion);
		ghostVisual.root.quaternion.copy(firstQuaternion).slerp(secondQuaternion, gAlpha);

		if (lastGapIndex === firstIndex)
			return;
		lastGapIndex = firstIndex;
		const gapCell = document.getElementById("compare-gap");
		if (!gapCell)
			return;
		const target = primaryProgress[firstIndex];
		const ghostTime = ghostTimeAtArc(ghostProgress, ghostLap, target, carTime);
		const deltaNode = gapCell.children[1];
		const timesNode = gapCell.children[2];
		const compareStats = window.__TMNF_VIEWER_STATS.compare;
		compareStats.tick = firstIndex;
		compareStats.arcM = target;
		compareStats.ghostDistanceM = ghostVisual.root.position.distanceTo(interpolatedPosition);
		if (ghostTime === null) {
			deltaNode.textContent = "ghost never here";
			deltaNode.className = "cell-delta none";
			timesNode.textContent = `${Math.round(target)} m along route`;
			compareStats.gapMs = null;
		} else {
			const gap = carTime - ghostTime;
			deltaNode.textContent = `${formatDelta(gap)} s`;
			deltaNode.className = `cell-delta ${deltaClass(gap)}`;
			timesNode.textContent = `${Math.round(target)} m along route`;
			compareStats.gapMs = gap;
		}
		for (const cell of compareCells.children) {
			if (cell.dataset.primaryTick === undefined || cell.dataset.primaryTick === "")
				continue;
			cell.classList.toggle("passed", Number(cell.dataset.primaryTick) <= firstIndex);
		}
	}

	function formatTime(milliseconds) {
		const total = Math.max(0, Math.round(milliseconds));
		const minutes = Math.floor(total / 60000);
		const seconds = Math.floor((total % 60000) / 1000);
		const millis = total % 1000;
		return `${String(minutes).padStart(2, "0")}:` +
			`${String(seconds).padStart(2, "0")}.` +
			`${String(millis).padStart(3, "0")}`;
	}

	function updateHud(record, index) {
		const speedKmh = Math.max(0, Math.round(record[FIELD.speed] * 3.6));
		speedValue.textContent = String(speedKmh).padStart(3, "0");
		rpmValue.textContent = `${Math.round(record[FIELD.rpm]).toLocaleString()} RPM`;
		rpmFill.style.width = `${THREE.MathUtils.clamp(record[FIELD.rpm] / maximumRpm * 100, 0, 100)}%`;
		const gear = record[FIELD.gear];
		gearValue.textContent = gear < 0 ? "R" : gear === 0 ? "N" : String(gear);
		raceTimeValue.textContent = formatTime(record[FIELD.time]);
		tickValue.textContent =
			`${String(index + 1).padStart(4, "0")} / ${String(replay.tickCount).padStart(4, "0")}`;
		timeReadout.textContent = formatTime(record[FIELD.time]);
		const contactMask = record[FIELD.contact];
		const slidingMask = record[FIELD.sliding];
		for (let wheel = 0; wheel < 4; ++wheel) {
			const contact = (contactMask & (1 << wheel)) !== 0;
			const sliding = (slidingMask & (1 << wheel)) !== 0;
			wheelNodes[wheel].classList.toggle("contact", contact && !sliding);
			wheelNodes[wheel].classList.toggle("sliding", sliding);
			const glow = wheelVisuals[wheel].glow;
			if (glow) {
				glow.visible = sliding;
				glow.material.opacity = sliding ? 0.72 : 0;
			}
		}
	}

	function updateReplayVisuals() {
		renderDirty = true;
		if (!replay || !carRoot)
			return;
		const firstIndex = Math.min(Math.floor(playhead), replay.tickCount - 1);
		const secondIndex = Math.min(firstIndex + 1, replay.tickCount - 1);
		const alpha = playhead - firstIndex;
		const a = replay.ticks[firstIndex];
		const b = replay.ticks[secondIndex];
		firstPosition.set(a[FIELD.x], a[FIELD.y], a[FIELD.z]);
		secondPosition.set(b[FIELD.x], b[FIELD.y], b[FIELD.z]);
		interpolatedPosition.lerpVectors(firstPosition, secondPosition, alpha);
		gameQuaternion(a, firstQuaternion);
		gameQuaternion(b, secondQuaternion);
		interpolatedQuaternion.copy(firstQuaternion).slerp(secondQuaternion, alpha);
		carRoot.position.copy(interpolatedPosition);
		carRoot.quaternion.copy(interpolatedQuaternion);
		currentSpeedMps = THREE.MathUtils.lerp(a[FIELD.speed], b[FIELD.speed], alpha);

		for (let wheel = 0; wheel < 4; ++wheel) {
			const visual = wheelVisuals[wheel];
			visual.steerRoot.rotation.y = THREE.MathUtils.lerp(
				a[FIELD.wheelSteer + wheel],
				b[FIELD.wheelSteer + wheel],
				alpha,
			);
			const damper = THREE.MathUtils.lerp(
				a[FIELD.wheelDamper + wheel],
				b[FIELD.wheelDamper + wheel],
				alpha,
			);
			visual.suspensionRoot.position.y =
				visual.baseY + (damper - wheelDamperMid[wheel]) * wheelDamperScale[wheel];
			const wheelPhase = THREE.MathUtils.lerp(
				wheelPhases[wheel][firstIndex],
				wheelPhases[wheel][secondIndex],
				alpha,
			);
			if (visual.rollingRoot) {
				visual.rollingRoot.rotation.x = wheelPhase;
			} else {
				visual.tire.rotation.x = wheelPhase;
				visual.rim.rotation.x = wheelPhase;
			}
		}

		const progress = firstIndex / Math.max(replay.tickCount - 1, 1);
		scrubber.value = String(firstIndex);
		scrubber.style.setProperty("--progress", `${progress * 100}%`);
		trailMaterial.uniforms.uProgress.value = progress;
		if (lastHudIndex !== firstIndex) {
			updateHud(a, firstIndex);
			lastHudIndex = firstIndex;
		}
		updateGhost(firstIndex);
		drawMinimap(a, interpolatedQuaternion);
	}

	function clearLiveGhosts() {
		for (const ghost of liveGhosts) {
			ghost.traverse((object) => {
				if (object.material)
					object.material.dispose();
			});
			if (worldGroup)
				worldGroup.remove(ghost);
		}
		liveGhosts = [];
	}

	function ensureLiveGhosts(count) {
		if (liveGhosts.length === count)
			return;
		clearLiveGhosts();
		for (let index = 0; index < count; ++index) {
			const ghost = stylizedCarVisual.clone(true);
			ghost.name = `Live ghost car ${index + 1}`;
			ghost.visible = true;
			ghost.traverse((object) => {
				if (!object.material)
					return;
				object.material = object.material.clone();
				object.material.transparent = true;
				object.material.opacity = 0.28;
				object.material.depthWrite = false;
				if (object.material.color)
					object.material.color.lerp(coolColor, 0.58);
			});
			liveGhosts.push(ghost);
			worldGroup.add(ghost);
		}
	}

	function updateLiveHud(state, sequence) {
		const speedKmh = Math.max(0, Math.round(state.speedMps * 3.6));
		speedValue.textContent = String(speedKmh).padStart(3, "0");
		rpmValue.textContent = `${Math.round(state.rpm).toLocaleString()} RPM`;
		rpmFill.style.width =
			`${THREE.MathUtils.clamp(state.rpm / maximumRpm * 100, 0, 100)}%`;
		gearValue.textContent =
			state.gear < 0 ? "R" : state.gear === 0 ? "N" : String(state.gear);
		raceTimeValue.textContent = formatTime(state.raceTimeMs);
		tickValue.textContent =
			`${String(state.tick).padStart(5, "0")} · ENV ${state.envId}`;
		const lag = liveNewestSequence - sequence;
		timeReadout.textContent =
			`LIVE · seq ${sequence} · ${lag} behind` + (liveTailNote ? ` · ${liveTailNote}` : "");
		for (let wheel = 0; wheel < 4; ++wheel) {
			const telemetry = state.wheels[wheel];
			wheelNodes[wheel].classList.toggle(
				"contact", telemetry.contact && !telemetry.sliding,
			);
			wheelNodes[wheel].classList.toggle("sliding", telemetry.sliding);
			const glow = wheelVisuals[wheel].glow;
			if (glow) {
				glow.visible = telemetry.sliding;
				glow.material.opacity = telemetry.sliding ? 0.72 : 0;
			}
		}
	}

	// Index of the buffered frame with the largest sequence <= `sequence`.
	function liveFrameIndexAt(sequence) {
		let low = 0;
		let high = liveFrames.length - 1;
		while (low < high) {
			const middle = (low + high + 1) >> 1;
			if (liveFrames[middle].sequence <= sequence)
				low = middle;
			else
				high = middle - 1;
		}
		return low;
	}

	function updateLiveVisuals(deltaSeconds) {
		if (!carRoot || liveFrames.length === 0)
			return;
		const firstIndex = liveFrameIndexAt(livePlayhead);
		const secondIndex = Math.min(firstIndex + 1, liveFrames.length - 1);
		const frame = liveFrames[firstIndex];
		const nextFrame = liveFrames[secondIndex];
		const span = nextFrame.sequence - frame.sequence;
		const alpha = span > 0
			? THREE.MathUtils.clamp((livePlayhead - frame.sequence) / span, 0, 1)
			: 0;
		const followIndex = chooseFollowedEnv(frame);
		const state = frame.envs[followIndex];
		const nextState = nextFrame.envs.find(
			candidate => candidate.envId === state.envId,
		) || state;

		firstPosition.set(...state.position);
		secondPosition.set(...nextState.position);
		interpolatedPosition.lerpVectors(firstPosition, secondPosition, alpha);
		gameQuaternionValues(state.quaternion, firstQuaternion);
		gameQuaternionValues(nextState.quaternion, secondQuaternion);
		interpolatedQuaternion.copy(firstQuaternion).slerp(secondQuaternion, alpha);
		carRoot.position.copy(interpolatedPosition);
		carRoot.quaternion.copy(interpolatedQuaternion);
		currentSpeedMps = THREE.MathUtils.lerp(
			state.speedMps, nextState.speedMps, alpha,
		);

		for (let wheel = 0; wheel < 4; ++wheel) {
			const telemetry = state.wheels[wheel];
			const nextTelemetry = nextState.wheels[wheel];
			const visual = wheelVisuals[wheel];
			visual.steerRoot.rotation.y = telemetry.steer === null
				? 0
				: THREE.MathUtils.lerp(
					telemetry.steer, nextTelemetry.steer, alpha,
				);
			const damper = THREE.MathUtils.lerp(
				telemetry.damper, nextTelemetry.damper, alpha,
			);
			visual.suspensionRoot.position.y =
				visual.baseY + (damper - wheelDamperMid[wheel]) * wheelDamperScale[wheel];
			liveWheelPhases[wheel] += telemetry.speed * deltaSeconds;
			if (visual.rollingRoot)
				visual.rollingRoot.rotation.x = liveWheelPhases[wheel];
			else {
				visual.tire.rotation.x = liveWheelPhases[wheel];
				visual.rim.rotation.x = liveWheelPhases[wheel];
			}
		}

		ensureLiveGhosts(Math.max(0, frame.envs.length - 1));
		let ghostSlot = 0;
		for (let index = 0; index < frame.envs.length; ++index) {
			if (index === followIndex)
				continue;
			const ghostState = frame.envs[index];
			const ghost = liveGhosts[ghostSlot++];
			ghost.position.set(...ghostState.position);
			gameQuaternionValues(ghostState.quaternion, ghost.quaternion);
		}

		if (liveRenderedFrame !== frame) {
			updateLiveHud(state, frame.sequence);
			liveRenderedFrame = frame;
			renderLiveState();
			const stats = window.__TMNF_VIEWER_STATS.live;
			if (stats.sequence !== null && frame.sequence < stats.sequence)
				stats.rewinds++;
			stats.sequence = frame.sequence;
			stats.raceTimeMs = state.raceTimeMs;
			stats.lagFrames = liveNewestSequence - frame.sequence;
		}
		liveMapRecord[FIELD.x] = interpolatedPosition.x;
		liveMapRecord[FIELD.y] = interpolatedPosition.y;
		liveMapRecord[FIELD.z] = interpolatedPosition.z;
		drawMinimap(liveMapRecord, interpolatedQuaternion);
	}

	// Index into frame.envs of the car the chase camera follows. With
	// auto-follow on, that is the env with the largest route progress, kept
	// until another env leads it by FOLLOW_BEST_HYSTERESIS_M so two cars
	// trading the lead do not make the camera flap.
	function chooseFollowedEnv(frame) {
		let index = 0;
		if (followBest) {
			let best = 0;
			for (let i = 1; i < frame.envs.length; ++i) {
				if (frame.envs[i].distance > frame.envs[best].distance)
					best = i;
			}
			const current = frame.envs.findIndex(
				candidate => candidate.envId === liveFollowedEnvId,
			);
			index = current >= 0 &&
				frame.envs[current].distance + FOLLOW_BEST_HYSTERESIS_M >= frame.envs[best].distance
				? current
				: best;
		}
		const envId = frame.envs[index].envId;
		if (envId !== liveFollowedEnvId) {
			liveFollowedEnvId = envId;
			cameraJustChanged = true;
			liveFollowing.textContent = followBest
				? `env ${envId} · leading`
				: `env ${envId} · stream slot 0`;
			window.__TMNF_VIEWER_STATS.live.following = envId;
		}
		return index;
	}

	function setFollowBest(enabled) {
		followBest = enabled;
		followBestButton.classList.toggle("active", enabled);
		followBestButton.setAttribute("aria-pressed", String(enabled));
		liveFollowedEnvId = null;
	}

	// The live panel's state line: connection state, how far behind the
	// newest frame playback sits, and the tail note (resync, dry buffer,
	// stall). The timeline is hidden in live mode, so this is the HUD.
	let liveStateLabel = "";
	function renderLiveState() {
		let text = liveStateLabel;
		if (liveRenderedFrame && liveStateLabel === "Connected")
			text += ` · ${liveNewestSequence - liveRenderedFrame.sequence} behind`;
		if (liveTailNote)
			text += ` · ${liveTailNote}`;
		liveState.lastChild.textContent = text;
	}

	function setLiveState(label, className) {
		liveState.className = className;
		liveStateLabel = label;
		renderLiveState();
		window.__TMNF_VIEWER_STATS.live.state = label;
	}

	function updateLivePanel(meta) {
		liveTrack.textContent = meta.trackName;
		liveUpdate.textContent = Number(meta.updateCount).toLocaleString();
		liveFinishes.textContent = Number(meta.finishes).toLocaleString();
		liveBest.textContent =
			meta.bestLapMs === null ? "--" : formatTime(meta.bestLapMs);
		liveDistance.textContent = meta.distanceMean === null
			? "--"
			: `${Math.round(meta.distanceMean).toLocaleString()} m`;
	}

	function recordTerminations(frames) {
		for (const frame of frames) {
			for (const state of frame.envs) {
				if (!state.reset)
					continue;
				const key = `${frame.sequence}:${state.envId}`;
				if (terminationKeys.has(key))
					continue;
				terminationKeys.add(key);
				terminationItems.unshift({
					envId: state.envId,
					reason: state.terminationReason || "unknown",
					distance: state.distance,
					raceTimeMs: state.raceTimeMs,
				});
			}
		}
		terminationItems = terminationItems.slice(0, 6);
		terminationKeys = new Set([...terminationKeys].slice(-128));
		terminationFeed.replaceChildren(...terminationItems.map((entry) => {
			const item = document.createElement("li");
			const swatch = document.createElement("span");
			swatch.className = "env-swatch";
			swatch.style.background = `hsl(${(entry.envId * 47) % 360} 70% 62%)`;
			const reason = document.createElement("span");
			reason.className = `reason reason-${entry.reason}`;
			reason.textContent = entry.reason;
			item.append(
				swatch,
				`env ${entry.envId} · `,
				reason,
				` · ${Math.round(entry.distance)} m · ${formatTime(entry.raceTimeMs)}`,
			);
			return item;
		}));
	}

	// Errors that no amount of retrying fixes (bad URL, wrong protocol,
	// misregistered track) stop the live session instead of showing
	// "Reconnecting" forever.
	function failLive(message) {
		const error = new Error(message);
		error.liveFatal = true;
		throw error;
	}

	// Validated spectate base URL, or a thrown fatal error.
	function liveBaseUrlFor(value) {
		let url;
		try {
			url = new URL(value);
		} catch {
			failLive(`Live URL ${JSON.stringify(value)} is not a URL.`);
		}
		if (!["http:", "https:"].includes(url.protocol))
			failLive(`Live URL must use HTTP or HTTPS, not ${url.protocol}`);
		url.pathname = url.pathname.replace(/\/+$/, "");
		url.search = "";
		url.hash = "";
		return url.href.replace(/\/$/, "");
	}

	function liveBaseUrl() {
		return liveBaseUrlFor(liveUrlInput.value);
	}

	function resetLiveBuffer() {
		liveFrames = [];
		livePlayhead = -1;
		liveNewestSequence = -1;
		liveNewestChangedAt = 0;
		liveRateSamples = [];
		liveLastPollAt = 0;
		livePollGapMs = 0;
		liveRenderedFrame = null;
		setLiveTailNote("", 0);
		const stats = window.__TMNF_VIEWER_STATS.live;
		stats.sequence = null;
		stats.raceTimeMs = null;
		stats.lagFrames = null;
	}

	// Merge a poll's frames into the buffer: ascending, unique sequences,
	// nothing older than what is already held (the server sends ascending
	// unique frames, so this is a cheap guard, not a sort of every poll).
	function appendLiveFrames(frames) {
		const newestHeld = liveFrames.length > 0 ? liveFrames[liveFrames.length - 1].sequence : -1;
		const fresh = frames
			.filter(frame => Number.isInteger(frame.sequence) && frame.sequence > newestHeld)
			.sort((a, b) => a.sequence - b.sequence)
			.filter((frame, index, all) => index === 0 || frame.sequence !== all[index - 1].sequence);
		if (fresh.length === 0)
			return;
		liveFrames.push(...fresh);
		if (livePlayhead < 0) {
			// First frames of a stream: start a tail behind the newest one.
			livePlayhead = Math.max(liveFrames[0].sequence, liveFrames[liveFrames.length - 1].sequence - liveTailFrames());
		}
		// Frames behind the playhead are history; keep one for interpolation.
		const playedIndex = liveFrameIndexAt(livePlayhead);
		if (playedIndex > 1)
			liveFrames.splice(0, playedIndex - 1);
		if (liveFrames.length > LIVE_BUFFER_FRAMES)
			liveFrames.splice(0, liveFrames.length - LIVE_BUFFER_FRAMES);
	}

	// Tail depth in frames: enough to cover the longest recent poll gap plus
	// two frames, at least LIVE_TAIL_FRAMES. A slow renderer or a slow link
	// stretches the gap between polls; sitting further behind is the price
	// of never running dry between them.
	function liveTailFrames() {
		const interval = Math.max(1, Number(liveMeta?.decisionIntervalMs || 50));
		return THREE.MathUtils.clamp(Math.ceil(livePollGapMs / interval) + 2, LIVE_TAIL_FRAMES, 30);
	}

	// Decisions per second the trainer produced over the last few seconds,
	// as a multiple of real time (1.0 = one decision per decisionIntervalMs).
	function liveTrainerRate() {
		if (liveRateSamples.length < 2 || !liveMeta)
			return null;
		const first = liveRateSamples[0];
		const last = liveRateSamples[liveRateSamples.length - 1];
		const seconds = (last.at - first.at) / 1000;
		if (seconds <= 0)
			return null;
		const interval = Math.max(1, Number(liveMeta.decisionIntervalMs || 50));
		return (last.sequence - first.sequence) / seconds / (1000 / interval);
	}

	// Advance the live playhead at 1x. Never moves backwards: when the buffer
	// runs dry it holds the newest frame (trainer slower than real time, or
	// stalled), when it overflows it jumps forward to a tail behind the newest
	// frame (trainer faster than real time), and the HUD names both.
	function advanceLivePlayhead(deltaSeconds) {
		if (liveFrames.length === 0 || livePlayhead < 0)
			return;
		const interval = Math.max(1, Number(liveMeta?.decisionIntervalMs || 50));
		const newest = liveFrames[liveFrames.length - 1].sequence;
		const stats = window.__TMNF_VIEWER_STATS.live;
		const now = performance.now();
		let next = livePlayhead + deltaSeconds * 1000 / interval;
		if (next > newest) {
			next = newest;
			// Dry because the trainer produces slower than real time (or has
			// stalled), as opposed to a late poll while the trainer is ahead:
			// only the former is worth a note.
			const rate = liveTrainerRate();
			const trainerDry = rate !== null && rate < 0.9;
			if (trainerDry && livePlayhead < newest)
				stats.waits++;
			if (trainerDry && !liveTailNote.startsWith("stalled"))
				setLiveTailNote("buffer dry, holding newest frame", now + 1500);
		} else if (newest - next > LIVE_MAX_LAG_FRAMES) {
			const target = newest - liveTailFrames();
			const skipped = Math.round(target - next);
			next = target;
			stats.resyncs++;
			const rate = liveTrainerRate();
			setLiveTailNote(
				`resynced, skipped ${skipped} frames` +
				(rate === null ? "" : ` (trainer ${rate.toFixed(1)}x real time)`),
				now + 4000,
			);
		} else if (liveTailNote && !liveTailNote.startsWith("stalled") && now > liveTailNoteUntil) {
			setLiveTailNote("", 0);
		}
		livePlayhead = Math.max(livePlayhead, next);
	}

	function setLiveTailNote(note, until) {
		liveTailNote = note;
		liveTailNoteUntil = until;
		renderLiveState();
		window.__TMNF_VIEWER_STATS.live.tailNote = note;
	}

	async function ensureLiveScene(meta, generation) {
		const sceneUrl = TRACK_SCENES[meta.trackId];
		if (!sceneUrl)
			failLive(`No committed scene for live track ${String(meta.trackId)}.`);
		if (currentRun && currentRun.track_id !== meta.trackId) {
			failLive(
				`Spectate stream is on track ${meta.trackId} but run ` +
				`${currentRun.run_id} is registered on ${currentRun.track_id}.`,
			);
		}
		if (sceneData?.gameVisuals?.scene !== meta.trackId) {
			const data = await fetchScene(sceneUrl);
			if (generation !== liveGeneration || viewerMode !== "live")
				return false;
			await installScene(data, `${meta.trackName} live`);
			sceneSelect.value = sceneUrl;
		}
		const ribbon = worldGroup?.getObjectByName("Measured speed trajectory");
		if (ribbon)
			ribbon.visible = false;
		ensureLiveGhosts(Math.max(0, meta.envIndices.length - 1));
		return true;
	}

	async function pollLive(generation) {
		try {
			const baseUrl = liveBaseUrl();
			const previousStream = liveStreamId;
			const previousNewest = liveFrames.length > 0
				? liveFrames[liveFrames.length - 1].sequence : -1;
			// Once attached, these reads are independent. Starting them
			// together avoids two browser/event-loop round trips per poll,
			// especially when software rendering makes each frame expensive.
			const [meta, prefetched] = await Promise.all([
				fetchJson(`${baseUrl}/spectate/meta`, "Live metadata"),
				previousStream === null ? null : fetchJson(
					`${baseUrl}/spectate/frames?since=${previousNewest}`, "Live frames"),
			]);
			if (generation !== liveGeneration || viewerMode !== "live")
				return;
			if (meta.protocolVersion !== 1)
				failLive("Live protocol version 1 is required.");
			if (meta.streamId !== liveStreamId) {
				liveStreamId = meta.streamId;
				resetLiveBuffer();
				liveWheelPhases = [0, 0, 0, 0];
				terminationKeys.clear();
				terminationItems = [];
				terminationFeed.replaceChildren();
			}
			if (!await ensureLiveScene(meta, generation))
				return;
			liveMeta = meta;
			updateLivePanel(meta);
			// Continue from the newest buffered frame; the first request takes
			// the trainer's recent ring so playback starts a tail behind.
			const newestHeld = liveFrames.length > 0
				? liveFrames[liveFrames.length - 1].sequence
				: Number(meta.latestSequence) - LIVE_TAIL_FRAMES - 4;
			// A restart invalidates the previous stream's sequence cursor.
			const payload = prefetched !== null && previousStream === meta.streamId
				? prefetched
				: await fetchJson(
					`${baseUrl}/spectate/frames?since=${Math.max(-1, newestHeld)}`, "Live frames",
				);
			if (generation !== liveGeneration || viewerMode !== "live")
				return;
			if (payload.protocolVersion !== 1)
				failLive("Live frame protocol version 1 is required.");
			const now = performance.now();
			// Longest recent gap between frame polls, decaying towards the
			// current one, sizes the tail.
			if (liveLastPollAt > 0)
				livePollGapMs = Math.max(now - liveLastPollAt, livePollGapMs * 0.9);
			liveLastPollAt = now;
			const latest = Number(payload.latestSequence);
			if (latest > liveNewestSequence) {
				liveRateSamples.push({ at: now, sequence: latest });
				liveRateSamples = liveRateSamples.filter(sample => now - sample.at <= 4000);
				liveNewestSequence = latest;
				liveNewestChangedAt = now;
				if (liveTailNote.startsWith("stalled"))
					setLiveTailNote("", 0);
			} else if (liveNewestChangedAt === 0) {
				liveNewestChangedAt = now;
			}
			if (payload.frames.length > 0) {
				recordTerminations(payload.frames);
				appendLiveFrames(payload.frames);
			}
			const stalledFor = (now - liveNewestChangedAt) / 1000;
			if (liveNewestSequence >= 0 && stalledFor * 1000 >= LIVE_STALL_MS) {
				setLiveState(`Stalled (no new frames for ${Math.round(stalledFor)} s)`, "stalled");
				setLiveTailNote("stalled, holding last frame", Infinity);
			} else if (liveFrames.length === 0) {
				setLiveState("Waiting for frames", "connected");
			} else {
				setLiveState("Connected", "connected");
			}
			status.textContent =
				`Live · ${meta.trackName} · ${meta.envIndices.length} cars · ` +
				`${meta.tickRateHz} Hz physics`;
			if (liveFailures > 0)
				clearNotice();
			liveFailures = 0;
			window.__TMNF_VIEWER_STATS.ready = true;
			window.__TMNF_VIEWER_STATS.scene = `${meta.trackName} live`;
			window.__TMNF_VIEWER_STATS.live.url = liveUrlInput.value;
			window.__TMNF_VIEWER_STATS.live.trainerRate = liveTrainerRate();
		} catch (error) {
			if (generation !== liveGeneration || viewerMode !== "live")
				return;
			console.warn(error);
			if (error.liveFatal) {
				liveGeneration++;
				livePollTimer = null;
				liveStopped = true;
				setLiveState("Stopped", "ended");
				status.textContent = `Live session stopped: ${error.message}`;
				setRunsCollapsed(false);
				showNotice(`${error.message} Pick a run from the list.`);
				return;
			}
			liveFailures++;
			setLiveState("Reconnecting", "reconnecting");
			status.textContent = "Live stream unavailable. Reconnecting…";
			if (liveFailures === LIVE_FAILURES_BEFORE_FALLBACK) {
				// The spectate port stopped answering: surface the run list so
				// the researcher can pick another run while we keep retrying.
				setRunsCollapsed(false);
				showNotice(
					`Spectate server ${liveUrlInput.value} is not answering ` +
					`(${error.message}). Reconnecting; pick another run from the list ` +
					"or wait for the trainer to come back.",
				);
			}
		} finally {
			if (generation === liveGeneration && viewerMode === "live") {
				livePollTimer = window.setTimeout(
					() => pollLive(generation),
					liveFailures >= LIVE_FAILURES_BEFORE_FALLBACK ? 1000 : 250,
				);
			}
		}
	}

	function startLive() {
		viewerMode = "live";
		viewerModeSelect.value = "live";
		document.body.classList.add("live-mode");
		playing = false;
		playButton.textContent = "Play";
		liveGeneration++;
		liveFailures = 0;
		liveFollowedEnvId = null;
		liveDetached = false;
		liveStopped = false;
		liveStreamId = null;
		resetLiveBuffer();
		if (livePollTimer !== null)
			window.clearTimeout(livePollTimer);
		if (ghostData)
			detachGhost();
		ghostSelect.value = "";
		setLiveState("Connecting", "reconnecting");
		pollLive(liveGeneration);
	}

	function stopLive() {
		viewerMode = "replay";
		viewerModeSelect.value = "replay";
		document.body.classList.remove("live-mode");
		liveGeneration++;
		liveDetached = false;
		liveStopped = false;
		if (livePollTimer !== null)
			window.clearTimeout(livePollTimer);
		livePollTimer = null;
		resetLiveBuffer();
		clearLiveGhosts();
	}

	// Called by the registry poller for the spectated run.
	// * Entry no longer live: the run ended or its trainer crashed, so stop
	//   hammering its port and say so instead of "Reconnecting" forever.
	// * Entry live again after that (heartbeat gap, machine suspend): re-attach.
	// * spectate_url changed (trainer restarted on a new port with the same
	//   run id): follow the registry, not the URL copied at click time.
	function reconcileLiveRun() {
		if (viewerMode !== "live" || !currentRun || !runsPayload)
			return;
		const entry = runsPayload.runs.find(run => run.run_id === currentRun.run_id);
		if (entry?.live) {
			const urlChanged = typeof entry.spectate_url === "string" && entry.spectate_url !== "" &&
				entry.spectate_url !== liveUrlInput.value;
			if (liveDetached || (liveStopped && urlChanged)) {
				clearNotice();
				status.textContent = `Run ${entry.run_id} is live again; re-attaching.`;
				spectateRun(entry);
				return;
			}
			if (urlChanged) {
				const previous = liveUrlInput.value;
				currentRun = entry;
				liveUrlInput.value = entry.spectate_url;
				liveFailures = 0;
				liveStreamId = null;
				resetLiveBuffer();
				clearNotice();
				status.textContent = `Trainer for ${entry.run_id} moved from ${previous} to ${entry.spectate_url}; following.`;
				renderRuns();
			}
			return;
		}
		if (liveDetached)
			return;
		const reason = !entry
			? "it was removed from the registry"
			: entry.error
				? `its run.json is invalid (${entry.error})`
				: entry.status === "running"
					? `no heartbeat for ${Math.round(entry.heartbeat_age_s)} s`
					: `status ${entry.status}`;
		liveGeneration++;
		if (livePollTimer !== null)
			window.clearTimeout(livePollTimer);
		livePollTimer = null;
		liveDetached = true;
		setLiveState("Run ended", "ended");
		status.textContent = `Run ${currentRun.run_id} is no longer live (${reason}).`;
		setRunsCollapsed(false);
		showNotice(
			`Run ${currentRun.run_id} is no longer live: ${reason}. ` +
			"Its replays, if any, are in the run list; if it comes back the viewer re-attaches.",
		);
		renderRuns();
	}

	function updateCamera(deltaSeconds) {
		if (!carRoot)
			return;
		const mode = cameraSelect.value;
		const cut = cameraJustChanged;
		if (mode === "orbit") {
			const sinPolar = Math.sin(orbit.polar);
			orbitPosition.set(
				orbit.target.x + orbit.radius * sinPolar * Math.sin(orbit.azimuth),
				orbit.target.y + orbit.radius * Math.cos(orbit.polar),
				orbit.target.z + orbit.radius * sinPolar * Math.cos(orbit.azimuth),
			);
			camera.position.copy(orbitPosition);
			camera.up.set(0, 1, 0);
			camera.lookAt(orbit.target);
		} else if (mode === "chase") {
			carForward.set(0, 0, 1).applyQuaternion(carRoot.quaternion).normalize();
			// Smooth only the heading the camera hangs off; the distance to
			// the car is rigid, otherwise smoothing lag scales with speed and
			// reads as the camera zooming out.
			carForward.y *= 0.35;
			carForward.normalize();
			const headingBlend = cut ? 1 : 1 - Math.exp(-deltaSeconds * 4.5);
			chaseHeading.lerp(carForward, headingBlend).normalize();
			const backDistance = 8.4;
			desiredCamera.copy(carRoot.position).addScaledVector(chaseHeading, -backDistance);
			desiredCamera.y += 3.15;
			desiredTarget.copy(carRoot.position).addScaledVector(chaseHeading, 6.5);
			desiredTarget.y += 0.65;
			camera.position.copy(desiredCamera);
			chaseTarget.copy(desiredTarget);
			camera.up.set(0, 1, 0);
			camera.lookAt(chaseTarget);
		} else if (mode === "top") {
			desiredCamera.copy(carRoot.position);
			desiredCamera.y += 88;
			const blend = 1 - Math.exp(-deltaSeconds * 7);
			if (cut)
				camera.position.copy(desiredCamera);
			else
				camera.position.lerp(desiredCamera, blend);
			camera.up.set(0, 0, 1);
			camera.lookAt(carRoot.position);
		} else {
			if (!reducedMotion.matches)
				beautyAngle += deltaSeconds * 0.14;
			desiredCamera.set(
				carRoot.position.x + Math.sin(beautyAngle) * 9.5,
				carRoot.position.y + 3.2,
				carRoot.position.z + Math.cos(beautyAngle) * 9.5,
			);
			camera.position.copy(desiredCamera);
			desiredTarget.copy(carRoot.position);
			desiredTarget.y += 0.65;
			camera.up.set(0, 1, 0);
			camera.lookAt(desiredTarget);
		}

		const speedFov = fixedFov ?? 54;
		const nextFov = cut
			? speedFov
			: THREE.MathUtils.lerp(camera.fov, speedFov, 1 - Math.exp(-deltaSeconds * 3.8));
		if (Math.abs(nextFov - camera.fov) > 0.005) {
			camera.fov = nextFov;
			camera.updateProjectionMatrix();
		}

		sun.position.copy(carRoot.position).add(sunOffset);
		sun.target.position.copy(carRoot.position);
		sun.target.updateMatrixWorld();
		cameraJustChanged = false;
	}

	async function decodeResponse(response, gzip) {
		if (!response.ok)
			fail(`Scene request failed with HTTP ${response.status}.`);
		if (!gzip)
			return response.text();
		if (typeof DecompressionStream === "undefined")
			fail("This browser cannot decompress gzip scene files.");
		const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
		return new Response(stream).text();
	}

	async function fetchScene(url) {
		const response = await fetch(url, { cache: "no-store" });
		const text = await decodeResponse(response, url.endsWith(".gz"));
		try {
			return JSON.parse(text);
		} catch (error) {
			fail(`${url.split("/").pop()} is not valid JSON (${error.message}). ` +
				"If a trainer is still writing it, retry in a moment.");
		}
	}

	async function loadSceneUrl(url, sourceName = url.split("/").pop(), run = null) {
		try {
			window.__TMNF_VIEWER_STATS.ready = false;
			status.textContent = `Loading ${sourceName}…`;
			const data = await fetchScene(url);
			validateScene(data);
			if (run && data.gameVisuals.scene !== run.track_id) {
				fail(
					`Replay ${sourceName} is on track ${data.gameVisuals.scene} but ` +
					`run.json registers run ${run.run_id} on track ${run.track_id}.`,
				);
			}
			await installScene(data, sourceName);
			return true;
		} catch (error) {
			showError(error);
			return false;
		}
	}

	// --- Run registry ------------------------------------------------------

	function setRunsCollapsed(collapsed) {
		document.body.classList.toggle("runs-collapsed", collapsed);
		runsToggle.textContent = collapsed ? "Show" : "Hide";
		runsToggle.setAttribute("aria-expanded", String(!collapsed));
	}

	function setRunsState(label, className) {
		runsState.className = className;
		runsState.lastChild.textContent = label;
		window.__TMNF_VIEWER_STATS.runs.state = label;
	}

	function setQuery(values) {
		const url = new URL(window.location.href);
		for (const key of ["run", "replay", "live", "ghost", "scene"]) {
			if (key in values) {
				if (values[key] === null || values[key] === undefined || values[key] === "")
					url.searchParams.delete(key);
				else
					url.searchParams.set(key, values[key]);
			}
		}
		window.history.replaceState(null, "", url);
	}

	function requireRun(run, label) {
		if (!run || typeof run !== "object")
			fail(`${label} is not a run object.`);
		const missing = ["run_id", "track_id", "track_name", "status", "replays", "summary"]
			.filter(key => !(key in run));
		if (missing.length > 0)
			fail(`${label} is missing fields: ${missing.join(", ")}.`);
		if (!RUN_ID_PATTERN.test(run.run_id))
			fail(`${label} has an invalid run_id ${JSON.stringify(run.run_id)}.`);
		if (!Array.isArray(run.replays))
			fail(`${label} replays must be an array.`);
		if (!(run.track_id in TRACK_SCENES))
			fail(`${label} names track ${run.track_id}, which has no committed scene.`);
		// The writer emits offsets and refuses naive timestamps; so does the
		// server, and so does this direct read, so liveness and dates cannot
		// be interpreted differently on the two sides.
		for (const key of ["started_at", "heartbeat_at", "finished_at"]) {
			const value = run[key];
			if (value === null || value === undefined)
				continue;
			if (typeof value !== "string" || Number.isNaN(Date.parse(value)) || !ISO_OFFSET_PATTERN.test(value))
				fail(`${label} ${key} ${JSON.stringify(value)} is not an ISO 8601 timestamp with a UTC offset.`);
		}
		return run;
	}

	async function fetchRun(runId) {
		if (!RUN_ID_PATTERN.test(runId))
			fail(`Invalid run id ${JSON.stringify(runId)}.`);
		const run = await fetchJson(`runs/${runId}/run.json`, `Run ${runId}`);
		return requireRun(run, `runs/${runId}/run.json`);
	}

	function formatElapsed(seconds) {
		const total = Math.max(0, Math.round(seconds));
		const hours = Math.floor(total / 3600);
		const minutes = Math.floor((total % 3600) / 60);
		if (hours > 0)
			return `${hours}h ${String(minutes).padStart(2, "0")}m`;
		return `${minutes}m ${String(total % 60).padStart(2, "0")}s`;
	}

	function formatDate(iso) {
		const date = new Date(iso);
		if (Number.isNaN(date.getTime()))
			return String(iso);
		return date.toLocaleString(undefined, {
			month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
		});
	}

	function serverNowMs() {
		return runsPayload ? Date.parse(runsPayload.now) : Date.now();
	}

	function bestLapText(run) {
		const best = run.summary.best_lap_ms;
		return best === null || best === undefined ? "--" : formatTime(best);
	}

	// summary is {} until the trainer has finished its first update.
	function summaryCount(run, key) {
		const value = run.summary[key];
		return value === undefined || value === null ? "--" : Number(value).toLocaleString();
	}

	function metaText(parts) {
		const fragment = document.createDocumentFragment();
		parts.forEach(([label, value], index) => {
			if (index > 0)
				fragment.append(" · ");
			const strong = document.createElement("strong");
			strong.textContent = value;
			fragment.append(strong, ` ${label}`);
		});
		return fragment;
	}

	function buildErrorCard(run) {
		const item = document.createElement("li");
		item.className = "run-row error";
		const title = document.createElement("div");
		title.className = "run-title";
		const track = document.createElement("span");
		track.className = "track";
		track.textContent = run.track_id ? `${run.track_id} · run.json invalid` : "run.json invalid";
		const badge = document.createElement("span");
		badge.className = "badge failed";
		badge.textContent = "invalid";
		title.append(track, badge);
		const id = document.createElement("div");
		id.className = "run-id";
		id.textContent = run.run_id;
		const error = document.createElement("div");
		error.className = "run-error";
		error.textContent = run.error;
		item.append(title, id, error);
		return item;
	}

	function buildLiveRow(run) {
		const item = document.createElement("li");
		item.className = "run-row live";
		item.tabIndex = 0;
		item.setAttribute("role", "button");
		if (currentRun?.run_id === run.run_id && viewerMode === "live")
			item.classList.add("selected");
		const title = document.createElement("div");
		title.className = "run-title";
		const track = document.createElement("span");
		track.className = "track";
		track.textContent = `${run.track_id} · ${run.algorithm}`;
		const badge = document.createElement("span");
		badge.className = "badge live";
		badge.textContent = "live";
		title.append(track, badge);
		const elapsed = (serverNowMs() - Date.parse(run.started_at)) / 1000;
		const meta = document.createElement("div");
		meta.className = "run-meta";
		meta.append(metaText([
			["upd", summaryCount(run, "updates")],
			["fin", summaryCount(run, "finishes")],
			["best", bestLapText(run)],
			["elapsed", formatElapsed(elapsed)],
		]));
		const id = document.createElement("div");
		id.className = "run-id";
		id.textContent = `${run.run_id} · seed ${run.seed} · ${run.spectate_url || "no spectate URL"}`;
		item.append(title, meta, id);
		const activate = () => spectateRun(run);
		item.addEventListener("click", activate);
		item.addEventListener("keydown", (event) => {
			if (event.key === "Enter" || event.key === " ") {
				event.preventDefault();
				activate();
			}
		});
		return item;
	}

	function buildReplayRow(run, replay) {
		const row = document.createElement("li");
		row.className = "replay-row";
		const file = replay.scene.slice("replays/".length);
		if (currentRun?.run_id === run.run_id && currentReplayScene === replay.scene)
			row.classList.add("selected");
		if (ghostData?.url === `runs/${run.run_id}/${replay.scene}`)
			row.classList.add("ghosted");
		const label = document.createElement("button");
		label.type = "button";
		label.className = "replay-label";
		label.textContent = `${replay.label} · min ${Math.round(Number(replay.minute))}`;
		label.title = replay.scene;
		label.addEventListener("click", (event) => {
			event.stopPropagation();
			loadRunReplay(run, file);
		});
		const lap = document.createElement("span");
		lap.className = "replay-lap";
		lap.textContent = replay.lap_ms === null ? "--" : formatTime(replay.lap_ms);
		const ghostButton = document.createElement("button");
		ghostButton.type = "button";
		ghostButton.className = "ghost-button";
		ghostButton.textContent = "Ghost";
		ghostButton.title = "Draw this replay as a ghost on the loaded track";
		ghostButton.addEventListener("click", (event) => {
			event.stopPropagation();
			const path = `runs/${run.run_id}/${replay.scene}`;
			loadGhost(path, `${run.run_id} / ${replay.label}`, path);
		});
		row.append(label, lap, ghostButton);
		return row;
	}

	function buildPastRow(run) {
		const item = document.createElement("li");
		item.className = "run-row";
		item.tabIndex = 0;
		item.setAttribute("role", "button");
		item.setAttribute("aria-expanded", String(expandedRunId === run.run_id));
		if (currentRun?.run_id === run.run_id && viewerMode !== "live")
			item.classList.add("selected");
		const title = document.createElement("div");
		title.className = "run-title";
		const track = document.createElement("span");
		track.className = "track";
		track.textContent = `${run.track_id} · ${run.algorithm}`;
		const badge = document.createElement("span");
		const stale = run.status === "running";
		badge.className = `badge ${stale ? "stale" : run.status}`;
		badge.textContent = stale
			? `stale · ${formatElapsed(run.heartbeat_age_s)} silent`
			: run.status;
		title.append(track, badge);
		const started = Date.parse(run.started_at);
		const ended = Date.parse(run.finished_at ?? run.heartbeat_at);
		const meta = document.createElement("div");
		meta.className = "run-meta";
		meta.append(metaText([
			["best", bestLapText(run)],
			["upd", summaryCount(run, "updates")],
			["fin", summaryCount(run, "finishes")],
			["", formatElapsed((ended - started) / 1000)],
			["", formatDate(run.started_at)],
		]));
		const id = document.createElement("div");
		id.className = "run-id";
		id.textContent =
			`${run.run_id} · seed ${run.seed} · ${run.replays.length} replay` +
			`${run.replays.length === 1 ? "" : "s"}`;
		item.append(title, meta, id);
		if (expandedRunId === run.run_id) {
			const list = document.createElement("ul");
			list.className = "replay-list";
			if (run.replays.length === 0) {
				const empty = document.createElement("li");
				empty.className = "empty";
				empty.textContent = "No replays recorded for this run.";
				list.append(empty);
			}
			for (const replay of run.replays)
				list.append(buildReplayRow(run, replay));
			item.append(list);
		}
		const toggle = () => {
			expandedRunId = expandedRunId === run.run_id ? null : run.run_id;
			renderRuns();
		};
		item.addEventListener("click", toggle);
		item.addEventListener("keydown", (event) => {
			if (event.target !== item)
				return;
			if (event.key === "Enter" || event.key === " ") {
				event.preventDefault();
				toggle();
			}
		});
		return item;
	}

	function comparePast(a, b) {
		const sort = runsSort.value;
		if (sort === "lap") {
			const lapA = a.summary.best_lap_ms ?? Infinity;
			const lapB = b.summary.best_lap_ms ?? Infinity;
			if (lapA !== lapB)
				return lapA - lapB;
		} else if (sort === "track") {
			if (a.track_id !== b.track_id)
				return a.track_id < b.track_id ? -1 : 1;
		}
		return Date.parse(b.started_at) - Date.parse(a.started_at);
	}

	function emptyItem(text) {
		const item = document.createElement("li");
		item.className = "empty";
		item.textContent = text;
		return item;
	}

	function renderRuns() {
		if (!runsPayload) {
			liveRunsList.replaceChildren(emptyItem("Registry not loaded."));
			pastRunsList.replaceChildren();
			return;
		}
		const valid = runsPayload.runs.filter(run => !run.error);
		const broken = runsPayload.runs.filter(run => run.error);
		const live = valid.filter(run => run.live);
		const past = valid.filter(run => !run.live).sort(comparePast);
		liveCount.textContent = String(live.length);
		liveRunsList.replaceChildren(
			...(live.length === 0 ? [emptyItem("No trainer is publishing a heartbeat.")] : []),
			...live.map(buildLiveRow),
		);
		pastRunsList.replaceChildren(
			...broken.map(buildErrorCard),
			...past.map(buildPastRow),
			...(past.length + broken.length === 0 ? [emptyItem("No past runs in build/runs/index.json.")] : []),
		);
		window.__TMNF_VIEWER_STATS.runs.live = live.length;
		window.__TMNF_VIEWER_STATS.runs.past = past.length;
		window.__TMNF_VIEWER_STATS.runs.errors = broken.length;
	}

	async function pollRuns() {
		try {
			const response = await fetch("api/runs", { cache: "no-store" });
			if (response.status === 404) {
				setRunsState("Registry unavailable", "degraded");
				liveRunsList.replaceChildren(emptyItem(
					"This server has no /api/runs. Start the viewer with " +
					"python3 tools/serve_viewer.py to browse training runs.",
				));
				pastRunsList.replaceChildren();
				return;
			}
			const payload = await response.json();
			if (response.status === 503) {
				// index.json is mid-rewrite or missing: keep the last good list.
				setRunsState(`Registry unreadable, retrying (${payload.error})`, "degraded");
				if (!runsPayload)
					liveRunsList.replaceChildren(emptyItem(payload.error));
				return;
			}
			if (!response.ok || !Array.isArray(payload.runs))
				fail(`/api/runs returned HTTP ${response.status} without a runs array.`);
			runsPayload = payload;
			setRunsState(`${payload.runs.filter(run => run.live).length} live · ${payload.runs.length} total`, "connected");
			renderRuns();
			reconcileLiveRun();
		} catch (error) {
			console.warn(error);
			setRunsState("Registry offline, retrying", "degraded");
		} finally {
			runsTimer = window.setTimeout(pollRuns, RUNS_POLL_MS);
		}
	}

	async function loadRunReplay(run, file) {
		const scene = `replays/${file}`;
		if (!REPLAY_SCENE_PATTERN.test(scene)) {
			showNotice(`Replay file name ${JSON.stringify(file)} is not allowed.`);
			return;
		}
		const replay = run.replays.find(entry => entry.scene === scene);
		if (!replay) {
			showNotice(
				`Run ${run.run_id} has no replay ${scene}. Available: ` +
				`${run.replays.map(entry => entry.scene).join(", ") || "none"}.`,
			);
			return;
		}
		if (viewerMode === "live")
			stopLive();
		currentRun = run;
		currentReplayScene = scene;
		expandedRunId = run.run_id;
		const loaded = await loadSceneUrl(
			`runs/${run.run_id}/${scene}`,
			`${run.run_id} / ${replay.label}`,
			run,
		);
		if (loaded)
			setQuery({ run: run.run_id, replay: file, live: null, scene: null });
		renderRuns();
	}

	// Show a run's replay list with its track on screen: the landing spot for
	// a live link whose run is no longer live.
	async function showRunReplays(run, message) {
		if (viewerMode === "live")
			stopLive();
		currentRun = run;
		currentReplayScene = null;
		expandedRunId = run.run_id;
		setRunsCollapsed(false);
		if (!sceneData) {
			await loadSceneUrl(TRACK_SCENES[run.track_id], `${run.track_name} (${run.run_id} track)`);
			sceneSelect.value = TRACK_SCENES[run.track_id];
		}
		setQuery({ run: run.run_id, live: null, replay: null, scene: null });
		renderRuns();
		showNotice(message);
	}

	async function spectateRun(run) {
		// Liveness is the server's call (heartbeat age on its clock); without
		// a registry answer yet, a running status is the best available guess.
		await runsFirstPoll;
		const entry = runsPayload?.runs.find(candidate => candidate.run_id === run.run_id);
		const live = entry ? entry.live : run.status === "running";
		if (typeof run.spectate_url !== "string" || run.spectate_url === "" || !live) {
			const replays = run.replays.length === 0
				? "It recorded no replays."
				: `Its ${run.replays.length} replay${run.replays.length === 1 ? " is" : "s are"} listed in the run panel.`;
			const why = !live
				? run.status === "running"
					? `is not live (no heartbeat for ${Math.round(entry?.heartbeat_age_s ?? 0)} s)`
					: `is not live (status ${run.status})`
				: "publishes no spectate_url";
			await showRunReplays(run, `Run ${run.run_id} ${why}, so it cannot be watched live. ${replays}`);
			return false;
		}
		try {
			liveBaseUrlFor(run.spectate_url);
		} catch (error) {
			await showRunReplays(run, `Run ${run.run_id} has an unusable spectate_url: ${error.message}`);
			return false;
		}
		if (viewerMode === "live")
			stopLive();
		currentRun = run;
		currentReplayScene = null;
		liveUrlInput.value = run.spectate_url;
		clearNotice();
		renderRuns();
		// Put the run's track on screen before the first spectate poll so an
		// unreachable trainer still leaves a canvas to look at.
		if (sceneData?.gameVisuals.scene !== run.track_id) {
			const loaded = await loadSceneUrl(TRACK_SCENES[run.track_id], `${run.track_name} live`);
			if (!loaded || currentRun !== run)
				return false;
			sceneSelect.value = TRACK_SCENES[run.track_id];
		}
		startLive();
		setQuery({ run: run.run_id, live: "1", replay: null, ghost: null, scene: null });
		renderRuns();
		return true;
	}

	// Clean-render deep link switches (headless screenshots): hud=0 hides
	// every panel, overlays=0 hides the racing line and gates, camera=<mode>
	// picks the camera, tick=<n> parks the playhead, cam=px,py,pz,tx,ty,tz
	// places the orbit camera at a world position looking at a target,
	// fov=<degrees> pins the field of view. scene= also accepts any file
	// under scenes/, not only the committed dropdown entries.
	function applyRenderParameters(parameters) {
		if (parameters.get("hud") === "0") {
			for (const panel of document.querySelectorAll(".instrument, #error"))
				panel.style.setProperty("display", "none", "important");
		}
		if (parameters.get("overlays") === "0") {
			document.body.classList.add("overlays-hidden");
			applyOverlayVisibility();
		}
		const cameraMode = parameters.get("camera");
		if (cameraMode !== null) {
			if (![...cameraSelect.options].some(entry => entry.value === cameraMode))
				fail(`camera=${cameraMode} is not a camera mode.`);
			cameraSelect.value = cameraMode;
			cameraJustChanged = true;
		}
		const fov = parameters.get("fov");
		if (fov !== null) {
			fixedFov = Number(fov);
			if (!(fixedFov > 1 && fixedFov < 179))
				fail(`fov=${fov} is not a field of view in degrees.`);
		}
		const cam = parameters.get("cam");
		if (cam !== null) {
			const values = cam.split(",").map(Number);
			if (values.length !== 6 || values.some(value => !Number.isFinite(value)))
				fail(`cam=${cam} must be px,py,pz,tx,ty,tz.`);
			orbit.target.set(values[3], values[4], values[5]);
			const dx = values[0] - values[3];
			const dy = values[1] - values[4];
			const dz = values[2] - values[5];
			orbit.radius = Math.hypot(dx, dy, dz);
			orbit.polar = Math.acos(dy / orbit.radius);
			orbit.azimuth = Math.atan2(dx, dz);
			cameraSelect.value = "orbit";
			cameraJustChanged = true;
		}
		const tick = parameters.get("tick");
		if (tick !== null && replay) {
			playhead = Number(tick);
			if (!(Number.isInteger(playhead) && playhead >= 0 && playhead < replay.tickCount))
				fail(`tick=${tick} is outside 0..${replay.tickCount - 1}.`);
			playing = false;
			updateReplayVisuals();
		}
	}
	// Headless frame sequencers load the scene once and re-apply camera/tick
	// per frame instead of navigating (a full scene load per frame).
	window.__TMNF_APPLY_RENDER_PARAMETERS = applyRenderParameters;

	async function openDeepLink(parameters) {
		const runId = parameters.get("run");
		const replayFile = parameters.get("replay");
		const live = parameters.get("live");
		const ghost = parameters.get("ghost");
		const scenePath = parameters.get("scene");
		try {
			if (runId !== null) {
				const run = await fetchRun(runId);
				if (live === "1") {
					// A run that is not (or no longer) live lands on its replay
					// list with the track on screen instead of an empty canvas.
					if (await spectateRun(run))
						return;
				} else {
					if (replayFile === null)
						fail(`Deep link for run ${runId} needs replay=<file> or live=1.`);
					await loadRunReplay(run, replayFile);
				}
			} else if (live !== null) {
				// Legacy form: ?live=<spectate URL>. Validated before anything
				// starts polling so a bad URL cannot leave "Reconnecting" on an
				// empty canvas.
				if (live === "1" || live === "")
					fail("live=1 needs run=<id>; the legacy form is live=<spectate URL>.");
				liveBaseUrlFor(live);
				liveUrlInput.value = live;
				currentRun = null;
				startLive();
				return;
			} else {
				const option = [...sceneSelect.options].find(entry => entry.value === scenePath);
				if (scenePath !== null && !option && !SCENE_PATH_PATTERN.test(scenePath))
					fail(`${scenePath} is not a committed scene.`);
				if (option)
					sceneSelect.value = scenePath;
				await loadSceneUrl((option || scenePath === null) ? sceneSelect.value : scenePath);
			}
			applyRenderParameters(parameters);
			if (ghost !== null && sceneData) {
				if (!GHOST_PATH_PATTERN.test(ghost) || ghost.includes(".."))
					fail(`Ghost path ${JSON.stringify(ghost)} is not allowed.`);
				await loadGhost(ghost, ghost.replace(/^(scenes|runs)\//, ""), ghost);
			}
		} catch (error) {
			showNotice(error);
		}
		// A bad deep link (unknown run, missing replay, obsolete scene) must
		// not leave an empty canvas: load the default committed scene and
		// keep the link's error on screen.
		if (!sceneData && viewerMode !== "live") {
			const notice = errorBox.style.display === "block" ? errorBox.textContent : "";
			await loadSceneUrl(sceneSelect.value);
			if (notice)
				showNotice(notice);
		}
	}

	async function loadSceneFile(file) {
		try {
			window.__TMNF_VIEWER_STATS.ready = false;
			status.textContent = `Loading ${file.name}…`;
			let text;
			if (file.name.endsWith(".gz")) {
				if (typeof DecompressionStream === "undefined")
					fail("This browser cannot decompress gzip scene files.");
				const stream = file.stream().pipeThrough(new DecompressionStream("gzip"));
				text = await new Response(stream).text();
			} else {
				text = await file.text();
			}
			await installScene(JSON.parse(text), file.name);
		} catch (error) {
			showError(error);
		}
	}

	function togglePlayback() {
		if (viewerMode === "live" || !replay)
			return;
		if (playhead >= replay.tickCount - 1)
			playhead = 0;
		playing = !playing;
		playButton.textContent = playing ? "Pause" : "Play";
	}

	function loadCommittedScene() {
		if (viewerMode === "live")
			stopLive();
		currentRun = null;
		currentReplayScene = null;
		setQuery({ run: null, replay: null, live: null, scene: sceneSelect.value });
		renderRuns();
		loadSceneUrl(sceneSelect.value);
	}

	function connectManualLive() {
		try {
			liveBaseUrlFor(liveUrlInput.value);
		} catch (error) {
			viewerModeSelect.value = viewerMode;
			showNotice(error);
			return;
		}
		currentRun = null;
		currentReplayScene = null;
		setQuery({ run: null, replay: null, ghost: null, scene: null, live: liveUrlInput.value });
		startLive();
		renderRuns();
	}

	playButton.addEventListener("click", togglePlayback);
	viewerModeSelect.addEventListener("change", () => {
		if (viewerModeSelect.value === "live")
			connectManualLive();
		else
			loadCommittedScene();
	});
	liveConnectButton.addEventListener("click", connectManualLive);
	liveUrlInput.addEventListener("keydown", (event) => {
		if (event.key === "Enter")
			connectManualLive();
	});
	followBestButton.addEventListener("click", () => setFollowBest(!followBest));
	runsToggle.addEventListener("click", () => {
		setRunsCollapsed(!document.body.classList.contains("runs-collapsed"));
	});
	runsSort.addEventListener("change", renderRuns);
	ghostSelect.addEventListener("change", () => {
		if (ghostSelect.value === "") {
			detachGhost();
			setQuery({ ghost: null });
			renderRuns();
			return;
		}
		const option = ghostSelect.selectedOptions[0];
		loadGhost(ghostSelect.value, option.textContent.replace(/^Ghost · /, ""), ghostSelect.value);
	});
	compareClear.addEventListener("click", () => {
		ghostSelect.value = "";
		detachGhost();
		setQuery({ ghost: null });
		renderRuns();
	});
	errorBox.addEventListener("click", clearNotice);
	speedSelect.addEventListener("change", () => {
		playbackSpeed = Number(speedSelect.value);
	});
	cameraSelect.addEventListener("change", () => {
		cameraJustChanged = true;
	});
	visualModeSelect.addEventListener("change", applyVisualMode);
	overlayToggle.addEventListener("click", () => {
		document.body.classList.toggle("overlays-hidden");
		applyOverlayVisibility();
	});
	sceneSelect.addEventListener("change", loadCommittedScene);
	scrubber.addEventListener("input", () => {
		playhead = Number(scrubber.value);
		playing = false;
		playButton.textContent = "Play";
		updateReplayVisuals();
		cameraJustChanged = true;
	});
	fileInput.addEventListener("change", () => {
		if (fileInput.files.length === 1) {
			if (viewerMode === "live")
				stopLive();
			currentRun = null;
			currentReplayScene = null;
			setQuery({ run: null, replay: null, live: null, scene: null });
			renderRuns();
			loadSceneFile(fileInput.files[0]);
		}
		fileInput.value = "";
	});
	document.querySelector(".file-button").addEventListener("keydown", (event) => {
		if (event.key === "Enter" || event.key === " ") {
			event.preventDefault();
			fileInput.click();
		}
	});
	window.addEventListener("keydown", (event) => {
		if (event.code === "Space" && event.target === document.body) {
			event.preventDefault();
			togglePlayback();
		}
	});

	renderer.domElement.addEventListener("pointerdown", (event) => {
		if (cameraSelect.value !== "orbit" || event.button !== 0)
			return;
		orbit.dragging = true;
		orbit.x = event.clientX;
		orbit.y = event.clientY;
		renderer.domElement.setPointerCapture(event.pointerId);
	});
	renderer.domElement.addEventListener("pointermove", (event) => {
		if (!orbit.dragging)
			return;
		const dx = event.clientX - orbit.x;
		const dy = event.clientY - orbit.y;
		orbit.x = event.clientX;
		orbit.y = event.clientY;
		orbit.azimuth -= dx * 0.006;
		orbit.polar = THREE.MathUtils.clamp(
			orbit.polar + dy * 0.005,
			0.12,
			Math.PI * 0.48,
		);
	});
	renderer.domElement.addEventListener("pointerup", (event) => {
		orbit.dragging = false;
		renderer.domElement.releasePointerCapture(event.pointerId);
	});
	renderer.domElement.addEventListener("wheel", (event) => {
		if (cameraSelect.value !== "orbit")
			return;
		event.preventDefault();
		orbit.radius = THREE.MathUtils.clamp(
			orbit.radius * Math.exp(event.deltaY * 0.001),
			8,
			2500,
		);
	}, { passive: false });

	window.addEventListener("resize", () => {
		renderDirty = true;
		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
		renderer.setSize(window.innerWidth, window.innerHeight);
	});

	function animate(now) {
		requestAnimationFrame(animate);
		const wallDeltaSeconds = (now - previousFrameTime) / 1000;
		const deltaSeconds = Math.min(wallDeltaSeconds, 0.1);
		previousFrameTime = now;
		if (viewerMode === "live") {
			if (liveFrames.length > 0) {
				// Live time is wall time: a slow renderer (or a hidden tab)
				// must advance by the real elapsed time to stay on the tail
				// instead of drifting behind and resyncing.
				advanceLivePlayhead(Math.min(wallDeltaSeconds, 2));
				updateLiveVisuals(deltaSeconds);
			}
		} else if (playing && replay) {
			playhead += deltaSeconds * playbackSpeed * (1000 / replay.tickMs);
			if (playhead >= replay.tickCount - 1) {
				playhead = replay.tickCount - 1;
				playing = false;
				playButton.textContent = "Play";
			}
			updateReplayVisuals();
		}
		updateCamera(deltaSeconds);
		// A paused replay has no changing pixels once the camera settles.
		// Continue the lightweight clock/input loop, but avoid re-submitting
		// thousands of static meshes (and both shadow/render passes) at idle.
		const cameraMoved = renderedCameraPosition.distanceToSquared(camera.position) > 1e-12 ||
			1 - Math.abs(renderedCameraQuaternion.dot(camera.quaternion)) > 1e-12 ||
			renderedFov !== camera.fov;
		if (viewerMode !== "live" && !playing && !renderDirty && !cameraMoved) {
			fpsValue.textContent = "Paused";
			window.__TMNF_VIEWER_STATS.fps = 0;
			fpsFrameCount = 0;
			fpsWindowStart = now;
			return;
		}
		renderDirty = false;
		renderedCameraPosition.copy(camera.position);
		renderedCameraQuaternion.copy(camera.quaternion);
		renderedFov = camera.fov;
		// Two render passes per frame: reset the counters once so the stats
		// report the whole frame rather than the car pass alone.
		renderer.info.reset();
		camera.layers.set(0);
		renderer.autoClear = true;
		renderer.render(scene, camera);
		// A scene background colour forces a clear on every render call, so
		// drop it for the car pass to keep the world pass's colour and depth.
		const background = scene.background;
		scene.background = null;
		camera.layers.set(CAR_LAYER);
		renderer.autoClear = false;
		renderer.render(scene, camera);
		renderer.autoClear = true;
		scene.background = background;
		camera.layers.set(0);

		window.__TMNF_VIEWER_STATS.renderedFrames = (window.__TMNF_VIEWER_STATS.renderedFrames || 0) + 1;
		window.__TMNF_VIEWER_STATS.triangles = renderer.info.render.triangles;
		window.__TMNF_VIEWER_STATS.drawCalls = renderer.info.render.calls;
		fpsFrameCount++;
		const fpsElapsed = now - fpsWindowStart;
		if (fpsElapsed >= 600) {
			const fps = fpsFrameCount * 1000 / fpsElapsed;
			fpsValue.textContent = `${Math.round(fps)} FPS`;
			window.__TMNF_VIEWER_STATS.fps = fps;
			fpsFrameCount = 0;
			fpsWindowStart = now;
		}
	}

	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	requestAnimationFrame(animate);
	setFollowBest(true);
	runsFirstPoll = pollRuns();
	openDeepLink(new URLSearchParams(window.location.search));
})();
