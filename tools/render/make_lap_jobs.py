#!/usr/bin/env python3
# Per-frame chase camera jobs for the lap video. Same camera as hero pick 2
# (7.5 m behind, 3.5 m to the side along velocity, 0.3 m up, fov 62), with
# exponential smoothing on camera position and look target across frames.
import json
import os
import sys

import numpy as np

ROOT = "/home/adityas/Projects/TMNF-C"
RUN = "a01_ppo_baseline_20m_s1_head"
REPLAY = "final_env25380ms.json"
BASE = f"http://127.0.0.1:8801/index.html?run={RUN}&replay={REPLAY}&hud=0&overlays=0"

ticks = json.load(open(f"{ROOT}/build/runs/{RUN}/replays/{REPLAY}"))["lap"]["ticks"]
N = len(ticks)

BACK, HEIGHT, AHEAD, TARGET_UP, SIDE, FOV = 6.0, 2.3, 10.0, 0.7, 0.0, 58
ALPHA = 0.15
STEP = 2
LAST_TICK = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2688, N - 1)
W, H = 1920, 1080
OUT = "/tmp/tweet/frames_top"


def quat_rotate(q, v):
    x, y, z, w = q
    qv = np.array([x, y, z])
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def pos_at(tick):
    return np.array(ticks[min(max(tick, 0), N - 1)][1:4])


def heading_at(tick):
    r = ticks[min(tick, N - 1)]
    v = pos_at(tick + 3) - pos_at(tick - 3)
    q = (r[5], r[6], r[7], r[4])
    # Standing-start heading from the trajectory itself (the car launches
    # straight): the quaternion-derived axis pointed backwards.
    fwd = pos_at(150) - pos_at(0)
    fwd = fwd / np.linalg.norm(fwd)
    # Chase direction = car forward axis; blend toward the velocity direction
    # only above 15 m/s (weight ramps to 1 at 45 m/s). Velocity is undefined
    # at the standing start.
    speed = np.linalg.norm(v) / 0.06  # 6 ticks of 10 ms
    w = min(max((speed - 15.0) / 30.0, 0.0), 1.0)
    v = w * (v / max(np.linalg.norm(v), 1e-9)) + (1.0 - w) * fwd
    v[1] = 0.0
    return v / np.linalg.norm(v)


def raw_cam(tick):
    pos = pos_at(tick)
    heading = heading_at(tick)
    right = np.cross(heading, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    cam = pos - heading * BACK + np.array([0.0, HEIGHT, 0.0]) + right * SIDE
    tgt = pos + heading * AHEAD + np.array([0.0, TARGET_UP, 0.0])
    return cam, tgt


os.makedirs(OUT, exist_ok=True)
jobs = []
# Smoothing the absolute camera position lags 12 m at 380 km/h (alpha 0.15,
# 50 fps), so the smoothing is applied to the camera/target offsets relative
# to the car instead: heading jitter is filtered, the car stays 7.5 m away.
cam_off = tgt_off = None
for i, tick in enumerate(range(0, LAST_TICK + 1, STEP)):
    pos = pos_at(tick)
    cam, tgt = raw_cam(tick)
    if cam_off is None:
        cam_off, tgt_off = cam - pos, tgt - pos
    else:
        cam_off = cam_off + ALPHA * ((cam - pos) - cam_off)
        tgt_off = tgt_off + ALPHA * ((tgt - pos) - tgt_off)
    cam_s, tgt_s = pos + cam_off, pos + tgt_off
    values = ",".join(f"{v:.3f}" for v in (*cam_s, *tgt_s))
    url = f"{BASE}&tick={min(tick, N - 1)}&cam={values}&fov={FOV}"
    jobs.append({"url": url, "out": f"{OUT}/{i:05d}.png", "width": W, "height": H, "frames": 3, "timeout": 60000})

which = sys.argv[2] if len(sys.argv) > 2 else "all"
if which == "smoke":
    jobs = [jobs[0], jobs[1], jobs[200], jobs[len(jobs) // 2], jobs[-1]]
    out = "/tmp/tweet/lap_jobs_smoke.json"
elif which.startswith("half"):
    half = int(which[4:])
    jobs = jobs[half::2]
    out = f"/tmp/tweet/lap_jobs_half{half}.json"
else:
    out = "/tmp/tweet/lap_jobs.json"
json.dump(jobs, open(out, "w"), indent=1)
print(f"ticks in replay: {N}; frames total: {(LAST_TICK // STEP) + 1}; wrote {len(jobs)} jobs to {out}")
