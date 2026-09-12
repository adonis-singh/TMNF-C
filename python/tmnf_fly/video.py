"""Compose the "fruit fly brain drives TrackMania" video from a drive log (tmnf_fly.drive npz).

    python -m tmnf_fly.video local/fly/drive/reflex_a01.npz local/fly/reflex_a01.mp4 [--fps 50] [--size 1920 1080]
                             [--stills 3.0,12.5]
    python -m tmnf_fly.video local/fly/drive/observe_a04.npz local/fly/mb_a04_lap.mp4 --track a04 \
        --real-view local/fly/mb_a04_chase.mp4 --mushroom-body local/fly/mb_a04_showcase.npz --training-minutes 5

Per video frame (one vision frame of the log, 20 ms):
    left 55 %      both compound eyes, Mollweide, one hexagon per ommatidium; HUD line below
                   (--real-view video.mp4: the game view rendered from the same lap goes on top of the eyes,
                    one frame per log frame)
    right top      MaleCNS skeleton render coloured by rate (BrainRenderer, slow orbit)
    right bottom   track map with the car's trail, and 3 s traces of DNp01, LC4 L/R, HS L/R; with
                   --mushroom-body (a tmnf_fly.mb_replay showcase of the same lap) the KC / MBON / DAN
                   activity is drawn on the brain and the traces show MBON value, PAM / PPL1 dopamine, LC4
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from tmnf_fly import ROOT, skeletons  # noqa: E402
from tmnf_fly.brain_render import BrainRenderer, Camera  # noqa: E402
from tmnf_fly.eye import Eye, hexagon_patches, mollweide  # noqa: E402
from tmnf_rl.tracks import track_spec  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from generate_route_centerline import parse_track, route_sections  # noqa: E402

TICK_DT = 0.01
TRACE_SECONDS = 3.0
ORBIT_DEG_PER_S = 2.0
MATERIAL_ROAD = 16
FLOOR_SLAB_AREA = 500.0   # m^2; stadium floor slabs are single triangles far larger than any block face
MAP_MARGIN = 40.0         # m around the route centerline
ACTIVITY_PERCENTILE = 99.0
ACTIVITY_MIN_SCALE = 1e-3
FONT = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"

TITLE = "MaleCNS v1.0 fruit fly connectome driving TrackMania Nations Forever"
FOOTER = ("eye: micro-CT ommatidia (Zhao 2025) | optic lobe: flyvis | brain: MaleCNS wiring, rate model | "
          "controls: DNp01 brake, LC4/LPLC2 + HS steer")

TITLE_MB = "MaleCNS v1.0 fruit fly mushroom body learns a TrackMania track by dopamine  ({minutes} min of trial and error)"
FOOTER_MB = ("learning: PN>KC>MBON wiring from the connectome, KC->MBON synapses depressed by dopamine (PAM reward / PPL1 "
             "punishment, reward-prediction error) | eye + optic lobe + brain shown as the fly would see the lap")

BG = (10, 11, 16)
TEXT = (190, 192, 200)
DIM = (96, 100, 115)
GRID = (52, 54, 66)
ROAD = (150, 151, 158)
BLOCK = (36, 38, 50)
CENTERLINE = (58, 110, 165)
TRAIL = (255, 204, 0)
CAR = (255, 255, 255)
BRAKE_ON = (255, 70, 70)
COL_L = (80, 200, 255)
COL_R = (255, 130, 200)
COL_GF = (255, 200, 80)


def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), px)


def load_log(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        log = {k: z[k] for k in z.files}
    for key in ("tick", "x", "y", "z", "speed", "steer", "brake", "progress", "eye",
                "DNp01/L", "DNp01/R", "LC4/L", "LC4/R", "HS/L", "HS/R"):
        if key not in log:
            raise KeyError(f"{path}: missing {key}")
    if log["eye"].ndim != 3 or log["eye"].shape[1] != 2:
        raise ValueError(f"{path}: eye has shape {log['eye'].shape}, expected (T, 2, 852)")
    return log


def merge_mushroom_body(log: dict[str, np.ndarray], path: Path, sets: dict) -> dict[str, np.ndarray]:
    """Overlay a tmnf_fly.mb_replay showcase episode on an observer log of the same lap: per-decision
    KC / MBON / DAN activity becomes per-frame activity of the rendered skeleton sets, and the value /
    dopamine series are added for the traces. Frame i of the log is tick 2i; decision d covers ticks
    sum(executed_ticks[:d]) .. sum(executed_ticks[:d+1])."""
    with np.load(path) as z:
        mb = {k: z[k] for k in z.files}
    if not int(mb["total_ticks"]) - 2 <= int(log["tick"][-1]) < int(mb["total_ticks"]):
        raise ValueError(f"{path}: lap is {int(mb['total_ticks'])} ticks, the observer log ends at tick {int(log['tick'][-1])}")
    starts = np.concatenate([[0], np.cumsum(mb["executed_ticks"])])
    decision = np.searchsorted(starts, log["tick"], side="right") - 1
    decision = np.clip(decision, 0, len(mb["action_idx"]) - 1)
    out = dict(log)
    for name, body_key, act in (("KC", "kc_body", mb["kc"].astype(np.float32)), ("MBON", "mbon_body", mb["mbon"]),
                                ("DAN", "dan_body", mb["dan"])):
        index = {int(b): i for i, b in enumerate(mb[body_key])}
        cols = np.array([index[int(b)] for b in sets[name].body_ids])
        out[f"neurons/{name}"] = act[decision][:, cols]
    out["MB/value"] = mb["value"][decision]
    out["MB/PAM"] = mb["dan"][:, mb["dan_class"] == 0].sum(axis=1)[decision]
    out["MB/PPL1"] = mb["dan"][:, mb["dan_class"] == 1].sum(axis=1)[decision]
    return out


def read_centerline(route: Path) -> np.ndarray:
    """(N, 3) world xyz of the route centerline (section 4: x, y, z, arc, halfwidth, leg)."""
    data = route.read_bytes()
    offset, count, stride = route_sections(data)[4]
    if stride != 24:
        raise ValueError(f"{route}: centerline stride {stride}, expected 24")
    rows = np.array([struct.unpack_from("<fffffI", data, offset + i * stride) for i in range(count)])
    return rows[:, :3]


# ---------------------------------------------------------------- static rasters

def _figure(w: int, h: int):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.axis("off")
    return fig, ax


def _grab(fig) -> np.ndarray:
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


class EyePanel:
    """Static Mollweide grid + per-pixel ommatidium label map; per frame a LUT gather."""

    def __init__(self, eye: Eye, w: int, h: int):
        self.w, self.h = w, h
        s2 = np.sqrt(2.0)
        pad = 0.06
        xr, yr = 2 * s2 + pad, s2 + pad
        if h / w > yr / xr:
            yr = xr * h / w
        else:
            xr = yr * w / h
        self.lim = (-xr, xr, -yr, yr)
        patches = hexagon_patches(eye)
        idx = np.arange(1, len(patches) + 1)
        codes = np.stack([(idx >> 16) & 255, (idx >> 8) & 255, idx & 255], axis=1) / 255.0
        fig, ax = _figure(w, h)
        ax.add_collection(PolyCollection(patches, facecolors=codes, edgecolors="none", antialiased=False))
        ax.set_xlim(self.lim[:2]); ax.set_ylim(self.lim[2:])
        img = _grab(fig).astype(np.int64)
        label = ((img[..., 0] << 16) | (img[..., 1] << 8) | img[..., 2]) - 1
        if label.shape != (h, w):
            raise RuntimeError(f"label map is {label.shape}, expected {(h, w)}")
        self.rows, self.cols = np.nonzero(label >= 0)
        self.labels = label[self.rows, self.cols]
        # the eyes' fields overlap in front (binocular zone), so a few hexagons are covered by the other eye's
        self.hidden = len(patches) - len(np.unique(self.labels))
        if self.hidden > 0.02 * len(patches):
            raise RuntimeError(f"{self.hidden} ommatidia cover no pixel; increase the eye panel size")

        fig, ax = _figure(w, h)
        theta = np.linspace(0, 2 * np.pi, 361)
        ax.plot(2 * s2 * np.cos(theta), s2 * np.sin(theta), color=np.array(GRID) / 255, lw=1.0)
        for az in (-90, 0, 90):
            el = np.radians(np.linspace(-90, 90, 91))
            d = np.stack([np.cos(el) * np.cos(np.radians(az)), -np.cos(el) * np.sin(np.radians(az)), np.sin(el)], 1)
            m = mollweide(d)
            ax.plot(m[:, 0], m[:, 1], color=np.array(GRID) / 255, lw=0.7)
        ax.plot([-2 * s2, 2 * s2], [0, 0], color=np.array(GRID) / 255, lw=0.7)
        ax.set_xlim(self.lim[:2]); ax.set_ylim(self.lim[2:])
        self.background = _grab(fig)
        self.background[(self.background == 0).all(-1)] = BG
        t = np.linspace(0, 1, 256)[:, None]
        self.lut = (np.array(BG) * (1 - t) + np.array([232, 234, 240]) * t).astype(np.uint8)

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        x0, x1, y0, y1 = self.lim
        return (x - x0) / (x1 - x0) * self.w, (y1 - y) / (y1 - y0) * self.h

    def draw(self, canvas: np.ndarray, x0: int, y0: int, luminance: np.ndarray) -> None:
        """Write the hexagons into canvas with the panel's top-left at (x0, y0); luminance (2, 852), left eye first."""
        lum = np.clip(luminance.reshape(-1), 0.0, 1.0)
        code = (lum * 255 + 0.5).astype(np.uint8)
        canvas[self.rows + y0, self.cols + x0] = self.lut[code[self.labels]]


class TrackMap:
    def __init__(self, track: Path, centerline: np.ndarray, w: int, h: int):
        tris = parse_track(track)
        xyz = np.array([[[p.x, p.y, p.z] for p in (t.a, t.b, t.c)] for t in tris], dtype=np.float64)
        mat = np.array([t.material for t in tris])
        lo = centerline[:, [0, 2]].min(0) - MAP_MARGIN
        hi = centerline[:, [0, 2]].max(0) + MAP_MARGIN
        span = hi - lo
        self.scale = min(w / span[0], h / span[1])
        self.w, self.h = int(round(span[0] * self.scale)), int(round(span[1] * self.scale))
        self.lo, self.hi = lo, hi
        area = 0.5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        centre = xyz.mean(1)[:, [0, 2]]
        keep = (area < FLOOR_SLAB_AREA) & (centre > lo).all(1) & (centre < hi).all(1)
        xz = xyz[:, :, [0, 2]]
        fig, ax = _figure(self.w, self.h)
        ax.add_collection(PolyCollection(xz[keep & (mat != MATERIAL_ROAD)], facecolors=np.array(BLOCK) / 255, edgecolors="none"))
        ax.add_collection(PolyCollection(xz[keep & (mat == MATERIAL_ROAD)], facecolors=np.array(ROAD) / 255, edgecolors="none"))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        self.image = _grab(fig)
        self.image[(self.image == 0).all(-1)] = BG
        if self.image.shape[:2] != (self.h, self.w):
            raise RuntimeError(f"map raster is {self.image.shape}, expected {(self.h, self.w)}")

    def to_pixel(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """World (x, z) -> map pixel (col, row), world z up on screen."""
        return np.stack([(x - self.lo[0]) * self.scale, (self.hi[1] - z) * self.scale], axis=-1)


# ---------------------------------------------------------------- composer

def load_real_view(path: Path, count: int, width: int) -> np.ndarray:
    """(count, h, width, 3) game-view frames resized to the panel width; one per log frame, else error."""
    reader = imageio.get_reader(str(path))
    frames = []
    for frame in reader:
        if not frames:
            h = int(round(frame.shape[0] * width / frame.shape[1])) // 2 * 2
        frames.append(np.asarray(Image.fromarray(frame[:, :, :3]).resize((width, h), Image.LANCZOS)))
        if len(frames) > count:
            break
    reader.close()
    if len(frames) != count:
        raise ValueError(f"{path}: {len(frames)}{'+' if len(frames) > count else ''} frames, the log has {count}")
    return np.stack(frames)


class Composer:
    def __init__(self, log: dict[str, np.ndarray], track: str, size: tuple[int, int], fps: int,
                 real_view: Path | None = None, mushroom_body: Path | None = None, training_minutes: int = 0):
        self.fps = fps
        self.training_minutes = training_minutes
        sets = skeletons.load()
        if mushroom_body is not None:
            log = merge_mushroom_body(log, mushroom_body, sets)
        self.log = log
        self.mb_mode = mushroom_body is not None
        W, H = size
        self.W, self.H = W, H
        self.T = len(log["tick"])
        u = H / 1080  # layout unit
        self.f_title, self.f_label, self.f_hud, self.f_foot = _font(int(22 * u)), _font(int(15 * u)), _font(int(19 * u)), _font(int(14 * u))
        top, bottom = int(44 * u), H - int(34 * u)
        pad = int(12 * u)
        left_w = int(0.55 * W)
        hud_h = int(46 * u)
        # real view (optional) above the eyes
        self.real = None
        eye_top = top
        if real_view is not None:
            self.real = load_real_view(real_view, self.T, left_w - 2 * pad)
            self.real_box = (pad, top, left_w - pad, top + self.real.shape[1])
            eye_top = self.real_box[3] + pad
        # eye
        self.eye_box = (pad, eye_top, left_w - pad, bottom - hud_h)              # x0, y0, x1, y1
        self.hud_box = (pad, bottom - hud_h, left_w - pad, bottom)
        # brain
        right_x0 = left_w + pad
        brain_h = int(0.50 * (bottom - top))
        self.brain_box = (right_x0, top, W - pad, top + brain_h)
        # bottom right: map left, traces right
        by0 = top + brain_h + pad
        traces_w = int(0.34 * (W - pad - right_x0))
        self.map_box = (right_x0, by0, W - pad - traces_w - pad, bottom)
        self.trace_box = (W - pad - traces_w, by0, W - pad, bottom)

        eye = Eye.load()
        self.eye_panel = EyePanel(eye, self.eye_box[2] - self.eye_box[0], self.eye_box[3] - self.eye_box[1])
        spec = track_spec(track)
        self.track_name = spec.name
        self.centerline = read_centerline(spec.route_path(ROOT))
        self.track_map = TrackMap(spec.track_path(ROOT), self.centerline,
                                  self.map_box[2] - self.map_box[0], self.map_box[3] - self.map_box[1])
        self.map_origin = (self.map_box[0] + (self.map_box[2] - self.map_box[0] - self.track_map.w) // 2,
                           self.map_box[1] + (self.map_box[3] - self.map_box[1] - self.track_map.h) // 2)
        self.trail = self.track_map.to_pixel(log["x"], log["z"]) + np.array(self.map_origin)

        self.set_names = list(sets)
        self.n_neurons = sum(len(s) for s in sets.values())
        self.scale = {}
        for name in self.set_names:
            key = f"neurons/{name}"
            if key not in log:
                raise KeyError(f"log lacks {key}")
            if log[key].shape != (self.T, len(sets[name])):
                raise ValueError(f"{key} has shape {log[key].shape}, expected {(self.T, len(sets[name]))}")
            self.scale[name] = max(float(np.percentile(log[key], ACTIVITY_PERCENTILE)), ACTIVITY_MIN_SCALE)
        bw, bh = self.brain_box[2] - self.brain_box[0], self.brain_box[3] - self.brain_box[1]
        self.brain = BrainRenderer(sets, Camera.frontal(orbit_deg_per_s=ORBIT_DEG_PER_S), size=(bw, bh))

        # traces: (label, [(key or pair, colour)], scale)
        gf = 0.5 * (log["DNp01/L"] + log["DNp01/R"])
        if self.mb_mode:
            self.traces = [
                ("MBON value  (approach - avoid)", [(log["MB/value"], COL_GF)]),
                ("dopamine  PAM + / PPL1 -", [(log["MB/PAM"], COL_L), (log["MB/PPL1"], COL_R)]),
                ("LC4  loom  L / R", [(log["LC4/L"], COL_L), (log["LC4/R"], COL_R)]),
            ]
        else:
            self.traces = [
                ("DNp01  giant fiber  (brake)", [(gf, COL_GF)]),
                ("LC4  loom  L / R", [(log["LC4/L"], COL_L), (log["LC4/R"], COL_R)]),
                ("HS  optic flow  L / R", [(log["HS/L"], COL_L), (log["HS/R"], COL_R)]),
            ]
        self.trace_scale = [max(max(float(np.abs(s).max()) for s, _ in rows), 1e-9) for _, rows in self.traces]
        self.trace_signed = [any(float(s.min()) < 0 for s, _ in rows) for _, rows in self.traces]
        self.window = int(round(TRACE_SECONDS * fps))
        self.static = self._static_canvas()
        self.brain_seconds = 0.0

    # -- static layer

    def _static_canvas(self) -> np.ndarray:
        canvas = np.empty((self.H, self.W, 3), np.uint8)
        canvas[:] = BG
        x0, y0, x1, y1 = self.eye_box
        canvas[y0:y1, x0:x1] = self.eye_panel.background
        mx, my = self.map_origin
        canvas[my:my + self.track_map.h, mx:mx + self.track_map.w] = self.track_map.image
        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img)
        u = self.H / 1080
        title = TITLE_MB.format(minutes=self.training_minutes) if self.mb_mode else TITLE
        d.text((self.eye_box[0], int(10 * u)), title, font=self.f_title, fill=TEXT)
        d.text((self.eye_box[0], self.H - int(26 * u)), FOOTER_MB if self.mb_mode else FOOTER, font=self.f_foot, fill=DIM)
        # eye labels
        s2 = np.sqrt(2.0)
        for x, y, text, anchor in ((-2 * s2, s2 + 0.02, "left eye", "lb"), (0.0, s2 + 0.02, "front", "mb"),
                                   (2 * s2, s2 + 0.02, "right eye", "rb")):
            px, py = self.eye_panel.to_pixel(x, y)
            d.text((x0 + px, y0 + py), text, font=self.f_label, fill=DIM, anchor=anchor)
        if self.real is not None:
            d.text((self.real_box[0] + int(8 * u), self.real_box[3] - int(6 * u)), "game view, same lap",
                   font=self.f_label, fill=DIM, anchor="lb")
        d.text((self.map_box[0], self.map_box[1] + int(4 * u)), f"track {self.track_name}  |  route, trail",
               font=self.f_label, fill=DIM, anchor="lt")
        # centerline
        cl = self.track_map.to_pixel(self.centerline[:, 0], self.centerline[:, 2]) + np.array(self.map_origin)
        d.line([tuple(p) for p in cl], fill=CENTERLINE, width=max(1, int(1.5 * u)))
        # trace axes
        tx0, ty0, tx1, ty1 = self.trace_box
        row_h = (ty1 - ty0) // len(self.traces)
        for i, (label, _) in enumerate(self.traces):
            ry0 = ty0 + i * row_h
            d.text((tx0, ry0 + int(4 * u)), label, font=self.f_label, fill=DIM)
            d.line([(tx0, ry0 + row_h - int(8 * u)), (tx1, ry0 + row_h - int(8 * u))], fill=GRID, width=1)
        d.text((tx1, ty1 - int(2 * u)), f"last {TRACE_SECONDS:.0f} s", font=self.f_label, fill=DIM, anchor="rt")
        return np.asarray(img).copy()

    # -- per frame

    def activity(self, i: int) -> dict[str, np.ndarray]:
        return {name: np.clip(self.log[f"neurons/{name}"][i] / self.scale[name], 0.0, 1.0).astype(np.float32)
                for name in self.set_names}

    def frame(self, i: int) -> np.ndarray:
        log = self.log
        canvas = self.static.copy()
        if self.real is not None:
            rx0, ry0, rx1, ry1 = self.real_box
            canvas[ry0:ry1, rx0:rx1] = self.real[i]
        self.eye_panel.draw(canvas, self.eye_box[0], self.eye_box[1], log["eye"][i])
        t0 = time.perf_counter()
        brain = self.brain.frame(self.activity(i), i / self.fps)
        self.brain_seconds += time.perf_counter() - t0
        bx0, by0, bx1, by1 = self.brain_box
        if brain.shape != (by1 - by0, bx1 - bx0, 3):
            raise RuntimeError(f"brain frame is {brain.shape}, expected {(by1 - by0, bx1 - bx0, 3)}")
        canvas[by0:by1, bx0:bx1] = brain

        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img)
        u = self.H / 1080
        d.text((bx0 + int(8 * u), by0 + int(6 * u)),
               f"MaleCNS  {self.n_neurons} neurons, rate / p99  |  DN, LC, LPLC, HS/VS, T4/T5, L1/Mi1, KC, MBON, DAN",
               font=self.f_label, fill=DIM)
        # HUD
        hx0, hy0, hx1, hy1 = self.hud_box
        speed_kmh = float(log["speed"][i]) * 3.6
        race_s = float(log["tick"][i]) * TICK_DT
        hud = f"{speed_kmh:6.1f} km/h    t {race_s:6.2f} s    progress {float(log['progress'][i]):7.1f} m"
        cy = (hy0 + hy1) // 2
        d.text((hx0, cy), hud, font=self.f_hud, fill=TEXT, anchor="lm")
        bar_w, bar_h = int(220 * u), int(14 * u)
        sx1 = hx1 - int(40 * u)
        sx0 = sx1 - bar_w
        d.text((sx0 - int(12 * u), cy), "steer", font=self.f_label, fill=DIM, anchor="rm")
        d.rectangle([sx0, cy - bar_h // 2, sx1, cy + bar_h // 2], outline=GRID, width=1)
        mid = (sx0 + sx1) // 2
        steer = float(np.clip(log["steer"][i], -1.0, 1.0))
        end = mid + int(steer * bar_w / 2)
        d.rectangle([min(mid, end), cy - bar_h // 2 + 2, max(mid, end), cy + bar_h // 2 - 2], fill=TRAIL)
        d.line([(mid, cy - bar_h // 2), (mid, cy + bar_h // 2)], fill=DIM, width=1)
        brk = bool(log["brake"][i])
        bxa = sx0 - int(150 * u)
        d.text((bxa - int(12 * u), cy), "brake", font=self.f_label, fill=DIM, anchor="rm")
        d.rectangle([bxa, cy - bar_h // 2, bxa + int(60 * u), cy + bar_h // 2],
                    outline=BRAKE_ON if brk else GRID, fill=BRAKE_ON if brk else None, width=1)
        # trail + car
        if i >= 1:
            d.line([tuple(p) for p in self.trail[:i + 1]], fill=TRAIL, width=max(1, int(2 * u)))
        cx, cy2 = self.trail[i]
        r = int(5 * u)
        d.ellipse([cx - r, cy2 - r, cx + r, cy2 + r], fill=CAR, outline=TRAIL, width=2)
        # traces
        tx0, ty0, tx1, ty1 = self.trace_box
        row_h = (ty1 - ty0) // len(self.traces)
        lo = i - self.window + 1
        idx = np.arange(max(lo, 0), i + 1)
        xs = tx0 + (idx - lo) / (self.window - 1) * (tx1 - tx0)
        head = int(22 * u)
        for k, (_, rows) in enumerate(self.traces):
            ry0 = ty0 + k * row_h
            base = ry0 + row_h - int(8 * u)
            amp = row_h - head - int(10 * u)
            if self.trace_signed[k]:
                base -= amp // 2
                amp //= 2
                d.line([(tx0, base), (tx1, base)], fill=GRID, width=1)
            for series, colour in rows:
                ys = base - np.clip(series[idx] / self.trace_scale[k], -1.0, 1.0) * amp
                if len(idx) >= 2:
                    d.line(list(zip(xs.tolist(), ys.tolist())), fill=colour, width=max(1, int(1.5 * u)))
            if k == 0 and not self.mb_mode:
                on = idx[log["brake"][idx] != 0]
                for j, xb in zip(on, tx0 + (on - lo) / (self.window - 1) * (tx1 - tx0)):
                    d.line([(xb, base - int(4 * u)), (xb, base)], fill=BRAKE_ON, width=2)
        return np.asarray(img)

    def close(self) -> None:
        self.brain.close()


def write_video(composer: Composer, out: Path, fps: int, crf: int = 18) -> tuple[float, float]:
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), format="FFMPEG", mode="I", fps=fps, codec="libx264", pixelformat="yuv420p",
                                output_params=["-crf", str(crf), "-preset", "medium"], macro_block_size=8)
    compose_s = 0.0
    t_start = time.perf_counter()
    for i in range(composer.T):
        t0 = time.perf_counter()
        frame = composer.frame(i)
        compose_s += time.perf_counter() - t0
        writer.append_data(frame)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{composer.T} frames, {(i + 1) / compose_s:.1f} composed/s", file=sys.stderr, flush=True)
    writer.close()
    return compose_s, time.perf_counter() - t_start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080))
    parser.add_argument("--track", default="a01")
    parser.add_argument("--stills", default="", help="comma-separated times in seconds; PNGs next to the video")
    parser.add_argument("--real-view", type=Path, help="game-view video of the same lap, one frame per log frame")
    parser.add_argument("--mushroom-body", type=Path, help="tmnf_fly.mb_replay showcase npz of the same lap")
    parser.add_argument("--training-minutes", type=int, default=0, help="shown in the title in mushroom-body mode")
    args = parser.parse_args()
    W, H = args.size
    if W % 8 or H % 8:
        raise ValueError(f"size {W}x{H} must be a multiple of 8")
    log = load_log(args.log)
    t0 = time.perf_counter()
    composer = Composer(log, args.track, (W, H), args.fps, real_view=args.real_view, mushroom_body=args.mushroom_body,
                        training_minutes=args.training_minutes)
    print(f"setup {time.perf_counter() - t0:.1f} s; {composer.T} frames = {composer.T / args.fps:.2f} s at {args.fps} fps, {W}x{H}",
          file=sys.stderr, flush=True)
    try:
        stills = []
        for s in (t.strip() for t in args.stills.split(",") if t.strip()):
            t = float(s)
            i = int(round(t * args.fps))
            if not 0 <= i < composer.T:
                raise ValueError(f"still at {t} s is outside the log ({composer.T / args.fps:.2f} s)")
            path = args.out.with_name(f"{args.out.stem}_t{t:06.2f}s.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(path, composer.frame(i))
            stills.append(path)
            print(f"still {path}", file=sys.stderr, flush=True)
        composer.brain_seconds = 0.0
        compose_s, wall_s = write_video(composer, args.out, args.fps)
    finally:
        composer.close()
    n = composer.T
    print(f"{args.out}: {n} frames, {W}x{H}, {n / args.fps:.2f} s at {args.fps} fps\n"
          f"composition {n / compose_s:.2f} frames/s ({1000 * compose_s / n:.1f} ms/frame, of which brain render "
          f"{1000 * composer.brain_seconds / n:.1f} ms); with encode {n / wall_s:.2f} frames/s, wall {wall_s:.1f} s",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
