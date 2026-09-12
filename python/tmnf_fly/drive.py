"""Closed loop: car -> compound eye -> flyvis optic lobe -> MaleCNS brain -> descending neurons -> controls.

Per 20 ms (flyvis's training dt): the retina renders both eyes at the car's
pose, each eye's 852 measured ommatidia are resampled onto flyvis's 721-column
lattice rotated onto that eye's field (mirrored for the left eye), flyvis
integrates one frame, the 54 optic-lobe cell types it models are clamped onto
the same-typed MaleCNS neurons column by column, and the MaleCNS rate model
runs 20 x 1 ms substeps over the real wiring. The controls come from named
descending neurons only:

    brake  <- DNp01 (Giant Fiber, the looming escape command)
    steer  <- LC4/LPLC2 left-right imbalance (escape away from the looming side)
              + HS-cell left-right imbalance (optomotor heading stabilisation)
    gas    <- on, off while braking

Gains and thresholds are the engineered part (`Reflexes`); everything upstream
of the descending neurons is measured anatomy plus flyvis's fitted dynamics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from tmnf_fly import FLY_DIR, ROOT, brain_graph
from tmnf_fly.brain_model import BrainModel
from tmnf_fly.eye import Eye
from tmnf_fly.flyvis_periphery import FlyvisPeriphery, lattice_plane, plane_to_directions
from tmnf_fly.pose import car_poses
from tmnf_fly.retina import Retina
from tmnf_rl.env import TmnfVectorEnv
from tmnf_rl.inputs import InputSchedule
from tmnf_rl.tracks import track_spec

GRAPH = FLY_DIR / "brain_full.npz"
VISION_DT = 0.02          # flyvis training dt; two physics ticks
EYE_HEIGHT = 1.0          # metres above the car body centre
SETTLE_FRAMES = 40           # 0.8 s of vision at the start pose before the first control
MAX_LATTICE_GAP_DEG = 6.0  # a lattice column further than this from any ommatidium sees nothing
# flyvis type -> MaleCNS type; absent: Mi3, Tm28; single-cell/non-columnar: CT1(*), Lawf1/2, Am(Lai)
TYPE_MAP = {t: t for t in (
    "R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3", "L4", "L5", "C2", "C3", "Mi1", "Mi2", "Mi4", "Mi9",
    "Mi10", "Mi13", "Mi14", "Mi15", "T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c",
    "T5d", "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20", "Tm30", "TmY3",
    "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15", "TmY18")}
TYPE_MAP.update({"R7": "R7", "R8": "R8"})
for _r in ("R1", "R2", "R3", "R4", "R5", "R6"):
    TYPE_MAP[_r] = "R1-R6"


@dataclass
class Reflexes:
    clamp_gain: float = 0.5      # flyvis relu(V) -> MaleCNS clamped rate
    adapt_tau_s: float = 0.15    # clamped rates are the increase over this running mean (transient responses)
    brake_threshold: float = 2.5   # brake when the Giant Fiber rises above this multiple of its 1 s running mean
    brake_floor: float = 0.001     # ... and above this absolute rate
    loom_gain: float = 1.0       # steer per unit normalised (LC4+LPLC2) left-right asymmetry
    loom_floor: float = 0.01     # LC4+LPLC2 total rate at which the loom steer reaches full weight
    flow_gain: float = 10.0      # steer per unit (HS right - HS left) mean rate; negative = centering
    speed_cap_mps: float = 25.0  # the throttle reflex: no gas above this speed
    steer_limit: float = 1.0


def rotation_onto(target: np.ndarray) -> np.ndarray:
    """Rotation taking head +x to `target` (unit), keeping +z as up as possible."""
    x = target / np.linalg.norm(target)
    z = np.array([0.0, 0.0, 1.0]); z = z - z @ x * x; z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=1)


class Vision:
    """Retina -> per-eye flyvis lattice -> MaleCNS column vectors."""

    def __init__(self, eye: Eye, periphery: FlyvisPeriphery, graph: dict[str, np.ndarray], device: torch.device):
        self.eye, self.periphery, self.device = eye, periphery, device
        lattice_dirs = plane_to_directions(lattice_plane())
        self.lattice = {}
        self.resample = {}
        self.column_map = {}
        for side, dirs, cols in (("R", eye.dirs_right, eye.malecns_column_right),
                                 ("L", eye.dirs_left, eye.malecns_column_left)):
            mirror = np.array([1.0, -1.0, 1.0])
            # flyvis T4a prefers lattice -x and MaleCNS T4a is the front-to-back subtype (it carries
            # 40,693 of the T4->HS synapses), so lattice -x must point toward the back of the eye:
            # the lattice is laid onto the right eye with x reversed, and mirrored for the left eye.
            oriented = lattice_dirs * mirror
            if side == "R":
                rotated = oriented @ rotation_onto(dirs.mean(0)).T
            else:
                rotated = (oriented @ rotation_onto(dirs.mean(0) * mirror).T) * mirror
            self.lattice[side] = rotated
            self.resample[side] = self._idw(rotated, dirs)
            self.column_map[side] = self._columns(rotated, dirs, cols, graph[f"columns/{side}"])

    def _idw(self, lattice: np.ndarray, measured: np.ndarray, k: int = 3) -> torch.Tensor:
        ang = np.arccos(np.clip(lattice @ measured.T, -1, 1))
        nn = np.argpartition(ang, k - 1, axis=1)[:, :k]
        w = 1.0 / (np.take_along_axis(ang, nn, axis=1) + 1e-6)
        w /= w.sum(1, keepdims=True)
        mat = np.zeros(ang.shape, np.float32)
        np.put_along_axis(mat, nn, w.astype(np.float32), axis=1)
        mat[ang.min(1) > np.deg2rad(MAX_LATTICE_GAP_DEG)] = 0.0  # outside the eye: grey (0.5) added below
        self._outside = torch.as_tensor(mat.sum(1) == 0, device=self.device)
        return torch.as_tensor(mat, device=self.device)

    def _columns(self, lattice: np.ndarray, measured: np.ndarray, omm_cols: np.ndarray,
                 table: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """For each MaleCNS column with an ommatidium: the nearest lattice column within the gap limit."""
        lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(table)}
        col_idx, lat_idx = [], []
        ang = np.arccos(np.clip(measured @ lattice.T, -1, 1))  # (852, 721)
        for m in range(len(measured)):
            key = (int(omm_cols[m, 0]), int(omm_cols[m, 1]))
            if key not in lookup:
                continue
            j = int(ang[m].argmin())
            if ang[m, j] <= np.deg2rad(MAX_LATTICE_GAP_DEG):
                col_idx.append(lookup[key]); lat_idx.append(j)
        return (torch.as_tensor(col_idx, device=self.device, dtype=torch.long),
                torch.as_tensor(lat_idx, device=self.device, dtype=torch.long))

    def lattice_input(self, luminance: np.ndarray) -> torch.Tensor:
        """(2, 852) luminance -> (2, 721) lattice luminance, left eye first, grey outside the eye."""
        out = []
        for i, side in enumerate(("L", "R")):
            lum = torch.as_tensor(luminance[i], dtype=torch.float32, device=self.device)
            x = self.resample[side] @ lum
            out.append(torch.where(self._outside_of(side), torch.full_like(x, 0.5), x))
        return torch.stack(out)

    def _outside_of(self, side: str) -> torch.Tensor:
        return (self.resample[side].sum(1) == 0)

    def column_values(self, side: str, lattice_activity: torch.Tensor, n_columns: int) -> torch.Tensor:
        col_idx, lat_idx = self.column_map[side]
        values = torch.zeros(n_columns, device=self.device)
        values[col_idx] = lattice_activity[lat_idx]
        return values


class FlyDriver:
    def __init__(self, track: str, device: str = "cuda:0", reflexes: Reflexes | None = None,
                 sun: np.ndarray | None = None, stuck_grace_ticks: int = 500, off_track_grace_ticks: int = 100,
                 action_space: str = "analog"):
        self.device = torch.device(device)
        self.reflexes = reflexes or Reflexes()
        self.action_space = action_space
        # A reflex driver is slow; give the race the budget a 6 m/s average needs.
        self.env = TmnfVectorEnv(1, track=track, thread_count=1, action_repeat=1, action_space=action_space,
                                 stuck_grace_ticks=stuck_grace_ticks, off_track_grace_ticks=off_track_grace_ticks,
                                 max_race_ticks=60000, horizon_ticks=60000)
        self.eye = Eye.load()
        self.retina = Retina.from_track(track_spec(track).track_path(ROOT), self.eye,
                                        sun if sun is not None else np.array([0.5, 1.2, 0.3]))
        self.periphery = FlyvisPeriphery(self.device, batch_size=2)  # batch = (left eye, right eye)
        self.graph = brain_graph.load_graph(GRAPH)
        self.brain = BrainModel(self.graph, self.device)
        self.vision = Vision(self.eye, self.periphery, self.graph, self.device)
        self.types = [(f, m) for f, m in TYPE_MAP.items()
                      if f in self.periphery.layer_index and f"pop/{m}/R" in self.graph]
        self.log: list[dict] = []
        # Per-neuron rates of the rendered skeleton sets (tmnf_fly.skeletons), recorded every vision frame.
        sets = json.loads((FLY_DIR / "skeleton_sets.json").read_text())
        body_index = {int(b): i for i, b in enumerate(self.graph["body_id"])}
        self.render_sets = {name: torch.as_tensor([body_index[b] for b in ids], device=self.device)
                            for name, ids in sets.items()}
        self.render_log: list[dict[str, np.ndarray]] = []

    def reset(self) -> None:
        self.env.reset()
        self.periphery.reset(2)
        # Rates are clamped as the increase over the uniform-grey steady state: flyvis's fitted
        # resting voltages give many types a tonic relu(V) that would otherwise drive the loom
        # detectors and the Giant Fiber continuously.
        self.rest = {t: torch.relu(v) for t, v in self.periphery.activity().items()}
        self.adapted = {t: v.clone() for t, v in self.rest.items()}
        self.brain.reset()
        self.log.clear()
        self.render_log.clear()
        self.eye_log: list[np.ndarray] = []
        self.actions_log: list[tuple[float, int, int]] = []  # exact per-tick controls for the viewer replay
        self.speed = 0.0
        self.gf_slow = None

    def see_and_think(self) -> dict[str, float]:
        pos, rot, _ = car_poses(self.env)
        eye_pos = pos[0] + rot[0][:, 2] * EYE_HEIGHT
        lum = self.retina.render(eye_pos, rot[0])                      # (2, 852)
        act = self.periphery.step(self.vision.lattice_input(lum), VISION_DT)
        gain = self.reflexes.clamp_gain
        alpha = VISION_DT / self.reflexes.adapt_tau_s
        for flyvis_type, malecns_type in self.types:
            rate = torch.relu(act[flyvis_type])
            adapted = self.adapted[flyvis_type]
            adapted += alpha * (rate - adapted)
            rates = torch.relu(rate - adapted) * gain                     # (2, 721) transient increase
            for i, side in enumerate(("L", "R")):
                values = self.vision.column_values(side, rates[i], self.brain.n_columns[side])
                self.brain.clamp(f"{malecns_type}/{side}", self.brain.scatter_columns(side, malecns_type, values))
        self.brain.step({}, int(round(VISION_DT / self.brain.dt)))
        read = {}
        for name in ("DNp01", "LC4", "LPLC2", "HS", "DNa02", "DNp15", "DNp09", "T4a", "T5a", "LC10", "DN"):
            for side in ("L", "R"):
                read[f"{name}/{side}"] = self.brain.mean_rate(f"{name}/{side}")
        read["luminance_mean"] = float(lum.mean())
        self.render_log.append({name: self.brain.r[idx, 0].cpu().numpy() for name, idx in self.render_sets.items()})
        self.last_luminance = lum
        return read

    def controls(self, read: dict[str, float]) -> dict[str, np.ndarray]:
        r = self.reflexes
        loom_l = read["LC4/L"] + read["LPLC2/L"]
        loom_r = read["LC4/R"] + read["LPLC2/R"]
        # Escape away from the looming side; the asymmetry is normalised so a head-on wall (both
        # sides rising together) still resolves to whichever side is freer, and scaled by the total
        # so quiet scenes do not steer.
        total = loom_l + loom_r
        loom = (loom_l - loom_r) / (total + r.loom_floor) * min(1.0, total / r.loom_floor)
        flow = read["HS/R"] - read["HS/L"]
        steer = float(np.clip(r.loom_gain * loom + r.flow_gain * flow, -r.steer_limit, r.steer_limit))
        gf = (read["DNp01/L"] + read["DNp01/R"]) * 0.5
        self.gf_slow = gf if self.gf_slow is None else self.gf_slow + (VISION_DT / 1.0) * (gf - self.gf_slow)
        brake = gf > max(r.brake_floor, r.brake_threshold * self.gf_slow)
        gas = (not brake) and self.speed < r.speed_cap_mps
        return {"steer": np.array([steer], np.float32), "gas": np.array([1 if gas else 0], np.int8),
                "brake": np.array([1 if brake else 0], np.int8)}

    def run(self, max_ticks: int, scripted=None, record: Path | None = None) -> dict:
        """Drive one episode. `scripted(tick) -> controls` overrides the reflexes (calibration)."""
        self.reset()
        # Let the eye and brain adapt to the static start scene before the car moves: the scene
        # onset is itself a full-field transient that would otherwise fire the Giant Fiber.
        for _ in range(SETTLE_FRAMES):
            self.see_and_think()
        started = time.perf_counter()
        action = {"steer": np.zeros(1, np.float32), "gas": np.ones(1, np.int8), "brake": np.zeros(1, np.int8)}
        read = None
        outcome = {"ticks": 0, "reason": None, "race_time_ms": None}
        for tick in range(max_ticks):
            if tick % 2 == 0:
                read = self.see_and_think()
                action = scripted(tick) if scripted is not None else self.controls(read)
                pos, _, vel = car_poses(self.env)
                self.speed = float(np.linalg.norm(vel[0]))
                self.eye_log.append(self.last_luminance.copy())
                steer, brake = control_summary(action)
                self.log.append({"tick": tick, **read, "steer": steer, "brake": brake, "speed": float(np.linalg.norm(vel[0])),
                                 "x": float(pos[0][0]), "y": float(pos[0][1]), "z": float(pos[0][2]),
                                 "progress": float(self.env.observations["race"][0, 4])})
            elif scripted is not None:
                action = scripted(tick)
            if self.action_space == "analog":
                self.actions_log.append((float(action["steer"][0]), int(action["gas"][0]), int(action["brake"][0])))
            _, _, terminated, truncated, info = self.env.step(action)
            if terminated[0] or truncated[0]:
                outcome = {"ticks": tick + 1, "reason": int(info["termination_reason"][0]),
                           "race_time_ms": int(info["race_time_ms"][0])}
                break
        outcome["ticks"] = outcome["ticks"] or max_ticks
        outcome["wall_s"] = time.perf_counter() - started
        outcome["progress_max"] = max(e["progress"] for e in self.log) if self.log else 0.0
        if record is not None:
            record.parent.mkdir(parents=True, exist_ok=True)
            keys = list(self.log[0])
            frames = self.render_log[SETTLE_FRAMES:]  # aligned with self.log (one entry per vision frame)
            np.savez(record, **{k: np.array([e[k] for e in self.log]) for k in keys},
                     **{f"neurons/{name}": np.stack([f[name] for f in frames]) for name in self.render_sets},
                     eye=np.stack(self.eye_log), outcome=json.dumps(outcome), reflexes=json.dumps(asdict(self.reflexes)))
            if self.actions_log:
                schedule = InputSchedule("analog", np.asarray(self.actions_log, dtype=np.float32))
                outcome["inputs_sha256"] = schedule.write(record.with_suffix(".inputs.bin"))
        return outcome

    def close(self) -> None:
        self.env.close()


def control_summary(action) -> tuple[float, int]:
    """(steer in [-1, 1], brake flag) for an analog dict or a discrete action array (tmnf discrete
    actions: index % 3 is steer left/none/right, index // 3 in {2, 3} brakes)."""
    if isinstance(action, dict):
        return float(action["steer"][0]), int(action["brake"][0])
    a = int(np.asarray(action).reshape(-1)[0])
    return float(a % 3 - 1), int(a // 3 in (2, 3))


def scripted_discrete(actions: np.ndarray):
    """Replay a recorded per-tick discrete schedule; holds the last action past its end."""
    def f(tick):
        return np.array([actions[min(tick, len(actions) - 1)]], dtype=np.int32)
    return f


def scripted_turn(steer: float, start_tick: int = 100):
    def f(tick):
        s = steer if tick >= start_tick else 0.0
        return {"steer": np.array([s], np.float32), "gas": np.ones(1, np.int8), "brake": np.zeros(1, np.int8)}
    return f


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="a01")
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--calibrate", action="store_true", help="scripted straight / left / right runs, log only")
    parser.add_argument("--replay-inputs", type=Path, help="observer mode: replay a discrete per-tick schedule (.inputs.bin) while the eye and brain run")
    parser.add_argument("--out", type=Path, default=FLY_DIR / "drive")
    parser.add_argument("--loom-gain", type=float, default=Reflexes.loom_gain)
    parser.add_argument("--flow-gain", type=float, default=Reflexes.flow_gain)
    parser.add_argument("--brake-threshold", type=float, default=Reflexes.brake_threshold)
    parser.add_argument("--clamp-gain", type=float, default=Reflexes.clamp_gain)
    parser.add_argument("--speed-cap", type=float, default=Reflexes.speed_cap_mps)
    parser.add_argument("--adapt-tau", type=float, default=Reflexes.adapt_tau_s)
    parser.add_argument("--loom-floor", type=float, default=Reflexes.loom_floor)
    args = parser.parse_args()
    reflexes = Reflexes(clamp_gain=args.clamp_gain, adapt_tau_s=args.adapt_tau, brake_threshold=args.brake_threshold,
                        loom_gain=args.loom_gain, loom_floor=args.loom_floor, flow_gain=args.flow_gain,
                        speed_cap_mps=args.speed_cap)
    if args.replay_inputs:
        # Observer: the schedule decides the outcome, never the env's grace timers.
        driver = FlyDriver(args.track, reflexes=reflexes, action_space="discrete",
                           stuck_grace_ticks=1_000_000, off_track_grace_ticks=1_000_000)
    else:
        driver = FlyDriver(args.track, reflexes=reflexes)
    print(f"types clamped: {len(driver.types)}; columns mapped L {len(driver.vision.column_map['L'][0])} "
          f"R {len(driver.vision.column_map['R'][0])}", flush=True)
    try:
        if args.replay_inputs:
            from tmnf_rl.inputs import decode_schedule
            schedule = decode_schedule(args.replay_inputs.read_bytes())
            if schedule.mode != "discrete":
                raise ValueError("observer replay expects a discrete schedule")
            outcome = driver.run(schedule.tick_count, scripted=scripted_discrete(schedule.actions),
                                 record=args.out / f"observe_{args.track}.npz")
            print(outcome, flush=True)
        elif args.calibrate:
            for name, script in (("straight", scripted_turn(0.0)), ("left", scripted_turn(-0.5)), ("right", scripted_turn(0.5))):
                outcome = driver.run(args.ticks, scripted=script, record=args.out / f"calib_{name}.npz")
                print(name, outcome, flush=True)
        else:
            outcome = driver.run(args.ticks, record=args.out / f"reflex_{args.track}.npz")
            print(outcome, flush=True)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
