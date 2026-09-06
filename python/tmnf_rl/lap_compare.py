"""Where does one lap lose time against another? Replay both, align by arc.

usage: python -m tmnf_rl.lap_compare REFERENCE.inputs.bin OTHER.inputs.bin \
           [--track a01] [--bin-m 100] [--physics-library PATH] [--json OUT]

Both schedules are TMNFRaceInputs files (discrete or analog, detected from
the records). Each is replayed tick by tick in one environment; every tick's
route arc position, speed, lateral offset, wheel contacts and inputs are
kept. The laps are aligned by arc position (the first tick at which the car
passes each metre of the route) so the time-behind curve is a function of
where on the track the car is, not of the clock.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmnf_rl.env import TERMINATION_NAMES, TmnfVectorEnv
from tmnf_rl.inputs import InputSchedule, decode_schedule
from tmnf_rl.tracks import project_root

FINISH_REASON = 1
# race observation columns (env.py: three controls, then the race fields)
ARC, PROGRESS, LATERAL, HALF_WIDTH, REMAINING, ELAPSED, NEXT_CP = 3, 4, 5, 6, 7, 8, 9


@dataclass
class LapTrace:
    name: str
    mode: str
    ticks: int
    finish_ms: int | None
    termination: str | None
    checkpoint_ticks: list[int]
    arc: np.ndarray  # [ticks + 1] route arc position after each tick (index 0 = start)
    speed: np.ndarray
    lateral: np.ndarray
    half_width: np.ndarray
    contacts: np.ndarray  # wheels on the ground, 0..4
    height: np.ndarray  # world y
    position: np.ndarray  # [ticks + 1, 3] world position
    steer: np.ndarray  # signed, -1..1 (discrete: -1/0/1)
    gas: np.ndarray
    brake: np.ndarray


def replay(name: str, schedule: InputSchedule, *, track: str, library: Path) -> LapTrace:
    env = TmnfVectorEnv(
        1, track=track, thread_count=1, action_repeat=1, action_space=schedule.mode, library_path=library
    )
    try:
        obs, _ = env.reset()
        n = schedule.tick_count
        arc = np.zeros(n + 1, np.float64)
        speed = np.zeros(n + 1)
        lateral = np.zeros(n + 1)
        half_width = np.zeros(n + 1)
        contacts = np.zeros(n + 1)
        height = np.zeros(n + 1)
        position = np.zeros((n + 1, 3))
        steer = np.zeros(n)
        gas = np.zeros(n)
        brake = np.zeros(n)

        def record(i: int, race: np.ndarray, vehicle: np.ndarray) -> None:
            arc[i] = race[PROGRESS]
            speed[i] = float(np.linalg.norm(vehicle[7:10]))
            lateral[i] = race[LATERAL]
            half_width[i] = race[HALF_WIDTH]
            contacts[i] = float((vehicle[21:25] >= 0.5).sum())
            height[i] = vehicle[1]
            position[i] = vehicle[0:3]

        record(0, obs["race"][0], obs["vehicle"][0])
        previous_cp = float(obs["race"][0, NEXT_CP])
        cp_ticks: list[int] = []
        finish_ms = None
        reason = None
        for tick in range(n):
            if schedule.mode == "discrete":
                a = int(schedule.actions[tick])
                action = np.array([a], dtype=np.int64)
                steer[tick] = (a % 3) - 1
                gas[tick] = float(a // 3 in (1, 3))
                brake[tick] = float(a // 3 in (2, 3))
            else:
                s, g, b = schedule.actions[tick]
                action = {
                    "steer": np.array([s], dtype=np.float32),
                    "gas": np.array([int(g)], dtype=np.int8),
                    "brake": np.array([int(b)], dtype=np.int8),
                }
                steer[tick], gas[tick], brake[tick] = float(s), float(g), float(b)
            obs, _, terminated, truncated, info = env.step(action)
            ended = bool(terminated[0] or truncated[0])
            race = env.final_observations["race"][0] if ended else obs["race"][0]
            vehicle = env.final_observations["vehicle"][0] if ended else obs["vehicle"][0]
            record(tick + 1, race, vehicle)
            cp = float(race[NEXT_CP])
            if cp > previous_cp:
                cp_ticks.append(tick + 1)
            previous_cp = cp
            if ended:
                reason = int(info["termination_reason"][0])
                if reason == FINISH_REASON:
                    finish_ms = int(info["race_time_ms"][0])
                elif tick != n - 1:
                    raise RuntimeError(f"{name}: episode failed at tick {tick + 1} of {n}")
                n = tick + 1  # a finish before the file's end: the rest is padding
                break
        return LapTrace(
            name, schedule.mode, n, finish_ms, TERMINATION_NAMES.get(reason) if reason else None,
            cp_ticks, arc[: n + 1], speed[: n + 1], lateral[: n + 1], half_width[: n + 1], contacts[: n + 1],
            height[: n + 1], position[: n + 1], steer[:n], gas[:n], brake[:n],
        )
    finally:
        env.close()


def time_at_arc(trace: LapTrace, grid: np.ndarray) -> np.ndarray:
    """First tick (fractional, in ms) at which the car reaches each grid arc."""
    arc = np.maximum.accumulate(trace.arc)  # monotone: ignore the rare backwards metre
    ticks = np.arange(arc.size, dtype=np.float64)
    out = np.interp(grid, arc, ticks, right=np.nan)
    out[grid > arc[-1]] = np.nan
    return out * 10.0


def at_arc(values: np.ndarray, trace: LapTrace, grid: np.ndarray) -> np.ndarray:
    arc = np.maximum.accumulate(trace.arc)
    return np.interp(grid, arc, values[: arc.size], right=np.nan)


def section_table(ref: LapTrace, other: LapTrace, bin_m: float) -> list[dict]:
    end = float(min(np.max(ref.arc), np.max(other.arc)))
    edges = np.arange(0.0, end + bin_m, bin_m)
    edges[-1] = min(edges[-1], end)
    grid = np.arange(0.0, end, 1.0)
    t_ref, t_other = time_at_arc(ref, grid), time_at_arc(other, grid)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (grid >= lo) & (grid < hi)
        if not sel.any():
            continue
        i0, i1 = int(np.flatnonzero(sel)[0]), int(np.flatnonzero(sel)[-1])
        behind_in = float(t_other[i0] - t_ref[i0])
        behind_out = float(t_other[i1] - t_ref[i1])

        def stats(trace: LapTrace) -> dict:
            tick_lo = int(np.nanmin(time_at_arc(trace, np.array([lo])) / 10.0))
            tick_hi = int(np.nanmin(time_at_arc(trace, np.array([hi - 1e-6])) / 10.0))
            tick_hi = max(tick_hi, tick_lo + 1)
            s = slice(tick_lo, tick_hi)
            n = tick_hi - tick_lo
            steps = np.diff(trace.position[tick_lo : tick_hi + 1], axis=0)
            path_m = float(np.linalg.norm(steps, axis=1).sum())
            arc_m = float(trace.arc[tick_hi] - trace.arc[tick_lo])
            return {
                "ticks": n,
                "path_m": path_m,
                "extra_path_m": path_m - arc_m,
                "mean_speed": path_m / (n * 0.01),
                "speed_in": float(trace.speed[tick_lo]),
                "speed_min": float(trace.speed[s].min()),
                "speed_out": float(trace.speed[tick_hi]),
                "lateral_mean": float(np.mean(trace.lateral[s])),
                "lateral_abs_max": float(np.max(np.abs(trace.lateral[s]))),
                "half_width_mean": float(np.mean(trace.half_width[s])),
                "airborne_ticks": int((trace.contacts[s] == 0).sum()),
                "brake_ticks": int(trace.brake[s].sum()),
                "gas_off_ticks": int(n - trace.gas[s].sum()),
                "steer_left_ticks": int((trace.steer[s] < -0.05).sum()),
                "steer_right_ticks": int((trace.steer[s] > 0.05).sum()),
                "steer_abs_mean": float(np.mean(np.abs(trace.steer[s]))),
            }

        rows.append({
            "from_m": float(lo), "to_m": float(hi),
            "behind_in_ms": behind_in, "behind_out_ms": behind_out,
            "lost_ms": behind_out - behind_in,
            "ref": stats(ref), "other": stats(other),
        })
    return rows


def format_table(rows: list[dict], ref_name: str, other_name: str) -> str:
    lines = [
        f"| section (m) | behind at exit (ms) | lost here (ms) | ticks {ref_name} / {other_name} "
        f"| path m (extra) {ref_name} | {other_name} | speed in / min / out {ref_name} | {other_name} "
        f"| lateral mean / |max| {ref_name} | {other_name} | air / brake / gas-off ticks {ref_name} | {other_name} |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        a, b = r["ref"], r["other"]
        lines.append(
            f"| {r['from_m']:.0f}-{r['to_m']:.0f} | {r['behind_out_ms']:+.0f} | {r['lost_ms']:+.0f} "
            f"| {a['ticks']} / {b['ticks']} "
            f"| {a['path_m']:.1f} ({a['extra_path_m']:+.1f}) | {b['path_m']:.1f} ({b['extra_path_m']:+.1f}) "
            f"| {a['speed_in']:.1f} / {a['speed_min']:.1f} / {a['speed_out']:.1f} "
            f"| {b['speed_in']:.1f} / {b['speed_min']:.1f} / {b['speed_out']:.1f} "
            f"| {a['lateral_mean']:+.1f} / {a['lateral_abs_max']:.1f} | {b['lateral_mean']:+.1f} / {b['lateral_abs_max']:.1f} "
            f"| {a['airborne_ticks']} / {a['brake_ticks']} / {a['gas_off_ticks']} "
            f"| {b['airborne_ticks']} / {b['brake_ticks']} / {b['gas_off_ticks']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("other", type=Path)
    parser.add_argument("--track", default="a01")
    parser.add_argument("--bin-m", type=float, default=100.0)
    parser.add_argument("--physics-library", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    root = project_root()
    library = args.physics_library or root / "build" / "libtmnf_physics.so"
    ref = replay(args.reference.stem, decode_schedule(args.reference.read_bytes()), track=args.track, library=library)
    other = replay(args.other.stem, decode_schedule(args.other.read_bytes()), track=args.track, library=library)
    for t in (ref, other):
        print(
            f"{t.name}: {t.mode}, {t.ticks} ticks, {t.termination} at {t.finish_ms} ms, "
            f"checkpoints at ticks {t.checkpoint_ticks}, peak speed {t.speed.max():.1f} m/s, "
            f"route end {t.arc.max():.1f} m"
        )
    rows = section_table(ref, other, args.bin_m)
    print(format_table(rows, ref.name, other.name))
    if args.json:
        args.json.write_text(json.dumps({
            "reference": {"name": ref.name, "finish_ms": ref.finish_ms, "checkpoint_ticks": ref.checkpoint_ticks},
            "other": {"name": other.name, "finish_ms": other.finish_ms, "checkpoint_ticks": other.checkpoint_ticks},
            "sections": rows,
        }, indent=1))


if __name__ == "__main__":
    main()
