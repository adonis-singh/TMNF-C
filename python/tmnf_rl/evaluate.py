"""Evaluate a policy from the official start: ``python -m tmnf_rl.evaluate``.

Accepts a run id (uses ``<runs_root>/<run_id>/policy.pt``), a ``policy.pt`` or
a ``checkpoints/latest.pt``. Reports finish rate, median/best lap, exact
checkpoint splits of the best lap (from a tick-by-tick replay of the recorded
schedule) and optionally exports the inputs and a viewer scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tmnf_rl.utils import enforce_cpu_affinity, require_single_gpu_env

require_single_gpu_env()

import torch  # noqa: E402

from tmnf_rl import provenance  # noqa: E402
from tmnf_rl.agents.ppo import Agent, make_env, select_device  # noqa: E402
from tmnf_rl.agents.td3 import OffPolicyAgent  # noqa: E402
from tmnf_rl.config import config_from_dict  # noqa: E402
from tmnf_rl.evaluation import evaluate_full_start, replay_schedule  # noqa: E402
from tmnf_rl.registry import RunRegistry  # noqa: E402
from tmnf_rl.replays import export_scene  # noqa: E402
from tmnf_rl.tracks import project_root, track_spec  # noqa: E402
from tmnf_rl.utils import write_json_atomic  # noqa: E402


def resolve_policy_path(target: str, runs_root: Path) -> Path:
    path = Path(target)
    if path.is_file():
        return path.resolve()
    candidate = runs_root / target / "policy.pt"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"{target!r} is neither a policy file nor a run id under {runs_root}"
    )


def load_policy(path: Path, device: torch.device) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    if "config" not in state:
        raise RuntimeError(f"{path} has no 'config' entry")
    if "normalizer" in state:
        raise RuntimeError(
            f"{path} carries a running observation normalizer; policies trained "
            "before the fixed-scale encoder cannot be evaluated by this trainer"
        )
    stored_config = dict(state["config"])
    migrations = []
    # The pure-RL B05 checkpoint predates removal of demonstrations. Its
    # unused empty field is safe to retire; a nonempty field remains an error
    # rather than silently relabelling a demonstration-assisted checkpoint.
    if stored_config.get("demonstration_inputs") == "":
        del stored_config["demonstration_inputs"]
        migrations.append("removed unused demonstration_inputs field")
    config = config_from_dict(stored_config)
    if config.algorithm == "td3":
        agent = OffPolicyAgent(
            config.hidden_size, config.td3_critic_width, config.td3_bins,
            *state["summary"]["return_support"], config.td3_exploration_std, config.td3_pedal_epsilon, config.encoder_version,
        ).to(device)
    else:
        agent = Agent(config.arch, config.hidden_size, config.action_space, encoder_version=config.encoder_version).to(device)
    agent.load_state_dict(state["agent"])
    agent.eval()
    meta = {
        "policy_path": str(path),
        "policy_sha256": provenance.sha256_file(path),
        "training_physics_sha256": (
            state.get("physics_sha256")
            or (state.get("summary") or {}).get("physics_sha256")
        ),
        "training_run_id": (state.get("summary") or {}).get("run_id") or state.get("run_id"),
        "checkpoint_update": (state.get("counters") or {}).get("update"),
        "config_migrations": migrations,
    }
    return agent, config.to_dict(), meta


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="tmnf_rl.evaluate")
    parser.add_argument("target", help="run id, policy.pt or checkpoints/latest.pt")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument(
        "--episodes", type=int, default=256,
        help="episodes for the sampled-policy evaluation; the greedy trajectory uses one round of --envs",
    )
    parser.add_argument("--sample-seed", type=int, default=0, help="Torch seed for the sampled evaluation")
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--physics-library", type=Path, default=None)
    parser.add_argument("--export-scene", type=Path, default=None)
    parser.add_argument("--write-inputs", type=Path, default=None)
    parser.add_argument(
        "--inputs-ticks", type=int, default=None,
        help="pad the written schedule with released inputs to this many ticks",
    )
    parser.add_argument("--output", type=Path, default=None, help="write JSON report here")
    parser.add_argument("--no-splits", action="store_true", help="skip the tick replay")
    args = parser.parse_args(argv)
    if args.envs <= 0 or args.episodes <= 0:
        raise ValueError("envs and episodes must be positive")
    outputs = [p.resolve() for p in (
        args.output, args.write_inputs, args.export_scene,
        args.export_scene.with_suffix(".inputs.bin") if args.export_scene else None,
    ) if p is not None]
    if len(outputs) != len(set(outputs)):
        raise ValueError("output paths must be distinct, including the scene's .inputs.bin file")
    enforce_cpu_affinity()

    root = project_root()
    runs_root = args.runs_root or (root / "build" / "runs")
    policy_path = resolve_policy_path(args.target, runs_root)
    torch.set_num_threads(1)
    device, gpu_name = select_device()
    agent, config_dict, meta = load_policy(policy_path, device)
    config = config_from_dict(config_dict)
    physics_library = (
        args.physics_library.resolve()
        if args.physics_library is not None
        else (root / config.physics_library).resolve()
    )
    physics_sha256 = provenance.sha256_file(physics_library)
    env = make_env(config, args.envs, physics_library)
    try:
        # Greedy: one round of the environments is the whole trajectory set
        # (F29); --episodes drives the sampled evaluation.
        result = evaluate_full_start(
            agent, env, args.envs, device, pedal_hold_decisions=config.pedal_hold_decisions
        )
        sampled = evaluate_full_start(
            agent, env, args.episodes, device, sampled=True, sample_seed=args.sample_seed,
            pedal_hold_decisions=config.pedal_hold_decisions,
        )
    finally:
        env.close()

    report: dict[str, Any] = {
        "event": "evaluate",
        "target": args.target,
        **meta,
        "track": config.track,
        "action_space": config.action_space,
        "action_repeat": config.action_repeat,
        "envs": args.envs,
        "episodes": args.episodes,
        "gpu": gpu_name,
        "physics_library": str(physics_library),
        "physics_sha256": physics_sha256,
        "physics_matches_training": (
            meta["training_physics_sha256"] == physics_sha256
            if meta["training_physics_sha256"]
            else None
        ),
        "result": result.to_json(),
        "sampled": sampled.to_json(),
        "sample_seed": args.sample_seed,
        "best_lap": None,
    }
    if result.best_schedule is not None and result.best_lap_ms is not None:
        best: dict[str, Any] = {
            "lap_ms": result.best_lap_ms,
            "ticks": result.best_schedule.tick_count,
            "schedule_sha256": result.best_schedule.sha256(),
        }
        if not args.no_splits:
            replay = replay_schedule(
                result.best_schedule,
                track=config.track,
                library_path=physics_library,
                max_race_ticks=env.max_race_ticks,
                horizon_ticks=env.horizon_ticks,
                off_track_grace_ticks=config.off_track_grace_ticks,
                stuck_grace_ticks=config.stuck_grace_ticks,
                stuck_progress_epsilon=config.stuck_progress_epsilon,
                discount_per_tick=config.discount_per_tick,
                root=root,
            )
            if replay["finish_ms"] != result.best_lap_ms:
                raise RuntimeError(
                    f"tick replay of the recorded schedule finished at "
                    f"{replay['finish_ms']} ms, evaluation said {result.best_lap_ms} ms"
                )
            best["checkpoint_splits_ms"] = replay["checkpoint_splits_ms"]
            best["checkpoint_count"] = replay["checkpoint_count"]
        if args.write_inputs is not None:
            best["inputs"] = str(args.write_inputs)
            best["inputs_sha256"] = result.best_schedule.write(
                args.write_inputs, args.inputs_ticks
            )
            best["inputs_ticks"] = args.inputs_ticks or result.best_schedule.tick_count
        if args.export_scene is not None:
            info = export_scene(
                spec=track_spec(config.track, root),
                schedule=result.best_schedule,
                output=args.export_scene,
                expected_finish_ms=result.best_lap_ms,
                root=root,
            )
            best["scene"] = info["scene"]
            best["scene_inputs"] = info["inputs"]
            best["world_lap_ms"] = info["finish_ms"]
            best["matches_world"] = info["matches_env"]
            best["scene_checkpoint_ticks"] = info["scene_checkpoint_ticks"]
        report["best_lap"] = best
    if args.output is not None:
        write_json_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
