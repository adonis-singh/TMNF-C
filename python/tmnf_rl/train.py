"""Registered PPO training run: ``python -m tmnf_rl.train [--config f] [flags]``.

Every run lives under ``<runs_root>/<run_id>/`` and is listed in
``<runs_root>/index.json``. ``--resume RUN_ID`` continues from
``checkpoints/latest.pt``.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from tmnf_rl.utils import enforce_cpu_affinity, require_single_gpu_env

require_single_gpu_env()

import torch  # noqa: E402

from tmnf_rl import provenance  # noqa: E402
from tmnf_rl.agents.ppo import (  # noqa: E402
    PPOTrainer,
    make_env,
    seed_everything,
    make_train_env,
    select_device,
)
from tmnf_rl.agents.td3 import OffPolicyTrainer  # noqa: E402
from tmnf_rl.config import TrainConfig, config_from_dict, parse_train_args  # noqa: E402
from tmnf_rl.evaluation import EvaluationResult  # noqa: E402
from tmnf_rl.registry import (  # noqa: E402
    HeartbeatThread,
    RunRecord,
    RunRegistry,
    new_run_payload,
    pid_start_ticks,
)
from tmnf_rl.replays import export_scene, require_exporter  # noqa: E402
from tmnf_rl.spectate import SpectateBuffer, SpectateServer  # noqa: E402
from tmnf_rl.tracks import project_root, track_spec  # noqa: E402
from tmnf_rl.utils import utc_now_iso, write_json_atomic  # noqa: E402


class RegistrySink:
    """Trainer side channel: run.json, evals/*.json, replays/*.json."""

    def __init__(
        self,
        *,
        record: RunRecord,
        config: TrainConfig,
        run_dir: Path,
        root: Path,
        spectate_buffer: SpectateBuffer | None,
        provenance_info: dict[str, Any],
    ) -> None:
        self.record = record
        self.config = config
        self.run_dir = run_dir
        self.root = root
        self.spectate_buffer = spectate_buffer
        self.provenance_info = provenance_info
        self.spec = track_spec(config.track, root)
        self.best_replay_lap_ms: int | None = min(
            (int(replay["lap_ms"]) for replay in record.payload["replays"]),
            default=None,
        )
        self.latest_result: EvaluationResult | None = None
        self.latest_minute: float | None = None
        self.exported_latest = False

    def on_start(self, event: dict[str, Any]) -> None:
        print(
            json.dumps({**event, "provenance": self.provenance_info}, sort_keys=True),
            flush=True,
        )
        # The budgets actually used (derived per track when args say 0).
        self.record.stage(
            environment={
                "route_length_m": event["route_length_m"],
                "lap_count": event["lap_count"],
                "horizon_ticks": event["horizon_ticks"],
                "max_race_ticks": event["max_race_ticks"],
            }
        )

    def on_update(self, row: dict[str, Any], summary: dict[str, Any]) -> None:
        del row
        self.record.stage(summary=summary)

    def on_checkpoint(self, path: Path, update: int) -> None:
        self.record.stage(
            checkpoint={
                "path": str(path.relative_to(self.run_dir)),
                "update": update,
                "saved_at": utc_now_iso(),
            }
        )

    def on_evaluation(
        self, evaluation: dict[str, Any], result: EvaluationResult, sampled: EvaluationResult,
        policy: dict[str, Any] | None = None,
    ) -> None:
        minute = float(evaluation["scheduled_minutes"])
        evals_dir = self.run_dir / "evals"
        payload = {
            "run_id": self.record.run_id,
            "scheduled_minutes": minute,
            "wall_time_s": evaluation["wall_time_s"],
            "physics_steps": evaluation["physics_steps"],
            "policy": policy,
            "row": evaluation,
            # `result` is the greedy trajectory (the replayed lap); `sampled`
            # is the policy sampled from the same start.
            "result": result.to_json(),
            "sampled": sampled.to_json(),
            "replay": None,
        }
        self.latest_result = result
        self.latest_minute = minute
        self.exported_latest = False
        if result.best_lap_ms is not None and (
            self.best_replay_lap_ms is None or result.best_lap_ms < self.best_replay_lap_ms
        ):
            payload["replay"] = self.export_replay(
                label=f"eval_{minute:g}m", result=result, minute=minute
            )
            self.exported_latest = True
        write_json_atomic(evals_dir / f"{minute:g}.json", payload)
        self.record.stage(summary=self.record.payload.get("summary", {}))

    def export_replay(
        self, *, label: str, result: EvaluationResult, minute: float
    ) -> dict[str, Any]:
        assert result.best_schedule is not None and result.best_lap_ms is not None
        scene_name = f"{label}_env{result.best_lap_ms}ms.json"
        scene_path = self.run_dir / "replays" / scene_name
        info = export_scene(
            spec=self.spec,
            schedule=result.best_schedule,
            output=scene_path,
            expected_finish_ms=result.best_lap_ms,
            root=self.root,
        )
        replay = {
            "label": label,
            # lap_ms is the vec-env lap the evaluation measured; world_lap_ms
            # is what the game-validated World path does with the same inputs
            # (None if it does not finish). Since native d09dc00 they must
            # agree; a mismatch is a physics regression (footgun F2).
            "lap_ms": int(result.best_lap_ms),
            "world_lap_ms": info["finish_ms"],
            "matches_world": bool(info["matches_env"]),
            "scene": f"replays/{scene_name}",
            "inputs": f"replays/{Path(info['inputs']).name}",
            "inputs_sha256": info["inputs_sha256"],
            "ticks": info["tick_count"],
            "scene_checkpoint_ticks": info["scene_checkpoint_ticks"],
            "minute": minute,
            "exported_at": utc_now_iso(),
        }
        if not replay["matches_world"]:
            print(
                json.dumps(
                    {
                        "event": "replay_lap_mismatch",
                        "env_lap_ms": replay["lap_ms"],
                        "world_lap_ms": replay["world_lap_ms"],
                        "footgun": "F2 vec-env vs World path disagree: physics regression",
                    }
                ),
                flush=True,
            )
        self.record.append_replay(replay)
        self.best_replay_lap_ms = int(result.best_lap_ms)
        print(json.dumps({"event": "replay_exported", **replay}, sort_keys=True), flush=True)
        return replay

    def discard_after(self, elapsed_seconds: float) -> dict[str, Any]:
        """Drop evaluations and replays from the timeline a resume discards.

        `_truncate_csv` removes the CSV rows written after the checkpoint;
        the `evals/<minute>.json` files, the `replays/*` files and the
        `run.json.replays` entries of those minutes would otherwise survive
        and seed `best_replay_lap_ms` from a lap the resumed timeline never
        drove. The evaluations at those minutes are regenerated.
        """
        discarded_minutes: list[float] = []
        evals_dir = self.run_dir / "evals"
        for path in sorted(evals_dir.glob("*.json")) if evals_dir.is_dir() else []:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if float(payload["wall_time_s"]) > elapsed_seconds:
                discarded_minutes.append(float(payload["scheduled_minutes"]))
                path.unlink()
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for replay in self.record.payload["replays"]:
            if float(replay["minute"]) in discarded_minutes:
                dropped.append(replay)
                for key in ("scene", "inputs"):
                    (self.run_dir / replay[key]).unlink(missing_ok=True)
            else:
                kept.append(replay)
        self.record.stage(replays=kept)
        self.best_replay_lap_ms = min((int(r["lap_ms"]) for r in kept), default=None)
        return {
            "evaluations": discarded_minutes,
            "replays": [r["scene"] for r in dropped],
        }

    def on_finish(self) -> dict[str, Any] | None:
        result = self.latest_result
        if (
            result is None
            or result.best_lap_ms is None
            or result.best_schedule is None
            or self.exported_latest
        ):
            return None
        assert self.latest_minute is not None
        return self.export_replay(label="final", result=result, minute=self.latest_minute)


def main(argv: list[str] | None = None) -> None:
    script_start = time.perf_counter()
    root = project_root()
    parsed = parse_train_args(argv, prog="tmnf_rl.train")
    config, overrides, resume_id = parsed.config, parsed.overrides, parsed.resume
    enforce_cpu_affinity()
    require_exporter(root)

    # The run directory is locked for the whole process lifetime, before any
    # environment or CUDA work: two trainers on one run (a duplicate --run-id
    # or two --resume) would otherwise interleave metrics.csv and race on
    # latest.pt. The kernel releases the lock when this process dies.
    if resume_id is not None:
        runs_root = Path(overrides.get("runs_root", TrainConfig.runs_root))
        registry = RunRegistry(root / runs_root if not runs_root.is_absolute() else runs_root)
        run_lock = registry.lock_run(resume_id)
        record = registry.open_run(resume_id)
        if record.payload["status"] == "running":
            raise RuntimeError(
                f"run {resume_id!r} is still marked running; refusing to resume a "
                "live or unreaped run (rebuild the index to reap dead runs)"
            )
        stored = dict(record.payload["args"])
        stored.update(overrides)
        config = config_from_dict(stored)
        run_id = resume_id
        run_dir = registry.run_dir(run_id)
        checkpoint_path = run_dir / "checkpoints" / "latest.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"{checkpoint_path} does not exist; nothing to resume")
    else:
        assert config is not None
        runs_root = Path(config.runs_root)
        registry = RunRegistry(root / runs_root if not runs_root.is_absolute() else runs_root)
        run_id = config.run_id or registry.generate_run_id(
            config.track, config.algorithm, config.seed
        )
        config.run_id = run_id
        run_lock = registry.claim_run(run_id)
        run_dir = registry.run_dir(run_id)
        record = None
        checkpoint_path = None

    # Everything up to the run.json write runs under the claim; a failure
    # before then (bad track, refused numerics, env abort) must not leave a
    # lock-only directory that blocks the run id.
    try:
        physics_library = Path(config.physics_library)
        if not physics_library.is_absolute():
            physics_library = root / physics_library
        physics_library = physics_library.resolve()
        # Pin CUDA numerics before the CUDA context exists and before provenance
        # is collected, so run.json records the pinned state.
        trainer_rng = seed_everything(config.seed)
        device, gpu_name = select_device()
        provenance_info = provenance.collect(
            physics_library, seed=config.seed, root=root, torch_module=torch
        )
        if record is not None and record.payload["physics_sha256"] != provenance_info["physics_sha256"]:
            raise RuntimeError(
                "physics library differs from the run being resumed: "
                f"{record.payload['physics_sha256']} -> {provenance_info['physics_sha256']}"
            )
        # A resume across a python/tmnf_rl change produces rows the run's
        # code_sha256 did not produce. Refuse unless asked; when asked,
        # the change is recorded under code_changes below.
        code_changed = (
            record is not None
            and record.payload.get("code_sha256_current", record.payload["code_sha256"])
            != provenance_info["code_sha256"]
        )
        if code_changed and not parsed.ignore_code_hash:
            assert record is not None
            raise RuntimeError(
                "python/tmnf_rl differs from the run being resumed "
                f"({record.payload.get('code_sha256_current', record.payload['code_sha256'])} -> "
                f"{provenance_info['code_sha256']}); pass --ignore-code-hash to continue "
                "anyway (recorded under run.json code_changes)"
            )

        spec = track_spec(config.track, root)
        eval_env = make_env(config, config.eval_num_envs, physics_library)
        env = make_train_env(config, physics_library, eval_env, device)

        spectate_buffer: SpectateBuffer | None = None
        spectate_server: SpectateServer | None = None
        spectate_url: str | None = None
        if config.spectate:
            # The viewer picks the track scene from trackId (its TRACK_SCENES
            # table); the trainer does not know or care which scenes are committed.
            spectate_buffer = SpectateBuffer(
                num_envs=config.num_envs,
                stream_envs=4,
                action_repeat=config.action_repeat,
                max_race_ticks=env.max_race_ticks,
                initial_meta={
                    "streamId": f"{run_id}-{os.getpid()}-{time.time_ns()}",
                    "trackId": env.track_id,
                    "trackName": env.track_name,
                    "updateCount": 0,
                    "wallTimeSeconds": 0.0,
                    "finishes": 0,
                    "bestLapMs": None,
                    "distanceMean": None,
                    "run": {
                        "runId": run_id,
                        "seed": config.seed,
                        "numEnvs": config.num_envs,
                        "outputDir": str(run_dir),
                    },
                },
            )
            spectate_server = SpectateServer(config.spectate, spectate_buffer)
            spectate_server.start()
            spectate_url = f"http://localhost:{spectate_server.port}"

        if record is None:
            record = registry.create_run(
                new_run_payload(
                    run_id=run_id,
                    track_id=spec.track_id,
                    track_name=spec.name,
                    algorithm=config.algorithm,
                    seed=config.seed,
                    args=config.to_dict(),
                    provenance=provenance_info,
                    spectate_url=spectate_url,
                    run_dir=run_dir,
                )
            )
    except BaseException:
        if checkpoint_path is None and record is None:
            run_lock.abandon()
        raise

    sink = RegistrySink(
        record=record,
        config=config,
        run_dir=run_dir,
        root=root,
        spectate_buffer=spectate_buffer,
        provenance_info=provenance_info,
    )
    trainer_class = OffPolicyTrainer if config.algorithm == "td3" else PPOTrainer
    trainer = trainer_class(
        config,
        device=device,
        gpu_name=gpu_name,
        env=env,
        eval_env=eval_env,
        run_dir=run_dir,
        physics_sha256=provenance_info["physics_sha256"],
        code_sha256=provenance_info["code_sha256"],
        run_id=run_id,
        sink=sink,
        trainer_rng=trainer_rng,
        script_start=script_start,
    )
    heartbeat = HeartbeatThread(record, registry)
    status = "failed"
    if checkpoint_path is not None:
        # Validate against the checkpoint before touching run.json: a refused
        # resume must leave the finished run's record exactly as it was.
        try:
            trainer.setup_resume(checkpoint_path)
        except BaseException:
            trainer.close()
            eval_env.close()
            env.close()
            if spectate_server is not None:
                spectate_server.close()
            raise
        resumed_at = utc_now_iso()
        discarded = sink.discard_after(trainer.resumed_from["elapsed_seconds"])
        code_changes = list(record.payload.get("code_changes", []))
        if code_changed:
            previous = record.payload.get("code_sha256_current", record.payload["code_sha256"])
            code_changes.append(
                {
                    "at": resumed_at,
                    "from_update": trainer.update,
                    "code_sha256_before": previous,
                    "code_sha256_after": provenance_info["code_sha256"],
                }
            )
            print(
                json.dumps(
                    {
                        "event": "code_change",
                        "run_id": run_id,
                        "from_update": trainer.update,
                        "code_sha256_before": previous,
                        "code_sha256_after": provenance_info["code_sha256"],
                    }
                ),
                flush=True,
            )
        # code_sha256 stays the hash that produced update 1; code_sha256_current
        # is the hash producing rows from here on.
        record.update(
            status="running",
            finished_at=None,
            failure=None,
            pid=os.getpid(),
            pid_start_ticks=pid_start_ticks(os.getpid()),
            hostname=platform.node(),
            spectate_url=spectate_url,
            args=config.to_dict(),
            code_sha256_current=provenance_info["code_sha256"],
            code_changes=code_changes,
            resumes=[
                *record.payload["resumes"],
                {
                    "resumed_at": resumed_at,
                    "checkpoint": str(checkpoint_path.relative_to(run_dir)),
                    "from_update": trainer.update,
                    "discarded": discarded,
                    "provenance": provenance_info,
                },
            ],
        )
        registry.rebuild_index()
    try:
        if checkpoint_path is None:
            trainer.setup_fresh()
        heartbeat.start()
        summary = trainer.run()
        final_replay = sink.on_finish()
        summary["final_replay"] = final_replay
        record.finish("finished", summary=summary, spectate_url=None)
        status = "finished"
        print(json.dumps({"event": "complete", **summary}, sort_keys=True), flush=True)
    except BaseException as error:
        failure = "".join(traceback.format_exception(error)).strip()
        # The cause goes to stderr first: when the failure is a full disk the
        # run.json write below fails too and would otherwise be the only
        # traceback anyone sees.
        print(f"tmnf_rl.train: run {run_id} failed:\n{failure}", file=sys.stderr, flush=True)
        record.finish("failed", failure=failure, spectate_url=None)
        raise
    finally:
        heartbeat.close()
        if spectate_server is not None:
            spectate_server.close()
        trainer.close()
        eval_env.close()
        env.close()
        registry.rebuild_index()
        run_lock.release()
        print(json.dumps({"event": "registry", "run_id": run_id, "status": status}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
