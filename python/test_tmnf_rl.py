"""Platform tests: each one is an attack on a component that could mislead.

Run from the project root; trainer tests use physical GPU 0 (the RTX 5090,
``CUDA_VISIBLE_DEVICES=0``) and the reserved CPU set. Native bugs outside python/ are
recorded as ``xfail(strict=True)`` while open, so a fix turns into a failure
that says "remove the guard"; F1 and F2 went through that cycle and are hard
tests now.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Earlier evaluation tests create a CUDA context. The in-process encoder and
# gradient checks must establish the trainer's cuBLAS workspace before that.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.sched_setaffinity(0, {*range(0, 14), *range(16, 30)})

import numpy as np
import pytest
import torch

from tmnf_rl import registry as registry_module
from tmnf_rl import encoder
from tmnf_rl.agents.ppo import EVALUATION_NAMES, METRIC_NAMES, TIMING_COLUMNS, Agent, count_parameters
from tmnf_rl.config import ConfigError, TrainConfig, config_from_dict, parse_train_args
from tmnf_rl.env import TERMINATION_NAMES, TmnfVectorEnv
from tmnf_rl.evaluation import EvaluationIntegrityError, evaluate_full_start
from tmnf_rl.inputs import RECORD, InputSchedule, decode_discrete_schedule, quantize_steer
from tmnf_rl.protocol import aggregate
from tmnf_rl.protocol import summarize as protocol_summarize
from tmnf_rl.registry import RunRegistry, infer_trainer, new_run_payload, validate_run_payload
from tmnf_rl.reproduce import compare_rows, read_rows
from tmnf_rl.tracks import project_root, track_catalogue
from tmnf_rl.utils import write_json_atomic


ROOT = project_root()
PYTHON = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "python"), "CUDA_VISIBLE_DEVICES": "0"}
TINY = [
    "--num-envs", "16", "--num-steps", "16", "--thread-count", "4",
    "--eval-num-envs", "4", "--eval-episodes", "4", "--eval-interval-minutes", "1000",
    "--duration-minutes", "0", "--no-staggered-phases", "--stuck-grace-ticks", "60",
]


def provenance_stub() -> dict:
    return {
        "git_commit": "0" * 40,
        "git_dirty": False,
        "code_sha256": "c" * 64,
        "physics_sha256": "p" * 64,
    }


def make_payload(registry: RunRegistry, run_id: str, **extra) -> dict:
    payload = new_run_payload(
        run_id=run_id, track_id="a01", track_name="A01-Race", algorithm="ppo", seed=1,
        args=TrainConfig(run_id=run_id).to_dict(), provenance=provenance_stub(),
        spectate_url=None, run_dir=registry.run_dir(run_id),
    )
    payload.update(extra)
    return payload


def train(runs_root: Path, run_id: str, *extra: str, check: bool = True) -> subprocess.CompletedProcess:
    command = [
        PYTHON, "-m", "tmnf_rl.train", *TINY, "--runs-root", str(runs_root),
        "--run-id", run_id, *extra,
    ]
    return subprocess.run(command, cwd=ROOT, env=ENV, check=check, capture_output=True, text=True)


# --------------------------------------------------------------------- registry


def test_registry_concurrent_writers_keep_index_valid(tmp_path: Path) -> None:
    """Attack: N trainers create runs and rebuild the index simultaneously."""
    root = tmp_path / "runs"
    RunRegistry(root)
    writer = tmp_path / "writer.py"
    writer.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, sys.argv[3])\n"
        "from tmnf_rl.registry import RunRegistry, new_run_payload\n"
        "from tmnf_rl.config import TrainConfig\n"
        "reg = RunRegistry(Path(sys.argv[1]))\n"
        "for i in range(5):\n"
        "    rid = f'w{sys.argv[2]}_r{i}'\n"
        "    rec = reg.create_run(new_run_payload(run_id=rid, track_id='a01', track_name='A01-Race',\n"
        "        algorithm='ppo', seed=i, args=TrainConfig(run_id=rid).to_dict(),\n"
        "        provenance={'git_commit': '0'*40, 'git_dirty': False, 'code_sha256': 'c'*64,\n"
        "        'physics_sha256': 'p'*64}, spectate_url=None, run_dir=reg.run_dir(rid)))\n"
        "    for k in range(3):\n"
        "        rec.update(summary={'updates': k})\n"
        "        reg.rebuild_index()\n"
        "    rec.finish('finished', summary={'updates': 3})\n"
        "    reg.rebuild_index()\n"
    )
    stop = threading.Event()
    reads = {"count": 0, "errors": []}

    def reader() -> None:
        while not stop.is_set():
            path = root / "index.json"
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    assert isinstance(payload["runs"], list)
                    reads["count"] += 1
                except Exception as error:  # noqa: BLE001
                    reads["errors"].append(repr(error))
            time.sleep(0.002)

    thread = threading.Thread(target=reader)
    thread.start()
    processes = [
        subprocess.Popen(
            [PYTHON, str(writer), str(root), str(n), str(ROOT / "python")],
            cwd=ROOT, env=ENV,
        )
        for n in range(6)
    ]
    for process in processes:
        assert process.wait() == 0
    stop.set()
    thread.join()
    assert reads["errors"] == []
    assert reads["count"] > 0
    index = RunRegistry(root).read_index()
    assert len(index["runs"]) == 30
    assert all(entry["status"] == "finished" for entry in index["runs"])
    assert not any(entry.get("invalid") for entry in index["runs"])
    assert not list(root.glob(".index.json.*.tmp"))


def test_registry_reaps_killed_run_and_keeps_live_run(tmp_path: Path, monkeypatch) -> None:
    """Attack: a trainer is SIGKILLed mid-heartbeat and stays 'running' forever."""
    root = tmp_path / "runs"
    registry = RunRegistry(root)
    script = tmp_path / "victim.py"
    script.write_text(
        "import sys, time\nfrom pathlib import Path\nsys.path.insert(0, sys.argv[2])\n"
        "from tmnf_rl.registry import RunRegistry, new_run_payload\n"
        "from tmnf_rl.config import TrainConfig\n"
        "reg = RunRegistry(Path(sys.argv[1]))\n"
        "reg.create_run(new_run_payload(run_id='victim', track_id='a01', track_name='A01-Race',\n"
        "    algorithm='ppo', seed=1, args=TrainConfig(run_id='victim').to_dict(),\n"
        "    provenance={'git_commit': '0'*40, 'git_dirty': False, 'code_sha256': 'c'*64,\n"
        "    'physics_sha256': 'p'*64}, spectate_url=None, run_dir=reg.run_dir('victim')))\n"
        "print('ready', flush=True)\ntime.sleep(60)\n"
    )
    victim = subprocess.Popen(
        [PYTHON, str(script), str(root), str(ROOT / "python")],
        cwd=ROOT, env=ENV, stdout=subprocess.PIPE, text=True,
    )
    assert victim.stdout is not None and victim.stdout.readline().strip() == "ready"
    victim.send_signal(signal.SIGKILL)
    victim.wait()
    live = registry.create_run(make_payload(registry, "live"))
    del live

    # Fresh heartbeat: nothing is reaped yet, even though the pid is dead.
    index = registry.rebuild_index()
    by_id = {entry["run_id"]: entry for entry in index["runs"]}
    assert by_id["victim"]["status"] == "running" and by_id["victim"]["stale"] is False

    monkeypatch.setattr(registry_module, "STALE_HEARTBEAT_SECONDS", 0.0)
    index = registry.rebuild_index()
    by_id = {entry["run_id"]: entry for entry in index["runs"]}
    assert by_id["victim"]["status"] == "failed"
    assert "heartbeat lost" in by_id["victim"]["failure"]
    on_disk = json.loads((root / "victim" / "run.json").read_text())
    assert on_disk["status"] == "failed" and on_disk["finished_at"] == on_disk["heartbeat_at"]
    # The live run (this pid) is stale by the zero threshold but must not be reaped.
    assert by_id["live"]["status"] == "running" and by_id["live"]["stale"] is True


def test_reaper_sees_through_pid_reuse_and_clock_jumps_and_mark_failed_overrides(tmp_path: Path) -> None:
    """Attack (review R6/F28): synthetic run.json files the old reaper parked
    in `running` forever: a reused pid (another live process), a heartbeat an
    hour in the future with a dead pid, and a run from another host."""
    root = tmp_path / "runs"
    registry = RunRegistry(root)
    hour = 3600.0
    now = time.time()
    stale_at = datetime.fromtimestamp(now - 2 * hour, tz=timezone.utc).isoformat()
    future_at = datetime.fromtimestamp(now + hour, tz=timezone.utc).isoformat()
    # A pid that no longer exists, with its start time unknown to the record.
    gone = subprocess.Popen([PYTHON, "-c", "pass"])
    gone.wait()
    cases = {
        # pid alive (ours) but a different start time: the pid was reused.
        "pid_reused": {"pid": os.getpid(), "pid_start_ticks": 1, "heartbeat_at": stale_at},
        # heartbeat in the future, process gone: a forward clock jump.
        "clock_future": {"pid": gone.pid, "heartbeat_at": future_at},
        "other_host": {"pid": 12345, "hostname": "elsewhere", "heartbeat_at": stale_at},
        "live": {"pid": os.getpid(), "pid_start_ticks": registry_module.pid_start_ticks(os.getpid())},
    }
    for run_id, fields in cases.items():
        payload = make_payload(registry, run_id)
        payload.update(fields)
        (root / run_id).mkdir()
        write_json_atomic(root / run_id / "run.json", payload)
    by_id = {entry["run_id"]: entry for entry in registry.rebuild_index()["runs"]}
    assert by_id["pid_reused"]["status"] == "failed" and "heartbeat lost" in by_id["pid_reused"]["failure"]
    assert by_id["clock_future"]["status"] == "failed"
    assert by_id["clock_future"]["finished_at"] < future_at, "finished_at must not sit in the future"
    assert by_id["other_host"]["status"] == "running" and by_id["other_host"]["stale"] is True
    assert by_id["live"]["status"] == "running" and by_id["live"]["stale"] is False
    assert by_id["live"]["trainer"] == "tmnf_rl.train"

    # Operator override: refused while the run looks alive, unless forced.
    with pytest.raises(RuntimeError, match="looks alive"):
        registry.mark_failed("live", reason="test", force=False)
    assert json.loads((root / "live" / "run.json").read_text())["status"] == "running"
    marked = registry.mark_failed("other_host", reason="host is gone", force=False)
    assert marked["status"] == "failed" and "host is gone" in marked["failure"]
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.registry", "--runs-root", str(root), "mark-failed", "live", "--force", "--reason", "operator"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    on_disk = json.loads((root / "live" / "run.json").read_text())
    assert on_disk["status"] == "failed" and on_disk["failure"] == "marked failed by operator (forced): operator"
    with pytest.raises(RuntimeError, match="not running"):
        registry.mark_failed("live", reason="again", force=True)


def test_index_names_the_trainer_of_foreign_runs(tmp_path: Path) -> None:
    """F26: a run written by another program through the registry format has
    no `trainer` field; the index names its entry point instead of blank."""
    registry = RunRegistry(tmp_path / "runs")
    foreign = make_payload(registry, "ladder")
    del foreign["trainer"]
    foreign["algorithm"] = "pilot_mlp"
    foreign["provenance"]["command_line"] = ["/venv/bin/python", "/repo/research/nn_pilots/ppo_pilot.py", "--gpu"]
    (registry.root / "ladder").mkdir(parents=True)
    write_json_atomic(registry.root / "ladder" / "run.json", foreign)
    # Our own runs from before the field: `-m tmnf_rl.train` records the module
    # file as argv[0]; the thin CLI records train_ppo.py.
    legacy_ours = make_payload(registry, "legacy_ours")
    del legacy_ours["trainer"]
    legacy_ours["provenance"]["command_line"] = ["/venv/bin/python", str(ROOT / "python" / "tmnf_rl" / "train.py"), "--seed", "1"]
    (registry.root / "legacy_ours").mkdir(parents=True)
    write_json_atomic(registry.root / "legacy_ours" / "run.json", legacy_ours)
    registry.create_run(make_payload(registry, "ours"))
    by_id = {entry["run_id"]: entry for entry in registry.rebuild_index()["runs"]}
    assert by_id["ladder"]["trainer"] == "unknown:ppo_pilot.py"
    assert by_id["legacy_ours"]["trainer"] == "tmnf_rl.train"
    assert by_id["ours"]["trainer"] == "tmnf_rl.train"
    assert infer_trainer(["python", "/x/python/train_ppo.py"]) == "tmnf_rl.train"
    assert infer_trainer(["python"]) == "unknown"
    table = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.registry", "--runs-root", str(registry.root)],
        cwd=ROOT, env=ENV, capture_output=True, text=True, check=True,
    ).stdout
    header, *rows = table.splitlines()
    assert header.split()[:3] == ["run_id", "trainer", "physics"]
    assert any(row.split()[:3] == ["ladder", "unknown:ppo_pilot.py", "p" * 8] for row in rows), table


def test_registry_survives_malformed_run_json_and_heartbeat_continues(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Attack: a foreign run.json with a naive (offset-less) timestamp.

    ``datetime.now(utc) - naive`` raised TypeError inside rebuild_index, which
    killed the heartbeat thread of every trainer sharing the runs root; their
    runs then looked dead. Rebuild must flag the bad run and go on, and the
    heartbeat must outlive any rebuild exception.
    """
    root = tmp_path / "runs"
    registry = RunRegistry(root)
    live = registry.create_run(make_payload(registry, "live"))
    naive = make_payload(registry, "naive")
    naive["heartbeat_at"] = "2026-09-01T20:00:00"
    naive["started_at"] = "2026-09-01T20:00:00"
    (root / "naive").mkdir()
    write_json_atomic(root / "naive" / "run.json", naive)
    (root / "garbage").mkdir()
    (root / "garbage" / "run.json").write_text("{not json")

    index = registry.rebuild_index()
    by_id = {entry["run_id"]: entry for entry in index["runs"]}
    assert by_id["live"]["status"] == "running" and "invalid" not in by_id["live"]
    assert by_id["naive"]["invalid"] is True and "no UTC offset" in by_id["naive"]["failure"]
    assert by_id["garbage"]["invalid"] is True
    assert by_id["naive"]["status"] == by_id["garbage"]["status"] == "failed"
    # The malformed file is reported, not rewritten or reaped.
    assert json.loads((root / "naive" / "run.json").read_text())["status"] == "running"

    # Heartbeat thread: even a rebuild that raises outright must not stop it.
    calls = {"n": 0}

    def exploding_rebuild() -> dict:
        calls["n"] += 1
        raise RuntimeError("index on fire")

    monkeypatch.setattr(registry, "rebuild_index", exploding_rebuild)
    before = live.payload["heartbeat_at"]
    thread = registry_module.HeartbeatThread(live, registry, interval_seconds=0.05)
    thread.start()
    time.sleep(1.1)
    assert thread.is_alive()
    thread.close()
    assert thread.beats >= 3 and thread.rebuild_failures == thread.beats == calls["n"]
    assert json.loads((root / "live" / "run.json").read_text())["heartbeat_at"] >= before
    assert "index rebuild failed (RuntimeError: index on fire)" in capsys.readouterr().err


def test_heartbeat_survives_a_full_disk_and_leaves_no_temp_file(tmp_path: Path) -> None:
    """Attack (review R5/F27): the heartbeat write itself fails (ENOSPC,
    simulated with RLIMIT_FSIZE). Before: the thread died after the first
    failed beat, the run looked dead, and `.run.json.<pid>.tmp` stayed behind.
    Runs in a subprocess so the file-size limit does not touch pytest."""
    script = r"""
import json, os, resource, sys, time
sys.path.insert(0, sys.argv[2])
from pathlib import Path
from tmnf_rl import registry as registry_module
from tmnf_rl.registry import RunRegistry, new_run_payload
root = Path(sys.argv[1]); registry = RunRegistry(root)
payload = new_run_payload(run_id="victim", track_id="a01", track_name="A01-Race", algorithm="ppo", seed=1,
    args={}, provenance={"git_commit": "0"*40, "git_dirty": False, "code_sha256": "c"*64, "physics_sha256": "p"*64},
    spectate_url=None, run_dir=root / "victim")
record = registry.create_run(payload)
thread = registry_module.HeartbeatThread(record, registry, interval_seconds=0.05)
thread.start()
while thread.beats < 3:
    time.sleep(0.01)
soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (512, hard))
while thread.heartbeat_failures < 3:
    time.sleep(0.01)
alive_during = thread.is_alive()
files_during = sorted(p.name for p in (root / "victim").iterdir())
valid_during = json.loads((root / "victim" / "run.json").read_text())["run_id"] == "victim"
resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
beats_before_recovery = thread.beats
while thread.beats < beats_before_recovery + 3:
    time.sleep(0.01)
thread.close()
print(json.dumps({"alive_during": alive_during, "files_during": files_during, "valid_during": valid_during,
    "failures": thread.heartbeat_failures, "beats": thread.beats, "last_error": thread.last_error}))
"""
    result = subprocess.run(
        [PYTHON, "-c", script, str(tmp_path / "runs"), str(ROOT / "python")],
        cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["alive_during"] is True and report["failures"] >= 3
    assert report["files_during"] == ["run.json"], report["files_during"]
    assert report["valid_during"] is True and report["beats"] >= 6
    assert "OSError" in report["last_error"] and "File too large" in report["last_error"]
    assert "heartbeat write failed (OSError" in result.stderr


def test_run_payload_validation_rejects_contract_violations(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs")
    payload = make_payload(registry, "ok")
    validate_run_payload(payload)
    bad_status = {**payload, "status": "done"}
    with pytest.raises(ValueError):
        validate_run_payload(bad_status)
    missing = dict(payload)
    del missing["heartbeat_at"]
    with pytest.raises(ValueError):
        validate_run_payload(missing)
    bad_replay = {**payload, "replays": [{"label": "x", "lap_ms": 1, "scene": "s"}]}
    with pytest.raises(ValueError):
        validate_run_payload(bad_replay)


def test_atomic_json_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.json"
    write_json_atomic(target, {"x": 1})
    write_json_atomic(target, {"x": 2})
    assert json.loads(target.read_text()) == {"x": 2}
    assert [path.name for path in target.parent.iterdir()] == ["b.json"]


# ----------------------------------------------------------------------- config


def test_config_rejects_unknown_keys_and_wrong_types() -> None:
    """Attack: a typo in a config file silently keeps the default."""
    with pytest.raises(ConfigError, match="unknown config keys"):
        config_from_dict({"num_env": 128})
    with pytest.raises(ConfigError, match="must be an integer"):
        config_from_dict({"num_envs": "256"})
    with pytest.raises(ConfigError, match="must be an integer"):
        config_from_dict({"num_envs": True})
    with pytest.raises(ConfigError, match="must be a boolean"):
        config_from_dict({"staggered_phases": 1})
    with pytest.raises(ConfigError, match="must be a number"):
        config_from_dict({"learning_rate": "2.5e-4"})
    with pytest.raises(ConfigError):
        config_from_dict({"num_envs": 100, "num_minibatches": 3, "num_steps": 1})
    config = config_from_dict({"learning_rate": 1})
    assert isinstance(config.learning_rate, float)


def test_evaluate_loads_retired_empty_demo_field_without_relabelling_demos(tmp_path: Path) -> None:
    from tmnf_rl.evaluate import load_policy

    agent = Agent("mlp", 64, "discrete")
    config = TrainConfig(hidden_size=64).to_dict()
    config["demonstration_inputs"] = ""
    path = tmp_path / "policy.pt"
    torch.save({"config": config, "agent": agent.state_dict()}, path)
    loaded, parsed, meta = load_policy(path, torch.device("cpu"))
    assert "demonstration_inputs" not in parsed
    assert meta["config_migrations"] == ["removed unused demonstration_inputs field"]
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in agent.state_dict().items())
    config["demonstration_inputs"] = "historical-expert.inputs.bin"
    torch.save({"config": config, "agent": agent.state_dict()}, path)
    with pytest.raises(ConfigError, match="demonstration_inputs"):
        load_policy(path, torch.device("cpu"))


@pytest.mark.parametrize("flags", [
    ["--write-inputs", "lap.inputs.bin", "--export-scene", "lap.json"],
    ["--output", "lap.json", "--export-scene", "lap.json"],
    ["--output", "lap.inputs.bin", "--write-inputs", "lap.inputs.bin"],
])
def test_evaluate_rejects_colliding_outputs_before_loading_policy(flags: list[str]) -> None:
    from tmnf_rl.evaluate import main

    with pytest.raises(ValueError, match="output paths must be distinct"):
        main(["nonexistent-policy.pt", *flags])


def test_cli_overrides_config_file_and_records_explicit_flags(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"num_envs": 32, "seed": 7, "staggered_phases": False}))
    parsed = parse_train_args(["--config", str(path), "--seed", "9", "--staggered-phases"])
    config = parsed.config
    assert parsed.resume is None and config is not None and parsed.ignore_code_hash is False
    assert config.num_envs == 32 and config.seed == 9 and config.staggered_phases is True
    assert parsed.overrides == {"seed": 9, "staggered_phases": True}
    path.write_text(json.dumps({"num_envss": 32}))
    with pytest.raises(ConfigError):
        parse_train_args(["--config", str(path)])
    parsed = parse_train_args(["--resume", "abc", "--max-updates", "5", "--ignore-code-hash"])
    assert parsed.resume == "abc" and parsed.overrides == {"max_updates": 5}
    assert parsed.config is None and parsed.ignore_code_hash is True
    with pytest.raises(SystemExit):
        parse_train_args(["--ignore-code-hash", "--max-updates", "5"])


def test_train_cli_fails_fast_on_config_typo(tmp_path: Path) -> None:
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"num_env": 16}))
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--config", str(path)],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unknown config keys" in result.stderr


# ----------------------------------------------------------------------- inputs


def test_committed_policy_lap_schedule_round_trips() -> None:
    payload = (ROOT / "oracle" / "results" / "policy_lap_inputs.bin").read_bytes()
    schedule = decode_discrete_schedule(payload, 2527)
    assert schedule.tick_count == 2527
    assert schedule.to_bytes(2700) == payload
    from_decisions = InputSchedule.from_decisions(
        "discrete", list(schedule.actions[::5]), [5] * (2527 // 5) + [2]
    )
    assert from_decisions.tick_count == 2527


def test_quantize_steer_matches_native_roundf() -> None:
    lib = TmnfVectorEnv(1, thread_count=1)._lib
    import ctypes

    lib.TmnfVecEnv_QuantizeAnalogSteer.argtypes = [ctypes.c_float]
    lib.TmnfVecEnv_QuantizeAnalogSteer.restype = ctypes.c_int32
    for steer in (0.0, 1.0, -1.0, 2.5 / 65536.0, -2.5 / 65536.0, 0.3333, -0.7071, 1e-7):
        native = lib.TmnfVecEnv_QuantizeAnalogSteer(ctypes.c_float(steer))
        assert quantize_steer(steer) == float(np.float32(-native / 65536.0)), steer
    with pytest.raises(ValueError):
        quantize_steer(1.5)


# --------------------------------------------------------------------- protocol


def test_inexact_eval_interval_still_matches_the_protocol_minutes(tmp_path: Path) -> None:
    """Review section 5: the trainer accumulated `interval * 60` seconds
    (0.03 min -> 1.7999999999999998 s -> 0.029999999999999995 min) while the
    protocol matched `round(k * interval, 6)` exactly, so summarize raised on
    any interval whose seconds are inexact."""
    runs_root = tmp_path / "runs"
    train(
        runs_root, "sched_s1", "--seed", "1", "--num-envs", "8", "--max-updates", "40",
        "--eval-interval-minutes", "0.03",
    )
    rows = list(csv.DictReader((runs_root / "sched_s1" / "evaluations.csv").open()))
    minutes = [row["scheduled_minutes"] for row in rows]
    assert minutes[:2] == ["0.0", "0.03"], minutes
    assert all(len(minute.split(".")[1]) <= 2 for minute in minutes), minutes
    assert {"0.json", "0.03.json"} <= {p.name for p in (runs_root / "sched_s1" / "evals").glob("*.json")}
    summary = protocol_summarize(
        name="sched", run_ids={1: "sched_s1"}, minutes=0.03, registry=RunRegistry(runs_root)
    )
    assert [row["minute"] for row in summary["checkpoints"]] == [0.0, 0.03]


def test_protocol_aggregate_reports_finishing_seeds_and_iqr() -> None:
    def row(minute: float, rate: float, median: float | None, greedy: int) -> dict[str, str]:
        return {
            "scheduled_minutes": str(minute),
            "eval_fullstart/finished": str(greedy),
            "eval_fullstart/distance_mean": "1000",
            "eval_fullstart/best_lap_ms": "" if not greedy else str(median),
            "eval_sampled/finish_rate": str(rate),
            "eval_sampled/distance_mean": "1000",
            "eval_sampled/median_lap_ms": "" if median is None else str(median),
            "eval_sampled/best_lap_ms": "" if median is None else str(median - 10),
        }

    evaluations = {
        1: [row(0.0, 0.0, None, 0), row(5.0, 0.75, 25000, 1)],
        2: [row(0.0, 0.0, None, 0), row(5.0, 0.0, None, 0)],
        # F29: the greedy trajectory stalled while the sampled policy finished
        # half its episodes; both are reported, neither is a 0/256 "rate".
        3: [row(0.0, 0.0, None, 0), row(5.0, 0.5, 26000, 0)],
    }
    table = aggregate(evaluations, [0.0, 5.0])
    assert table[0]["finishing_seeds"] == 0 and table[0]["median_lap_ms"]["median"] is None
    assert table[1]["finishing_seeds"] == 2 and table[1]["greedy_finished_seeds"] == 1
    assert table[1]["median_lap_ms"]["median"] == 25500.0
    assert table[1]["greedy_lap_ms"]["n"] == 1 and table[1]["greedy_lap_ms"]["median"] == 25000.0
    assert table[1]["finish_rate"]["median"] == 0.5 and table[1]["finish_rate"]["q1"] == 0.25
    with pytest.raises(ValueError, match="expected exactly one"):
        aggregate(evaluations, [0.0, 5.0, 10.0])


# ------------------------------------------------------------------- evaluation


def test_evaluate_refuses_restored_environment() -> None:
    """Attack: report lap times from an eval env that was seeded from snapshots."""
    device = torch.device("cuda:0")
    env = TmnfVectorEnv(2, thread_count=2, action_repeat=5)
    try:
        env.reset()
        blob = env.capture([0])[0]
        env.restore([1], [blob])
        agent = Agent("mlp", 16).to(device)
        with pytest.raises(EvaluationIntegrityError, match="restored"):
            evaluate_full_start(agent, env, 1, device)
    finally:
        env.close()


def test_evaluate_counts_only_full_episodes_it_stepped() -> None:
    """Every counted episode must have been stepped from tick 0 in Python."""
    device = torch.device("cuda:0")
    env = TmnfVectorEnv(4, thread_count=4, action_repeat=5, stuck_grace_ticks=60)
    try:
        agent = Agent("mlp", 16).to(device)
        result = evaluate_full_start(agent, env, 4, device)
        assert result.mode == "greedy" and result.finished is False
        assert result.episodes == 4 and result.finishes == 0
        assert sum(result.termination_reason_counts.values()) == 4
        # Every native reason (finish, timeout, off_track, stuck, fell) has a
        # name; an unnamed enum value would surface in evals/*.json as a digit.
        assert all(int(key) in TERMINATION_NAMES for key in result.termination_reason_counts)
        assert len(set(result.distances)) == 1, "greedy episodes are one trajectory"
        # Sampled evaluation: different trajectories, same result for the same
        # seed, and the global Torch RNG streams come back untouched (F29).
        torch.manual_seed(123)
        cpu_before, cuda_before = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        sampled = evaluate_full_start(agent, env, 8, device, sampled=True, sample_seed=7)
        assert torch.equal(torch.get_rng_state(), cpu_before)
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_before)
        assert sampled.mode == "sampled" and sampled.finished is None and sampled.episodes == 8
        assert len(set(sampled.distances)) > 1
        again = evaluate_full_start(agent, env, 8, device, sampled=True, sample_seed=7)
        assert again.distances == sampled.distances
        other = evaluate_full_start(agent, env, 8, device, sampled=True, sample_seed=8)
        assert other.distances != sampled.distances
        assert set(sampled.row()) == {name for name in EVALUATION_NAMES if name.startswith("eval_sampled/")}
        assert set(result.row()) == {
            name for name in EVALUATION_NAMES
            if name.startswith("eval_fullstart/") or name == "termination_reason_counts"
        }
    finally:
        env.close()


# ---------------------------------------------------------------------- encoder


def _effective_inputs(record: tuple) -> tuple[float, int, int]:
    """The game's input mapper (src/vehicle.c
    CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs) on one
    TMNFRaceInputs record: which timestamped field is in effect."""
    (left_t, _, left, right_t, _, right, analog_t, _, analog,
     accelerate_t, _, accelerate, brake_t, _, brake, gas_analog_t, _, gas_analog) = record
    latest = max(accelerate_t, brake_t)
    event = gas_analog_t
    if event == latest and accelerate == 0 and brake == 0 and abs(gas_analog) > 0.01:
        event = latest + 1
    if event <= latest:
        gas, braking = int(accelerate != 0), int(brake != 0)
    elif gas_analog >= 0.3:
        gas, braking = 1, 0
    elif gas_analog <= -0.3:
        gas, braking = 0, 1
    else:
        gas, braking = 0, 0
    latest = max(left_t, right_t)
    event = analog_t
    if event == latest and left == 0 and right == 0 and abs(analog) > 0.01:
        event = latest + 1
    if event <= latest:
        steer = -1.0 if left else (1.0 if right else 0.0)
    else:
        steer = float(np.float32(-np.float32(analog)))
    return steer, gas, braking


def _drive_schedule(track_id: str, inputs: bytes, *, analog: bool, ticks: int) -> np.ndarray:
    """Step one env tick by tick through a recorded schedule; return every
    policy observation seen (the terminal one included)."""
    kwargs = dict(track=track_id, thread_count=1, action_repeat=1, stuck_grace_ticks=10_000_000)
    respawn_action = False
    if analog:
        records = [RECORD.unpack_from(inputs, i * RECORD.size) for i in range(ticks)]
        # Word 16 is TMNFRaceInputs.respawn (analysis/respawn.md).
        respawn_action = any(record[16] != 0 for record in records)
    env = TmnfVectorEnv(
        1, action_space="analog" if analog else "discrete", respawn_action=respawn_action, **kwargs
    )
    try:
        env.reset()
        rows = []
        if not analog:
            actions = decode_discrete_schedule(inputs, ticks).actions
        for tick in range(ticks):
            if analog:
                steer, gas, brake = _effective_inputs(records[tick])
                action = {
                    "steer": np.array([steer], dtype=np.float32),
                    "gas": np.array([gas], dtype=np.int8),
                    "brake": np.array([brake], dtype=np.int8),
                }
                if respawn_action:
                    action["respawn"] = np.array([int(records[tick][16] != 0)], dtype=np.int8)
            else:
                action = np.array([int(actions[tick])])
            _, _, terminated, truncated, info = env.step(action)
            if terminated[0] or truncated[0]:
                rows.append(env.policy_final_observations[0].copy())
                assert int(info["termination_reason"][0]) == 1, f"{track_id}: schedule did not finish"
                assert tick == ticks - 1, f"{track_id}: finished at tick {tick + 1}, schedule has {ticks}"
                break
            rows.append(env.policy_observations[0].copy())
        return np.stack(rows)
    finally:
        env.close()


# Features that depend on the route reference: excluded for A08, whose
# reference is the south out-and-back while the record loops the north road
# 154 half-widths away (analysis/rl_env.md, Validation limits).
_ROUTE_FEATURES = {"lateral", "half_width", "lateral_ratio"} | {
    name for name in encoder.FEATURE_NAMES if name.startswith("geo")
}


SCALE_BOUND = 3.2  # asinh(59 m / 5 m) = 3.16: the A01 policy lap 59 m beside the line at the 5 m slot


def test_encoder_features_stay_within_scale_on_record_lines() -> None:
    """Every flat feature stays within about 3 in magnitude on the seven
    world-record replays (E04's with its respawn) and the exported A01/B05
    policy laps. Before the
    scales were fixed the 5 m lookahead slot read x/d up to 8.7 (review,
    finding 11); the wall-ride and open-surface widths on E01 (256 m), the
    181 m/s record speed and the A01 policy lap's 59 m excursion set them."""
    manifest = [
        line.split("|")
        for line in (ROOT / "oracle" / "results" / "wr" / "manifest.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(manifest) == 7
    laps: dict[str, np.ndarray] = {}
    for row in manifest:
        track_id, finish_ms = row[0], int(row[6])
        inputs = (ROOT / "oracle" / "results" / "wr" / f"{track_id}_wr_inputs.bin").read_bytes()
        laps[f"{track_id}_wr"] = _drive_schedule(track_id, inputs, analog=True, ticks=finish_ms // 10)
    for run_id, track_id in (("f33_a01_tmaxscaled_20m_s1", "a01"), ("f33_b05_tmaxscaled_20m_s1", "b05")):
        for path in sorted((ROOT / "artifacts" / "runs" / run_id / "replays").glob("*_env*ms.inputs.bin")):
            lap_ms = int(path.name.split("_env")[1].split("ms")[0])
            laps[f"{track_id}_{path.name}"] = _drive_schedule(
                track_id, path.read_bytes(), analog=False, ticks=lap_ms // 10
            )
    assert len(laps) == 7 + 3 + 4
    over: list[tuple[str, str, float]] = []
    for label, observations in laps.items():
        features = encoder.encode_flat(torch.from_numpy(observations))
        assert features.shape == (observations.shape[0], encoder.FLAT_FEATURES)
        assert torch.isfinite(features).all()
        maxima = features.abs().max(dim=0).values
        for name, value in zip(encoder.FEATURE_NAMES, maxima.tolist(), strict=True):
            if label == "a08_wr" and name in _ROUTE_FEATURES:
                continue
            if value > SCALE_BOUND:
                over.append((label, name, value))
    assert over == [], over
    # The A08 record line is what the route reference cannot follow: the
    # lateral feature reads above 4 there (hundreds of metres before asinh).
    # Route work, not scale.
    a08 = encoder.encode_flat(torch.from_numpy(laps["a08_wr"]))
    assert a08[:, encoder.FEATURE_NAMES.index("lateral")].abs().max() > 4.0


def test_encoder_ignores_position_clock_and_progress() -> None:
    """World position, arc length, unwrapped progress, remaining distance,
    the race clock, the checkpoint fraction and the lap fraction (obs 0-2,
    38, 39, 42-45) must not reach the network (review, finding 5)."""
    env = TmnfVectorEnv(4, thread_count=1, action_repeat=5)
    try:
        env.reset()
        rng = np.random.default_rng(1)
        for _ in range(40):
            env.step(rng.integers(0, 12, size=4))
        observation = torch.from_numpy(env.policy_observations.copy())
    finally:
        env.close()
    baseline = encoder.encode_flat(observation)
    assert encoder.FLAT_FEATURES == 193 and baseline.shape == (4, 193)
    perturbed = observation.clone()
    perturbed[:, list(encoder.EXCLUDED_COLUMNS)] += torch.linspace(1.0, 900.0, len(encoder.EXCLUDED_COLUMNS))
    assert torch.equal(encoder.encode_flat(perturbed), baseline)
    # Every other column does reach the features.
    for column in range(observation.shape[1]):
        if column in encoder.EXCLUDED_COLUMNS:
            continue
        nudged = observation.clone()
        nudged[:, column] += 1.0  # whole units: material, gear and turbo type are rounded
        assert not torch.equal(encoder.encode_flat(nudged), baseline), column
    # The critic alone reads the remaining distance (obs 42), nothing else
    # from the excluded columns.
    context = encoder.critic_context(observation)
    assert context.shape == (4, encoder.CRITIC_FEATURES)
    assert torch.equal(context[:, 0], observation[:, 42] / encoder.REMAINING_SCALE)
    assert torch.equal(encoder.critic_context(perturbed)[:, 0], (observation[:, 42] + torch.linspace(1.0, 900.0, 9)[5]) / encoder.REMAINING_SCALE)
    # Both arms consume the same encoder output; parameter counts are the
    # pilot's (MLP 256x2 on 193 inputs; transformer s about 0.8M) plus the
    # critic's one extra input.
    assert count_parameters(Agent("mlp", 256)) == 193 * 256 + 256 + 256 * 256 + 256 + 12 * 256 + 12 + (256 + 1) + 1
    transformer = Agent("transformer_s", 256)
    assert 780_000 < count_parameters(transformer) < 820_000
    logits = transformer.trunk(encoder.encode(observation))
    assert logits.shape == (4, 128)
    with pytest.raises(ValueError, match="arch must be one of"):
        Agent("cnn", 16)


# ------------------------------------------------------------- native footguns


def _lockstep_after_cross_restore(ticks: int = 400) -> int | None:
    """Restore env 0's snapshot into env 1, step both identically, return the
    first tick whose observations differ (None if identical)."""
    env = TmnfVectorEnv(2, track="a01", thread_count=1, action_repeat=1)
    try:
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(300):
            env.step(rng.integers(0, 12, size=2))
        blob = env.capture([0])[0]
        env.restore([1], [blob])
        assert env.raw_observations[0].tobytes() == env.raw_observations[1].tobytes()
        for tick in range(1, ticks + 1):
            action = int(rng.integers(0, 12))
            env.step(np.array([action, action]))
            if env.raw_observations[0].tobytes() != env.raw_observations[1].tobytes():
                return tick
        return None
    finally:
        env.close()


def test_cross_env_restore_matches_source_env() -> None:
    """F1: a snapshot moved to another environment must continue bit-identically.

    Before src/vec_env.c 5d59a19 the restore kept the source environment's
    contact-context pointers and diverged at tick 2; a Python pointer
    transplant covered it until the native snapshots became pointer-free.
    """
    assert _lockstep_after_cross_restore() is None


def test_restore_into_never_stepped_instance_matches_source() -> None:
    """Resume restores checkpointed snapshots into a fresh process's instances,
    which have never stepped. Before src/vec_env.c 5d59a19 those diverged at
    the first step (a Python warm_up shim covered it); now cold == warm."""
    kwargs = dict(thread_count=4, action_repeat=1, stuck_grace_ticks=60)
    source = TmnfVectorEnv(8, **kwargs)
    source.reset()
    rng = np.random.default_rng(0)
    for _ in range(200):
        source.step(rng.integers(0, 12, size=8))
    blobs = list(source.capture())
    target = TmnfVectorEnv(8, **kwargs)
    target.restore(np.arange(8, dtype=np.uint32), blobs)
    try:
        for _ in range(300):
            action = rng.integers(0, 12, size=8)
            source.step(action)
            target.step(action)
            assert source.raw_results.tobytes() == target.raw_results.tobytes()
    finally:
        source.close()
        target.close()


def _replay_committed_lap_against_game(env: TmnfVectorEnv) -> tuple[int | None, int | None]:
    raw = (ROOT / "oracle" / "results" / "policy_lap.bin").read_bytes()
    game = np.array(
        [np.frombuffer(raw[k * 1668 + 56 : k * 1668 + 68], dtype="<f4") for k in range(2600)]
    )
    schedule = decode_discrete_schedule(
        (ROOT / "oracle" / "results" / "policy_lap_inputs.bin").read_bytes(), 2700
    )
    first_mismatch = None
    finish_ms = None
    for tick in range(2599):
        observations, _, terminated, truncated, info = env.step(
            np.array([int(schedule.actions[tick])])
        )
        source = env.final_observations if terminated[0] or truncated[0] else observations
        if first_mismatch is None and not np.array_equal(source["vehicle"][0, :3], game[tick + 1]):
            first_mismatch = tick + 1
        if terminated[0] or truncated[0]:
            if int(info["termination_reason"][0]) == 1:
                finish_ms = int(info["race_time_ms"][0])
            break
    return first_mismatch, finish_ms


def test_env_first_episode_replays_game_capture_of_committed_policy_lap() -> None:
    """F2: the vec-env's first episode is byte-exact against the game capture
    (in-game 25,270 ms). Fixed natively in src/vec_env.c 5d59a19."""
    env = TmnfVectorEnv(1, track="a01", thread_count=1, action_repeat=1)
    try:
        env.reset()
        assert _replay_committed_lap_against_game(env) == (None, 25270)
    finally:
        env.close()


def test_env_post_reset_episode_replays_game_capture() -> None:
    """F2b: an episode after a reset that follows physics steps (the regime of
    every training episode but the first) is byte-exact against the game too.
    Between 5d59a19 and d09dc00 it diverged at tick 2 by 0.5 mm."""
    env = TmnfVectorEnv(1, track="a01", thread_count=1, action_repeat=1)
    try:
        env.reset()
        env.step(np.array([4]))
        env.reset()
        assert _replay_committed_lap_against_game(env) == (None, 25270)
    finally:
        env.close()


def test_env_autoreset_episode_replays_game_capture() -> None:
    """Same as above through the native same-step autoreset instead of reset()."""
    env = TmnfVectorEnv(
        1, track="a01", thread_count=1, action_repeat=1, stuck_grace_ticks=60
    )
    try:
        env.reset()
        terminated = np.zeros(1, dtype=bool)
        while not terminated[0]:
            _, _, terminated, _, _ = env.step(np.array([1]))
        assert _replay_committed_lap_against_game(env) == (None, 25270)
    finally:
        env.close()


SNAPSHOT_START_FLAGS = [
    "--track", "a10", "--seed", "3", "--snapshot-start-fraction", "0.5",
    "--snapshot-capture-interval-ticks", "20", "--snapshot-capture-episode-fraction", "1.0",
    "--horizon-ticks", "300",
]


def test_snapshot_start_runs_are_bit_reproducible_resume_and_never_touch_eval(tmp_path: Path) -> None:
    """Attacks on snapshot starts (F9): (1) two same-seed runs must match every
    cell although half the resets restore pool states; (2) a resume must
    continue them bit-exactly (pool, candidates and in-flight captures are
    checkpointed); (3) the evaluation environment is never restored and every
    evaluated episode starts at tick 0 (the evaluator asserts it)."""
    runs_root = tmp_path / "runs"
    for run_id in ("snap_a", "snap_b"):
        train(runs_root, run_id, "--max-updates", "24", *SNAPSHOT_START_FLAGS,
              "--eval-interval-minutes", "0.01")
    rows_a = read_rows(runs_root / "snap_a" / "metrics.csv", 24)
    rows_b = read_rows(runs_root / "snap_b" / "metrics.csv", 24)
    comparison = compare_rows(rows_a, rows_b)
    assert comparison["matched"] is True, comparison["first_mismatches"]
    summary_a = json.loads((runs_root / "snap_a" / "summary.json").read_text())
    summary_b = json.loads((runs_root / "snap_b" / "summary.json").read_text())
    assert summary_a["reset_choice_sha256"] == summary_b["reset_choice_sha256"]
    assert summary_a["snapshot_starts"] > 0 and summary_a["snapshot_episodes"] > 0
    assert summary_a["snapshot_pool"]["snapshot_pool_states"] > 0
    assert summary_a["snapshot_pool"]["snapshot_pool_refreshes"] == 2
    assert int(rows_a[-1]["train/snapshot_starts"]) == summary_a["snapshot_starts"]
    # (3) the trainer's evaluation environment was never restored, and the
    # evaluations it produced (one per update here) all passed the evaluator's
    # tick-0 / native-tick-count assertions.
    assert summary_a["eval_env_restore_count"] == 0
    evaluations = read_rows(runs_root / "snap_a" / "evaluations.csv", 1000)
    assert len(evaluations) >= 3
    assert all(int(row["eval_fullstart/episodes"]) == 4 for row in evaluations)
    # (2) resume from update 12 reproduces updates 13-24.
    train(runs_root, "snap_half", "--max-updates", "12", *SNAPSHOT_START_FLAGS,
          "--eval-interval-minutes", "0.01")
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", "snap_half", "--runs-root", str(runs_root),
         "--max-updates", "24"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    rows_half = read_rows(runs_root / "snap_half" / "metrics.csv", 24)
    comparison = compare_rows(rows_a, rows_half)
    assert comparison["compared_updates"] == 24
    assert comparison["matched"] is True, comparison["first_mismatches"]
    summary_half = json.loads((runs_root / "snap_half" / "summary.json").read_text())
    assert summary_half["reset_choice_sha256"] == summary_a["reset_choice_sha256"]


def test_trainer_refuses_to_evaluate_on_its_training_environment() -> None:
    from tmnf_rl.agents.ppo import PPOTrainer
    env = object()
    with pytest.raises(ValueError, match="never-restored"):
        PPOTrainer(
            TrainConfig(), device=torch.device("cpu"), gpu_name="x", env=env, eval_env=env,  # type: ignore[arg-type]
            run_dir=Path("/nonexistent"), physics_sha256="p", code_sha256="c", run_id="r",
            sink=None, trainer_rng=np.random.default_rng(0), script_start=0.0,  # type: ignore[arg-type]
        )


# -------------------------------------------------------- trainer / reproduce


@pytest.mark.parametrize("count", [32, 256, 4096])
def test_graph_encoder_preserves_bits_rng_and_retained_features(count: int) -> None:
    from tmnf_rl.agents.ppo import pin_cuda_numerics

    pin_cuda_numerics()
    observations = torch.randn((count, 81), device="cuda")
    observations[:, 3:7] = torch.nn.functional.normalize(observations[:, 3:7], dim=1)
    encode = encoder.FlatEncoder()
    rng = torch.cuda.get_rng_state()
    first = encode(observations)
    expected = encoder.encode_flat(observations)
    assert torch.equal(first, expected)
    observations.mul_(0.5)
    second = encode(observations)
    assert torch.equal(second, encoder.encode_flat(observations))
    assert torch.equal(first, expected), "graph replay overwrote an earlier forward"
    assert torch.equal(rng, torch.cuda.get_rng_state())


@pytest.mark.parametrize("mode", ["discrete", "analog"])
def test_preencoded_policy_preserves_outputs_and_gradients(mode: str) -> None:
    from tmnf_rl.agents.ppo import pin_cuda_numerics

    pin_cuda_numerics()
    agent = Agent("mlp", 64, mode).cuda()
    observations = torch.randn((256, 81), device="cuda")
    observations[:, 3:7] = torch.nn.functional.normalize(observations[:, 3:7], dim=1)
    # Encode before permutation as the PPO loop does, then retain two
    # forwards before backward to expose graph output aliasing.
    permutation = torch.randperm(256, device="cuda")
    actions = torch.zeros((256,) if mode == "discrete" else (256, 3), device="cuda",
                          dtype=torch.long if mode == "discrete" else torch.float32)
    features = agent._flat_encoder(observations)
    results, gradients = [], []
    for cached in [False, True]:
        agent.zero_grad()
        forwards = []
        for indices in [permutation[:128], permutation[128:]]:
            forwards.append(agent.get_action_and_value(
                observations[indices], actions[indices],
                encoded=features[indices] if cached else None))
        loss = sum(part.sum() for output in forwards for part in output[1:])
        loss.backward()
        results.append([part.detach().clone() for output in forwards for part in output])
        gradients.append([p.grad.clone() for p in agent.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*results))
    assert all(torch.equal(a, b) for a, b in zip(*gradients))


def test_graph_advantages_preserve_truncations_and_changed_rollout_buffers() -> None:
    from tmnf_rl.advantages import AdvantageEstimator, generalized_advantage

    shape = (16, 32)
    buffers = [torch.randn(shape, device="cuda"), torch.rand(shape, device="cuda"),
               torch.rand(shape, device="cuda") < 0.2, torch.rand(shape, device="cuda") < 0.3,
               torch.randn(shape, device="cuda"), torch.randn(shape, device="cuda")]
    estimate = AdvantageEstimator()
    saved = None
    for repeat in range(3):
        buffers[0].add_(1)
        if repeat == 2:
            buffers = [x.clone() for x in buffers]
        next_value = torch.randn(32, device="cuda")
        expected = generalized_advantage(*buffers, next_value, 0.95)
        actual = estimate(*buffers, next_value, gae_lambda=0.95)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        if saved is not None:
            assert all(torch.equal(a, b) for a, b in zip(*saved))
        saved = (actual, expected)


def test_export_summary_avoids_loading_track_geometry(tmp_path: Path, monkeypatch) -> None:
    from tmnf_rl import replays
    from tmnf_rl.tracks import track_spec

    def unexpected_parse(*args):
        raise AssertionError("export parsed the complete scene to read its lap summary")

    monkeypatch.setattr(replays, "scene_lap_summary", unexpected_parse)
    scene = tmp_path / "lap.json"
    info = replays.export_scene(spec=track_spec("a04"),
                               schedule=InputSchedule.from_decisions("discrete", [4], [20]),
                               output=scene, expected_finish_ms=0)
    lap = json.loads(scene.read_text())["lap"]
    assert info["scene_checkpoint_ticks"] == lap["checkpointTicks"]
    assert info["finish_ms"] == (lap["finishTimeMs"] or None)
    assert not info["world_finished"]


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory) -> tuple[Path, str]:
    runs_root = tmp_path_factory.mktemp("runs")
    run_id = "tiny_a01_s1"
    train(runs_root, run_id, "--max-updates", "6")
    return runs_root, run_id


def test_train_writes_registry_contract(trained_run: tuple[Path, str]) -> None:
    runs_root, run_id = trained_run
    run = json.loads((runs_root / run_id / "run.json").read_text())
    validate_run_payload(run)
    assert run["status"] == "finished" and run["finished_at"] is not None
    assert run["track_id"] == "a01" and run["track_name"] == "A01-Race"
    assert len(run["git_commit"]) == 40 and len(run["physics_sha256"]) == 64
    assert run["summary"]["updates"] == 6 and run["summary"]["eval_fullstart"] is not None
    eval_summary = run["summary"]["eval_fullstart"]
    assert eval_summary["finished"] is False and eval_summary["finish_rate"] == 0.0
    assert eval_summary["sampled_episodes"] == 4
    minute0 = json.loads((runs_root / run_id / "evals" / "0.json").read_text())
    assert minute0["result"]["mode"] == "greedy" and minute0["result"]["episodes"] == 4
    assert minute0["sampled"]["mode"] == "sampled" and minute0["sampled"]["episodes"] == 4
    assert run["checkpoint"]["update"] == 6
    assert run["provenance"]["torch_version"] == torch.__version__
    assert run["provenance"]["command_line"][1].endswith("tmnf_rl/train.py")
    assert "--run-id" in run["provenance"]["command_line"]
    assert (runs_root / run_id / "checkpoints" / "latest.pt").is_file()
    assert (runs_root / run_id / "evals" / "0.json").is_file()
    assert (runs_root / run_id / "policy.pt").is_file()
    rows = read_rows(runs_root / run_id / "metrics.csv", 100)
    assert len(rows) == 6 and list(rows[0]) == METRIC_NAMES
    index = json.loads((runs_root / "index.json").read_text())
    assert [entry["run_id"] for entry in index["runs"]] == [run_id]


def test_reproduce_matches_and_refuses_other_physics(trained_run: tuple[Path, str]) -> None:
    """Attack: 'reproduce' silently compares runs made with different binaries."""
    runs_root, run_id = trained_run
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.reproduce", run_id, "--updates", "6", "--runs-root", str(runs_root)],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads((runs_root / f"{run_id}_repro6" / "reproduce.json").read_text())
    assert report["matched"] is True and report["compared_updates"] == 6
    assert report["code_matched"] is True

    tampered = json.loads((runs_root / run_id / "run.json").read_text())
    tampered["physics_sha256"] = "f" * 64
    run_dir = runs_root / "tampered"
    run_dir.mkdir()
    tampered["run_id"] = "tampered"
    tampered["args"]["run_id"] = "tampered"
    write_json_atomic(run_dir / "run.json", tampered)
    (run_dir / "metrics.csv").write_bytes((runs_root / run_id / "metrics.csv").read_bytes())
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.reproduce", "tampered", "--updates", "2", "--runs-root", str(runs_root)],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "physics library differs" in result.stderr

    tampered["physics_sha256"] = json.loads((runs_root / run_id / "run.json").read_text())["physics_sha256"]
    tampered["code_sha256"] = "e" * 64
    write_json_atomic(run_dir / "run.json", tampered)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.reproduce", "tampered", "--updates", "2", "--runs-root", str(runs_root)],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "python/tmnf_rl differs" in result.stderr

    # F24: a run recorded under different CUDA numerics or a different GPU is
    # refused by name, before any replica is trained.
    tampered["code_sha256"] = json.loads((runs_root / run_id / "run.json").read_text())["code_sha256"]
    tampered["provenance"]["numerics"]["matmul_allow_tf32"] = True
    tampered["provenance"]["cuda_device_name"] = "NVIDIA GeForce RTX 3060"
    write_json_atomic(run_dir / "run.json", tampered)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.reproduce", "tampered", "--updates", "2", "--runs-root", str(runs_root)],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0
    gpu_name = torch.cuda.get_device_name(0)
    assert f"cuda_device_name 'NVIDIA GeForce RTX 3060' -> '{gpu_name}'" in result.stderr
    assert "numerics.matmul_allow_tf32 True -> False" in result.stderr
    assert not (runs_root / "tampered_repro2").exists()


def test_analog_arm_scores_the_stored_pre_tanh_steer_exactly() -> None:
    """The analog head stores the pre-tanh steer sample. Re-scoring the stored
    action must return the rollout log-probability bit for bit (the update's
    first ratio is then exactly 1), and the env receives tanh of it."""
    from tmnf_rl.spaces import environment_action

    torch.manual_seed(0)
    agent = Agent("mlp", 32, "analog")
    observation = torch.randn(64, 81)
    with torch.no_grad():
        action, log_probability, entropy, _ = agent.get_action_and_value(observation)
        again, rescored, _, _ = agent.get_action_and_value(observation, action)
    assert action.shape == (64, 3) and torch.equal(again, action)
    assert torch.equal(rescored, log_probability)
    assert torch.isfinite(entropy).all()
    env_action = environment_action(action, "analog")
    np.testing.assert_array_equal(env_action["steer"], torch.tanh(action[:, 0]).numpy())
    assert np.all(np.abs(env_action["steer"]) <= 1.0)
    assert set(np.unique(env_action["gas"])) <= {0, 1} and set(np.unique(env_action["brake"])) <= {0, 1}
    # Saturated samples (|pre-tanh| large) are the case atanh(clamp(.)) got wrong.
    saturated = action.clone()
    saturated[:, 0] = 12.0
    with torch.no_grad():
        _, saturated_log_probability, _, _ = agent.get_action_and_value(observation, saturated)
    assert torch.isfinite(saturated_log_probability).all()
    assert environment_action(saturated, "analog")["steer"].max() == np.float32(1.0)
    greedy = agent.get_deterministic_action(observation)
    assert torch.equal(greedy[:, 0], agent.steer_actor(agent.hidden(observation)).squeeze(-1))


def test_analog_arm_trains_reproduces_and_evaluates(tmp_path: Path) -> None:
    """The analog arm (tanh-Gaussian steer, Bernoulli gas/brake) goes through
    the same guards as the discrete one: bit-exact reproduce across processes,
    bit-exact resume, and an analog schedule out of the evaluator."""
    runs_root = tmp_path / "runs"
    run_id = "tiny_analog_s1"
    train(runs_root, run_id, "--max-updates", "5", "--action-space", "analog")
    run = json.loads((runs_root / run_id / "run.json").read_text())
    assert run["args"]["action_space"] == "analog"
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.reproduce", run_id, "--updates", "5", "--runs-root", str(runs_root)],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads((runs_root / f"{run_id}_repro5" / "reproduce.json").read_text())
    assert report["matched"] is True and report["compared_updates"] == 5
    half_id = "tiny_analog_s1_half"
    train(runs_root, half_id, "--max-updates", "3", "--action-space", "analog")
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", half_id, "--runs-root", str(runs_root), "--max-updates", "5"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    comparison = compare_rows(read_rows(runs_root / run_id / "metrics.csv", 5), read_rows(runs_root / half_id / "metrics.csv", 5))
    assert comparison["matched"] is True, comparison["first_mismatches"]
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.evaluate", run_id, "--envs", "2", "--episodes", "2", "--runs-root", str(runs_root), "--no-splits"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["result"]["episodes"] == 2 and report["action_space"] == "analog"


def test_target_kl_ends_an_update_early(tmp_path: Path) -> None:
    """target_kl ends an update's epochs once an epoch's mean approx_kl
    passes it: 1e-9 leaves exactly one epoch per update, 0 (off) all four,
    and the two runs differ from the first update on."""
    runs_root = tmp_path / "runs"
    train(runs_root, "kl_stop", "--max-updates", "3", "--action-space", "analog", "--target-kl", "1e-9")
    train(runs_root, "kl_free", "--max-updates", "3", "--action-space", "analog", "--target-kl", "0")
    stopped = read_rows(runs_root / "kl_stop" / "metrics.csv", 3)
    free = read_rows(runs_root / "kl_free" / "metrics.csv", 3)
    assert [int(r["train/update_epochs_run"]) for r in stopped] == [1, 1, 1]
    assert [int(r["train/update_epochs_run"]) for r in free] == [4, 4, 4]
    assert stopped[0]["train/entropy"] != free[0]["train/entropy"]


def test_transformer_arm_trains_reproduces_and_resumes(tmp_path: Path) -> None:
    """The second arm goes through the same guards as the MLP: bit-exact
    reproduce across processes (torch.compile, SDPA and the pinned numerics
    included), and a bit-exact resume."""
    runs_root = tmp_path / "runs"
    run_id = "tiny_tf_s1"
    train(runs_root, run_id, "--max-updates", "5", "--arch", "transformer_s", "--lr-warmup-updates", "2")
    run = json.loads((runs_root / run_id / "run.json").read_text())
    assert run["args"]["arch"] == "transformer_s" and run["summary"]["arch"] == "transformer_s"
    assert 780_000 < run["summary"]["params"] < 820_000
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.reproduce", run_id, "--updates", "5", "--runs-root", str(runs_root)],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads((runs_root / f"{run_id}_repro5" / "reproduce.json").read_text())
    assert report["matched"] is True and report["compared_updates"] == 5 and report["compared_columns"] == len(METRIC_NAMES) - len(TIMING_COLUMNS) == 33
    # Resume from update 3 reproduces updates 4-5.
    half_id = "tiny_tf_s1_half"
    train(runs_root, half_id, "--max-updates", "3", "--arch", "transformer_s", "--lr-warmup-updates", "2")
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", half_id, "--runs-root", str(runs_root), "--max-updates", "5"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    comparison = compare_rows(read_rows(runs_root / run_id / "metrics.csv", 5), read_rows(runs_root / half_id / "metrics.csv", 5))
    assert comparison["matched"] is True, comparison["first_mismatches"]
    # The arm is part of the checkpointed config: it cannot change on resume.
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", half_id, "--runs-root", str(runs_root), "--max-updates", "6", "--arch", "mlp"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "config differs from the checkpoint" in result.stderr
    # Evaluate loads the arm from policy.pt.
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.evaluate", run_id, "--envs", "2", "--episodes", "2", "--runs-root", str(runs_root), "--no-splits"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert json.loads(result.stdout.splitlines()[-1])["result"]["episodes"] == 2


PINNED_NUMERICS = {
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "deterministic_algorithms": True,
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None,
    "NVIDIA_TF32_OVERRIDE": None,
    "PYTHONHASHSEED": None,
}


def test_run_records_pinned_numerics_and_refuses_overrides(trained_run: tuple[Path, str], tmp_path: Path) -> None:
    """Attack (review R3): `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` changed
    `train/policy_loss` at update 1 (19 cells in 4 updates) and
    `CUBLAS_WORKSPACE_CONFIG=:16:8` 14 cells, with nothing in run.json saying
    why. Now every run records the pinned switches and refuses to start under
    an override."""
    runs_root, run_id = trained_run
    run = json.loads((runs_root / run_id / "run.json").read_text())
    provenance = run["provenance"]
    expected = dict(PINNED_NUMERICS, PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED"))
    assert provenance["numerics"] == expected
    assert provenance["cuda_device_name"] == torch.cuda.get_device_name(0)
    assert provenance["cuda_visible_devices"] == "0"
    assert provenance["cudnn_version"] == torch.backends.cudnn.version()
    for variable, value, message in (
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1' is set"),
        ("NVIDIA_TF32_OVERRIDE", "1", "NVIDIA_TF32_OVERRIDE='1' is set"),
        ("CUBLAS_WORKSPACE_CONFIG", ":16:8", "CUBLAS_WORKSPACE_CONFIG=':16:8' is set"),
    ):
        result = subprocess.run(
            [PYTHON, "-m", "tmnf_rl.train", *TINY, "--runs-root", str(tmp_path / "runs"),
             "--run-id", f"override_{variable}", "--max-updates", "1"],
            cwd=ROOT, env={**ENV, variable: value}, capture_output=True, text=True,
        )
        assert result.returncode != 0 and message in result.stderr, result.stderr[-1500:]
        assert not (tmp_path / "runs" / f"override_{variable}" / "run.json").exists()


def test_compare_rows_detects_single_cell_change() -> None:
    original = [{"update": "1", "wall_time_s": "1.0", "train/entropy": "2.0"}]
    replica = [{"update": "1", "wall_time_s": "9.0", "train/entropy": "2.0"}]
    assert compare_rows(original, replica)["matched"] is True
    replica[0]["train/entropy"] = "2.0000001"
    report = compare_rows(original, replica)
    assert report["matched"] is False and report["first_mismatches"][0]["column"] == "train/entropy"
    assert "wall_time_s" in TIMING_COLUMNS


def test_resume_is_bit_exact_and_guarded(trained_run: tuple[Path, str], tmp_path: Path) -> None:
    """Resume from update 3 must reproduce updates 4-6 of the uninterrupted run."""
    runs_root, full_id = trained_run
    half_id = "tiny_a01_s1_half"
    train(runs_root, half_id, "--max-updates", "3")

    # Guard: a running (unreaped) run cannot be resumed.
    run_path = runs_root / half_id / "run.json"
    payload = json.loads(run_path.read_text())
    finished = dict(payload)
    payload["status"] = "running"
    write_json_atomic(run_path, payload)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", half_id, "--runs-root", str(runs_root), "--max-updates", "6"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "still marked running" in result.stderr
    # Guard: a different physics library cannot continue the run.
    finished["physics_sha256"] = "a" * 64
    write_json_atomic(run_path, finished)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", half_id, "--runs-root", str(runs_root), "--max-updates", "6"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "physics library differs" in result.stderr
    # Guard: learner settings cannot change on resume.
    finished["physics_sha256"] = json.loads((runs_root / full_id / "run.json").read_text())["physics_sha256"]
    write_json_atomic(run_path, finished)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", half_id, "--runs-root", str(runs_root), "--max-updates", "6", "--learning-rate", "0.001"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "config differs from the checkpoint" in result.stderr
    # Guard (F25): python/tmnf_rl changed since the run started.
    finished["code_sha256"] = finished["code_sha256_current"] = "d" * 64
    write_json_atomic(run_path, finished)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", half_id, "--runs-root", str(runs_root), "--max-updates", "6"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "python/tmnf_rl differs from the run being resumed" in result.stderr
    assert json.loads(run_path.read_text()) == finished
    finished["code_sha256"] = finished["code_sha256_current"] = json.loads(
        (runs_root / full_id / "run.json").read_text()
    )["code_sha256"]
    # F30: evaluations and replays written after the checkpoint belong to the
    # timeline the resume discards. Plant one of each beyond the checkpoint's
    # elapsed time next to the real minute-0 evaluation.
    half_dir = runs_root / half_id
    minute0 = json.loads((half_dir / "evals" / "0.json").read_text())
    orphan_eval = {**minute0, "scheduled_minutes": 99.0, "wall_time_s": 1.0e9}
    write_json_atomic(half_dir / "evals" / "99.json", orphan_eval)
    (half_dir / "replays").mkdir(exist_ok=True)
    replays = []
    for minute, lap in ((0.0, 30_000), (99.0, 20_000)):
        (half_dir / "replays" / f"m{minute:g}.json").write_text("{}")
        (half_dir / "replays" / f"m{minute:g}.inputs.bin").write_bytes(b"\0")
        replays.append({
            "label": f"eval_{minute:g}m", "lap_ms": lap, "world_lap_ms": lap, "matches_world": True,
            "scene": f"replays/m{minute:g}.json", "inputs": f"replays/m{minute:g}.inputs.bin",
            "inputs_sha256": "0" * 64, "ticks": 1, "scene_checkpoint_ticks": [], "minute": minute,
            "exported_at": finished["started_at"],
        })
    finished["replays"] = replays
    write_json_atomic(run_path, finished)

    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", half_id, "--runs-root", str(runs_root), "--max-updates", "6"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    resumed = json.loads(run_path.read_text())
    assert resumed["status"] == "finished" and len(resumed["resumes"]) == 1
    assert resumed["resumes"][0]["discarded"] == {"evaluations": [99.0], "replays": ["replays/m99.json"]}
    assert not (half_dir / "evals" / "99.json").exists() and (half_dir / "evals" / "0.json").exists()
    assert not (half_dir / "replays" / "m99.json").exists() and not (half_dir / "replays" / "m99.inputs.bin").exists()
    assert [r["minute"] for r in resumed["replays"]] == [0.0]
    assert (half_dir / "replays" / "m0.json").exists()
    assert resumed["code_changes"] == [] and resumed["code_sha256_current"] == resumed["code_sha256"]
    assert resumed["summary"]["updates"] == 6
    assert resumed["summary"]["resumed_from"]["update"] == 3
    full_rows = read_rows(runs_root / full_id / "metrics.csv", 6)
    half_rows = read_rows(runs_root / half_id / "metrics.csv", 6)
    assert [row["update"] for row in half_rows] == ["1", "2", "3", "4", "5", "6"]
    comparison = compare_rows(full_rows, half_rows)
    assert comparison["matched"] is True, comparison["first_mismatches"]
    full_policy = torch.load(runs_root / full_id / "policy.pt", map_location="cpu", weights_only=False)
    half_policy = torch.load(runs_root / half_id / "policy.pt", map_location="cpu", weights_only=False)
    for name, tensor in full_policy["agent"].items():
        assert torch.equal(tensor, half_policy["agent"][name]), name
    assert "normalizer" not in full_policy and "normalizer" not in half_policy


def test_resume_across_code_change_is_recorded_and_not_reproducible(trained_run: tuple[Path, str], tmp_path: Path) -> None:
    """Attack (review R4): a 3-update run resumed to 5 after a comment was
    appended to agents/ppo.py exited 0 with no message, run.json kept the old
    top-level code_sha256, and protocol/reproduce would have trusted it."""
    runs_root, full_id = trained_run
    run_id = "tiny_a01_s1_codechange"
    train(runs_root, run_id, "--max-updates", "3")
    run_path = runs_root / run_id / "run.json"
    original = json.loads(run_path.read_text())
    original_hash = original["code_sha256"]
    # Stand in for an edited tree: the recorded hash no longer matches the code.
    original["code_sha256"] = original["code_sha256_current"] = "a" * 64
    write_json_atomic(run_path, original)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", run_id, "--runs-root", str(runs_root),
         "--max-updates", "5", "--ignore-code-hash"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"event": "code_change"')]
    assert len(events) == 1 and events[0]["from_update"] == 3
    resumed = json.loads(run_path.read_text())
    assert resumed["code_sha256"] == "a" * 64, "top-level hash must stay the one that produced update 1"
    assert resumed["code_sha256_current"] == original_hash
    assert len(resumed["code_changes"]) == 1
    change = resumed["code_changes"][0]
    assert change["from_update"] == 3 and change["code_sha256_before"] == "a" * 64
    assert change["code_sha256_after"] == original_hash and "at" in change
    # Rows 4-5 still match the uninterrupted run (the "change" was only a hash).
    comparison = compare_rows(read_rows(runs_root / full_id / "metrics.csv", 5), read_rows(runs_root / run_id / "metrics.csv", 5))
    assert comparison["matched"] is True, comparison["first_mismatches"]

    # reproduce: at most the pre-change updates can be claimed from code_sha256.
    resumed["code_sha256"] = original_hash  # pretend the tree is the pre-change one again
    write_json_atomic(run_path, resumed)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.reproduce", run_id, "--updates", "5", "--runs-root", str(runs_root)],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "resumed across a python/tmnf_rl change at update 3" in result.stderr
    # protocol: a seed with code_changes is refused outright.
    with pytest.raises(RuntimeError, match="resumed across python/tmnf_rl changes"):
        protocol_summarize(name="x", run_ids={1: run_id}, minutes=0.0, registry=RunRegistry(runs_root))


def test_concurrent_resumes_serialise_on_the_run_lock(tmp_path: Path) -> None:
    """Attack (review R2): two `--resume` of one run started together.

    Without the lock both passed the `status == running` check, both trained,
    metrics.csv held every post-checkpoint update twice (1,2,3,3,4,4,...) and
    run.json ended `failed` with an 8-update summary. Now exactly one trains
    and the other fails fast naming the holder's pid.
    """
    runs_root = tmp_path / "runs"
    train(runs_root, "dup", "--max-updates", "2", "--checkpoint-interval-minutes", "1000")
    command = [
        PYTHON, "-m", "tmnf_rl.train", "--resume", "dup", "--runs-root", str(runs_root),
        "--max-updates", "8",
    ]
    procs = [
        subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(2)
    ]
    outputs = [proc.communicate()[0] for proc in procs]
    codes = sorted(proc.returncode for proc in procs)
    assert codes == [0, 1], outputs
    loser = outputs[[proc.returncode for proc in procs].index(1)]
    winner_pid = procs[[proc.returncode for proc in procs].index(0)].pid
    assert f"locked by another trainer (pid {winner_pid} " in loser, loser[-2000:]
    rows = read_rows(runs_root / "dup" / "metrics.csv", 8)
    assert [row["update"] for row in rows] == [str(i) for i in range(1, 9)]
    run = json.loads((runs_root / "dup" / "run.json").read_text())
    assert run["status"] == "finished" and run["summary"]["updates"] == 8 and len(run["resumes"]) == 1
    assert not (runs_root / "dup" / "checkpoints" / "latest.pt.tmp").exists()


def test_run_lock_names_holder_and_dies_with_it(tmp_path: Path) -> None:
    """A SIGKILLed holder leaves no stale lock: flock is released by the kernel."""
    registry = RunRegistry(tmp_path / "runs")
    registry.create_run(make_payload(registry, "held"))
    script = (
        "import sys, time\nsys.path.insert(0, sys.argv[2])\n"
        "from pathlib import Path\nfrom tmnf_rl.registry import RunRegistry\n"
        "lock = RunRegistry(Path(sys.argv[1])).lock_run('held')\nprint('locked', flush=True)\ntime.sleep(60)\n"
    )
    holder = subprocess.Popen(
        [PYTHON, "-c", script, str(registry.root), str(ROOT / "python")],
        cwd=ROOT, env=ENV, stdout=subprocess.PIPE, text=True,
    )
    assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
    with pytest.raises(registry_module.RunLockError, match=f"pid {holder.pid} "):
        registry.lock_run("held")
    holder.send_signal(signal.SIGKILL)
    holder.wait()
    lock = registry.lock_run("held")
    assert f"pid {os.getpid()} " in (registry.root / "held" / "run.lock").read_text()
    lock.release()
    # A duplicate fresh --run-id is refused before any environment is built.
    with pytest.raises(FileExistsError, match="already exists"):
        registry.claim_run("held")
    started = time.perf_counter()
    result = train(registry.root, "held", "--max-updates", "1", check=False)
    assert result.returncode != 0 and "run directory already exists" in result.stderr
    assert time.perf_counter() - started < 15.0


def test_tracks_catalogue_matches_manifest() -> None:
    manifest_ids = {
        line.split("|")[0]
        for line in (ROOT / "oracle" / "tracks" / "manifest.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert set(track_catalogue()) == manifest_ids | {"a01"}
    assert {"a01", "a08", "a10", "b04", "c03", "e01"} <= set(track_catalogue())
    # F22 regression tracks must stay catalogued.
    assert {"d01", "e02", "e03"} <= set(track_catalogue())


def require_fixtures(track_id: str) -> None:
    """Skip visibly (pytest -rs) when a manifest track's files are not in this
    tree; the gate runs from a worktree of HEAD, so an onboarded-but-uncommitted
    track (F32) shows up here as a skip, never as a pass."""
    missing = track_catalogue()[track_id].missing_fixtures(ROOT)
    if missing:
        pytest.skip(f"{track_id}: fixtures not in this tree: {[p.name for p in missing]}")


SPEED_CAP_MPS = 277.77777099609375


@pytest.mark.parametrize("track_id", sorted(track_catalogue()))
def test_every_manifest_track_builds_an_env_with_a_derived_horizon(track_id: str) -> None:
    """Generic track loading: every committed track creates with the default
    config and derives both budgets natively (2 x laps x length at 50 m/s, no
    clamp, never below the speed-cap bound), and steps.

    F22: the old 12,000-tick default `max_race_ticks` clamped the horizon on
    12 of 27 tracks and made D01 (18.2 km, 5 laps) need a 546 km/h average;
    the derived budget must let a 90 km/h (25 m/s) average finish every track.
    """
    require_fixtures(track_id)
    env = TmnfVectorEnv(2, track=track_id, thread_count=2)
    try:
        total = env.route_length * env.lap_count
        minimum = int(np.ceil(total / SPEED_CAP_MPS / 0.01))
        derived = int(np.ceil(2.0 * total / 50.0 / 0.01))
        finish_ticks_at_25_mps = total / 25.0 / 0.01
        assert env.horizon_ticks == derived
        assert env.max_race_ticks == derived
        assert env.horizon_ticks >= finish_ticks_at_25_mps
        assert env.max_race_ticks >= finish_ticks_at_25_mps
        assert env.horizon_ticks >= minimum
        if track_id in {"d01", "e02", "e03"}:
            assert env.max_race_ticks > 12_000 and env.horizon_ticks > 12_000
        observations, _ = env.reset()
        env.step(np.array([4, 4]))
        assert np.isfinite(observations["vehicle"]).all()
    finally:
        env.close()


def test_elapsed_fraction_decodes_with_the_resolved_timeout() -> None:
    """race[8] is elapsed_ticks / resolved max_race_ticks; decoding it with the
    config value (0) or the old default (12,000) gives wrong capture ticks on
    every track whose derived budget is not 12,000."""
    require_fixtures("d01")
    env = TmnfVectorEnv(1, track="d01", thread_count=1, action_repeat=7)
    try:
        assert env.max_race_ticks == 74_358
        env.reset()
        for _ in range(3):
            env.step(np.array([1]))
        ticks = int(np.rint(env.observations["race"][0, 8] * env.max_race_ticks))
        assert ticks == 21
        assert int(np.rint(env.observations["race"][0, 8] * 12_000)) != 21
    finally:
        env.close()


def test_train_accepts_any_manifest_track(tmp_path: Path) -> None:
    """`--track <id>` for a track that no code path names explicitly."""
    require_fixtures("c01")
    runs_root = tmp_path / "runs"
    train(runs_root, "tiny_c01", "--max-updates", "2", "--track", "c01")
    run = json.loads((runs_root / "tiny_c01" / "run.json").read_text())
    assert run["track_id"] == "c01" and run["track_name"] == "C01-Race"
    assert run["environment"]["horizon_ticks"] == 7716
    assert run["environment"]["max_race_ticks"] == 7716
    assert run["environment"]["lap_count"] == 1
    assert run["args"]["horizon_ticks"] == 0 and run["args"]["staggered_phases"] is False
    assert run["summary"]["updates"] == 2


def test_failed_setup_abandons_the_claimed_run_directory(tmp_path: Path) -> None:
    """A refused start (here: a numerics override) leaves no lock-only run
    directory behind, so the same run id works on the retry."""
    runs_root = tmp_path / "runs"
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", *TINY, "--runs-root", str(runs_root), "--run-id", "retry", "--max-updates", "1"],
        cwd=ROOT, env={**ENV, "NVIDIA_TF32_OVERRIDE": "0"}, capture_output=True, text=True,
    )
    assert result.returncode != 0 and not (runs_root / "retry").exists()
    train(runs_root, "retry", "--max-updates", "1")
    assert json.loads((runs_root / "retry" / "run.json").read_text())["status"] == "finished"


CUDA_LIBRARY = ROOT / "build" / "libtmnf_cuda.so"


def _cuda_env_or_skip(num_envs: int, **kwargs):
    from tmnf_rl.cuda_env import TmnfCudaVectorEnv

    if not CUDA_LIBRARY.is_file():
        pytest.skip(f"{CUDA_LIBRARY} is not built")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return TmnfCudaVectorEnv(num_envs, library_path=CUDA_LIBRARY, device=torch.device("cuda:0"), **kwargs)


@pytest.mark.parametrize("track", [
    "a01", "a04", "desert-a1", "rally-a1", "snow-a1",
    "island-a1", "coast-a1", "bay-a1", "stadium-a1",
])
def test_cuda_env_steps_bit_identically_to_the_cpu_env_and_exchanges_snapshots(track: str) -> None:
    """P0: the device binding is the CPU env's RL mirror: same flat rows,
    rewards, discounts and terminations for the same actions (discrete and
    analog), and snapshots restore across the two envs in both directions."""
    n, repeat = 64, 5
    reference = TmnfVectorEnv(n, thread_count=4, action_repeat=repeat, track=track, library_path=CUDA_LIBRARY if CUDA_LIBRARY.is_file() else None)
    budgets = dict(max_race_ticks=reference.max_race_ticks, horizon_ticks=reference.horizon_ticks)
    cuda = _cuda_env_or_skip(n, action_repeat=repeat, action_space="discrete", track=track, **budgets)
    rng = np.random.default_rng(3)
    try:
        reference.reset()
        cuda.reset()
        np.testing.assert_array_equal(cuda.device_observations.cpu().numpy(), reference.policy_observations)
        for step in range(120):
            action = rng.integers(0, 12, size=n)
            reference.step(action)
            cuda.step(torch.as_tensor(action, device=cuda.device))
            np.testing.assert_array_equal(cuda.device_observations.cpu().numpy(), reference.policy_observations, err_msg=f"step {step}")
            np.testing.assert_array_equal(cuda.device_transitions.cpu().numpy(), reference.policy_transitions, err_msg=f"step {step}")
            np.testing.assert_array_equal(cuda.observations["race"], reference.observations["race"])
            np.testing.assert_array_equal(cuda.terminations, reference.terminations)
            np.testing.assert_array_equal(cuda.termination_reasons, reference.termination_reasons)
        # CPU -> CUDA and CUDA -> CPU restores continue identically.
        blobs = reference.capture(np.arange(0, n, 2, dtype=np.uint32))
        cuda.restore(np.arange(1, n, 2, dtype=np.uint32), blobs)
        cuda_blobs = cuda.capture(np.arange(0, n, 2, dtype=np.uint32))
        reference.restore(np.arange(1, n, 2, dtype=np.uint32), cuda_blobs)
        np.testing.assert_array_equal(cuda.device_observations.cpu().numpy(), reference.policy_observations)
        for _ in range(60):
            action = rng.integers(0, 12, size=n)
            reference.step(action)
            cuda.step(torch.as_tensor(action, device=cuda.device))
            np.testing.assert_array_equal(cuda.device_observations.cpu().numpy(), reference.policy_observations)
            np.testing.assert_array_equal(cuda.device_transitions.cpu().numpy(), reference.policy_transitions)
    finally:
        reference.close()
        cuda.close()

    reference = TmnfVectorEnv(n, thread_count=4, action_repeat=repeat, action_space="analog", track=track, library_path=CUDA_LIBRARY)
    cuda = _cuda_env_or_skip(n, action_repeat=repeat, action_space="analog", track=track, **budgets)
    try:
        reference.reset()
        cuda.reset()
        for step in range(120):
            steer = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
            gas = rng.integers(0, 2, size=n).astype(np.int8)
            brake = rng.integers(0, 2, size=n).astype(np.int8)
            reference.step({"steer": steer, "gas": gas, "brake": brake})
            device_action = torch.as_tensor(np.stack((steer, gas, brake), axis=1).astype(np.float32), device=cuda.device)
            cuda.step(device_action)
            np.testing.assert_array_equal(cuda.device_observations.cpu().numpy(), reference.policy_observations, err_msg=f"step {step}")
            np.testing.assert_array_equal(cuda.device_transitions.cpu().numpy(), reference.policy_transitions, err_msg=f"step {step}")
    finally:
        reference.close()
        cuda.close()


def test_cuda_env_training_mirrors_the_cpu_trainer_and_reproduces(tmp_path: Path) -> None:
    """P0: a tiny run on the CUDA env (env_device cuda, the CUDA library for
    both the training env and the CPU evaluation env) writes the same metric
    cells as the CPU-env trainer with the same seed, and reproduces itself."""
    if not CUDA_LIBRARY.is_file():
        pytest.skip(f"{CUDA_LIBRARY} is not built")
    runs_root = tmp_path / "runs"
    common = ["--physics-library", str(CUDA_LIBRARY), "--max-updates", "6", "--num-envs", "128"]
    train(runs_root, "cpu_a", *common, "--env-device", "cpu")
    train(runs_root, "cuda_a", *common, "--env-device", "cuda")
    train(runs_root, "cuda_b", *common, "--env-device", "cuda")
    cpu_a = read_rows(runs_root / "cpu_a" / "metrics.csv", 6)
    cuda_a = read_rows(runs_root / "cuda_a" / "metrics.csv", 6)
    cuda_b = read_rows(runs_root / "cuda_b" / "metrics.csv", 6)
    mirror = compare_rows(cpu_a, cuda_a)
    assert mirror["matched"] and mirror["compared_updates"] == 6, mirror["first_mismatches"]
    replica = compare_rows(cuda_a, cuda_b)
    assert replica["matched"], replica["first_mismatches"]
    run = json.loads((runs_root / "cuda_a" / "run.json").read_text())
    assert run["args"]["env_device"] == "cuda" and run["summary"]["updates"] == 6



# ------------------------------------------------------- Phase 6 exploration


def test_landing_novelty_and_pedal_hold_units() -> None:
    """LandingNovelty pays coef / sqrt(n) at the decision a car lands after
    >= 2 airborne decisions, keyed by (20 m bin, airborne decisions, 2 m/s
    bin); an episode end resets the airborne count and scores nothing.
    PedalHold decides pedals every `period` decisions and holds them between,
    restarting at an episode end."""
    from tmnf_rl.exploration import LandingNovelty, PedalHold, entropy_weights

    novelty = LandingNovelty(2, coef=0.5)
    obs = np.zeros((2, 81), dtype=np.float32)
    obs[:, 21:25] = 1.0  # on the ground
    obs[:, 7] = 40.0  # 40 m/s
    obs[:, 39] = 300.0
    ended = np.zeros(2, dtype=bool)
    assert not novelty.step(obs, ended).any()
    air = obs.copy(); air[:, 21:25] = 0.0
    for _ in range(3):
        assert not novelty.step(air, ended).any()
    assert list(novelty.air_decisions) == [3, 3]
    bonus = novelty.step(obs, np.array([False, True]))  # env 1's episode ended: no landing scored
    assert bonus[0] == pytest.approx(0.5) and bonus[1] == 0.0
    assert novelty.counts == {(15, 3, 20): 1} and novelty.landings == 1
    for _ in range(3):
        novelty.step(air, ended)
    bonus = novelty.step(obs, ended)
    assert bonus[0] == pytest.approx(0.5 / np.sqrt(2)) and bonus[1] == pytest.approx(0.5 / np.sqrt(3))
    state = novelty.state_dict()
    again = LandingNovelty(2, coef=0.5)
    again.load_state_dict(state)
    assert again.counts == novelty.counts and again.bonus_total == novelty.bonus_total

    hold = PedalHold(3, 2, torch.device("cpu"))
    mask = hold.mask()
    assert mask.tolist() == [True, True, True]
    first = torch.tensor([[0.1, 1.0, 0.0], [0.2, 0.0, 1.0], [0.3, 1.0, 1.0]])
    assert torch.equal(hold.apply(first, mask), first)
    hold.advance(np.array([False, False, True]))
    mask = hold.mask()
    assert mask.tolist() == [False, False, True]
    second = torch.tensor([[0.5, 0.0, 1.0], [0.6, 1.0, 0.0], [0.7, 0.0, 0.0]])
    applied = hold.apply(second, mask)
    assert torch.equal(applied[:, 0], second[:, 0])
    assert applied[:, 1:].tolist() == [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    hold.advance(np.zeros(3, dtype=bool))
    assert hold.mask().tolist() == [True, True, False]

    weights = entropy_weights(torch.tensor(obs), (200.0, 400.0), 4.0)
    assert weights.tolist() == [5.0, 5.0]
    assert entropy_weights(torch.tensor(obs), (400.0, 600.0), 4.0).tolist() == [1.0, 1.0]

    # (e) finish-time-only reward: 0 everywhere except a finishing decision,
    # which pays the unused budget in seconds; failures pay 0.
    from tmnf_rl.exploration import finish_time_rewards

    rewards = finish_time_rewards(
        np.array([True, True, False, True]), np.array([1, 3, 1, 4]),
        np.array([24_600, 5_000, 24_600, 80_000]), 8_839,
    )
    assert rewards.dtype == np.float32
    assert rewards.tolist() == pytest.approx([(8_839 - 2_460) / 100.0, 0.0, 0.0, 0.0])

    # Real observations: a full-gas straight run on A04 flies the 72 m jump
    # (21 airborne decisions at repeat 5) and lands once, at 173 m, 74 m/s.
    env = TmnfVectorEnv(1, track="a04", thread_count=1, action_repeat=5, action_space="analog")
    try:
        env.reset()
        novelty = LandingNovelty(1, coef=0.5)
        bonuses = []
        for _ in range(120):
            _, _, terminated, truncated, _ = env.step(
                {"steer": np.zeros(1, np.float32), "gas": np.ones(1, np.int8), "brake": np.zeros(1, np.int8)}
            )
            bonuses.append(float(novelty.step(env.policy_observations, terminated | truncated)[0]))
            if terminated[0] or truncated[0]:
                break
    finally:
        env.close()
    assert sum(1 for b in bonuses if b > 0) == 1 and max(bonuses) == pytest.approx(0.5)
    ((cell, count),) = novelty.counts.items()
    assert cell == (8, 21, 36) and count == 1 and novelty.landings == 1


def test_exploration_arms_train_reproduce_resume_and_change_the_run(tmp_path: Path) -> None:
    """Each Phase 6 arm (landing novelty, pedal hold, entropy boost, EMA +
    lr decay) runs on the analog head, differs from the plain run once it
    has anything to act on, and the
    combined arms reproduce bit-exactly and resume bit-exactly (their state is
    checkpointed). The pedal-hold run masks pedal log-probabilities and the
    lr-decay run logs the decayed rate; pedal hold is refused on the discrete
    head."""
    runs_root = tmp_path / "runs"
    base = ["--max-updates", "6", "--action-space", "analog", *SNAPSHOT_START_FLAGS]
    arms = {
        "novelty": ["--novelty-coef", "0.5"],
        "hold": ["--pedal-hold-decisions", "2"],
        "boost": ["--entropy-boost", "4", "--entropy-boost-window", "0,400"],
        "ema": ["--ema-decay", "0.9", "--lr-decay-updates", "4"],
        "finish_time": ["--finish-time-reward"],
    }
    train(runs_root, "plain", *base)
    plain = read_rows(runs_root / "plain" / "metrics.csv", 6)
    for name, flags in arms.items():
        train(runs_root, name, *base, *flags)
        rows = read_rows(runs_root / name / "metrics.csv", 6)
        if name in ("novelty", "finish_time"):
            # No car lands after >= 2 airborne decisions, and none finishes,
            # in six tiny updates here; the bonus and the reward switch are
            # exercised on synthetic inputs above and the runs must then
            # equal the plain one (the switch update stays 0).
            assert compare_rows(rows, plain)["matched"] is True
            if name == "finish_time":
                summary = json.loads((runs_root / name / "summary.json").read_text())
                assert summary["finish_time_switch_update"] == 0
            continue
        assert rows[0]["train/policy_loss"] != plain[0]["train/policy_loss"] or (
            rows[1]["train/policy_loss"] != plain[1]["train/policy_loss"]
        ), name
    ema = read_rows(runs_root / "ema" / "metrics.csv", 6)
    rates = [float(r["train/learning_rate"]) for r in ema]
    assert rates[0] == pytest.approx(2.5e-4) and rates[3] == pytest.approx(6.25e-5)
    assert rates[4] == pytest.approx(2.5e-5) and rates[5] == pytest.approx(2.5e-5)  # the 0.1 floor
    assert all(float(r["train/learning_rate"]) == pytest.approx(2.5e-4) for r in plain)
    assert all(int(r["train/novelty_cells"]) == 0 for r in plain)
    novelty_summary = json.loads((runs_root / "novelty" / "summary.json").read_text())
    assert novelty_summary["eval_env_restore_count"] == 0
    everything = [flag for flags in arms.values() for flag in flags]
    train(runs_root, "all_a", *base, *everything)
    train(runs_root, "all_b", *base, *everything)
    rows_a = read_rows(runs_root / "all_a" / "metrics.csv", 6)
    rows_b = read_rows(runs_root / "all_b" / "metrics.csv", 6)
    assert compare_rows(rows_a, rows_b)["matched"] is True
    train(runs_root, "all_half", "--max-updates", "3", "--action-space", "analog", *SNAPSHOT_START_FLAGS, *everything)
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", "all_half", "--runs-root", str(runs_root), "--max-updates", "6"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    rows_half = read_rows(runs_root / "all_half" / "metrics.csv", 6)
    comparison = compare_rows(rows_a, rows_half)
    assert comparison["compared_updates"] == 6 and comparison["matched"] is True, comparison["first_mismatches"]
    result = train(runs_root, "discrete_hold", "--max-updates", "1", "--pedal-hold-decisions", "2", check=False)
    assert result.returncode != 0 and "analog" in result.stderr


# ------------------------------------------------------------ off-policy arm


def test_td3_trainer_learns_from_replay_reproduces_and_evaluates(tmp_path: Path) -> None:
    """--algorithm td3: one env step of every environment per iteration, a
    device replay, twin HL-Gauss critics over the env's return support and a
    tanh actor. Two same-seed runs match cell for cell, the metrics table
    carries the off-policy columns, warm-up steps take uniform actions (no
    gradient steps), policy.pt evaluates through tmnf_rl.evaluate with the
    agent's own action conversion, the discrete head is refused and resume is
    refused."""
    from tmnf_rl.agents.td3 import LOG_EVERY_STEPS, TD3_METRIC_NAMES, OffPolicyAgent

    agent = OffPolicyAgent(32, 64, 11, -10.0, 10.0, 0.2, 0.1)
    distribution = agent.target_distribution(torch.tensor([0.0, 25.0]))
    assert distribution.shape == (2, 11) and torch.allclose(distribution.sum(dim=1), torch.ones(2))
    assert distribution[0].argmax() == 5 and distribution[1].argmax() == 10
    converted = agent.environment_action(torch.tensor([[0.3, 1.0], [-0.7, 2.0], [0.1, 3.0], [0.0, 0.0]]), "analog")
    assert converted["steer"].tolist() == pytest.approx([0.3, -0.7, 0.1, 0.0])
    assert converted["gas"].tolist() == [1, 0, 1, 0] and converted["brake"].tolist() == [0, 1, 1, 0]
    observation = torch.zeros((5, 81))
    with torch.no_grad():
        greedy = agent.act(observation)
        assert greedy.shape == (5, 2) and set(greedy[:, 1].tolist()) <= {0.0, 1.0, 2.0, 3.0}
        assert agent.q_values(agent.critics[0], observation, greedy[:, :1]).shape == (5, 4)

    runs_root = tmp_path / "runs"
    flags = ["--action-space", "analog", "--algorithm", "td3", "--td3-batch-size", "64",
             "--td3-warmup-steps", str(LOG_EVERY_STEPS), "--td3-replay-capacity", "4096", "--max-updates", "3"]
    train(runs_root, "td3_a", *flags)
    train(runs_root, "td3_b", *flags)
    rows_a = read_rows(runs_root / "td3_a" / "metrics.csv", 3)
    rows_b = read_rows(runs_root / "td3_b" / "metrics.csv", 3)
    assert list(rows_a[0].keys()) == TD3_METRIC_NAMES
    assert compare_rows(rows_a, rows_b)["matched"] is True
    assert int(rows_a[0]["train/gradient_steps"]) == 0 and int(rows_a[0]["train/replay_size"]) == 16 * LOG_EVERY_STEPS
    assert int(rows_a[2]["train/gradient_steps"]) == 2 * 2 * LOG_EVERY_STEPS
    assert float(rows_a[2]["train/critic_loss"]) > 0.0
    summary = json.loads((runs_root / "td3_a" / "summary.json").read_text())
    assert summary["algorithm"] == "td3" and summary["return_support"][0] < 0.0 < summary["return_support"][1]
    result = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.evaluate", str(runs_root / "td3_a" / "policy.pt"), "--runs-root", str(runs_root),
         "--envs", "4", "--episodes", "4"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["result"]["episodes"] == 4 and report["sampled"]["episodes"] == 4
    refused = train(runs_root, "td3_discrete", "--algorithm", "td3", "--max-updates", "1", check=False)
    assert refused.returncode != 0 and "analog" in refused.stderr
    resumed = subprocess.run(
        [PYTHON, "-m", "tmnf_rl.train", "--resume", "td3_a", "--runs-root", str(runs_root), "--max-updates", "4"],
        cwd=ROOT, env=ENV, capture_output=True, text=True,
    )
    assert resumed.returncode != 0 and "does not resume" in resumed.stderr
