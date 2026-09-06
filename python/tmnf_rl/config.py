"""Dataclass run configuration with JSON files and strict CLI overrides."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass
class TrainConfig:
    seed: int = 1
    track: str = "a01"
    algorithm: str = "ppo"
    num_envs: int = 256
    num_steps: int = 128
    action_repeat: int = 5
    action_space: str = "discrete"
    thread_count: int = 28
    duration_minutes: float = 45.0
    max_updates: int = 0
    learning_rate: float = 2.5e-4
    discount_per_tick: float = 0.9999
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Skip remaining epochs when mean approx_kl exceeds this; 0 disables it.
    target_kl: float = 0.05
    # MLP or tokenized transformer, both using the fixed-scale encoder.
    encoder_version: int = 1
    arch: str = "mlp"
    hidden_size: int = 256
    # Linear learning-rate warm-up over this many updates; 0 = none.
    lr_warmup_updates: int = 0
    # Zero derives the budget from route length, laps and reference speed.
    # Values below the speed-cap bound fail environment creation. Resolved
    # budgets are recorded in run.json.environment.
    max_race_ticks: int = 0
    horizon_ticks: int = 0
    # Optional pace for automatic budgets only; progress rewards remain at
    # their existing 50 m/s normalization. Zero retains legacy derivation.
    budget_reference_speed: float = 0.0
    off_track_grace_ticks: int = 100
    stuck_grace_ticks: int = 500
    # End an episode after this little progress per tick for
    # stuck_grace_ticks consecutive ticks.
    stuck_progress_epsilon: float = 0.02
    # Fraction of training resets drawn from the agent's own trajectory
    # snapshot pool. Zero uses only grid starts; evaluation always uses the grid.
    # Capture frequency and the fraction of episodes sampled are set below.
    snapshot_start_fraction: float = 0.5
    snapshot_capture_interval_ticks: int = 100
    snapshot_capture_episode_fraction: float = 0.25
    # Minimum continuation time before a failed episode ends. Shorter
    # exclusions permit recovery experiments using only our own snapshots;
    # terminal snapshots remain excluded. The established recipe uses 200.
    snapshot_prefailure_exclusion_ticks: int = 200
    # Learning rule: "ppo" (agents/ppo.py) or "td3" (agents/td3.py, the
    # off-policy large-batch actor-critic: one env step of every environment
    # per iteration, td3_updates_per_step critic updates of td3_batch_size
    # from a device replay of td3_replay_capacity transitions, actor every
    # td3_policy_delay critic steps, Polyak td3_tau, exploration N(0,
    # td3_exploration_std) on the tanh action plus uniform pedal coordinates
    # with probability td3_pedal_epsilon per decision, uniform actions for the first
    # td3_warmup_steps, twin HL-Gauss critics of td3_critic_width over
    # td3_bins). Analog only; no resume.
    td3_replay_capacity: int = 2_000_000
    td3_batch_size: int = 32_768
    td3_updates_per_step: int = 2
    td3_policy_delay: int = 2
    td3_tau: float = 0.005
    td3_exploration_std: float = 0.2
    td3_pedal_epsilon: float = 0.1
    td3_warmup_steps: int = 64
    td3_critic_width: int = 512
    td3_bins: int = 201
    # Optional exploration settings (tmnf_rl.exploration).
    # (a) Landing novelty: a landing after >= 2 airborne decisions adds
    # novelty_coef / sqrt(n) to that decision's reward, n the run's visits of
    # the (20 m progress bin, airborne decisions, 2 m/s speed bin) cell.
    novelty_coef: float = 0.0
    # (b) Pedal hold: gas and brake decided every pedal_hold_decisions
    # decisions of an episode and held between (steer every decision). 1 = off.
    pedal_hold_decisions: int = 1
    # (c) Entropy boost: rows whose unwrapped progress lies in
    # entropy_boost_window ("low,high" metres) weigh (1 + entropy_boost) in the
    # entropy term. Off when the window is empty.
    entropy_boost: float = 0.0
    entropy_boost_window: str = ""
    # (d) EMA policy weights, decay per update (evaluation, exports and
    # policy.pt use the average; 0 = live weights), and a linear learning-rate
    # decay to LR_DECAY_FLOOR x the base rate at lr_decay_updates (0 = none).
    ema_decay: float = 0.0
    lr_decay_updates: int = 0
    # (e) Finish-time-only reward: the native shaped reward (progress
    # potential) trains the policy until its first training finish; from
    # the next decision on every reward is 0 except the finishing decision,
    # which pays the unused race budget in seconds, (max_race_ticks - lap
    # ticks) / 100. Failures pay 0. Off by default.
    finish_time_reward: bool = False
    # Optionally stagger episode start phases.
    staggered_phases: bool = False
    stagger_max_seconds: float = 2.0
    eval_interval_minutes: float = 5.0
    eval_num_envs: int = 64
    # Episodes of the sampled-policy evaluation (actions drawn from the policy
    # with the run seed). The greedy evaluation runs one round of eval_num_envs:
    # it is a single trajectory, reported as `finished`, not a rate.
    eval_episodes: int = 256
    checkpoint_interval_minutes: float = 5.0
    # Where the training environments run: `cpu` (TmnfVectorEnv, thread_count
    # threads) or `cuda` (TmnfCudaVecEnv on the training GPU: device-resident
    # observations and actions, tmnf_rl.cuda_env; physics_library must be the
    # CUDA build, build/libtmnf_cuda.so). Evaluation environments are always
    # CPU environments of the same library. thread_count is the CPU env's.
    env_device: str = "cpu"
    physics_library: str = "build/libtmnf_physics.so"
    runs_root: str = "build/runs"
    run_id: str = ""
    spectate: int = 0

    def validate(self) -> None:
        if self.action_space not in ("discrete", "analog"):
            raise ConfigError("action_space must be discrete or analog")
        if self.arch not in ("mlp", "transformer_s"):
            raise ConfigError("arch must be mlp or transformer_s")
        if self.hidden_size <= 0 or self.lr_warmup_updates < 0:
            raise ConfigError("hidden_size must be positive and lr_warmup_updates non-negative")
        if self.ent_coef <= 0.0:
            raise ConfigError("ent_coef must be positive")
        batch_size = self.num_envs * self.num_steps
        if self.num_envs <= 0 or self.num_steps <= 0:
            raise ConfigError("num_envs and num_steps must be positive")
        if self.thread_count <= 0 or self.action_repeat <= 0:
            raise ConfigError("thread_count and action_repeat must be positive")
        if self.duration_minutes < 0.0 or self.max_updates < 0:
            raise ConfigError("duration_minutes and max_updates cannot be negative")
        if self.duration_minutes == 0.0 and self.max_updates == 0:
            raise ConfigError("duration_minutes or max_updates must stop the run")
        if self.spectate < 0 or self.spectate > 65_535:
            raise ConfigError("spectate port must be in [0, 65535]")
        if self.encoder_version not in (1, 2, 3):
            raise ConfigError("encoder_version must be 1 (legacy) or 2 (all materials) or 3 (gates)")
        if self.env_device not in ("cpu", "cuda"):
            raise ConfigError("env_device must be cpu or cuda")
        if self.env_device == "cuda" and self.spectate:
            raise ConfigError("spectate reads host observations; it needs env_device cpu")
        if batch_size % self.num_minibatches != 0:
            raise ConfigError("rollout batch must divide evenly into minibatches")
        if not 0.0 < self.discount_per_tick < 1.0:
            raise ConfigError("discount_per_tick must be strictly between zero and one")
        if self.snapshot_capture_interval_ticks <= 0:
            raise ConfigError("snapshot_capture_interval_ticks must be positive")
        if self.snapshot_prefailure_exclusion_ticks < 1:
            raise ConfigError("snapshot_prefailure_exclusion_ticks must be positive")
        if not 0.0 < self.snapshot_capture_episode_fraction <= 1.0:
            raise ConfigError("snapshot capture episode fraction must be in (0, 1]")
        if not 0.0 <= self.snapshot_start_fraction <= 1.0:
            raise ConfigError("snapshot_start_fraction must be in [0, 1]")
        if self.algorithm not in ("ppo", "td3"):
            raise ConfigError("algorithm must be ppo or td3")
        if self.algorithm == "td3":
            if self.action_space != "analog":
                raise ConfigError("td3 drives the analog action space")
            if min(self.td3_replay_capacity, self.td3_batch_size, self.td3_updates_per_step,
                   self.td3_policy_delay, self.td3_warmup_steps, self.td3_critic_width) < 1 or self.td3_bins < 2:
                raise ConfigError("td3_* sizes must be positive (td3_bins >= 2)")
            if not 0.0 < self.td3_tau <= 1.0 or self.td3_exploration_std < 0.0:
                raise ConfigError("td3_tau in (0, 1], td3_exploration_std >= 0")
            if not 0.0 <= self.td3_pedal_epsilon <= 1.0:
                raise ConfigError("td3_pedal_epsilon must be in [0, 1]")
            if self.td3_replay_capacity < self.num_envs:
                raise ConfigError("td3_replay_capacity must hold one step of every environment")
        if self.novelty_coef < 0.0:
            raise ConfigError("novelty_coef must be >= 0")
        if self.pedal_hold_decisions < 1:
            raise ConfigError("pedal_hold_decisions must be >= 1")
        if self.pedal_hold_decisions > 1 and self.action_space != "analog":
            raise ConfigError("pedal_hold_decisions requires the analog action space")
        if (self.entropy_boost > 0.0) != bool(self.entropy_boost_window):
            raise ConfigError("entropy_boost and entropy_boost_window go together")
        if self.entropy_boost < 0.0:
            raise ConfigError("entropy_boost must be >= 0")
        if self.entropy_boost_window:
            low, high = (float(part) for part in self.entropy_boost_window.split(","))
            if not low < high:
                raise ConfigError("entropy_boost_window must be 'low,high' with low < high")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ConfigError("ema_decay must be in [0, 1)")
        if self.lr_decay_updates < 0:
            raise ConfigError("lr_decay_updates must be >= 0")
        if (
            self.stagger_max_seconds < 0.0
            or self.eval_interval_minutes <= 0.0
            or self.eval_num_envs <= 0
            or self.eval_episodes <= 0
        ):
            raise ConfigError("stagger and evaluation settings must be positive")
        if self.checkpoint_interval_minutes <= 0.0:
            raise ConfigError("checkpoint_interval_minutes must be positive")
        if self.stuck_progress_epsilon <= 0.0:
            raise ConfigError("stuck_progress_epsilon must be positive")
        if self.max_race_ticks < 0 or self.horizon_ticks < 0:
            raise ConfigError("max_race_ticks and horizon_ticks cannot be negative; 0 derives per track")
        if not math.isfinite(self.budget_reference_speed) or self.budget_reference_speed < 0:
            raise ConfigError("budget_reference_speed must be zero or finite and positive")
        if self.run_id and ("/" in self.run_id or self.run_id.startswith(".")):
            raise ConfigError("run_id must be a plain directory name")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_type(name: str, value: Any, expected: type) -> Any:
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"config key {name!r} must be a boolean")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config key {name!r} must be an integer")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config key {name!r} must be a number")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"config key {name!r} must be a string")
        return value
    raise ConfigError(f"unsupported config field type for {name!r}")


def config_from_dict(values: dict[str, Any]) -> TrainConfig:
    """Build a TrainConfig; unknown keys and wrong types are fatal."""
    known = {field.name: field.type for field in fields(TrainConfig)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(
            f"unknown config keys {unknown}; known keys: {sorted(known)}"
        )
    resolved: dict[str, Any] = {}
    for name, value in values.items():
        expected = {"int": int, "float": float, "bool": bool, "str": str}[
            known[name] if isinstance(known[name], str) else known[name].__name__
        ]
        resolved[name] = _check_type(name, value, expected)
    config = TrainConfig(**resolved)
    config.validate()
    return config


def load_config_file(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"config file {path} must contain a JSON object")
    return payload


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="TMNF PPO trainer. Every run is registered under runs_root.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON file with TrainConfig keys; CLI flags override it",
    )
    parser.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="continue an existing run from its latest checkpoint",
    )
    parser.add_argument(
        "--ignore-code-hash", action="store_true",
        help="with --resume: continue although python/tmnf_rl changed since the run "
        "started (recorded under run.json code_changes; the run is then not "
        "reproducible from its top-level code_sha256, F25)",
    )
    for field in fields(TrainConfig):
        option = f"--{field.name.replace('_', '-')}"
        if field.type in ("bool", bool):
            parser.add_argument(
                option,
                dest=field.name,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
            )
        elif field.name == "action_space":
            parser.add_argument(
                option, dest=field.name, choices=("discrete", "analog"),
                default=argparse.SUPPRESS,
            )
        elif field.name == "arch":
            parser.add_argument(
                option, dest=field.name, choices=("mlp", "transformer_s"),
                default=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(
                option,
                dest=field.name,
                type={"int": int, "float": float, "str": str}[
                    field.type if isinstance(field.type, str) else field.type.__name__
                ],
                default=argparse.SUPPRESS,
            )
    return parser


@dataclass(frozen=True)
class TrainArgs:
    """Parsed trainer command line.

    With ``--resume`` ``config`` is None and ``overrides`` (config file plus
    explicit flags) is merged by the caller against the run's stored args.
    Without it ``overrides`` holds only the explicit flags.
    """

    config: TrainConfig | None
    overrides: dict[str, Any]
    resume: str | None
    ignore_code_hash: bool


def parse_train_args(argv: list[str] | None = None, prog: str | None = None) -> TrainArgs:
    parser = build_parser(prog)
    namespace = parser.parse_args(argv)
    overrides = {
        name: value
        for name, value in vars(namespace).items()
        if name not in ("config", "resume", "ignore_code_hash")
    }
    if namespace.ignore_code_hash and namespace.resume is None:
        parser.error("--ignore-code-hash only applies with --resume")
    values: dict[str, Any] = {}
    if namespace.config is not None:
        values.update(load_config_file(namespace.config))
    values.update(overrides)
    if namespace.resume is not None:
        return TrainArgs(None, values, namespace.resume, namespace.ignore_code_hash)
    return TrainArgs(config_from_dict(values), overrides, None, False)


__all__ = [
    "ConfigError",
    "TrainConfig",
    "build_parser",
    "config_from_dict",
    "load_config_file",
    "TrainArgs",
    "parse_train_args",
]
