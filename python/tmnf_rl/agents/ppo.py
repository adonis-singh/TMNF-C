"""CleanRL-style PPO for the native TMNF vector environment.

The rollout/update loop is the reference learner from the single-file trainer.
RNG consumption order is part of its contract: fixed-seed metric rows are
compared bit-for-bit by ``tmnf_rl.reproduce`` and by the refactor guard.
Anything added to the loop must not draw from ``torch``'s CUDA generator, the
global NumPy generator or ``trainer_rng`` unless the old code did.
"""

from __future__ import annotations

from tmnf_rl.encoder import observation_width

import csv
import hashlib
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli, Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from tmnf_rl.advantages import AdvantageEstimator
from tmnf_rl.config import TrainConfig
from tmnf_rl.policy_artifacts import retain_policy
from tmnf_rl.snapshot_starts import (
    OFFICIAL_START,
    REFRESH_UPDATES,
    SNAPSHOT_START,
    PoolState,
    SnapshotPool,
    Trajectory,
    trajectory_score,
)
from tmnf_rl.encoder import (
    CRITIC_FEATURES,
    flat_features,
    material_classes,
    GEAR_CLASSES,
    GEO_FEATURES,
    GEOMETRY_SLOTS,
    TURBO_FEATURES,
    VEHICLE_FEATURES,
    WHEEL_FEATURES,
    WHEEL_SLOTS,
    FlatEncoder,
    critic_context,
    encode,
    flatten,
)
from tmnf_rl.cuda_env import TmnfCudaVectorEnv
from tmnf_rl.env import FINISH_REASON, POLICY_TRANSITION_WIDTH, TmnfVectorEnv
from tmnf_rl.exploration import (
    LandingNovelty, PedalHold, ema_update, entropy_weights, finish_time_rewards, parse_window,
)

# Metres per second used to convert a finish's unused tick budget into route
# metres when ranking trajectories for the snapshot pool.
REFERENCE_SPEED = 50.0
from tmnf_rl.evaluation import EvaluationResult, evaluate_full_start
from tmnf_rl.spaces import (
    OBSERVATION_WIDTH,
    device_environment_action,
    environment_action,
    forward_action,
)

TrainingEnv = TmnfVectorEnv | TmnfCudaVectorEnv
from tmnf_rl.utils import write_json_atomic


CHECKPOINT_FORMAT = "tmnf-rl-ppo-checkpoint"
CHECKPOINT_VERSION = 2
# Stopping criteria and the spectate port may change when resuming; nothing
# that affects the learner or the environment may.
RESUME_MUTABLE_KEYS = frozenset({"duration_minutes", "max_updates", "spectate"})
# Floor of the linear learning-rate decay (config.lr_decay_updates).
LR_DECAY_FLOOR = 0.1

METRIC_NAMES = [
    "update",
    "wall_time_s",
    "decision_steps",
    "physics_steps",
    "train/decision_steps_per_second",
    "train/physics_steps_per_second",
    "train/fullstart_episodes",
    "train/snapshot_episodes",
    "train/fullstart_episode_return_mean_100",
    "train/fullstart_distance_mean_100",
    "train/fullstart_distance_p90_100",
    "train/fullstart_best_distance",
    "train/fullstart_checkpoints_mean_100",
    "train/fullstart_best_checkpoints",
    "train/fullstart_finishes",
    "train/fullstart_best_lap_ms",
    "train/snapshot_pool_states",
    "train/snapshot_pool_candidates",
    "train/snapshot_pool_refreshes",
    "train/snapshot_pool_max_progress",
    "train/snapshot_pool_best_score",
    "train/snapshot_starts",
    "train/snapshot_capture_calls",
    "train/snapshot_captured_states",
    "train/snapshot_restore_calls",
    "train/snapshot_restored_states",
    "train/snapshot_io_seconds",
    "train/novelty_cells",
    "train/novelty_landings",
    "train/novelty_bonus_total",
    "train/learning_rate",
    "train/policy_loss",
    "train/value_loss",
    "train/entropy",
    "train/approx_kl",
    "train/explained_variance",
    "train/update_epochs_run",
    "train/environment_seconds",
    "train/rollout_other_seconds",
    "train/optimization_seconds",
]
EVALUATION_NAMES = [
    "scheduled_minutes",
    "wall_time_s",
    "physics_steps",
    # Greedy trajectory from the official start (one trajectory, F21/F29).
    "eval_fullstart/episodes",
    "eval_fullstart/finished",
    "eval_fullstart/distance_mean",
    "eval_fullstart/best_distance",
    "eval_fullstart/median_lap_ms",
    "eval_fullstart/best_lap_ms",
    "eval_fullstart/physics_steps",
    "eval_fullstart/decision_steps",
    "eval_fullstart/evaluation_seconds",
    "eval_fullstart/physics_steps_per_second",
    "termination_reason_counts",
    # Sampled policy from the official start (eval_episodes episodes, fixed
    # seed): the finish rate that means something.
    "eval_sampled/episodes",
    "eval_sampled/finishes",
    "eval_sampled/finish_rate",
    "eval_sampled/distance_mean",
    "eval_sampled/distance_p90",
    "eval_sampled/best_distance",
    "eval_sampled/median_lap_ms",
    "eval_sampled/best_lap_ms",
    "eval_sampled/physics_steps",
    "eval_sampled/decision_steps",
    "eval_sampled/evaluation_seconds",
    "eval_sampled/physics_steps_per_second",
    "eval_sampled/termination_reason_counts",
]
# Wall-clock dependent columns. Everything else must be bit-identical between
# two runs of the same config, seed and physics library.
TIMING_COLUMNS = frozenset(
    {
        "wall_time_s",
        "train/decision_steps_per_second",
        "train/physics_steps_per_second",
        "train/snapshot_io_seconds",
        "train/environment_seconds",
        "train/rollout_other_seconds",
        "train/optimization_seconds",
    }
)


def layer_init(
    layer: nn.Linear, standard_deviation: float = np.sqrt(2), bias: float = 0.0
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, standard_deviation)
    nn.init.constant_(layer.bias, bias)
    return layer


ARCHITECTURES = ("mlp", "transformer_s")
ACTION_COUNT = 12


class MlpTrunk(nn.Module):
    """The pilot winner (`review_nn_mlpfixed_s1`): a tanh MLP, depth 2, width
    ``hidden_size``, on the flattened fixed-scale features; the policy and
    value heads share it."""

    def __init__(self, hidden_size: int, encoder_version: int = 1) -> None:
        super().__init__()
        self.encoder_version = encoder_version
        self.width = hidden_size
        self.network = nn.Sequential(
            layer_init(nn.Linear(flat_features(encoder_version), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.network(flatten(features, self.encoder_version))


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TransformerBlock(nn.Module):
    """Pre-norm residual block: RMSNorm, multi-head attention, RMSNorm, GELU
    MLP with 4x expansion, no biases, GPT-2 depth-scaled output projections."""

    def __init__(self, width: int, heads: int, depth: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = RMSNorm(width)
        self.fc1 = nn.Linear(width, ffn_mult * width, bias=False)
        self.fc2 = nn.Linear(ffn_mult * width, width, bias=False)
        self.heads = heads
        for module in (self.qkv, self.fc1):
            nn.init.normal_(module.weight, std=0.02)
        for module in (self.proj, self.fc2):
            nn.init.normal_(module.weight, std=0.02 / np.sqrt(2 * depth))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(c, dim=2)
        q = q.view(b, t, self.heads, c // self.heads).transpose(1, 2)
        k = k.view(b, t, self.heads, c // self.heads).transpose(1, 2)
        v = v.view(b, t, self.heads, c // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


TRANSFORMER_S = {"width": 128, "depth": 4, "heads": 4}
TOKEN_TYPES = 5  # geometry, wheel, vehicle, turbo, cls


class TransformerTrunk(nn.Module):
    """The pilot ladder's size s (0.8M parameters) with K=0 history tokens:
    8 geometry + 4 wheel + 1 vehicle + 1 turbo + CLS = 15 tokens, pre-norm
    blocks, the CLS output after a final RMSNorm feeds the heads."""

    def __init__(self, width: int, depth: int, heads: int, encoder_version: int = 1) -> None:
        super().__init__()
        self.width = width
        self.encoder_version = encoder_version
        if encoder_version == 3:
            self.gate_in = nn.Linear(19, width)
            self.gate_counts_in = nn.Linear(4, width)
        self.geo_in = nn.Linear(GEO_FEATURES, width)
        self.wheel_in = nn.Linear(WHEEL_FEATURES, width)
        self.material_emb = nn.Embedding(material_classes(encoder_version), width)
        self.vehicle_in = nn.Linear(VEHICLE_FEATURES, width)
        self.gear_emb = nn.Embedding(GEAR_CLASSES, width)
        self.turbo_in = nn.Linear(TURBO_FEATURES, width)
        self.type_emb = nn.Embedding(TOKEN_TYPES + (1 if encoder_version == 3 else 0), width)
        self.geo_pos = nn.Parameter(torch.zeros(GEOMETRY_SLOTS, width))
        self.wheel_pos = nn.Parameter(torch.zeros(WHEEL_SLOTS, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, heads, depth) for _ in range(depth)]
        )
        self.norm_out = RMSNorm(width)
        for parameter in (self.geo_pos, self.wheel_pos, self.cls):
            nn.init.normal_(parameter, std=0.02)
        for embedding in (self.material_emb, self.gear_emb, self.type_emb):
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        n = features["vehicle"].shape[0]
        types = self.type_emb.weight
        geo = self.geo_in(features["geo"]) + self.geo_pos + types[0]
        wheels = (
            self.wheel_in(features["wheels"])
            + self.material_emb(features["material"])
            + self.wheel_pos
            + types[1]
        )
        vehicle = (
            self.vehicle_in(features["vehicle"]) + self.gear_emb(features["gear"]) + types[2]
        ).unsqueeze(1)
        turbo = (self.turbo_in(features["turbo"]) + types[3]).unsqueeze(1)
        cls = self.cls.expand(n, 1, -1) + types[4]
        x = torch.cat((cls, vehicle, turbo, wheels, geo), dim=1)
        mask = None
        if self.encoder_version == 3:
            gate_tokens = self.gate_in(features["gates"]) + types[5]
            cls_counts = self.gate_counts_in(features["gate_counts"])
            x = torch.cat((x[:, :1] + cls_counts.unsqueeze(1), x[:, 1:], gate_tokens), dim=1)
            mask = torch.cat((torch.ones((n, 15), dtype=torch.bool, device=x.device),
                              features["gate_mask"]), dim=1)[:, None, None, :]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm_out(x[:, 0])


def build_trunk(arch: str, hidden_size: int, encoder_version: int = 1) -> nn.Module:
    if arch == "mlp":
        return MlpTrunk(hidden_size, encoder_version)
    if arch == "transformer_s":
        return TransformerTrunk(**TRANSFORMER_S, encoder_version=encoder_version)
    raise ValueError(f"arch must be one of {ARCHITECTURES}, got {arch!r}")


class Agent(nn.Module):
    """Policy and value on the fixed-scale encoder features.

    ``arch`` picks the trunk (`mlp`: width ``hidden_size``; `transformer_s`:
    the fixed 0.8M-parameter tokenized transformer, run through
    ``torch.compile``). Both consume the raw 81-float observation and encode
    it themselves, so there is exactly one input path and nothing to warm up,
    freeze or checkpoint. The value head additionally reads the remaining
    distance (`encoder.critic_context`); the policy head does not.
    """

    def __init__(self, arch: str, hidden_size: int, action_space: str = "discrete", *, cuda_graphs: bool = True, encoder_version: int = 1) -> None:
        super().__init__()
        if action_space not in {"discrete", "analog"}:
            raise ValueError("action_space must be discrete or analog")
        self.arch = arch
        self.action_space = action_space
        self.encoder_version = encoder_version
        self._flat_encoder = FlatEncoder(cuda_graphs=cuda_graphs, version=encoder_version)
        self.trunk = build_trunk(arch, hidden_size, encoder_version)
        width = self.trunk.width
        # torch.compile keeps the parameters on the plain module (state_dict
        # keys stay unprefixed); the compiled callable is not a submodule.
        self._trunk_forward = (
            torch.compile(self.trunk) if arch == "transformer_s" else self.trunk
        )
        if self.action_space == "discrete":
            self.actor = layer_init(nn.Linear(width, ACTION_COUNT), 0.01)
        else:
            self.steer_actor = layer_init(nn.Linear(width, 1), 0.01)
            self.steer_log_std = nn.Parameter(torch.zeros(1))
            self.binary_actor = layer_init(nn.Linear(width, 2), 0.01)
        self.critic = layer_init(nn.Linear(width + CRITIC_FEATURES, 1), 1.0)

    def hidden(self, observation: torch.Tensor) -> torch.Tensor:
        if self.arch == "mlp":
            return self.trunk.network(self._flat_encoder(observation))
        return self._trunk_forward(encode(observation, self.encoder_version))

    def value(self, hidden: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat((hidden, critic_context(observation)), dim=1)).squeeze(-1)

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(self.hidden(observation), observation)

    def get_deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(observation)
        if self.action_space == "discrete":
            return torch.argmax(self.actor(hidden), dim=1)
        # Analog actions are (pre-tanh steer, gas, brake); environment_action
        # applies the tanh. The mean of the Gaussian is the greedy steer.
        binary = (self.binary_actor(hidden) >= 0.0).to(torch.float32)
        return torch.cat((self.steer_actor(hidden), binary), dim=1)

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        pedal_mask: torch.Tensor | None = None,
        *,
        encoded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """pedal_mask (analog, tmnf_rl.exploration.PedalHold): rows where it is
        False carry held pedals; their Bernoulli log-probability and entropy
        are excluded, the steer term stays."""
        if encoded is not None:
            if self.arch != "mlp":
                raise ValueError("pre-encoded observations require the MLP trunk")
            hidden = self.trunk.network(encoded)
        else:
            hidden = self.hidden(observation)
        if self.action_space == "analog":
            return self._get_analog_action_and_value(hidden, observation, action, pedal_mask)
        if pedal_mask is not None:
            raise ValueError("pedal_mask is defined for the analog head")
        # validate_args reads the tensors back to the host (a stream sync per
        # call); the logits are finite by construction.
        distribution = Categorical(logits=self.actor(hidden), validate_args=False)
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.value(hidden, observation),
        )

    def _get_analog_action_and_value(
        self,
        hidden: torch.Tensor,
        observation: torch.Tensor,
        action: torch.Tensor | None,
        pedal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.steer_actor(hidden).squeeze(-1)
        steer_distribution = Normal(mean, self.steer_log_std.exp(), validate_args=False)
        binary_distribution = Bernoulli(logits=self.binary_actor(hidden), validate_args=False)
        # The stored action keeps the pre-tanh steer sample: recovering it
        # from the squashed value with atanh(clamp(.)) loses the tail beyond
        # |steer| = 1 - eps and shifts the update's log-probability against
        # the rollout's. environment_action applies the tanh for the env.
        if action is None:
            pre_tanh_steer = steer_distribution.sample()
            binary = binary_distribution.sample()
            action = torch.cat((pre_tanh_steer.unsqueeze(1), binary), dim=1)
        else:
            pre_tanh_steer = action[:, 0]
            binary = action[:, 1:]
        log_two = 0.6931471805599453
        steer_log_probability = (
            steer_distribution.log_prob(pre_tanh_steer)
            - 2.0
            * (
                log_two
                - pre_tanh_steer
                - F.softplus(-2.0 * pre_tanh_steer)
            )
        )
        binary_log_probability = binary_distribution.log_prob(binary).sum(dim=1)
        binary_entropy = binary_distribution.entropy().sum(dim=1)
        if pedal_mask is not None:
            decided = pedal_mask.to(binary_log_probability.dtype)
            binary_log_probability = binary_log_probability * decided
            binary_entropy = binary_entropy * decided
        log_probability = steer_log_probability + binary_log_probability
        entropy = -steer_log_probability + binary_entropy
        return (
            action,
            log_probability,
            entropy,
            self.value(hidden, observation),
        )


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())



def optional_mean(values: deque[float], count: int = 100) -> float | None:
    if not values:
        return None
    return float(np.mean(list(values)[-count:]))


def optional_percentile(
    values: deque[float], percentile: float, count: int = 100
) -> float | None:
    if not values:
        return None
    return float(np.percentile(list(values)[-count:], percentile))


def stagger_initial_phases(
    env: TmnfVectorEnv,
    rng: np.random.Generator,
    maximum_seconds: float,
) -> dict[str, Any]:
    maximum_decisions = int(np.rint(maximum_seconds * 100.0 / env.action_repeat))
    if maximum_decisions == 0:
        return {
            "enabled": False,
            "maximum_ticks": 0,
            "mean_ticks": 0.0,
            "restore_calls": 0,
            "restored_states": 0,
        }

    start_snapshots = env.capture()
    ages = rng.integers(0, maximum_decisions + 1, size=env.num_envs)
    actions = forward_action(env)
    restore_calls = 0
    restored_states = 0
    for completed in range(maximum_decisions):
        _, _, terminated, truncated, _ = env.step(actions)
        if np.any(terminated | truncated):
            raise RuntimeError("stagger warmup ended an environment")
        remaining = maximum_decisions - completed - 1
        indices = np.flatnonzero(ages == remaining)
        if indices.size:
            env.restore(
                indices,
                [start_snapshots[int(index)] for index in indices],
            )
            restore_calls += 1
            restored_states += int(indices.size)
    return {
        "enabled": True,
        "maximum_ticks": maximum_decisions * env.action_repeat,
        "mean_ticks": float(ages.mean() * env.action_repeat),
        "restore_calls": restore_calls,
        "restored_states": restored_states,
    }


class TrainerSink(Protocol):
    """Side channel for registry, spectate and replay export. No RNG use."""

    spectate_buffer: Any

    def on_start(self, event: dict[str, Any]) -> None: ...

    def on_update(self, row: dict[str, Any], summary: dict[str, Any]) -> None: ...

    def on_evaluation(
        self, evaluation: dict[str, Any], result: EvaluationResult, sampled: EvaluationResult,
        policy: dict[str, Any] | None = None,
    ) -> None: ...

    def on_checkpoint(self, path: Path, update: int) -> None: ...


class NullSink:
    spectate_buffer = None

    def on_start(self, event: dict[str, Any]) -> None:
        del event

    def on_update(self, row: dict[str, Any], summary: dict[str, Any]) -> None:
        del row, summary

    def on_evaluation(
        self, evaluation: dict[str, Any], result: EvaluationResult, sampled: EvaluationResult,
        policy: dict[str, Any] | None = None,
    ) -> None:
        del evaluation, result, sampled, policy

    def on_checkpoint(self, path: Path, update: int) -> None:
        del path, update


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    trainer_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    return trainer_rng


# cuBLAS reads this at handle creation; PyTorch requires it (or :16:8) when
# deterministic algorithms are on. :4096:8 is the documented choice.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# Environment variables that silently change CUDA float32 numerics. A trainer
# refuses to start when one of them would override what pin_cuda_numerics sets
# (F24): the same seed then produces different metrics with no record of why.
NUMERICS_OVERRIDE_VARIABLES = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")


def pin_cuda_numerics() -> None:
    """Pin every float32 CUDA numerics switch before the CUDA context exists.

    TF32 in cuBLAS/cuDNN, non-deterministic kernels and cuBLAS workspace size
    each changed `train/policy_loss` at update 1 in the review's reproduction
    (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`: 19 cells in 4 updates;
    `CUBLAS_WORKSPACE_CONFIG=:16:8`: 14). Called from `select_device`, which
    every CUDA entry point (train, evaluate, reproduce) goes through first.
    """
    for name in NUMERICS_OVERRIDE_VARIABLES:
        if name in os.environ:
            raise RuntimeError(
                f"{name}={os.environ[name]!r} is set; it overrides the pinned CUDA "
                "numerics and would make this run irreproducible. Unset it."
            )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is not None and workspace != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG={workspace!r} is set; runs are pinned to "
            f"{CUBLAS_WORKSPACE_CONFIG!r}. Unset it."
        )
    if torch.cuda.is_initialized() and workspace is None:
        raise RuntimeError(
            "CUDA was initialised before pin_cuda_numerics; CUBLAS_WORKSPACE_CONFIG "
            "cannot be applied any more"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def numerics_state() -> dict[str, Any]:
    """The switches pin_cuda_numerics controls, as recorded in provenance."""
    return {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        **{name: os.environ.get(name) for name in NUMERICS_OVERRIDE_VARIABLES},
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def select_device() -> tuple[torch.device, str]:
    """Pin the numerics and take the one visible GPU.

    Which physical GPU that is comes from ``CUDA_VISIBLE_DEVICES``
    (``require_single_gpu_env``); its name is recorded in provenance and
    compared by ``reproduce`` (F24: the same seed on another GPU model gives
    different last bits).
    """
    pin_cuda_numerics()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA visibility must contain exactly one GPU")
    device = torch.device("cuda:0")
    return device, torch.cuda.get_device_name(device)


def make_env(config: TrainConfig, num_envs: int, physics_library: Path) -> TmnfVectorEnv:
    return TmnfVectorEnv(
        num_envs,
        track=config.track,
        gate_observations=config.encoder_version == 3,
        thread_count=min(config.thread_count, num_envs),
        action_repeat=config.action_repeat,
        action_space=config.action_space,
        max_race_ticks=config.max_race_ticks,
        horizon_ticks=config.horizon_ticks,
        budget_reference_speed=config.budget_reference_speed,
        off_track_grace_ticks=config.off_track_grace_ticks,
        stuck_grace_ticks=config.stuck_grace_ticks,
        stuck_progress_epsilon=config.stuck_progress_epsilon,
        discount_per_tick=config.discount_per_tick,
        library_path=physics_library,
    )


def make_train_env(
    config: TrainConfig, physics_library: Path, eval_env: TmnfVectorEnv, device: torch.device
) -> TrainingEnv:
    """The training environments per ``config.env_device``. The CUDA env takes
    the race budgets the CPU evaluation env resolved for the track."""
    if config.env_device == "cpu":
        return make_env(config, config.num_envs, physics_library)
    return TmnfCudaVectorEnv(
        config.num_envs,
        track=config.track,
        gate_observations=config.encoder_version == 3,
        action_repeat=config.action_repeat,
        action_space=config.action_space,
        max_race_ticks=eval_env.max_race_ticks,
        horizon_ticks=eval_env.horizon_ticks,
        off_track_grace_ticks=config.off_track_grace_ticks,
        stuck_grace_ticks=config.stuck_grace_ticks,
        stuck_progress_epsilon=config.stuck_progress_epsilon,
        discount_per_tick=config.discount_per_tick,
        library_path=physics_library,
        device=device,
    )


def rng_states(trainer_rng: np.random.Generator, device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
        "trainer_rng": trainer_rng.bit_generator.state,
    }


def restore_rng_states(
    states: dict[str, Any], trainer_rng: np.random.Generator, device: torch.device
) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy_global"])
    torch.set_rng_state(states["torch_cpu"].cpu())
    torch.cuda.set_rng_state(states["torch_cuda"].cpu(), device)
    trainer_rng.bit_generator.state = states["trainer_rng"]


class PPOTrainer:
    # Columns of metrics.csv; the off-policy trainer (agents/td3.py) swaps
    # the loss columns.
    metric_names = METRIC_NAMES

    """Owns the loop state so checkpoints can capture and restore it."""

    def __init__(
        self,
        config: TrainConfig,
        *,
        device: torch.device,
        gpu_name: str,
        env: TrainingEnv,
        eval_env: TmnfVectorEnv,
        run_dir: Path,
        physics_sha256: str,
        code_sha256: str,
        run_id: str,
        sink: TrainerSink,
        trainer_rng: np.random.Generator,
        script_start: float,
    ) -> None:
        self.config = config
        self.device = device
        self.gpu_name = gpu_name
        self.env = env
        if eval_env is env:
            raise ValueError(
                "the evaluation environment must be a separate instance: training "
                "restores snapshot starts into its environments and a full-start lap "
                "may only come from a never-restored one (F9)"
            )
        self.eval_env = eval_env
        # CUDA training env: the policy reads the env's device buffers and
        # writes device actions; only the episode bookkeeping reaches the host.
        self.device_env = isinstance(env, TmnfCudaVectorEnv)
        if self.device_env != (config.env_device == "cuda"):
            raise ValueError("env_device does not match the training environment's type")
        self.run_dir = Path(run_dir)
        self.physics_sha256 = physics_sha256
        self.code_sha256 = code_sha256
        self.run_id = run_id
        self.sink = sink
        self.trainer_rng = trainer_rng
        self.script_start = script_start
        args = config

        self.agent = Agent(args.arch, args.hidden_size, args.action_space, encoder_version=config.encoder_version).to(device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=args.learning_rate, eps=1e-5
        )
        # Exploration arms (tmnf_rl.exploration), each None when off.
        self.novelty = LandingNovelty(args.num_envs, args.novelty_coef) if args.novelty_coef > 0.0 else None
        self.pedal_hold = (
            PedalHold(args.num_envs, args.pedal_hold_decisions, device)
            if args.pedal_hold_decisions > 1 else None
        )
        self.entropy_window = parse_window(args.entropy_boost_window) if args.entropy_boost_window else None
        self.ema_agent: Agent | None = None
        if args.ema_decay > 0.0:
            self.ema_agent = Agent(args.arch, args.hidden_size, args.action_space, encoder_version=config.encoder_version).to(device)
            self.ema_agent.load_state_dict(self.agent.state_dict())
            self.ema_agent.eval()
            for parameter in self.ema_agent.parameters():
                parameter.requires_grad_(False)
        self.learning_rate = args.learning_rate
        # (e) Finish-time-only reward: the update at which the first training
        # finish switched the reward stream (0 = shaped reward still active).
        self.finish_time_switch_update = 0
        # Snapshot starts (snapshot_start_fraction > 0): the pool of route
        # states and, per environment, the captures of the episode in flight.
        self.pool = SnapshotPool(
            route_length=env.route_length * env.lap_count, rng=trainer_rng,
            prefailure_exclusion_ticks=config.snapshot_prefailure_exclusion_ticks,
        )
        self.inflight: list[Trajectory] = [Trajectory() for _ in range(args.num_envs)]
        self.snapshot_starts = 0

        self.batch_size = args.num_envs * args.num_steps
        self.minibatch_size = self.batch_size // args.num_minibatches
        self.observations = torch.zeros(
            (args.num_steps, args.num_envs, observation_width(args.encoder_version)),
            dtype=torch.float32,
            device=device,
        )
        self.actions = torch.zeros(
            (
                (args.num_steps, args.num_envs)
                if args.action_space == "discrete"
                else (args.num_steps, args.num_envs, 3)
            ),
            dtype=torch.long if args.action_space == "discrete" else torch.float32,
            device=device,
        )
        self.log_probabilities = torch.zeros(
            (args.num_steps, args.num_envs), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros_like(self.log_probabilities)
        self.pedal_masks = torch.ones(
            (args.num_steps, args.num_envs), dtype=torch.bool, device=device
        )
        self.discounts = torch.zeros_like(self.log_probabilities)
        self.terminations = torch.zeros_like(self.log_probabilities, dtype=torch.bool)
        self.truncations = torch.zeros_like(self.terminations)
        self.truncation_values = torch.zeros_like(self.log_probabilities)
        self.values = torch.zeros_like(self.log_probabilities)
        # CPU env: host staging for the two per-step uploads. A copy from
        # pageable NumPy memory synchronises the stream; from pinned memory it
        # is asynchronous, and the action download that follows every forward
        # pass (`environment_action`) orders the next host write behind it.
        # The CUDA env has no uploads: its buffers are the tensors.
        if not self.device_env:
            self.pinned_transitions = torch.empty(
                (args.num_envs, POLICY_TRANSITION_WIDTH), dtype=torch.float32, pin_memory=True
            )
            self.pinned_observations = torch.empty(
                (args.num_envs, observation_width(args.encoder_version)), dtype=torch.float32, pin_memory=True
            )

        self.fullstart_returns: deque[float] = deque(maxlen=10_000)
        self.fullstart_distances: deque[float] = deque(maxlen=10_000)
        self.fullstart_checkpoints: deque[float] = deque(maxlen=10_000)
        self.fullstart_finishes = 0
        self.fullstart_best_lap_ms = 0
        self.fullstart_best_distance = 0.0
        self.fullstart_best_checkpoints = 0.0
        self.fullstart_episodes = 0
        self.snapshot_episodes = 0
        self.total_decision_steps = 0
        self.total_physics_steps = 0
        self.cumulative_env_seconds = 0.0
        self.cumulative_rollout_seconds = 0.0
        self.cumulative_optimization_seconds = 0.0
        self.snapshot_capture_calls = 0
        self.snapshot_captured_states = 0
        self.snapshot_restore_calls = 0
        self.snapshot_restored_states = 0
        self.snapshot_io_seconds = 0.0
        # Chained digest of every snapshot-start choice (checkpointed, so a
        # resumed run ends with the same digest as the uninterrupted one).
        self.reset_choice_sha256 = hashlib.sha256().hexdigest()
        self.start_kinds = np.full(args.num_envs, OFFICIAL_START, dtype=object)
        self.episode_max_progress = np.zeros(args.num_envs, dtype=np.float32)
        self.last_capture_ticks = np.zeros(args.num_envs, dtype=np.int32)
        self.capture_episodes = np.empty(args.num_envs, dtype=np.bool_)
        self.last_losses = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "update_epochs_run": 0,
            "explained_variance": 0.0,
        }
        self._advantage_estimator = AdvantageEstimator()
        self.update = 0
        self.stagger_info: dict[str, Any] = {}
        self.setup_seconds = 0.0
        self.run_start = 0.0
        self.next_evaluation_seconds = args.eval_interval_minutes * 60.0
        self.next_checkpoint_seconds = args.checkpoint_interval_minutes * 60.0
        self.latest_evaluation: dict[str, Any] | None = None
        self.best_eval_lap_ms: int | None = None
        self.resumed_from: dict[str, Any] | None = None
        self.current_raw_observation: torch.Tensor | None = None

        self.metrics_path = self.run_dir / "metrics.csv"
        self.evaluations_path = self.run_dir / "evaluations.csv"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.tensorboard: SummaryWriter | None = None
        self.metrics_file: Any = None
        self.metric_writer: csv.DictWriter[str] | None = None
        self.evaluations_file: Any = None
        self.evaluation_writer: csv.DictWriter[str] | None = None

    # ------------------------------------------------------------------ setup

    def _open_outputs(self, append: bool) -> None:
        self.tensorboard = SummaryWriter(self.run_dir / "tensorboard")
        mode = "a" if append else "w"
        self.metrics_file = self.metrics_path.open(mode, newline="", encoding="utf-8")
        self.metric_writer = csv.DictWriter(self.metrics_file, fieldnames=self.metric_names)
        self.evaluations_file = self.evaluations_path.open(
            mode, newline="", encoding="utf-8"
        )
        self.evaluation_writer = csv.DictWriter(
            self.evaluations_file, fieldnames=EVALUATION_NAMES
        )
        if not append:
            self.metric_writer.writeheader()
            self.evaluation_writer.writeheader()

    def setup_fresh(self) -> None:
        args = self.config
        self._open_outputs(append=False)
        self.env.reset()
        self.stagger_info = (
            stagger_initial_phases(self.env, self.trainer_rng, args.stagger_max_seconds)
            if args.staggered_phases
            else {
                "enabled": False,
                "maximum_ticks": 0,
                "mean_ticks": 0.0,
                "restore_calls": 0,
                "restored_states": 0,
            }
        )
        self.snapshot_restore_calls += int(self.stagger_info["restore_calls"])
        self.snapshot_restored_states += int(self.stagger_info["restored_states"])
        self.capture_episodes[:] = (
            self.trainer_rng.random(args.num_envs)
            < args.snapshot_capture_episode_fraction
        )
        self.current_raw_observation = self._observation_batch()
        self.episode_max_progress[:] = self.env.observations["race"][:, 4]
        self.last_capture_ticks[:] = np.rint(
            self.env.observations["race"][:, 8] * self.env.max_race_ticks
        ).astype(np.int32)
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - self.script_start
        self.run_start = time.perf_counter()

    def setup_resume(self, checkpoint_path: Path) -> None:
        args = self.config
        # Load on CPU so RNG state tensors and Adam's `step` scalars keep the
        # device they were saved from; load_state_dict moves parameters and
        # moments to the GPU itself.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError(f"{checkpoint_path} is not a PPO checkpoint")
        if state["version"] != CHECKPOINT_VERSION:
            raise RuntimeError(
                f"checkpoint version {state['version']} != {CHECKPOINT_VERSION}"
            )
        if state["physics_sha256"] != self.physics_sha256:
            raise RuntimeError(
                "physics library changed since the checkpoint: "
                f"{state['physics_sha256']} -> {self.physics_sha256}; refusing "
                "to resume across a different simulator"
            )
        if state["run_id"] != self.run_id:
            raise RuntimeError(
                f"checkpoint belongs to run {state['run_id']!r}, not {self.run_id!r}"
            )
        checkpoint_config = dict(state["config"])
        # Explicit defaults for fields absent from historical checkpoints.
        checkpoint_config.setdefault("encoder_version", 1)
        checkpoint_config.setdefault("snapshot_prefailure_exclusion_ticks", 200)
        checkpoint_config.setdefault("budget_reference_speed", 0.0)
        differences = sorted(
            key
            for key in set(checkpoint_config) | set(args.to_dict())
            if key not in RESUME_MUTABLE_KEYS
            and checkpoint_config.get(key) != args.to_dict().get(key)
        )
        if differences:
            raise RuntimeError(
                f"config differs from the checkpoint on {differences}; only "
                f"{sorted(RESUME_MUTABLE_KEYS)} may change on resume"
            )
        if state["env_snapshot_size"] != self.env.snapshot_size:
            raise RuntimeError("native snapshot size changed since the checkpoint")

        self.agent.load_state_dict(state["agent"])
        self.optimizer.load_state_dict(state["optimizer"])
        counters = state["counters"]
        for name, value in counters.items():
            setattr(self, name, value)
        self.fullstart_returns = deque(state["deques"]["returns"], maxlen=10_000)
        self.fullstart_distances = deque(state["deques"]["distances"], maxlen=10_000)
        self.fullstart_checkpoints = deque(state["deques"]["checkpoints"], maxlen=10_000)
        self.last_losses = dict(state["last_losses"])
        self.start_kinds = np.array(state["arrays"]["start_kinds"], dtype=object)
        self.pool.load_state_dict(state["snapshot_pool"])
        self.inflight = [
            Trajectory(states=[PoolState(bytes(blob), **meta) for blob, meta in states])
            for states in state["inflight"]
        ]
        self.episode_max_progress = np.asarray(state["arrays"]["episode_max_progress"], dtype=np.float32)
        self.last_capture_ticks = np.asarray(state["arrays"]["last_capture_ticks"], dtype=np.int32)
        self.capture_episodes = np.asarray(state["arrays"]["capture_episodes"], dtype=np.bool_)
        self.stagger_info = dict(state["stagger_info"])
        self.latest_evaluation = state["latest_evaluation"]
        self.best_eval_lap_ms = state["best_eval_lap_ms"]
        self.next_evaluation_seconds = float(state["next_evaluation_seconds"])
        self.next_checkpoint_seconds = float(state["next_checkpoint_seconds"])
        exploration = state["exploration"]
        if self.novelty is not None:
            self.novelty.load_state_dict(exploration["novelty"])
        if self.pedal_hold is not None:
            self.pedal_hold.load_state_dict(exploration["pedal_hold"])
        if self.ema_agent is not None:
            self.ema_agent.load_state_dict(exploration["ema_agent"])
        self.learning_rate = float(exploration["learning_rate"])

        self.env.restore(
            np.arange(self.env.num_envs, dtype=np.uint32),
            [bytes(blob) for blob in state["env_snapshots"]],
        )
        self.current_raw_observation = self._observation_batch()
        expected = np.asarray(state["current_raw_observation"], dtype=np.float32)
        if not np.array_equal(self.env.policy_observations, expected):
            raise RuntimeError(
                "restored environments do not reproduce the checkpointed "
                "observation batch"
            )
        restore_rng_states(state["rng"], self.trainer_rng, self.device)
        self._truncate_csv(self.metrics_path, "update", self.update)
        self._truncate_csv(
            self.evaluations_path, "wall_time_s", float(state["elapsed_seconds"])
        )
        self._open_outputs(append=True)
        self.resumed_from = {
            "checkpoint": str(checkpoint_path),
            "update": self.update,
            "elapsed_seconds": float(state["elapsed_seconds"]),
            "physics_sha256": state["physics_sha256"],
            "code_sha256_at_checkpoint": state["code_sha256"],
            "code_sha256_now": self.code_sha256,
        }
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - self.script_start
        self.run_start = time.perf_counter() - float(state["elapsed_seconds"])

    @staticmethod
    def _truncate_csv(path: Path, column: str, limit: float) -> None:
        """Drop rows written after the checkpoint so the CSV stays monotone."""
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        kept = [row for row in rows if float(row[column]) <= limit]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

    # ------------------------------------------------------------- checkpoint

    def save_checkpoint(self, elapsed: float) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "latest.pt"
        temporary = self.checkpoint_dir / "latest.pt.tmp"
        counters = {
            name: getattr(self, name)
            for name in (
                "update",
                "fullstart_finishes",
                "fullstart_best_lap_ms",
                "fullstart_best_distance",
                "fullstart_best_checkpoints",
                "fullstart_episodes",
                "snapshot_episodes",
                "snapshot_starts",
                "reset_choice_sha256",
                "total_decision_steps",
                "total_physics_steps",
                "cumulative_env_seconds",
                "cumulative_rollout_seconds",
                "cumulative_optimization_seconds",
                "snapshot_capture_calls",
                "snapshot_captured_states",
                "snapshot_restore_calls",
                "snapshot_restored_states",
                "snapshot_io_seconds",
                "setup_seconds",
                "finish_time_switch_update",
            )
        }
        assert self.current_raw_observation is not None
        state = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "physics_sha256": self.physics_sha256,
            "code_sha256": self.code_sha256,
            "elapsed_seconds": elapsed,
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_states(self.trainer_rng, self.device),
            "env_snapshots": list(self.env.capture()),
            "env_snapshot_size": self.env.snapshot_size,
            "current_raw_observation": self.env.policy_observations.copy(),
            "counters": counters,
            "deques": {
                "returns": list(self.fullstart_returns),
                "distances": list(self.fullstart_distances),
                "checkpoints": list(self.fullstart_checkpoints),
            },
            "snapshot_pool": self.pool.state_dict(),
            "inflight": [
                [(state.blob, state.meta()) for state in trajectory.states]
                for trajectory in self.inflight
            ],
            "arrays": {
                "start_kinds": list(self.start_kinds),
                "episode_max_progress": self.episode_max_progress.copy(),
                "last_capture_ticks": self.last_capture_ticks.copy(),
                "capture_episodes": self.capture_episodes.copy(),
            },
            "last_losses": dict(self.last_losses),
            "stagger_info": dict(self.stagger_info),
            "latest_evaluation": self.latest_evaluation,
            "best_eval_lap_ms": self.best_eval_lap_ms,
            "next_evaluation_seconds": self.next_evaluation_seconds,
            "next_checkpoint_seconds": self.next_checkpoint_seconds,
            "exploration": {
                "novelty": self.novelty.state_dict() if self.novelty is not None else None,
                "pedal_hold": self.pedal_hold.state_dict() if self.pedal_hold is not None else None,
                "ema_agent": self.ema_agent.state_dict() if self.ema_agent is not None else None,
                "learning_rate": self.learning_rate,
            },
        }
        if not np.array_equal(
            state["current_raw_observation"],
            self.current_raw_observation.cpu().numpy(),
        ):
            raise RuntimeError("checkpoint observation batch is out of sync")
        torch.save(state, temporary)
        temporary.replace(path)
        self.sink.on_checkpoint(path, self.update)
        return path

    # ------------------------------------------------------------------ loop

    def _write_row(self, row: dict[str, Any]) -> None:
        assert self.metric_writer is not None and self.tensorboard is not None
        self.metric_writer.writerow(row)
        self.metrics_file.flush()
        step = int(row["physics_steps"])
        for name, value in row.items():
            if (
                name not in {"update", "wall_time_s", "decision_steps", "physics_steps"}
                and isinstance(value, (int, float))
                and value is not None
            ):
                self.tensorboard.add_scalar(name, value, step)

    def _write_evaluation(self, row: dict[str, Any], physics_step: int) -> None:
        assert self.evaluation_writer is not None and self.tensorboard is not None
        self.evaluation_writer.writerow(row)
        self.evaluations_file.flush()
        for name, value in row.items():
            if name.startswith(("eval_fullstart/", "eval_sampled/")) and isinstance(value, (int, float)):
                self.tensorboard.add_scalar(name, value, physics_step)

    @property
    def policy_agent(self) -> Agent:
        """The weights evaluated and exported: the EMA average when on."""
        return self.ema_agent if self.ema_agent is not None else self.agent

    def _evaluate(self, scheduled_minutes: float) -> dict[str, Any]:
        # Greedy: one round of the eval environments is the whole trajectory
        # set (F29); more episodes would be the same trajectory again.
        greedy = evaluate_full_start(
            self.policy_agent, self.eval_env, self.eval_env.num_envs, self.device,
            pedal_hold_decisions=self.config.pedal_hold_decisions,
        )
        # Sampled: eval_episodes draws from the policy with the run seed; the
        # training RNG streams are restored afterwards.
        sampled = evaluate_full_start(
            self.policy_agent, self.eval_env, self.config.eval_episodes, self.device,
            sampled=True, sample_seed=self.config.seed,
            pedal_hold_decisions=self.config.pedal_hold_decisions,
        )
        evaluation = {
            "scheduled_minutes": scheduled_minutes,
            "wall_time_s": 0.0,
            "physics_steps": self.total_physics_steps,
            **greedy.row(),
            **sampled.row(),
        }
        agent = self.policy_agent
        policy = retain_policy(self.run_dir, {
            "agent": agent.state_dict(),
            "config": self.config.to_dict(),
            "physics_sha256": self.physics_sha256,
            "code_sha256": self.code_sha256,
            "run_id": self.run_id,
            "counters": {"update": self.update},
            "evaluation": {key: value for key, value in evaluation.items()
                           if key != "wall_time_s"},
            "summary": ({"return_support": [agent.v_min, agent.v_max]}
                        if self.config.algorithm == "td3" else {}),
        })
        evaluation["wall_time_s"] = time.perf_counter() - self.run_start
        self._write_evaluation(evaluation, self.total_physics_steps)
        print(json.dumps(evaluation, sort_keys=True), flush=True)
        self.latest_evaluation = evaluation
        if greedy.best_lap_ms is not None and (
            self.best_eval_lap_ms is None or greedy.best_lap_ms < self.best_eval_lap_ms
        ):
            self.best_eval_lap_ms = greedy.best_lap_ms
        self.sink.on_evaluation(evaluation, greedy, sampled, policy=policy)
        return evaluation

    def start_event(self) -> dict[str, Any]:
        return {
            "event": "start" if self.resumed_from is None else "resume",
            "run_id": self.run_id,
            "gpu": self.gpu_name,
            "physics_sha256": self.physics_sha256,
            "code_sha256": self.code_sha256,
            "arch": self.config.arch,
            "params": count_parameters(self.agent),
            "setup_seconds": self.setup_seconds,
            "staggered_phases": self.stagger_info,
            "resumed_from": self.resumed_from,
            "route_length_m": self.env.route_length,
            "lap_count": self.env.lap_count,
            "horizon_ticks": self.env.horizon_ticks,
            "max_race_ticks": self.env.max_race_ticks,
            "args": self.config.to_dict(),
        }

    def _observation_batch(self) -> torch.Tensor:
        """The current policy input: the CUDA env's own device buffer, or a
        device copy of the CPU env's flat rows."""
        if self.device_env:
            return self.env.device_observations
        return torch.as_tensor(self.env.policy_observations, device=self.device)

    def run(self) -> dict[str, Any]:
        args = self.config
        env = self.env
        device = self.device
        agent = self.agent
        snapshot_starts = args.snapshot_start_fraction > 0.0
        spectate_buffer = self.sink.spectate_buffer
        assert self.current_raw_observation is not None
        current_raw_observation = self.current_raw_observation
        episode_max_progress = self.episode_max_progress

        self.sink.on_start(self.start_event())
        if self.resumed_from is None:
            self._evaluate(0.0)

        while True:
            self.update += 1
            update = self.update
            torch.cuda.synchronize()
            rollout_start = time.perf_counter()
            update_env_seconds = 0.0

            for step in range(args.num_steps):
                self.observations[step] = current_raw_observation
                pedal_mask = self.pedal_hold.mask() if self.pedal_hold is not None else None
                with torch.no_grad():
                    action, log_probability, _, value = agent.get_action_and_value(
                        current_raw_observation, pedal_mask=pedal_mask
                    )
                if self.pedal_hold is not None:
                    action = self.pedal_hold.apply(action, pedal_mask)
                    self.pedal_masks[step] = pedal_mask
                self.actions[step] = action
                self.log_probabilities[step] = log_probability
                self.values[step] = value

                if self.device_env:
                    action_array = device_environment_action(action, args.action_space)
                else:
                    action_array = environment_action(action, args.action_space)
                env_start = time.perf_counter()
                _, _, terminated, truncated, info = env.step(action_array)
                update_env_seconds += time.perf_counter() - env_start

                if self.device_env:
                    transition = env.device_transitions
                else:
                    transition = self.pinned_transitions.copy_(
                        torch.from_numpy(env.policy_transitions)
                    ).to(device, non_blocking=True)
                self.rewards[step] = transition[:, 0]
                self.discounts[step] = transition[:, 1]
                self.terminations[step] = transition[:, 2]
                self.truncations[step] = transition[:, 3]
                self.truncation_values[step].zero_()

                ended = terminated | truncated
                if np.any(truncated & ~terminated):
                    if self.device_env:
                        final_raw = env.device_final_observations
                    else:
                        final_raw = torch.as_tensor(
                            env.policy_final_observations, device=device
                        )
                    truncated_indices = torch.as_tensor(
                        np.flatnonzero(truncated & ~terminated), device=device
                    )
                    with torch.no_grad():
                        final_values = agent.get_value(final_raw[truncated_indices])
                    self.truncation_values[step, truncated_indices] = final_values

                transition_ticks = env.executed_ticks
                if np.any((transition_ticks < 1) | (transition_ticks > args.action_repeat)):
                    raise RuntimeError("native transition discount encoded bad tick count")
                self.total_physics_steps += int(transition_ticks.sum())
                self.total_decision_steps += args.num_envs

                active_race = env.observations["race"]
                final_race = env.final_observations["race"]
                step_progress = active_race[:, 4].copy()
                step_progress[ended] = final_race[ended, 4]
                episode_max_progress = np.maximum(episode_max_progress, step_progress)
                if spectate_buffer is not None:
                    spectate_buffer.capture(
                        observations=env.observations,
                        final_observations=env.final_observations,
                        ended=ended,
                        termination_reasons=env.termination_reasons,
                        episode_ids=env.episode_ids,
                        progress=episode_max_progress,
                    )

                if snapshot_starts:
                    self._capture_step(active_race, ended)

                finish_time_active = self.finish_time_switch_update > 0
                if finish_time_active:
                    # (e) From the decision after the first training finish:
                    # the shaped reward is replaced by 0 everywhere except the
                    # finishing decision, which pays the unused race budget in
                    # seconds. The step that produced the first finish keeps
                    # its shaped reward; failures pay 0.
                    self.rewards[step] = torch.as_tensor(
                        finish_time_rewards(
                            ended, info["termination_reason"], info["race_time_ms"],
                            env.max_race_ticks,
                        ),
                        device=device,
                    )

                # Snapshot restores clear the live transition views. Consume
                # the finishing time before _handle_ended can restore slots.
                if np.any(ended):
                    episode_max_progress = self._handle_ended(
                        ended, info, final_race, episode_max_progress
                    )

                if self.device_env:
                    # The env's own buffer: read by the next forward pass and
                    # copied into the rollout before the next step overwrites it.
                    current_raw_observation = env.device_observations
                else:
                    current_raw_observation = self.pinned_observations.copy_(
                        torch.from_numpy(env.policy_observations)
                    ).to(device, non_blocking=True)
                if self.pedal_hold is not None:
                    self.pedal_hold.advance(ended)
                if self.novelty is not None:
                    flat = (
                        current_raw_observation.cpu().numpy()
                        if self.device_env else env.policy_observations
                    )
                    bonus = self.novelty.step(flat, ended)
                    if bonus.any():
                        self.rewards[step] += torch.as_tensor(bonus, device=device)

            with torch.no_grad():
                next_value = agent.get_value(current_raw_observation)
            torch.cuda.synchronize()
            rollout_seconds = time.perf_counter() - rollout_start
            self.cumulative_env_seconds += update_env_seconds
            self.cumulative_rollout_seconds += rollout_seconds

            if snapshot_starts and update % REFRESH_UPDATES == 0:
                self.pool.refresh()
            self._optimize(next_value)
            if self.ema_agent is not None:
                ema_update(self.ema_agent, agent, args.ema_decay)

            self.current_raw_observation = current_raw_observation
            self.episode_max_progress = episode_max_progress
            elapsed = time.perf_counter() - self.run_start
            row = self._metrics_row(elapsed)
            self._write_row(row)
            summary = self.summary(elapsed)
            if spectate_buffer is not None:
                spectate_buffer.update_meta(
                    {
                        "updateCount": update,
                        "wallTimeSeconds": elapsed,
                        "finishes": self.fullstart_finishes,
                        "bestLapMs": self.fullstart_best_lap_ms or None,
                        "distanceMean": optional_mean(self.fullstart_distances),
                        "fullstartEpisodes": self.fullstart_episodes,
                    }
                )
            self.sink.on_update(row, summary)
            print(json.dumps(row, sort_keys=True), flush=True)

            if elapsed >= self.next_evaluation_seconds:
                # Rounded to the same 6 decimals the protocol uses for its
                # expected minutes: the accumulated schedule is inexact for
                # intervals like 0.03 min (1.7999999999999998 s).
                self._evaluate(round(self.next_evaluation_seconds / 60.0, 6))
                self.next_evaluation_seconds += args.eval_interval_minutes * 60.0
                elapsed = time.perf_counter() - self.run_start

            duration_reached = (
                args.duration_minutes > 0.0 and elapsed >= args.duration_minutes * 60.0
            )
            update_limit_reached = args.max_updates > 0 and update >= args.max_updates
            if elapsed >= self.next_checkpoint_seconds or duration_reached or update_limit_reached:
                self.save_checkpoint(elapsed)
                while elapsed >= self.next_checkpoint_seconds:
                    self.next_checkpoint_seconds += args.checkpoint_interval_minutes * 60.0
            if duration_reached or update_limit_reached:
                break

        wall_time = time.perf_counter() - self.run_start
        summary = self.summary(wall_time)
        torch.save(
            {
                "agent": self.policy_agent.state_dict(),
                "args": args.to_dict(),
                "config": args.to_dict(),
                "summary": summary,
            },
            self.run_dir / "policy.pt",
        )
        write_json_atomic(self.run_dir / "summary.json", summary)
        return summary

    # --------------------------------------------------------- loop pieces

    def _capture_step(self, active_race: np.ndarray, ended: np.ndarray) -> None:
        """Capture a snapshot of every capturing environment whose episode has
        advanced snapshot_capture_interval_ticks since its last capture."""
        args = self.config
        env = self.env
        active_ticks = np.rint(active_race[:, 8] * env.max_race_ticks).astype(np.int32)
        capture_indices = np.flatnonzero(
            (~ended)
            & self.capture_episodes
            & (active_ticks - self.last_capture_ticks >= args.snapshot_capture_interval_ticks)
        )
        if not capture_indices.size:
            return
        snapshot_start = time.perf_counter()
        blobs = env.capture(capture_indices)
        self.snapshot_io_seconds += time.perf_counter() - snapshot_start
        self.snapshot_capture_calls += 1
        self.snapshot_captured_states += int(capture_indices.size)
        for index_value, blob in zip(capture_indices, blobs, strict=True):
            index = int(index_value)
            tick = int(active_ticks[index])
            self.inflight[index].states.append(
                PoolState(blob=blob, progress=float(active_race[index, 4]), capture_tick=tick, score=0.0)
            )
            self.last_capture_ticks[index] = tick

    def _handle_ended(
        self,
        ended: np.ndarray,
        info: dict[str, Any],
        final_race: np.ndarray,
        episode_max_progress: np.ndarray,
    ) -> np.ndarray:
        args = self.config
        env = self.env
        snapshot_starts = args.snapshot_start_fraction > 0.0
        restore_indices: list[int] = []
        restore_states: list[PoolState] = []
        ended_indices = np.flatnonzero(ended)
        for index_value in ended_indices:
            index = int(index_value)
            episode_return = float(info["completed_episode_return"][index])
            distance = float(
                np.clip(final_race[index, 4], 0.0, env.route_length * env.lap_count)
            )
            checkpoints = float(
                np.clip(
                    np.rint(final_race[index, 9] * env.checkpoint_count),
                    0,
                    env.checkpoint_count,
                )
            )
            reason = int(info["termination_reason"][index])
            lap_ms = int(info["race_time_ms"][index])
            if self.start_kinds[index] == OFFICIAL_START:
                self.fullstart_returns.append(episode_return)
                self.fullstart_distances.append(distance)
                self.fullstart_checkpoints.append(checkpoints)
                self.fullstart_episodes += 1
                self.fullstart_best_distance = max(self.fullstart_best_distance, distance)
                self.fullstart_best_checkpoints = max(
                    self.fullstart_best_checkpoints, checkpoints
                )
                if reason == FINISH_REASON:
                    self.fullstart_finishes += 1
                    if self.fullstart_best_lap_ms == 0 or lap_ms < self.fullstart_best_lap_ms:
                        self.fullstart_best_lap_ms = lap_ms
            else:
                self.snapshot_episodes += 1
            if (
                args.finish_time_reward
                and reason == FINISH_REASON
                and self.finish_time_switch_update == 0
            ):
                self.finish_time_switch_update = self.update

            if snapshot_starts:
                trajectory = self.inflight[index]
                if trajectory.states:
                    episode_ticks = int(info["completed_episode_ticks"][index])
                    trajectory.score = trajectory_score(
                        final_progress=distance,
                        finished=reason == FINISH_REASON,
                        episode_ticks=episode_ticks,
                        max_race_ticks=env.max_race_ticks,
                        reference_speed=REFERENCE_SPEED,
                    )
                    self.pool.add_trajectory(
                        trajectory, finished=reason == FINISH_REASON, episode_ticks=episode_ticks
                    )
                self.inflight[index] = Trajectory()

            # The native autoreset already put this environment at the grid
            # start; a snapshot start (below) replaces that.
            self.start_kinds[index] = OFFICIAL_START
            episode_max_progress[index] = 0.0
            self.last_capture_ticks[index] = 0
            self.capture_episodes[index] = (
                self.trainer_rng.random() < args.snapshot_capture_episode_fraction
            )
            if snapshot_starts and self.trainer_rng.random() < args.snapshot_start_fraction:
                chosen = self.pool.choose()
                if chosen is not None:
                    source, pool_index, state = chosen
                    restore_indices.append(index)
                    restore_states.append(state)
                    self.start_kinds[index] = SNAPSHOT_START
                    episode_max_progress[index] = state.progress
                    self.last_capture_ticks[index] = state.capture_tick
                    self.reset_choice_sha256 = hashlib.sha256(
                        bytes.fromhex(self.reset_choice_sha256)
                        + f"{index}:{source}:{pool_index}:{state.capture_tick}:{state.progress.hex()};".encode()
                    ).hexdigest()

        if restore_indices:
            snapshot_start = time.perf_counter()
            env.restore(restore_indices, [state.blob for state in restore_states])
            self.snapshot_io_seconds += time.perf_counter() - snapshot_start
            self.snapshot_restore_calls += 1
            self.snapshot_restored_states += len(restore_indices)
            self.snapshot_starts += len(restore_indices)
        return episode_max_progress

    def _optimize(self, next_value: torch.Tensor) -> None:
        args = self.config
        device = self.device
        agent = self.agent
        advantages, returns = self._advantage_estimator(
            self.rewards, self.discounts, self.terminations, self.truncations,
            self.truncation_values, self.values, next_value, gae_lambda=args.gae_lambda,
        )

        batch_observations = self.observations.reshape((-1, observation_width(self.config.encoder_version)))
        # The fixed encoder has no learned state or gradients. Compute each
        # rollout's features once, rather than again for every PPO epoch.
        batch_encoded = (
            agent._flat_encoder(batch_observations) if args.arch == "mlp" else None
        )
        batch_log_probabilities = self.log_probabilities.reshape(-1)
        batch_actions = self.actions.reshape(
            (-1,) if args.action_space == "discrete" else (-1, 3)
        )
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = self.values.reshape(-1)
        batch_pedal_masks = self.pedal_masks.reshape(-1) if self.pedal_hold is not None else None

        torch.cuda.synchronize()
        optimization_start = time.perf_counter()
        # Linear warm-up over the first lr_warmup_updates updates (0 = none);
        # the pilot transformer arms used 10, the MLP none.
        learning_rate = args.learning_rate * min(
            1.0, self.update / args.lr_warmup_updates if args.lr_warmup_updates else 1.0
        )
        if args.lr_decay_updates > 0:
            learning_rate *= max(LR_DECAY_FLOOR, 1.0 - (self.update - 1) / args.lr_decay_updates)
        self.learning_rate = learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        indices = np.arange(self.batch_size)
        entropy_sum = torch.zeros((), device=device)
        entropy_passes = 0
        epochs_run = 0
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            # Each NumPy advanced index used to trigger its own host-to-device
            # upload. Transfer the permutation once, retaining NumPy's exact
            # shuffle/RNG stream and the original minibatch order.
            device_indices = torch.as_tensor(indices, device=device)
            epoch_kl = torch.zeros((), device=device)
            for start in range(0, self.batch_size, self.minibatch_size):
                minibatch_indices = device_indices[start : start + self.minibatch_size]
                _, new_log_probability, entropy, new_value = agent.get_action_and_value(
                    batch_observations[minibatch_indices],
                    batch_actions[minibatch_indices],
                    pedal_mask=(
                        batch_pedal_masks[minibatch_indices]
                        if batch_pedal_masks is not None else None
                    ),
                    encoded=(batch_encoded[minibatch_indices] if batch_encoded is not None else None),
                )
                log_ratio = new_log_probability - batch_log_probabilities[minibatch_indices]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    epoch_kl += approx_kl

                minibatch_advantages = batch_advantages[minibatch_indices]
                minibatch_advantages = (
                    minibatch_advantages - minibatch_advantages.mean()
                ) / (minibatch_advantages.std() + 1e-8)
                policy_loss_unclipped = -minibatch_advantages * ratio
                policy_loss_clipped = -minibatch_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

                new_value = new_value.view(-1)
                value_loss_unclipped = (new_value - batch_returns[minibatch_indices]).square()
                value_delta = new_value - batch_values[minibatch_indices]
                clipped_value = batch_values[minibatch_indices] + torch.clamp(
                    value_delta, -args.clip_coef, args.clip_coef
                )
                value_loss_clipped = (clipped_value - batch_returns[minibatch_indices]).square()
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy_loss = entropy.mean()
                entropy_sum += entropy_loss.detach()
                entropy_passes += 1
                if self.entropy_window is not None:
                    entropy_loss = (
                        entropy
                        * entropy_weights(
                            batch_observations[minibatch_indices], self.entropy_window, args.entropy_boost
                        )
                    ).mean()
                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                self.optimizer.step()
            epochs_run += 1
            # Trust region (config.target_kl): an epoch whose mean approx_kl
            # passes it ends the update. One host sync per epoch.
            if args.target_kl > 0.0 and float(epoch_kl.item()) / args.num_minibatches > args.target_kl:
                break

        torch.cuda.synchronize()
        self.cumulative_optimization_seconds += time.perf_counter() - optimization_start
        with torch.no_grad():
            prediction_variance = torch.var(batch_returns)
            explained_variance = (
                1.0 - torch.var(batch_returns - batch_values) / prediction_variance
                if prediction_variance > 0
                else torch.tensor(float("nan"), device=device)
            )
        self.last_losses = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            # Mean over every minibatch pass of the update, not the last
            # minibatch alone.
            "entropy": float((entropy_sum / entropy_passes).item()),
            "approx_kl": float(approx_kl.item()),
            "explained_variance": float(explained_variance.item()),
            "update_epochs_run": epochs_run,
        }

    def _metrics_row(self, elapsed: float) -> dict[str, Any]:
        return {
            "update": self.update,
            "wall_time_s": elapsed,
            "decision_steps": self.total_decision_steps,
            "physics_steps": self.total_physics_steps,
            "train/decision_steps_per_second": self.total_decision_steps / elapsed,
            "train/physics_steps_per_second": self.total_physics_steps / elapsed,
            "train/fullstart_episodes": self.fullstart_episodes,
            "train/snapshot_episodes": self.snapshot_episodes,
            "train/fullstart_episode_return_mean_100": optional_mean(self.fullstart_returns),
            "train/fullstart_distance_mean_100": optional_mean(self.fullstart_distances),
            "train/fullstart_distance_p90_100": optional_percentile(
                self.fullstart_distances, 90.0
            ),
            "train/fullstart_best_distance": self.fullstart_best_distance,
            "train/fullstart_checkpoints_mean_100": optional_mean(self.fullstart_checkpoints),
            "train/fullstart_best_checkpoints": self.fullstart_best_checkpoints,
            "train/fullstart_finishes": self.fullstart_finishes,
            "train/fullstart_best_lap_ms": self.fullstart_best_lap_ms,
            **{f"train/{name}": value for name, value in self.pool.metrics().items()},
            "train/snapshot_starts": self.snapshot_starts,
            "train/snapshot_capture_calls": self.snapshot_capture_calls,
            "train/snapshot_captured_states": self.snapshot_captured_states,
            "train/snapshot_restore_calls": self.snapshot_restore_calls,
            "train/snapshot_restored_states": self.snapshot_restored_states,
            "train/snapshot_io_seconds": self.snapshot_io_seconds,
            **{
                f"train/{name}": value
                for name, value in (
                    self.novelty.metrics() if self.novelty is not None
                    else {"novelty_cells": 0, "novelty_landings": 0, "novelty_bonus_total": 0.0}
                ).items()
            },
            "train/learning_rate": self.learning_rate,
            **{f"train/{name}": value for name, value in self.last_losses.items()},
            "train/environment_seconds": self.cumulative_env_seconds,
            "train/rollout_other_seconds": (
                self.cumulative_rollout_seconds - self.cumulative_env_seconds
            ),
            "train/optimization_seconds": self.cumulative_optimization_seconds,
        }

    def summary(self, wall_time: float) -> dict[str, Any]:
        latest = self.latest_evaluation
        return {
            "run_id": self.run_id,
            "seed": self.config.seed,
            "gpu": self.gpu_name,
            "physics_sha256": self.physics_sha256,
            "code_sha256": self.code_sha256,
            "arch": self.config.arch,
            "params": count_parameters(self.agent),
            "setup_seconds": self.setup_seconds,
            "wall_time_s": wall_time,
            "updates": self.update,
            "decision_steps": self.total_decision_steps,
            "physics_steps": self.total_physics_steps,
            "decision_steps_per_second": (
                self.total_decision_steps / wall_time if wall_time > 0 else 0.0
            ),
            "physics_steps_per_second": (
                self.total_physics_steps / wall_time if wall_time > 0 else 0.0
            ),
            "fullstart_episodes": self.fullstart_episodes,
            "snapshot_episodes": self.snapshot_episodes,
            "episode_return_mean_100": optional_mean(self.fullstart_returns),
            "distance_mean_100": optional_mean(self.fullstart_distances),
            "distance_p90_100": optional_percentile(self.fullstart_distances, 90.0),
            "best_distance": self.fullstart_best_distance,
            "checkpoints_mean_100": optional_mean(self.fullstart_checkpoints),
            "best_checkpoints": self.fullstart_best_checkpoints,
            "finishes": self.fullstart_finishes,
            "best_lap_ms": self.fullstart_best_lap_ms or None,
            "eval_fullstart": (
                {
                    "scheduled_minutes": latest["scheduled_minutes"],
                    # Greedy trajectory: finished is the whole answer (F29).
                    "finished": bool(latest["eval_fullstart/finished"]),
                    "median_lap_ms": latest["eval_fullstart/median_lap_ms"],
                    "best_lap_ms": latest["eval_fullstart/best_lap_ms"],
                    "distance_mean": latest["eval_fullstart/distance_mean"],
                    # Sampled policy: the finish rate that is a rate.
                    "finish_rate": latest["eval_sampled/finish_rate"],
                    "sampled_episodes": latest["eval_sampled/episodes"],
                    "sampled_median_lap_ms": latest["eval_sampled/median_lap_ms"],
                    "sampled_best_lap_ms": latest["eval_sampled/best_lap_ms"],
                    "sampled_distance_mean": latest["eval_sampled/distance_mean"],
                }
                if latest is not None
                else None
            ),
            "eval_best_lap_ms": self.best_eval_lap_ms,
            "snapshot_pool": self.pool.metrics(),
            "snapshot_starts": self.snapshot_starts,
            "eval_env_restore_count": self.eval_env.restore_count,
            "snapshot_capture_calls": self.snapshot_capture_calls,
            "snapshot_captured_states": self.snapshot_captured_states,
            "snapshot_restore_calls": self.snapshot_restore_calls,
            "snapshot_restored_states": self.snapshot_restored_states,
            "snapshot_io_seconds": self.snapshot_io_seconds,
            "reset_choice_sha256": self.reset_choice_sha256,
            "staggered_phases": self.stagger_info,
            "finish_time_switch_update": self.finish_time_switch_update,
            "resumed_from": self.resumed_from,
            "environment_seconds": self.cumulative_env_seconds,
            "rollout_other_seconds": (
                self.cumulative_rollout_seconds - self.cumulative_env_seconds
            ),
            "optimization_seconds": self.cumulative_optimization_seconds,
            **self.last_losses,
        }

    def close(self) -> None:
        for handle in (self.metrics_file, self.evaluations_file):
            if handle is not None:
                handle.close()
        if self.tensorboard is not None:
            self.tensorboard.close()


__all__ = [
    "Agent",
    "CHECKPOINT_FORMAT",
    "EVALUATION_NAMES",
    "METRIC_NAMES",
    "ARCHITECTURES",
    "NullSink",
    "PPOTrainer",
    "TIMING_COLUMNS",
    "TrainerSink",
    "count_parameters",
    "make_env",
    "numerics_state",
    "pin_cuda_numerics",
    "seed_everything",
    "select_device",
    "stagger_initial_phases",
]
