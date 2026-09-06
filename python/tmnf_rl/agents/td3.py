"""Off-policy large-batch actor-critic on the vectorised env. One environment step of all N environments per
iteration, a device-resident replay buffer, twin distributional critics
(HL-Gauss categorical over the shaped return), a deterministic actor with
Gaussian exploration noise, Polyak targets, delayed actor updates.

Action representation (hybrid, P-DQN shape): the actor emits the steer
u = tanh(raw) in [-1, 1]; the critics take (state, steer) and output one
return distribution per pedal combination k in {none, gas, brake, both}.
Acting picks k = argmax_k Q(s, u, k) (epsilon-greedy over k), so the pedals
are learnt by Q-learning and the steer by the deterministic policy gradient
through Q(s, ., k). Exploration: N(0, sigma) on the steer, uniform k with
probability pedal_epsilon, and a small pre-tanh penalty keeps the steer
head off saturation.

Return support: the shaped return is bounded above by
route / 50 (a finish from the grid at infinite speed) and a failure costs at
most the budget's ticks, 0.01 (1 - gamma^Tmax) / (1 - gamma); the critic's
categorical support is [-budget cost, route / 50] (bins in config), so no
incumbent lap or hand-set range enters.

The trainer subclasses PPOTrainer for everything that is not the learning
rule: environments, snapshot starts from the agent's own trajectories,
evaluation from the grid, metrics, registry. No resume: the replay buffer is
not checkpointed (policy.pt and a light checkpoint are written).
"""

from __future__ import annotations

from tmnf_rl.encoder import observation_width

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tmnf_rl.agents.ppo import (
    METRIC_NAMES,
    OBSERVATION_WIDTH,
    REFRESH_UPDATES,
    PPOTrainer,
    build_trunk,
    layer_init,
)
from tmnf_rl.encoder import CRITIC_FEATURES, flat_features, critic_context, encode, encode_flat
from tmnf_rl.utils import write_json_atomic

PPO_LOSS_KEYS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "explained_variance",
    "update_epochs_run",
)
TD3_LOSS_KEYS = (
    "critic_loss",
    "actor_loss",
    "q_mean",
    "q_target_mean",
    "replay_size",
    "gradient_steps",
    "exploration_std",
)
TD3_METRIC_NAMES = [
    name for name in METRIC_NAMES if name.removeprefix("train/") not in PPO_LOSS_KEYS
] + [f"train/{name}" for name in TD3_LOSS_KEYS]
# Decisions per logged row: one "update" of the metrics table.
LOG_EVERY_STEPS = 32
TARGET_NOISE_STD = 0.2
TARGET_NOISE_CLIP = 0.5
HL_GAUSS_SIGMA_BINS = 0.75
PRE_TANH_PENALTY = 1e-3
# Pedal combinations the critic scores: (gas, brake).
PEDALS = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


class Actor(nn.Module):
    """Deterministic steer in [-1, 1]; ``raw`` is the pre-tanh value."""

    def __init__(self, hidden_size: int, encoder_version: int = 1) -> None:
        super().__init__()
        self.encoder_version = encoder_version
        self.trunk = build_trunk("mlp", hidden_size, encoder_version)
        self.head = layer_init(nn.Linear(self.trunk.width, 1), 0.01)

    def raw(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(encode(observation, self.encoder_version)))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.raw(observation))


class DistributionalCritic(nn.Module):
    """LayerNorm MLP (SimBa-lite) from (features, remaining distance, steer)
    to categorical logits over the return support, one row per pedal
    combination: output (N, 4, bins)."""

    def __init__(self, width: int, bins: int, encoder_version: int = 1) -> None:
        super().__init__()
        self.bins = bins
        self.encoder_version = encoder_version
        inputs = flat_features(encoder_version) + CRITIC_FEATURES + 1
        self.network = nn.Sequential(
            layer_init(nn.Linear(inputs, width)),
            nn.LayerNorm(width),
            nn.SiLU(),
            layer_init(nn.Linear(width, width)),
            nn.LayerNorm(width),
            nn.SiLU(),
            layer_init(nn.Linear(width, len(PEDALS) * bins), 0.01),
        )

    def forward(self, observation: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        features = torch.cat((encode_flat(observation, self.encoder_version), critic_context(observation), steer), dim=1)
        return self.network(features).view(observation.shape[0], len(PEDALS), self.bins)


class OffPolicyAgent(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        critic_width: int,
        bins: int,
        v_min: float,
        v_max: float,
        exploration_std: float,
        pedal_epsilon: float,
        encoder_version: int = 1,
    ) -> None:
        super().__init__()
        if bins < 2 or not v_min < v_max:
            raise ValueError("categorical support needs >= 2 bins and v_min < v_max")
        self.actor = Actor(hidden_size, encoder_version)
        self.critics = nn.ModuleList([DistributionalCritic(critic_width, bins, encoder_version) for _ in range(2)])
        width = (v_max - v_min) / (bins - 1)
        # Centers at linspace(v_min, v_max, bins); edges half a bin either side.
        self.register_buffer("centers", torch.linspace(v_min, v_max, bins))
        self.register_buffer("edges", torch.linspace(v_min - width / 2.0, v_max + width / 2.0, bins + 1))
        self.register_buffer("pedals", PEDALS.clone())
        self.sigma = HL_GAUSS_SIGMA_BINS * width
        self.v_min = v_min
        self.v_max = v_max
        self.exploration_std = exploration_std
        self.pedal_epsilon = pedal_epsilon

    def q_values(self, critic: nn.Module, observation: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        """(N, 4): the mean return of each pedal combination at (s, steer)."""
        return torch.softmax(critic(observation, steer), dim=2) @ self.centers

    def q_min(self, critics: nn.ModuleList, observation: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        return torch.min(self.q_values(critics[0], observation, steer), self.q_values(critics[1], observation, steer))

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        """Greedy action (N, 2): steer, pedal index of the twin-min best."""
        steer = self.actor(observation)
        pedal = self.q_min(self.critics, observation, steer).argmax(dim=1)
        return torch.cat((steer, pedal.unsqueeze(1).to(steer.dtype)), dim=1)

    def target_distribution(self, target: torch.Tensor) -> torch.Tensor:
        """HL-Gauss: the Gaussian of std sigma around the clipped target,
        integrated over the bins."""
        target = target.clamp(self.v_min, self.v_max).unsqueeze(1)
        cdf = 0.5 * (1.0 + torch.erf((self.edges.unsqueeze(0) - target) / (self.sigma * math.sqrt(2.0))))
        probabilities = cdf[:, 1:] - cdf[:, :-1]
        return probabilities / probabilities.sum(dim=1, keepdim=True)

    # ---- evaluator interface (tmnf_rl.evaluation)
    def get_deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.act(observation)

    def get_action_and_value(self, observation: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        return self.explore(observation), None, None, None

    def explore(self, observation: torch.Tensor) -> torch.Tensor:
        """N(0, exploration_std) on the steer, clipped; the pedal index is the
        twin-min argmax at the noisy steer, replaced by a uniform index with
        probability pedal_epsilon."""
        steer = self.actor(observation)
        steer = (steer + self.exploration_std * torch.randn_like(steer)).clamp(-1.0, 1.0)
        pedal = self.q_min(self.critics, observation, steer).argmax(dim=1)
        if self.pedal_epsilon > 0.0:
            redraw = torch.rand(observation.shape[0], device=observation.device) < self.pedal_epsilon
            uniform = torch.randint(0, len(PEDALS), (observation.shape[0],), device=observation.device)
            pedal = torch.where(redraw, uniform, pedal)
        return torch.cat((steer, pedal.unsqueeze(1).to(steer.dtype)), dim=1)

    def random_action(self, count: int, device: torch.device) -> torch.Tensor:
        steer = torch.rand((count, 1), device=device) * 2.0 - 1.0
        pedal = torch.randint(0, len(PEDALS), (count, 1), device=device).to(steer.dtype)
        return torch.cat((steer, pedal), dim=1)

    def environment_action(self, action: torch.Tensor, mode: str) -> dict[str, np.ndarray]:
        if mode != "analog":
            raise ValueError("the off-policy agent drives the analog action space")
        values = self.device_environment_action(action).cpu().numpy()
        return {
            "steer": values[:, 0],
            "gas": values[:, 1].astype(np.int8),
            "brake": values[:, 2].astype(np.int8),
        }

    def device_environment_action(self, action: torch.Tensor) -> torch.Tensor:
        """(steer, gas, brake) as the env takes them, from (steer, pedal index)."""
        values = action.detach()
        return torch.cat((values[:, :1], self.pedals[values[:, 1].long()]), dim=1)


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device, encoder_version: int = 1) -> None:
        self.capacity = capacity
        self.device = device
        self.observations = torch.zeros((capacity, observation_width(encoder_version)), device=device)
        self.next_observations = torch.zeros((capacity, observation_width(encoder_version)), device=device)
        self.actions = torch.zeros((capacity, 2), device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.discounts = torch.zeros(capacity, device=device)
        self.position = 0
        self.size = 0

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> None:
        count = observations.shape[0]
        if count > self.capacity:
            raise ValueError("more transitions per step than the replay holds")
        indices = (torch.arange(count, device=self.device) + self.position) % self.capacity
        self.observations[indices] = observations
        self.next_observations[indices] = next_observations
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.discounts[indices] = discounts
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.discounts[indices],
            self.next_observations[indices],
        )


class OffPolicyTrainer(PPOTrainer):
    metric_names = TD3_METRIC_NAMES

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = self.config
        device = self.device
        route = self.env.route_length * self.env.lap_count
        gamma = config.discount_per_tick
        budget_cost = 0.01 * (1.0 - gamma ** self.env.max_race_ticks) / (1.0 - gamma)
        # Floor: the whole budget's tick cost (the stuck and off-track rules
        # end a failure long before the deferred potential adds much); the
        # target distribution clamps anything below.
        v_min = -budget_cost
        v_max = route / 50.0
        agent_arguments = (
            config.hidden_size, config.td3_critic_width, config.td3_bins, v_min, v_max,
            config.td3_exploration_std, config.td3_pedal_epsilon, config.encoder_version,
        )
        self.agent = OffPolicyAgent(*agent_arguments).to(device)
        self.target = OffPolicyAgent(*agent_arguments).to(device)
        self.target.load_state_dict(self.agent.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = optim.Adam(self.agent.actor.parameters(), lr=config.learning_rate, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.agent.critics.parameters(), lr=config.learning_rate, eps=1e-5)
        self.replay = ReplayBuffer(config.td3_replay_capacity, device, config.encoder_version)
        self.gradient_steps = 0
        self.support = (v_min, v_max)
        self.last_losses = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "q_mean": 0.0,
            "q_target_mean": 0.0,
            "replay_size": 0,
            "gradient_steps": 0,
            "exploration_std": config.td3_exploration_std,
        }

    # ------------------------------------------------------------ overrides
    @property
    def policy_agent(self) -> OffPolicyAgent:  # type: ignore[override]
        return self.agent

    def setup_resume(self, checkpoint_path: Path) -> None:
        raise RuntimeError("the off-policy trainer does not resume (replay buffer is not checkpointed)")

    def save_checkpoint(self, elapsed: float) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "latest.pt"
        torch.save(
            {
                "format": "tmnf-td3-checkpoint",
                "run_id": self.run_id,
                "config": self.config.to_dict(),
                "physics_sha256": self.physics_sha256,
                "elapsed_seconds": elapsed,
                "agent": self.agent.state_dict(),
                "target": self.target.state_dict(),
                "counters": {"update": self.update, "gradient_steps": self.gradient_steps},
            },
            path,
        )
        self.sink.on_checkpoint(path, self.update)
        return path

    def start_event(self) -> dict[str, Any]:
        event = super().start_event()
        event["algorithm"] = "td3"
        event["return_support"] = list(self.support)
        return event

    def summary(self, wall_time: float) -> dict[str, Any]:
        summary = super().summary(wall_time)
        summary["algorithm"] = "td3"
        summary["return_support"] = list(self.support)
        summary["gradient_steps"] = self.gradient_steps
        return summary

    # ------------------------------------------------------------------ loop
    def run(self) -> dict[str, Any]:
        args = self.config
        env = self.env
        device = self.device
        agent = self.agent
        snapshot_starts = args.snapshot_start_fraction > 0.0
        assert self.current_raw_observation is not None
        observation = self.current_raw_observation
        episode_max_progress = self.episode_max_progress

        self.sink.on_start(self.start_event())
        self._evaluate(0.0)

        step = 0
        while True:
            torch.cuda.synchronize()
            rollout_start = time.perf_counter()
            update_env_seconds = 0.0
            for _ in range(LOG_EVERY_STEPS):
                step += 1
                with torch.no_grad():
                    if step <= args.td3_warmup_steps:
                        action = agent.random_action(args.num_envs, device)
                    else:
                        action = agent.explore(observation)
                if self.device_env:
                    action_array: Any = agent.device_environment_action(action)
                else:
                    action_array = agent.environment_action(action, "analog")
                env_start = time.perf_counter()
                _, _, terminated, truncated, info = env.step(action_array)
                update_env_seconds += time.perf_counter() - env_start

                if self.device_env:
                    transition = env.device_transitions
                    final_raw = env.device_final_observations
                    new_observation = env.device_observations
                else:
                    transition = torch.as_tensor(env.policy_transitions, device=device)
                    final_raw = torch.as_tensor(env.policy_final_observations, device=device)
                    new_observation = torch.as_tensor(env.policy_observations, device=device)
                ended = terminated | truncated
                ended_device = transition[:, 2].bool() | transition[:, 3].bool()
                next_observation = torch.where(ended_device.unsqueeze(1), final_raw, new_observation)
                discount = transition[:, 1] * (1.0 - transition[:, 2])
                self.replay.add(observation, action, transition[:, 0], discount, next_observation)
                self.last_losses["replay_size"] = self.replay.size

                transition_ticks = env.executed_ticks
                self.total_physics_steps += int(transition_ticks.sum())
                self.total_decision_steps += args.num_envs
                active_race = env.observations["race"]
                final_race = env.final_observations["race"]
                step_progress = active_race[:, 4].copy()
                step_progress[ended] = final_race[ended, 4]
                episode_max_progress = np.maximum(episode_max_progress, step_progress)
                if snapshot_starts:
                    self._capture_step(active_race, ended)
                if np.any(ended):
                    episode_max_progress = self._handle_ended(ended, info, final_race, episode_max_progress)
                observation = self._observation_batch().clone()

                if step > args.td3_warmup_steps:
                    self._learn()

            torch.cuda.synchronize()
            self.update += 1
            self.cumulative_env_seconds += update_env_seconds
            self.cumulative_rollout_seconds += time.perf_counter() - rollout_start
            if snapshot_starts and self.update % REFRESH_UPDATES == 0:
                self.pool.refresh()
            self.current_raw_observation = observation
            self.episode_max_progress = episode_max_progress
            elapsed = time.perf_counter() - self.run_start
            row = self._metrics_row(elapsed)
            self._write_row(row)
            summary = self.summary(elapsed)
            self.sink.on_update(row, summary)
            print(json.dumps(row, sort_keys=True), flush=True)

            if elapsed >= self.next_evaluation_seconds:
                self._evaluate(round(self.next_evaluation_seconds / 60.0, 6))
                self.next_evaluation_seconds += args.eval_interval_minutes * 60.0
                elapsed = time.perf_counter() - self.run_start
            duration_reached = args.duration_minutes > 0.0 and elapsed >= args.duration_minutes * 60.0
            update_limit_reached = args.max_updates > 0 and self.update >= args.max_updates
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
                "agent": agent.state_dict(),
                "args": args.to_dict(),
                "config": args.to_dict(),
                "summary": summary,
            },
            self.run_dir / "policy.pt",
        )
        write_json_atomic(self.run_dir / "summary.json", summary)
        return summary

    def _learn(self) -> None:
        args = self.config
        agent = self.agent
        target = self.target
        torch.cuda.synchronize()
        start = time.perf_counter()
        critic_loss_value = 0.0
        q_mean = 0.0
        q_target_mean = 0.0
        actor_loss_value = self.last_losses["actor_loss"]
        for _ in range(args.td3_updates_per_step):
            observations, actions, rewards, discounts, next_observations = self.replay.sample(args.td3_batch_size)
            steers = actions[:, :1]
            pedals = actions[:, 1].long()
            with torch.no_grad():
                noise = (torch.randn_like(steers) * TARGET_NOISE_STD).clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
                next_steers = (target.actor(next_observations) + noise).clamp(-1.0, 1.0)
                # Pedal at s': the twin-min argmax (double-Q through the min).
                next_q = agent.q_min(target.critics, next_observations, next_steers).max(dim=1).values
                y = rewards + discounts * next_q
                distribution = agent.target_distribution(y)
            rows = torch.arange(observations.shape[0], device=self.device)
            critic_loss = torch.zeros((), device=self.device)
            for critic in agent.critics:
                logits = critic(observations, steers)[rows, pedals]
                critic_loss = critic_loss - (distribution * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(agent.critics.parameters(), args.max_grad_norm)
            self.critic_optimizer.step()
            self.gradient_steps += 1
            critic_loss_value = float(critic_loss.item())
            q_target_mean = float(y.mean().item())
            if self.gradient_steps % args.td3_policy_delay == 0:
                raw = agent.actor.raw(observations)
                best_q = agent.q_values(agent.critics[0], observations, torch.tanh(raw)).max(dim=1).values
                actor_loss = -best_q.mean() + PRE_TANH_PENALTY * raw.square().mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.item())
                q_mean = -actor_loss_value
                with torch.no_grad():
                    for target_parameter, parameter in zip(target.parameters(), agent.parameters(), strict=True):
                        target_parameter.mul_(1.0 - args.td3_tau).add_(parameter, alpha=args.td3_tau)
        torch.cuda.synchronize()
        self.cumulative_optimization_seconds += time.perf_counter() - start
        self.last_losses = {
            "critic_loss": critic_loss_value,
            "actor_loss": actor_loss_value,
            "q_mean": q_mean if q_mean else self.last_losses["q_mean"],
            "q_target_mean": q_target_mean,
            "replay_size": self.replay.size,
            "gradient_steps": self.gradient_steps,
            "exploration_std": args.td3_exploration_std,
        }


__all__ = [
    "LOG_EVERY_STEPS",
    "OffPolicyAgent",
    "OffPolicyTrainer",
    "ReplayBuffer",
    "TD3_METRIC_NAMES",
]
