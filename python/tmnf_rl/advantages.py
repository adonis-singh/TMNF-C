"""Exact generalized advantage estimation with reusable CUDA launches."""

from __future__ import annotations

import torch


@torch.no_grad()
def generalized_advantage(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    truncation_values: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        following_value = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        bootstrap_value = torch.where(
            terminations[step],
            torch.zeros_like(following_value),
            torch.where(truncations[step], truncation_values[step], following_value),
        )
        delta = rewards[step] + discounts[step] * bootstrap_value - values[step]
        continues = ~(terminations[step] | truncations[step])
        last_advantage = delta + discounts[step] * gae_lambda * continues * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


class AdvantageEstimator:
    """Capture the original recurrence, retaining its rounding and ordering.

    Rollout buffers have stable addresses for a trainer's lifetime. Only the
    final value needs staging. Rebuild if those addresses or shapes change.
    This eliminates thousands of Python/kernel launches per update without
    introducing a fused or parallel approximation of the recurrence.
    """

    def __init__(self) -> None:
        self._key: tuple | None = None

    @torch.no_grad()
    def __call__(self, *args: torch.Tensor, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
        buffers, next_value = args[:-1], args[-1]
        if next_value.device.type != "cuda":
            return generalized_advantage(*args, gae_lambda)
        key = (gae_lambda, next_value.device, next_value.dtype, tuple(next_value.shape),
               tuple((x.data_ptr(), x.device, x.dtype, tuple(x.shape), x.stride()) for x in buffers))
        if key != self._key:
            self._next_value = torch.empty_like(next_value)
            self._next_value.copy_(next_value)
            stream = torch.cuda.Stream(device=next_value.device)
            current = torch.cuda.current_stream(next_value.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                generalized_advantage(*buffers, self._next_value, gae_lambda)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = generalized_advantage(*buffers, self._next_value, gae_lambda)
            self._graph, self._output, self._buffers = graph, output, buffers
            self._key = key
        self._next_value.copy_(next_value)
        self._graph.replay()
        return tuple(x.clone() for x in self._output)
