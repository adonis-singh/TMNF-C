"""Policy trunk whose topology is the MaleCNS v1.0 fruit fly connectome.

The graph comes from ``tools/build_connectome_graph.py``: every traced neuron
outside the optic lobes (71,618: central brain, visual projection neurons,
ventral nerve cord) and their connections above a synapse-count threshold,
signed by the predicted transmitter and normalised so each neuron's absolute
input sums to one. The wiring is fixed. What trains is what the wiring does
not specify: an input map from the observation encoder onto the sensory
(afferent) neurons, a per-neuron gain and bias, a shared channel-mixing
update, and a readout from the descending and motor (efferent) neurons.

Per decision the state starts at zero, the encoded observation is injected
into the afferent neurons, ``rounds`` synaptic propagations run

    h <- tanh(gain * (W h) + bias + inject + U h)

and the efferent states are read out as the policy/value features. This is
the FlyGM shape (arXiv 2602.17997) with the synaptic operator kept fixed and
no recurrent state across decisions, so PPO's rollout and minibatch code stay
unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


class FixedSparseMatmul(torch.autograd.Function):
    """y = W x for a fixed CSR ``W`` with a precomputed transpose, so the
    backward is one more sparse product instead of autograd's dense detour."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
        ctx.w_t = w_t
        return torch.sparse.mm(w, x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return torch.sparse.mm(ctx.w_t, grad.contiguous()), None, None


class ChannelMix(torch.autograd.Function):
    """y = h @ M^T over the [neurons, batch, channels] state. cuBLAS picks a
    split-K kernel for the weight gradient's 36M-row reduction that is 40x
    slower than a per-neuron bmm followed by a sum, so the backward does that."""

    @staticmethod
    def forward(ctx, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(h, m)
        return torch.matmul(h, m.t())

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        h, m = ctx.saved_tensors
        grad = grad.contiguous()
        grad_h = torch.matmul(grad, m)
        grad_m = torch.bmm(grad.transpose(1, 2), h).sum(0)
        return grad_h, grad_m


def load_graph(path: Path) -> dict[str, np.ndarray]:
    graph = dict(np.load(path, allow_pickle=True))
    graph["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return graph


class ConnectomeTrunk(nn.Module):
    width: int

    def __init__(self, graph_path: str | Path, in_features: int, hidden_size: int,
                 channels: int = 8, rounds: int = 3) -> None:
        super().__init__()
        graph = load_graph(Path(graph_path))
        n = int(len(graph["body_id"]))
        indptr = torch.from_numpy(graph["indptr"].astype(np.int64))
        indices = torch.from_numpy(graph["indices"].astype(np.int64))
        strength = torch.from_numpy(graph["strength"].astype(np.float32))
        w = torch.sparse_csr_tensor(indptr, indices, strength, size=(n, n))
        w_t = w.to_sparse_coo().t().coalesce().to_sparse_csr()
        self.register_buffer("w_crow", w.crow_indices(), persistent=False)
        self.register_buffer("w_col", w.col_indices(), persistent=False)
        self.register_buffer("w_val", w.values(), persistent=False)
        self.register_buffer("wt_crow", w_t.crow_indices(), persistent=False)
        self.register_buffer("wt_col", w_t.col_indices(), persistent=False)
        self.register_buffer("wt_val", w_t.values(), persistent=False)
        afferent = torch.from_numpy(np.flatnonzero(graph["afferent"]).astype(np.int64))
        efferent = torch.from_numpy(np.flatnonzero(graph["efferent"]).astype(np.int64))
        self.register_buffer("afferent", afferent, persistent=False)
        self.register_buffer("efferent", efferent, persistent=False)
        self.graph_sha256 = graph["sha256"]
        self.graph_path = str(graph_path)
        # Persistent, so a policy.pt refuses a trunk built from another graph.
        self.register_buffer("graph_digest", torch.tensor(list(bytes.fromhex(graph["sha256"])), dtype=torch.uint8))
        self.neurons, self.edges = n, int(len(indices))
        self.channels, self.rounds, self.width = channels, rounds, hidden_size

        self.encoder = nn.Sequential(nn.Linear(in_features, 64), nn.Tanh(), nn.Linear(64, channels))
        self.afferent_gate = nn.Parameter(torch.randn(len(afferent), channels) * 0.5)
        self.gain = nn.Parameter(torch.ones(n, 1, channels))
        self.bias = nn.Parameter(torch.zeros(n, 1, channels))
        self.mix = nn.Parameter(torch.empty(channels, channels))
        nn.init.orthogonal_(self.mix, 0.5)
        self.readout = nn.Sequential(nn.Linear(len(efferent) * channels, hidden_size), nn.Tanh())
        self._w = None
        self._w_t = None

    def _sparse(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w is None or self._w.device != self.w_val.device:
            n = self.neurons
            self._w = torch.sparse_csr_tensor(self.w_crow, self.w_col, self.w_val, size=(n, n))
            self._w_t = torch.sparse_csr_tensor(self.wt_crow, self.wt_col, self.wt_val, size=(n, n))
        return self._w, self._w_t

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        w, w_t = self._sparse()
        batch, c, n = flat.shape[0], self.channels, self.neurons
        encoded = self.encoder(flat)                                  # [B, C]
        inject = torch.zeros(n, batch, c, device=flat.device, dtype=flat.dtype)
        inject[self.afferent] = self.afferent_gate[:, None, :] * encoded[None, :, :]
        h = torch.tanh(inject)                                        # round 0: afferents only
        for _ in range(self.rounds):
            message = FixedSparseMatmul.apply(h.reshape(n, batch * c), w, w_t).view(n, batch, c)
            h = torch.tanh(self.gain * message + self.bias + inject + ChannelMix.apply(h, self.mix))
        out = h[self.efferent].permute(1, 0, 2).reshape(batch, -1)    # [B, E*C]
        return self.readout(out)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        recorded = state_dict.get(prefix + "graph_digest")
        if recorded is not None and not torch.equal(recorded.cpu(), self.graph_digest.cpu()):
            error_msgs.append(
                f"policy was trained on connectome graph {bytes(recorded.tolist()).hex()[:12]}, "
                f"this trunk loaded {self.graph_sha256[:12]}"
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


__all__ = ["ChannelMix", "ConnectomeTrunk", "FixedSparseMatmul", "load_graph"]
