"""Immutable, inference-only weights for each measured policy evaluation."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import torch


def retain_policy(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Save before publishing its evaluation. Never serialize optimizer/replay.

    Content addressing leaves discarded resume timelines intact without
    letting their weights overwrite the policy referenced by a newer result.
    Serialization is synchronous, so subsequent optimizer updates cannot
    mutate the saved state_dict's tensors. No RNG is consumed.
    """
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    folder = run_dir / "eval_policies"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}.pt"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"retained evaluation policy is corrupt: {path}")
    else:
        temporary = folder / f".{digest}.{os.getpid()}.tmp"
        try:
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(path.relative_to(run_dir)), "sha256": digest,
            "bytes": len(data), "update": payload["counters"]["update"]}
