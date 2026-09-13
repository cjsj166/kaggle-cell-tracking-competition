"""Shared loading for resumable training checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointInfo:
    """Training progress restored from a checkpoint."""

    epoch: int
    global_step: int


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> CheckpointInfo:
    """Restore a model and optional optimizer from a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "epoch", "global_step"}
    if optimizer is not None:
        required.add("optimizer_state_dict")
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        missing = sorted(required - checkpoint.keys()) if isinstance(checkpoint, dict) else []
        detail = f"; missing fields: {', '.join(missing)}" if missing else ""
        raise ValueError(f"Not a training checkpoint: {path}{detail}")

    try:
        info = CheckpointInfo(
            epoch=int(checkpoint["epoch"]),
            global_step=int(checkpoint["global_step"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training checkpoint metadata at {path}: {exc}") from exc

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training checkpoint state at {path}: {exc}") from exc
    return info
