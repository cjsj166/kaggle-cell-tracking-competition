"""Persisted settings shared by training and checkpoint validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Settings that define model and validation semantics for a training run."""

    config_version: int = 1
    unet_out_channels: int = 32
    unet_layers: tuple[int, ...] = (32, 64, 128)
    downsample: tuple[int, ...] = (1, 4, 4)
    window_size: int = 2
    det_loss_weight: float = 1.0
    det_neg_weight: float = 1e-2
    pool_kernel_um: float = 5.0


def save_training_config(config: TrainingConfig, path: Path) -> None:
    """Write all settings for a training run to JSON."""
    path.write_text(json.dumps(asdict(config), indent=2))


def load_training_config(path: Path) -> TrainingConfig:
    """Load a complete training configuration from JSON."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid training config at {path}: expected a JSON object")

    expected = {field.name for field in fields(TrainingConfig)}
    missing = sorted(expected - data.keys())
    unknown = sorted(data.keys() - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"Invalid training config at {path}: {'; '.join(details)}")

    try:
        for name in ("config_version", "unet_out_channels", "window_size"):
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        for name in ("det_loss_weight", "det_neg_weight", "pool_kernel_um"):
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            data[name] = float(value)
        for name in ("unet_layers", "downsample"):
            if not isinstance(data[name], list):
                raise TypeError(f"{name} must be a JSON list")
        data["unet_layers"] = tuple(int(value) for value in data["unet_layers"])
        data["downsample"] = tuple(int(value) for value in data["downsample"])
        return TrainingConfig(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training config at {path}: {exc}") from exc
