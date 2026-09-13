import json
from pathlib import Path

import pytest

from tracking_cellmot.training_config import (
    TrainingConfig,
    load_training_config,
    save_training_config,
)


def test_training_config_round_trip_preserves_every_value(tmp_path: Path) -> None:
    config = TrainingConfig(
        config_version=1,
        unet_out_channels=16,
        unet_layers=(16, 32),
        downsample=(2, 3, 4),
        window_size=3,
        det_loss_weight=2.5,
        det_neg_weight=0.25,
        pool_kernel_um=7.5,
    )
    path = tmp_path / "config.json"

    save_training_config(config, path)

    assert json.loads(path.read_text()) == {
        "config_version": 1,
        "unet_out_channels": 16,
        "unet_layers": [16, 32],
        "downsample": [2, 3, 4],
        "window_size": 3,
        "det_loss_weight": 2.5,
        "det_neg_weight": 0.25,
        "pool_kernel_um": 7.5,
    }
    loaded = load_training_config(path)
    assert loaded == config
    assert isinstance(loaded.unet_layers, tuple)
    assert isinstance(loaded.downsample, tuple)


def test_load_training_config_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config_version": 1}))

    with pytest.raises(ValueError, match="Invalid training config"):
        load_training_config(path)


def test_load_training_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    data = {
        "config_version": 1,
        "unet_out_channels": 32,
        "unet_layers": [32, 64, 128],
        "downsample": [1, 4, 4],
        "window_size": 2,
        "det_loss_weight": 1.0,
        "det_neg_weight": 0.01,
        "pool_kernel_um": 5.0,
        "unexpected": True,
    }
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="unknown fields: unexpected"):
        load_training_config(path)


def test_load_training_config_rejects_invalid_scalar_types(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    data = {
        "config_version": 1,
        "unet_out_channels": "many",
        "unet_layers": [32, 64, 128],
        "downsample": [1, 4, 4],
        "window_size": 2,
        "det_loss_weight": 1.0,
        "det_neg_weight": 0.01,
        "pool_kernel_um": 5.0,
    }
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="unet_out_channels"):
        load_training_config(path)
