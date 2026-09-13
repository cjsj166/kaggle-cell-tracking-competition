"""Checkpoint-range validation script tests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import validate_epochs as validation
from train_unet_transformer import ValidationLosses

from tracking_cellmot.checkpoints import CheckpointInfo
from tracking_cellmot.training_config import TrainingConfig, save_training_config


def test_get_checkpoint_paths_returns_inclusive_order(tmp_path: Path) -> None:
    for epoch in range(3, 6):
        (tmp_path / f"checkpoint_epoch_{epoch:04d}.pth").touch()

    checkpoints = validation.get_checkpoint_paths(tmp_path, 3, 5)

    assert [epoch for epoch, _ in checkpoints] == [3, 4, 5]
    assert [path.name for _, path in checkpoints] == [
        "checkpoint_epoch_0003.pth",
        "checkpoint_epoch_0004.pth",
        "checkpoint_epoch_0005.pth",
    ]


@pytest.mark.parametrize(
    ("start_epoch", "through_epoch", "message"),
    [
        (0, 2, "--start-epoch must be at least 1"),
        (3, 2, "--through-epoch must be greater than or equal"),
    ],
)
def test_get_checkpoint_paths_rejects_invalid_ranges(
    tmp_path: Path,
    start_epoch: int,
    through_epoch: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validation.get_checkpoint_paths(tmp_path, start_epoch, through_epoch)


def test_get_checkpoint_paths_reports_every_missing_epoch(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_epoch_0002.pth").touch()

    with pytest.raises(FileNotFoundError) as error:
        validation.get_checkpoint_paths(tmp_path, 1, 3)

    assert "checkpoint_epoch_0001.pth" in str(error.value)
    assert "checkpoint_epoch_0003.pth" in str(error.value)


def test_validate_checkpoints_uses_config_and_fresh_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoints = [(3, tmp_path / "three.pth"), (4, tmp_path / "four.pth")]
    config = TrainingConfig(
        unet_out_channels=16,
        unet_layers=(16, 32),
        det_loss_weight=2.0,
        det_neg_weight=0.2,
        pool_kernel_um=7.0,
    )
    constructed = []
    loaded = []
    prediction_objects = []
    logged = []

    class Model:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.unet = object()
            self.epoch = None

        def to(self, device):
            return self

    def load(path, model, optimizer=None):
        assert optimizer is None
        epoch = 3 if path.name == "three.pth" else 4
        model.epoch = epoch
        loaded.append((path, model))
        return CheckpointInfo(epoch=epoch, global_step=epoch * 10)

    def run(model, loader, device, **kwargs):
        assert kwargs["det_loss_weight"] == 2.0
        assert kwargs["det_neg_weight"] == 0.2
        assert kwargs["pool_kernel_um"] == 7.0
        prediction_objects.append(kwargs["predictions"])
        kwargs["predictions"]["epoch"] = model.epoch
        return ValidationLosses(1.0, 2.0, 3.0, 0.5, 0.6)

    def score(predictions):
        epoch = predictions["epoch"]
        return {
            "score": epoch / 10,
            "edge_jaccard": 0.2,
            "division_jaccard": 0.1,
        }

    monkeypatch.setattr(validation, "UNetNodeTransformer", Model)
    monkeypatch.setattr(validation, "load_training_checkpoint", load, raising=False)
    monkeypatch.setattr(validation, "run_validation", run)
    monkeypatch.setattr(validation, "score_tracking_predictions", score)
    monkeypatch.setattr(
        validation,
        "log_validation_to_tensorboard",
        lambda writer, losses, metrics, epoch: logged.append((losses, metrics, epoch)),
    )
    writer = SimpleNamespace(flush_calls=0)
    writer.flush = lambda: setattr(writer, "flush_calls", writer.flush_calls + 1)

    validation.validate_checkpoints(
        checkpoints,
        config,
        loader=object(),
        writer=writer,
        device=torch.device("cpu"),
        data_parallel=False,
    )

    assert constructed == [
        {
            "unet_out_channels": 16,
            "unet_layers": (16, 32),
            "pos_feat_dim": 32,
        },
        {
            "unet_out_channels": 16,
            "unet_layers": (16, 32),
            "pos_feat_dim": 32,
        },
    ]
    assert [path for path, _ in loaded] == [path for _, path in checkpoints]
    assert [entry[2] for entry in logged] == [3, 4]
    assert prediction_objects[0] is not prediction_objects[1]
    assert writer.flush_calls == 2


def test_validate_checkpoints_rejects_checkpoint_epoch_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Model:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(validation, "UNetNodeTransformer", Model)
    monkeypatch.setattr(
        validation,
        "load_training_checkpoint",
        lambda *args: CheckpointInfo(epoch=2, global_step=0),
        raising=False,
    )

    with pytest.raises(ValueError, match="expected 1, found 2"):
        validation.validate_checkpoints(
            [(1, tmp_path / "checkpoint_epoch_0001.pth")],
            TrainingConfig(),
            loader=object(),
            writer=object(),
            device=torch.device("cpu"),
            data_parallel=False,
        )


def test_main_loads_only_validation_files_and_uses_local_padding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"train": ["train"], "test": ["validation"]}]))
    output_dir = tmp_path / "weights" / "method" / "split_0"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_epoch_0001.pth").touch()
    save_training_config(TrainingConfig(), output_dir / "config.json")

    opened = []
    dataset_calls = []

    def load(path, **kwargs):
        opened.append((path, kwargs))
        return None, [SimpleNamespace(node_counts=[1, 1])]

    def make_dataset(video_data, **kwargs):
        dataset_calls.append((video_data, kwargs))
        return [object()]

    writer = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(validation, "WEIGHTS_PATH", tmp_path / "weights")
    monkeypatch.setattr(validation, "load_dataset_windows", load)
    monkeypatch.setattr(validation, "FrameWindowDataset", make_dataset)
    monkeypatch.setattr(validation, "DataLoader", lambda *args, **kwargs: object())
    monkeypatch.setattr(validation, "SummaryWriter", lambda **kwargs: writer)
    monkeypatch.setattr(validation, "validate_checkpoints", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--method",
            "method",
            "--data-dir",
            str(data_dir),
            "--splits",
            str(splits),
            "--start-epoch",
            "1",
            "--through-epoch",
            "1",
            "--num-workers",
            "0",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )

    validation.main()

    assert opened == [
        (
            data_dir / "validation",
            {"window_size": 2, "downsample": (1, 4, 4)},
        )
    ]
    assert dataset_calls[0][1] == {}


def test_main_requires_existing_split_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(validation, "WEIGHTS_PATH", tmp_path / "weights")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--data-dir",
            str(tmp_path),
            "--splits",
            str(tmp_path / "missing.json"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        validation.main()

    assert error.value.code == 2


def test_main_reports_checkpoint_metadata_errors_as_cli_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"train": [], "test": ["validation"]}]))
    writer = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(validation, "load_training_config", lambda path: TrainingConfig())
    monkeypatch.setattr(
        validation,
        "load_dataset_windows",
        lambda *args, **kwargs: (None, [SimpleNamespace(node_counts=[1, 1])]),
    )
    monkeypatch.setattr(validation, "FrameWindowDataset", lambda *args: [object()])
    monkeypatch.setattr(validation, "DataLoader", lambda *args, **kwargs: object())
    monkeypatch.setattr(validation, "get_checkpoint_paths", lambda *args: [])
    monkeypatch.setattr(validation, "SummaryWriter", lambda **kwargs: writer)
    monkeypatch.setattr(
        validation,
        "validate_checkpoints",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Checkpoint epoch mismatch: expected 1, found 2")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--data-dir",
            str(tmp_path),
            "--splits",
            str(splits),
            "--num-workers",
            "0",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        validation.main()

    assert error.value.code == 2
    assert "Checkpoint epoch mismatch" in capsys.readouterr().err
