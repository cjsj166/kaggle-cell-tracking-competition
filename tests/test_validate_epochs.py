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


def test_checkpoint_paths_returns_inclusive_order(tmp_path: Path) -> None:
    for epoch in range(3, 6):
        (tmp_path / f"checkpoint_epoch_{epoch:04d}.pth").touch()

    checkpoints = validation.checkpoint_paths(tmp_path, 3, 5)

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
def test_checkpoint_paths_rejects_invalid_ranges(
    tmp_path: Path,
    start_epoch: int,
    through_epoch: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validation.checkpoint_paths(tmp_path, start_epoch, through_epoch)


def test_checkpoint_paths_reports_every_missing_epoch(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_epoch_0002.pth").touch()

    with pytest.raises(FileNotFoundError) as error:
        validation.checkpoint_paths(tmp_path, 1, 3)

    assert "checkpoint_epoch_0001.pth" in str(error.value)
    assert "checkpoint_epoch_0003.pth" in str(error.value)


def test_resolve_fold_files_uses_training_split_layout(tmp_path: Path) -> None:
    splits = tmp_path / "dataset_splits.json"
    splits.write_text(json.dumps([
        {"train": ["train-a"], "test": ["val-a"]},
        {"train": ["train-b"], "test": ["val-b"]},
    ]))

    train_files, validation_files = validation.resolve_fold_files(tmp_path, splits, 1)

    assert train_files == [tmp_path / "train-b"]
    assert validation_files == [tmp_path / "val-b"]


def test_load_validation_data_uses_training_and_validation_for_padding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    counts = {"train": [2, 7], "validation": [3, 4]}

    def load(path: Path, **kwargs):
        assert kwargs == {"window_size": 2, "downsample": (1, 4, 4)}
        windows = [SimpleNamespace(node_counts=[count]) for count in counts[path.name]]
        return SimpleNamespace(zarr_path=path), windows

    monkeypatch.setattr(validation, "load_dataset_windows", load)
    validation_data, max_nodes = validation.load_validation_data(
        [tmp_path / "train"],
        [tmp_path / "validation"],
        window_size=2,
        downsample=(1, 4, 4),
    )

    assert max_nodes == 7
    assert validation_data[0][0].zarr_path == tmp_path / "validation"


def test_load_checkpoint_model_rejects_epoch_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_epoch_0001.pth"
    torch.save({"epoch": 2, "model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="expected 1, found 2"):
        validation.load_checkpoint_model(
            checkpoint,
            1,
            {},
            torch.device("cpu"),
            data_parallel=False,
        )


def test_validate_checkpoints_logs_one_point_at_each_checkpoint_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoints = [(3, tmp_path / "three.pth"), (4, tmp_path / "four.pth")]
    loaded = []
    prediction_objects = []
    logged = []

    def load_model(path, epoch, config, device, *, data_parallel):
        loaded.append((path, epoch, config, device, data_parallel))
        return SimpleNamespace(epoch=epoch)

    def run(model, loader, device, **kwargs):
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

    monkeypatch.setattr(validation, "load_checkpoint_model", load_model)
    monkeypatch.setattr(validation, "run_validation", run)
    monkeypatch.setattr(validation, "score_tracking_predictions", score)
    monkeypatch.setattr(
        validation,
        "log_validation_to_tensorboard",
        lambda writer, losses, metrics, epoch: logged.append((losses, metrics, epoch)),
    )
    writer = SimpleNamespace(flush_calls=0)

    def flush():
        writer.flush_calls += 1

    writer.flush = flush
    validation.validate_checkpoints(
        checkpoints,
        {"pool_kernel_um": 5.0},
        loader=object(),
        writer=writer,
        device=torch.device("cpu"),
        det_loss_weight=1.0,
        det_neg_weight=0.01,
        data_parallel=False,
    )

    assert [entry[1] for entry in loaded] == [3, 4]
    assert [entry[2] for entry in logged] == [3, 4]
    assert prediction_objects[0] is not prediction_objects[1]
    assert writer.flush_calls == 2
