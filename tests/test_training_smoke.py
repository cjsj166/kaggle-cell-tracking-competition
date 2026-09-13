"""Bounded dataset preparation and smoke CLI coverage."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import train_unet_transformer as training


def test_dataset_limit_applies_before_opening_files(monkeypatch, tmp_path):
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps([{"train": ["a", "b"], "test": ["c", "d"]}]))
    opened = []

    class LoadingComplete(Exception):
        pass

    def load(path, **kwargs):
        opened.append((path.name, kwargs["max_frames"]))
        if path.name == "c":
            raise LoadingComplete
        return None, []

    monkeypatch.setattr(training, "WEIGHTS_PATH", tmp_path / "weights")
    monkeypatch.setattr(training, "load_dataset_windows", load)
    with pytest.raises(LoadingComplete):
        training.train(tmp_path, 0, splits, max_datasets=1, max_frames=4)
    assert opened == [("a", 4), ("c", 4)]


def test_missing_split_is_created_before_training_reads_it(monkeypatch, tmp_path):
    splits = tmp_path / "splits.json"
    created = []

    class LoadingStarted(Exception):
        pass

    def create(data_dir, splits_file):
        created.append((data_dir, splits_file))
        splits_file.write_text(json.dumps([{"train": ["a"], "test": ["b"]}]))

    monkeypatch.setattr(training, "WEIGHTS_PATH", tmp_path / "weights")
    monkeypatch.setattr(training, "create_splits", create, raising=False)
    monkeypatch.setattr(
        training,
        "load_dataset_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(LoadingStarted),
    )

    with pytest.raises(LoadingStarted):
        training.train(tmp_path, 0, splits)

    assert created == [(tmp_path, splits)]


def test_cli_builds_training_config_and_forwards_limits(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--max-datasets",
            "1",
            "--unet-out-channels",
            "16",
            "--unet-layers",
            "16,32",
            "--downsample",
            "2,3,4",
            "--window-size",
            "3",
            "--det-loss-weight",
            "2.5",
            "--det-neg-weight",
            "0.25",
            "--pool-kernel-um",
            "7.5",
        ],
    )
    monkeypatch.setattr(training, "train", lambda **kwargs: calls.append(kwargs))
    training.main()
    assert calls[0]["max_datasets"] == 1
    assert "max_frames" not in calls[0]
    assert calls[0]["config"] == training.TrainingConfig(
        unet_out_channels=16,
        unet_layers=(16, 32),
        downsample=(2, 3, 4),
        window_size=3,
        det_loss_weight=2.5,
        det_neg_weight=0.25,
        pool_kernel_um=7.5,
    )


@pytest.mark.parametrize("args", [
    ["--max-datasets", "0"], ["--window-size", "1"],
])
def test_cli_rejects_invalid_limits(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["train", *args])
    with pytest.raises(SystemExit) as error:
        training.main()
    assert error.value.code == 2
