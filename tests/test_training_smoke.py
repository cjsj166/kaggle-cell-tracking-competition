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

    def load(path, **kwargs):
        opened.append((path.name, kwargs["max_frames"]))
        return None, []

    monkeypatch.setattr(training, "WEIGHTS_PATH", tmp_path / "weights")
    monkeypatch.setattr(training, "load_dataset_windows", load)
    with pytest.raises(ValueError, match="No usable train windows"):
        training.train(tmp_path, 0, splits, max_datasets=1, max_frames=4)
    assert opened == [("a", 4), ("c", 4)]


def test_cli_forwards_limits(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["train", "--max-datasets", "1"])
    monkeypatch.setattr(training, "train", lambda **kwargs: calls.append(kwargs))
    training.main()
    assert calls[0]["max_datasets"] == 1
    assert "max_frames" not in calls[0]


@pytest.mark.parametrize("args", [
    ["--max-datasets", "0"], ["--window-size", "1"],
])
def test_cli_rejects_invalid_limits(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["train", *args])
    with pytest.raises(SystemExit) as error:
        training.main()
    assert error.value.code == 2
