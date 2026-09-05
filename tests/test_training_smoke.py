"""Bounded dataset preparation and smoke CLI coverage."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import train_unet_transformer as training


@pytest.mark.parametrize("limit,expected", [(None, 10), (4, 4), (20, 10)])
def test_frame_limit_bounds_window_preparation(monkeypatch, limit, expected):
    graph = SimpleNamespace()
    graph.filter = lambda *args: SimpleNamespace(subgraph=lambda: graph)
    opened = []
    starts = []

    def open_dataset(*args, **kwargs):
        opened.append(kwargs)
        return SimpleNamespace(
            image_shape=(10, 8, 8, 8), tracks=graph, scale=(1, 1, 1),
            zarr_path=Path("video.zarr"), quantiles={"0.001": 0, "0.999": 1},
        )

    monkeypatch.setattr(training, "open_dataset", open_dataset)
    monkeypatch.setattr(training, "get_window_data", lambda graph, shape, t, *a, **kw: starts.append(t))
    meta, _ = training.load_dataset_windows(Path("video"), max_frames=limit)
    assert meta.image_shape[0] == expected
    assert starts == list(range(expected - 1))
    assert opened[0]["load_image"] is False


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
    monkeypatch.setattr(sys, "argv", ["train", "--max-datasets", "1", "--max-frames", "4"])
    monkeypatch.setattr(training, "train", lambda **kwargs: calls.append(kwargs))
    training.main()
    assert calls[0]["max_datasets"] == 1
    assert calls[0]["max_frames"] == 4


@pytest.mark.parametrize("args", [
    ["--max-datasets", "0"], ["--max-frames", "1"], ["--window-size", "1"],
])
def test_cli_rejects_invalid_limits(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["train", *args])
    with pytest.raises(SystemExit) as error:
        training.main()
    assert error.value.code == 2
