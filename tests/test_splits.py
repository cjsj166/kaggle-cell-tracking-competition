import json
from pathlib import Path

from tracking_cellmot.splits import create_splits


def test_create_splits_writes_deterministic_seed_zero_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index in range(10):
        (data_dir / f"sample-{index}.zarr").mkdir()
        (data_dir / f"sample-{index}.geff").touch()
    (data_dir / "unpaired.zarr").mkdir()

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    create_splits(data_dir, first_path)
    create_splits(data_dir, second_path)

    folds = json.loads(first_path.read_text())
    assert folds == [
        {
            "split": 0,
            "train": [
                "sample-8",
                "sample-1",
                "sample-5",
                "sample-3",
                "sample-4",
                "sample-2",
                "sample-0",
                "sample-9",
                "sample-6",
            ],
            "test": ["sample-7"],
        }
    ]
    assert second_path.read_text() == first_path.read_text()
