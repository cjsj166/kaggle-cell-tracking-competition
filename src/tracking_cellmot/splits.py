"""Training dataset split creation."""

from __future__ import annotations

import json
import random
from pathlib import Path


def create_splits(data_dir: Path, splits_file: Path) -> None:
    """Create and persist the deterministic default train/validation split."""
    stems = sorted(path.name[:-5] for path in data_dir.glob("*.zarr") if (data_dir / f"{path.name[:-5]}.geff").exists())
    random.Random(0).shuffle(stems)
    n_validation = max(1, len(stems) // 10)
    folds = [
        {
            "split": 0,
            "train": stems[n_validation:],
            "test": stems[:n_validation],
        }
    ]
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    splits_file.write_text(json.dumps(folds, indent=2))
