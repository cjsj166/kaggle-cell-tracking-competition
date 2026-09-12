#!/usr/bin/env python
"""Evaluate a range of training checkpoints and log TensorBoard curves.

This uses the same validation data, inference, scoring, and TensorBoard
logging functions as ``train_unet_transformer.py`` so an evaluated checkpoint
produces the same ``validation/*`` scalars as an automatic training-time
validation epoch.

Usage:
    uv run python scripts/validate_epochs.py \
        --data-dir data/train --split 0 \
        --start-epoch 1 --through-epoch 10
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from dataspec import DATASET_PATH, RUNS_PATH, WEIGHTS_PATH
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from train_unet_transformer import (
    DEFAULT_METHOD,
    FrameWindowData,
    FrameWindowDataset,
    VideoMeta,
    load_dataset_windows,
    log_validation_to_tensorboard,
    run_validation,
    score_tracking_predictions,
)

from tracking_cellmot.models import POS_EMBED_DIM, UNetNodeTransformer


def checkpoint_paths(
    checkpoint_dir: Path,
    start_epoch: int,
    through_epoch: int,
) -> list[tuple[int, Path]]:
    """Return every requested checkpoint in epoch order, failing on gaps."""
    if start_epoch < 1:
        raise ValueError("--start-epoch must be at least 1")
    if through_epoch < start_epoch:
        raise ValueError(
            "--through-epoch must be greater than or equal to --start-epoch"
        )

    checkpoints = [
        (epoch, checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth")
        for epoch in range(start_epoch, through_epoch + 1)
    ]
    missing = [path for _, path in checkpoints if not path.is_file()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing requested checkpoints:\n  {formatted}")
    return checkpoints


def resolve_fold_files(
    data_dir: Path,
    splits_file: Path,
    fold: int,
) -> tuple[list[Path], list[Path]]:
    """Resolve train and validation files exactly as the training script does."""
    if splits_file.exists():
        folds = json.loads(splits_file.read_text())
    else:
        stems = sorted(
            path.name[:-5]
            for path in data_dir.glob("*.zarr")
            if (data_dir / f"{path.name[:-5]}.geff").exists()
        )
        random.Random(0).shuffle(stems)
        n_val = max(1, len(stems) // 10)
        folds = [{"split": 0, "train": stems[n_val:], "test": stems[:n_val]}]
        print(
            f"No splits file at {splits_file}; generated seed-0 split "
            f"({len(stems) - n_val} train / {n_val} val)."
        )

    try:
        fold_data = folds[fold]
    except IndexError as exc:
        raise ValueError(f"Split {fold} is not present in {splits_file}") from exc
    return (
        [data_dir / name for name in fold_data["train"]],
        [data_dir / name for name in fold_data["test"]],
    )


def load_validation_data(
    train_files: list[Path],
    validation_files: list[Path],
    *,
    window_size: int,
    downsample: tuple[int, ...],
) -> tuple[list[tuple[VideoMeta, list[FrameWindowData]]], int]:
    """Load validation metadata and reproduce training's shared padding size."""

    def _load(
        files: list[Path], desc: str,
    ) -> list[tuple[VideoMeta, list[FrameWindowData]]]:
        data = []
        for path in tqdm(files, desc=desc, disable=False):
            data.append(
                load_dataset_windows(
                    path,
                    window_size=window_size,
                    downsample=downsample,
                )
            )
        return data

    train_video_data = _load(train_files, "train metadata")
    validation_video_data = _load(validation_files, "validation metadata")
    all_windows = [
        window
        for _, windows in train_video_data + validation_video_data
        for window in windows
    ]
    if not all_windows:
        raise ValueError("The selected split contains no valid frame windows")
    max_nodes = max(max(window.node_counts) for window in all_windows)
    return validation_video_data, max_nodes


def load_model_config(output_dir: Path) -> dict:
    """Load the architecture and validation settings saved by training."""
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing model config: {config_path}. "
            "Checkpoint validation cannot safely reconstruct the training model."
        )
    config = json.loads(config_path.read_text())
    required = {
        "unet_out_channels",
        "unet_layers",
        "downsample",
        "window_size",
        "pool_kernel_um",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Model config is missing required keys: {', '.join(missing)}")
    return config


def load_checkpoint_model(
    checkpoint_path: Path,
    expected_epoch: int,
    config: dict,
    device: torch.device,
    *,
    data_parallel: bool,
) -> UNetNodeTransformer:
    """Reconstruct a model and load one canonical epoch checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Not a training checkpoint: {checkpoint_path}")
    actual_epoch = checkpoint.get("epoch")
    if actual_epoch != expected_epoch:
        raise ValueError(
            f"Checkpoint epoch mismatch for {checkpoint_path}: "
            f"expected {expected_epoch}, found {actual_epoch}"
        )

    model = UNetNodeTransformer(
        unet_out_channels=int(config["unet_out_channels"]),
        unet_layers=[int(value) for value in config["unet_layers"]],
        pos_feat_dim=4 * POS_EMBED_DIM,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model.unet = nn.DataParallel(model.unet)
    return model


def validate_checkpoints(
    checkpoints: list[tuple[int, Path]],
    config: dict,
    loader: DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    *,
    det_loss_weight: float,
    det_neg_weight: float,
    data_parallel: bool,
) -> None:
    """Evaluate checkpoints and append one matching validation point per epoch."""
    for epoch, checkpoint_path in checkpoints:
        print(f"Evaluating epoch {epoch}: {checkpoint_path}")
        model = load_checkpoint_model(
            checkpoint_path,
            epoch,
            config,
            device,
            data_parallel=data_parallel,
        )
        predictions = {}
        losses = run_validation(
            model,
            loader,
            device,
            det_loss_weight=det_loss_weight,
            det_neg_weight=det_neg_weight,
            pool_kernel_um=float(config["pool_kernel_um"]),
            predictions=predictions,
        )
        metrics = score_tracking_predictions(predictions)
        log_validation_to_tensorboard(writer, losses, metrics, epoch)
        writer.flush()
        print(
            f"  epoch={epoch} score={metrics['score']:.4f} "
            f"loss={losses.loss:.4f} edge_jaccard={metrics['edge_jaccard']:.4f} "
            f"division_jaccard={metrics['division_jaccard']:.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate epoch checkpoints with training-time validation logging."
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--data-dir", type=Path, default=DATASET_PATH)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--through-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--det-loss-weight", type=float, default=1e0)
    parser.add_argument("--det-neg-weight", type=float, default=1e-2)
    parser.add_argument(
        "--single-gpu",
        dest="data_parallel",
        action="store_false",
        default=True,
        help="Disable multi-GPU validation.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="TensorBoard run directory. Defaults to the same timestamp layout as training.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    data_dir = args.data_dir
    splits_file = args.splits or data_dir / "dataset_splits.json"
    output_dir = WEIGHTS_PATH / args.method / f"split_{args.split}"
    try:
        checkpoints = checkpoint_paths(
            output_dir / "checkpoints",
            args.start_epoch,
            args.through_epoch,
        )
        config = load_model_config(output_dir)
        train_files, validation_files = resolve_fold_files(
            data_dir,
            splits_file,
            args.split,
        )
        validation_video_data, max_nodes = load_validation_data(
            train_files,
            validation_files,
            window_size=int(config["window_size"]),
            downsample=tuple(int(value) for value in config["downsample"]),
        )
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    validation_ds = FrameWindowDataset(validation_video_data, max_nodes=max_nodes)
    validation_loader = DataLoader(
        validation_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.run_dir or (
        RUNS_PATH
        / args.method
        / f"split_{args.split}"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    print(f"Fold {args.split}: {len(train_files)} train, {len(validation_files)} validation")
    print(f"Using device: {device}")
    print(f"TensorBoard logs: {run_dir}")

    writer = SummaryWriter(log_dir=str(run_dir))
    try:
        validate_checkpoints(
            checkpoints,
            config,
            validation_loader,
            writer,
            device,
            det_loss_weight=args.det_loss_weight,
            det_neg_weight=args.det_neg_weight,
            data_parallel=args.data_parallel,
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
