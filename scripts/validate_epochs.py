#!/usr/bin/env python
"""Evaluate a range of training checkpoints and log TensorBoard curves.

This uses the configuration, validation data, inference, scoring, and
TensorBoard logging semantics saved or used by ``train_unet_transformer.py``.

Usage:
    uv run python scripts/validate_epochs.py \
        --data-dir data/train --split 0 \
        --start-epoch 1 --through-epoch 10
"""

from __future__ import annotations

import argparse
import json
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
    FrameWindowDataset,
    load_dataset_windows,
    log_validation_to_tensorboard,
    run_validation,
    score_tracking_predictions,
)

from tracking_cellmot.checkpoints import load_training_checkpoint
from tracking_cellmot.models import POS_EMBED_DIM, UNetNodeTransformer
from tracking_cellmot.training_config import TrainingConfig, load_training_config


def get_checkpoint_paths(
    checkpoint_dir: Path,
    start_epoch: int,
    through_epoch: int,
) -> list[tuple[int, Path]]:
    """Return every requested checkpoint in epoch order, failing on gaps."""
    if start_epoch < 1:
        raise ValueError("--start-epoch must be at least 1")
    if through_epoch < start_epoch:
        raise ValueError("--through-epoch must be greater than or equal to --start-epoch")

    checkpoints = [
        (epoch, checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth") for epoch in range(start_epoch, through_epoch + 1)
    ]
    missing = [path for _, path in checkpoints if not path.is_file()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing requested checkpoints:\n  {formatted}")
    return checkpoints


def validate_checkpoints(
    checkpoints: list[tuple[int, Path]],
    config: TrainingConfig,
    loader: DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    *,
    data_parallel: bool,
) -> None:
    """Evaluate checkpoints and append one matching validation point per epoch."""
    for epoch, checkpoint_path in checkpoints:
        print(f"Evaluating epoch {epoch}: {checkpoint_path}")
        model = UNetNodeTransformer(
            unet_out_channels=config.unet_out_channels,
            unet_layers=config.unet_layers,
            pos_feat_dim=4 * POS_EMBED_DIM,
        )
        checkpoint_info = load_training_checkpoint(checkpoint_path, model)
        if checkpoint_info.epoch != epoch:
            raise ValueError(
                f"Checkpoint epoch mismatch for {checkpoint_path}: expected {epoch}, found {checkpoint_info.epoch}"
            )

        model.to(device)
        if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
            model.unet = nn.DataParallel(model.unet)

        predictions = {}
        losses = run_validation(
            model,
            loader,
            device,
            det_loss_weight=config.det_loss_weight,
            det_neg_weight=config.det_neg_weight,
            pool_kernel_um=config.pool_kernel_um,
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
    parser = argparse.ArgumentParser(description="Evaluate epoch checkpoints with training-time validation logging.")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--data-dir", type=Path, default=DATASET_PATH)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--through-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
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
        config = load_training_config(output_dir / "config.json")
        folds = json.loads(splits_file.read_text())
        try:
            fold_data = folds[args.split]
        except IndexError as exc:
            raise ValueError(f"Split {args.split} is not present in {splits_file}") from exc
        validation_files = [data_dir / name for name in fold_data["test"]]
        validation_video_data = [
            load_dataset_windows(
                path,
                window_size=config.window_size,
                downsample=config.downsample,
            )
            for path in tqdm(validation_files, desc="validation", disable=False)
        ]
        validation_ds = FrameWindowDataset(validation_video_data)
        checkpoints = get_checkpoint_paths(
            output_dir / "checkpoints",
            args.start_epoch,
            args.through_epoch,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

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
        RUNS_PATH / args.method / f"split_{args.split}" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    print(f"Fold {args.split}: {len(validation_files)} validation datasets")
    print(f"Using device: {device}")
    print(f"TensorBoard logs: {run_dir}")

    writer = SummaryWriter(log_dir=str(run_dir))
    try:
        try:
            validate_checkpoints(
                checkpoints,
                config,
                validation_loader,
                writer,
                device,
                data_parallel=args.data_parallel,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
