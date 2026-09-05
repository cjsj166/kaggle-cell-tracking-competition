#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Later arguments override these defaults.
uv run python scripts/train_unet_transformer.py \
    --method unet_transformer_smoke --split 0 \
    --epochs 1 --batch-size 1 --max-iters 2 --num-workers 0 \
    --max-datasets 1 --max-frames 4 --single-gpu "$@"
