"""Validation loss, competition metric, and TensorBoard logging tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import evaluate as evaluate_script
import predict_unet_transformer as prediction
import train_unet_transformer as training


class _ValidationModel:
    def eval(self) -> None:
        pass

    def encode(self, imgs: torch.Tensor):
        batch_size, window_size = imgs.shape[:2]
        features = torch.zeros(batch_size, window_size, 1, 1, 1, 1)
        logits = [torch.zeros(batch_size, 1, 1, 1, 1) for _ in range(window_size)]
        return features, logits

    def _index_features(self, features, coords, mask):
        return torch.zeros(coords.shape[0], coords.shape[1], 1)

    def predict_edges(self, source_features, target_features, *args):
        return torch.zeros(
            source_features.shape[0],
            source_features.shape[1],
            target_features.shape[1],
        )


def test_evaluate_returns_all_three_validation_losses(monkeypatch: pytest.MonkeyPatch) -> None:
    batch_size, window_size = 2, 2
    batch = {
        "imgs": torch.zeros(batch_size, window_size, 1, 1, 1),
        "coords": torch.zeros(batch_size, window_size, 1, 3),
        "pos_feats": torch.zeros(batch_size, window_size, 1, 32),
        "masks": torch.ones(batch_size, window_size, 1, dtype=torch.bool),
        "targets": torch.ones(batch_size, window_size - 1, 1, 1),
        "image_shape": torch.tensor([[2, 1, 1, 1]] * batch_size),
        "voxel_size": torch.ones(batch_size, 3),
        "downsample": torch.ones(batch_size, 3),
    }

    def fake_detect_and_match(det_logits, gt_coords, mask, *args, **kwargs):
        batch = det_logits.shape[0]
        coords = torch.zeros(batch, 1, 3)
        pos = torch.zeros(batch, 1, 32)
        detected = torch.ones(batch, 1, dtype=torch.bool)
        matches = [torch.tensor([0]) for _ in range(batch)]
        return coords, pos, detected, matches

    monkeypatch.setattr(training, "detect_and_match", fake_detect_and_match)
    monkeypatch.setattr(
        training,
        "compute_detection_loss",
        lambda *args, **kwargs: torch.tensor(2.0),
    )
    monkeypatch.setattr(training, "_evaluate_pair", lambda *args: (3.0, 1, 2))

    result = training.evaluate(
        _ValidationModel(), [batch], torch.device("cpu"), det_loss_weight=0.5,
    )

    assert result.edge_loss == pytest.approx(3.0)
    assert result.det_loss == pytest.approx(2.0)
    assert result.loss == pytest.approx(4.0)
    assert result.edge_accuracy == pytest.approx(0.5)
    assert result.detection_node_recall == pytest.approx(1.0)


def test_full_video_metrics_include_counts_and_over_detection_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int16)
    edges = [(0, 1, 0.9, 0.0)]
    gt_graph = prediction.build_graph(coords, edges)

    monkeypatch.setattr(prediction, "predict_video", lambda *args, **kwargs: (coords, edges))
    monkeypatch.setattr(
        training,
        "open_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            tracks=gt_graph,
            scale=(1.0, 1.0, 1.0),
            image_shape=(2, 1, 1, 1),
        ),
    )
    monkeypatch.setattr(evaluate_script, "_read_estimated_n_total", lambda path: 2.0)

    with pytest.warns(UserWarning, match="No divisions"):
        result = training.evaluate_tracking_metrics(
            _ValidationModel(),
            [Path("video")],
            torch.device("cpu"),
            window_size=2,
            downsample=(1, 1, 1),
            pool_kernel_um=5.0,
        )

    assert result["edge_jaccard"] == pytest.approx(1.0)
    assert result["adj_edge_jaccard"] == pytest.approx(1.0)
    assert np.isnan(result["division_jaccard"])
    assert result["edge_tp"] == 1
    assert result["edge_fp"] == 0
    assert result["edge_fn"] == 0
    assert result["num_pred_nodes"] == 2
    assert result["total_node_ratio"] == pytest.approx(0.0)
    assert result["node_count_adjustment"] == pytest.approx(1.0)
    assert result["over_detection_penalty"] == pytest.approx(0.0)


def test_tensorboard_logging_writes_every_validation_value() -> None:
    calls: list[tuple[str, float, int]] = []
    writer = SimpleNamespace(
        add_scalar=lambda tag, value, step: calls.append((tag, value, step)),
    )
    losses = training.ValidationLosses(1.0, 2.0, 3.0, 0.75, 0.8)
    metrics = {
        "score": 0.9,
        "edge_jaccard": 0.8,
        "adj_edge_jaccard": 0.7,
        "division_jaccard": 1.0,
        "over_detection_penalty": 0.1,
    }

    training.log_validation_to_tensorboard(writer, losses, metrics, epoch=4)

    tags = {tag for tag, _, _ in calls}
    assert tags == {
        "validation/edge_loss",
        "validation/det_loss",
        "validation/loss",
        "validation/edge_accuracy",
        "validation/detection_node_recall",
        *(f"validation/{name}" for name in metrics),
    }
    assert all(step == 4 for _, _, step in calls)
