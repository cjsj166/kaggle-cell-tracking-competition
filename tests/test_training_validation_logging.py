"""Validation loss, competition metric, and TensorBoard logging tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import train_unet_transformer as training
from tracking_cellmot.edge_prediction import build_graph


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


@pytest.mark.parametrize("valid_pairs", [0, 1, 2])
def test_run_validation_returns_all_three_validation_losses(
    monkeypatch: pytest.MonkeyPatch, valid_pairs: int,
) -> None:
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
        "video_id": [f"video-{i}" for i in range(batch_size)],
        "t_start": torch.zeros(batch_size, dtype=torch.long),
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
    pair_results = iter([(3.0, 1, 2)] * valid_pairs + [(0.0, 0, 0)] * (batch_size - valid_pairs))
    monkeypatch.setattr(training, "_evaluate_pair", lambda *args: next(pair_results))

    result = training.run_validation(
        _ValidationModel(), [batch], torch.device("cpu"), det_loss_weight=0.5, det_neg_weight=0.1,
        predictions={},
    )

    expected_edge_loss = 3.0 if valid_pairs else 0.0
    assert result.edge_loss == pytest.approx(expected_edge_loss)
    assert result.det_loss == pytest.approx(2.0)
    assert result.loss == pytest.approx(expected_edge_loss + 1.0)
    assert result.edge_accuracy == pytest.approx(0.5 if valid_pairs else 0.0)
    assert result.detection_node_recall == pytest.approx(1.0)


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_evaluate_pair_without_active_rows(shape: tuple[int, int]) -> None:
    assert training._evaluate_pair(torch.zeros(shape), torch.zeros(shape)) == (0.0, 0, 0)


@pytest.mark.parametrize("empty_pairs", [0, 1, 5])
def test_empty_pairs_do_not_dilute_real_validation_loss(
    monkeypatch: pytest.MonkeyPatch, empty_pairs: int,
) -> None:
    batch_size = 1 + empty_pairs
    targets = torch.zeros(batch_size, 1, 2, 2)
    targets[0, 0] = torch.eye(2)
    batch = {
        "imgs": torch.zeros(batch_size, 2, 1, 1, 1),
        "coords": torch.zeros(batch_size, 2, 2, 3),
        "pos_feats": torch.zeros(batch_size, 2, 2, 32),
        "masks": torch.ones(batch_size, 2, 2, dtype=torch.bool),
        "targets": targets,
        "image_shape": torch.tensor([[2, 1, 1, 1]] * batch_size),
        "voxel_size": torch.ones(batch_size, 3),
        "downsample": torch.ones(batch_size, 3),
        "video_id": [f"video-{i}" for i in range(batch_size)],
        "t_start": torch.zeros(batch_size, dtype=torch.long),
    }

    def fake_detect_and_match(det_logits, gt_coords, mask, *args, **kwargs):
        return (
            gt_coords, torch.zeros(batch_size, 2, 32), mask,
            [torch.arange(2) for _ in range(batch_size)],
        )

    monkeypatch.setattr(training, "detect_and_match", fake_detect_and_match)
    monkeypatch.setattr(training, "compute_detection_loss", lambda *args: torch.tensor(2.0))
    result = training.run_validation(
        _ValidationModel(), [batch], torch.device("cpu"),
        det_loss_weight=0.5, det_neg_weight=0.1, predictions={},
    )

    # With zero logits and two candidates, focal BCE is 0.25 * log(2).
    # Adding unannotated pairs must not divide this by (1 + empty_pairs).
    expected = 0.25 * np.log(2)
    assert result.edge_loss == pytest.approx(expected)
    assert result.det_loss == pytest.approx(2.0)
    assert result.loss == pytest.approx(expected + 1.0)


@pytest.mark.parametrize("wrap_unet", [False, True])
def test_resume_restores_cpu_model_and_optimizer_before_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    wrap_unet: bool,
) -> None:
    events = []

    class TinyModel(torch.nn.Module):
        def __init__(self, unet=None, **kwargs):
            super().__init__()
            self.unet = unet if unet is not None else torch.nn.Linear(1, 1)

        def load_state_dict(self, state, **kwargs):
            assert not isinstance(self.unet, torch.nn.DataParallel)
            assert all(value.device.type == "cpu" for value in state.values())
            events.append("load")
            return super().load_state_dict(state, **kwargs)

        def to(self, *args, **kwargs):
            events.append("to")
            return super().to(*args, **kwargs)

    source = TinyModel(torch.nn.Linear(1, 1))
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.012)
    source.unet(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    expected = {key: value.clone() for key, value in source.state_dict().items()}
    resume = tmp_path / "resume.pth"
    torch.save({
        "model_state_dict": expected,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 1,
        "global_step": 7,
    }, resume)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "WEIGHTS_PATH", tmp_path / "weights")
    monkeypatch.setattr(training, "RUNS_PATH", tmp_path / "runs")
    monkeypatch.setattr(training, "UNetNodeTransformer", TinyModel)
    monkeypatch.setattr(training, "load_dataset_windows", lambda *args, **kwargs: (
        None, [SimpleNamespace(node_counts=[1, 1])],
    ))
    monkeypatch.setattr(training, "FrameWindowDataset", lambda *args, **kwargs: [0])

    def fake_train_epoch(model, loader, restored_optimizer, *args, **kwargs):
        assert events == ["load", "to"]
        assert kwargs["global_step"] == 7
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key])
        assert restored_optimizer.param_groups[0]["lr"] == 0.012
        for actual, original in zip(restored_optimizer.state.values(), optimizer.state.values(), strict=True):
            for key in original:
                torch.testing.assert_close(actual[key], original[key])
        if wrap_unet:
            model.unet = torch.nn.DataParallel(model.unet)
        return 1.0, 2.0, 8

    monkeypatch.setattr(training, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(training, "run_validation", lambda *args, **kwargs: (
        training.ValidationLosses(1.0, 2.0, 3.0, 1.0, 1.0)
    ))
    monkeypatch.setattr(training, "score_tracking_predictions", lambda *args, **kwargs: {
        "score": 0.25,
        "edge_jaccard": 1.0,
        "adj_edge_jaccard": 1.0,
        "division_jaccard": 1.0,
        "total_node_ratio": 1.0,
    })

    model = training.train(
        data_dir=tmp_path, fold=0, splits_file=tmp_path / "unused.json",
        debug_video=tmp_path / "video", resume=resume, n_epochs=2, num_workers=0,
    )

    assert events == ["load", "to", "load", "to"]
    saved = torch.load(
        tmp_path / "weights/unet_transformer/split_0/checkpoints/checkpoint_epoch_0002.pth",
        map_location="cpu", weights_only=True,
    )
    assert saved["epoch"] == 2
    assert saved["global_step"] == 8
    assert "Best competition score: 0.2500" in capsys.readouterr().out
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])


@pytest.mark.parametrize("estimates", [[1.0], [2.0], [4.0], [2.0, 4.0]])
def test_full_video_metrics_include_counts_and_node_ratio(
    monkeypatch: pytest.MonkeyPatch, estimates: list[float],
) -> None:
    coords = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int16)
    edges = [(0, 1, 0.9, 0.0)]
    gt_graph = build_graph(coords, edges)

    monkeypatch.setattr(
        training,
        "open_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            tracks=gt_graph,
            scale=(1.0, 1.0, 1.0),
            image_shape=(2, 1, 1, 1),
        ),
    )
    estimate_iter = iter(estimates)
    monkeypatch.setattr(training, "_read_estimated_n_total", lambda path: next(estimate_iter))

    with pytest.warns(UserWarning, match="No divisions"):
        result = training.score_tracking_predictions(
            {f"video_{i}": SimpleNamespace(
                result=lambda: (coords, edges), seen_frames={0, 1}, seen_pairs={(0, 1)},
            ) for i in range(len(estimates))},
        )

    assert result["edge_jaccard"] == pytest.approx(1.0)
    expected_ratio = (2 * len(estimates) - sum(estimates)) / sum(estimates)
    assert result["adj_edge_jaccard"] == pytest.approx(
        sum(1 - 0.1 * (2 - n) / n for n in estimates) / len(estimates),
    )
    assert np.isnan(result["division_jaccard"])
    assert result["edge_tp"] == len(estimates)
    assert result["edge_fp"] == 0
    assert result["edge_fn"] == 0
    assert result["num_pred_nodes"] == 2 * len(estimates)
    assert result["estimated_num_nodes"] == sum(estimates)
    assert result["total_node_ratio"] == pytest.approx(expected_ratio)
    assert "node_count_adjustment" not in result
    assert "over_detection_penalty" not in result


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
        "total_node_ratio": 0.1,
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
