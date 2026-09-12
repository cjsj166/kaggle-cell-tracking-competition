"""Regression coverage for window inference shared by loss and movie metrics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_unet_transformer as training
from tracking_cellmot.models import UNetNodeTransformer
from tracking_cellmot.edge_prediction import VideoPredictionAccumulator, build_graph


class RecordingModel:
    _index_features = UNetNodeTransformer._index_features

    def __init__(self):
        self.encodes = 0
        self.pairs = []

    def eval(self):
        pass

    def encode(self, imgs):
        self.encodes += 1
        features = imgs.unsqueeze(2)
        return features, list(features.unbind(dim=1))

    def predict_edges(self, src_features, dst_features, src, dst, *args):
        self.pairs.append((src.clone(), src_features.clone()))
        return torch.zeros(src.shape[0], src.shape[1], dst.shape[1])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_run_validation_reuses_unet_and_preserves_loss_across_batches_and_movies(batch_size):
    batches = []
    for video, start in [("a", 0), ("a", 1), ("b", 0)]:
        imgs = torch.full((1, 2, 1, 1, 3), -1.)
        imgs[..., start] = 1.
        coords = torch.zeros(1, 2, 1, 3)
        coords[..., 2] = start
        batches.append({
            "imgs": imgs, "coords": coords,
            "pos_feats": torch.zeros(1, 2, 1, 32),
            "masks": torch.ones(1, 2, 1, dtype=torch.bool),
            "targets": torch.ones(1, 1, 1, 1),
            "image_shape": torch.tensor([[3, 1, 1, 3]]),
            "voxel_size": torch.ones(1, 3), "downsample": torch.ones(1, 3),
            "video_id": [video], "t_start": torch.tensor([start]),
        })
    batches = [
        {key: (sum([sample[key] for sample in group], []) if key == "video_id"
               else torch.cat([sample[key] for sample in group]))
         for key in group[0]}
        for offset in range(0, len(batches), batch_size)
        for group in [batches[offset:offset + batch_size]]
    ]
    reference = training.run_validation(
        RecordingModel(), batches, torch.device("cpu"), 0.1, 0.01,
        pool_kernel_um=1., predictions={},
    )
    model = RecordingModel()
    predictions = {}
    actual = training.run_validation(
        model, batches, torch.device("cpu"), 0.1, 0.01,
        pool_kernel_um=1., predictions=predictions,
    )
    assert actual == reference
    assert model.encodes == len(batches)
    assert predictions["a"].seen_frames == {0, 1, 2}
    assert predictions["b"].seen_frames == {0, 1}
    assert len(predictions["a"].result()[0]) == 3
    assert len(predictions["b"].result()[0]) == 2


def test_metrics_exclude_unvisited_frames_and_transitions(monkeypatch):
    coords = np.array([[t, 0, 0, 0] for t in range(6)])
    gt = build_graph(coords, [(t, t + 1, 1., 0.) for t in range(5)])
    # Frame 5 and pairs 1->2, 4->5 were not evaluated.
    selected = coords[:5]
    edges = [(0, 1, 1., 0.), (2, 3, 1., 0.), (3, 4, 1., 0.)]
    monkeypatch.setattr(training, "open_dataset", lambda *a, **k: SimpleNamespace(
        tracks=gt, scale=(1., 1., 1.), image_shape=(6, 1, 1, 1),
    ))
    def no_full_movie_estimate(*args):
        pytest.fail("A whole-movie estimate cannot be used for partial coverage")
    monkeypatch.setattr(training, "_read_estimated_n_total", no_full_movie_estimate)
    with pytest.warns(UserWarning, match="No divisions"):
        result = training.score_tracking_predictions({"movie": SimpleNamespace(
            result=lambda: (selected, edges), seen_frames=set(range(5)),
            seen_pairs={(0, 1), (2, 3), (3, 4)},
        )})
    assert result["edge_tp"] == 3
    assert result["edge_fn"] == 0
    assert result["edge_jaccard"] == 1.
    assert np.isnan(result["total_node_ratio"])


def test_validation_requires_explicit_loss_weights():
    with pytest.raises(TypeError):
        training.run_validation(RecordingModel(), [], torch.device("cpu"))


def test_validation_requires_prediction_accumulator():
    with pytest.raises(TypeError):
        training.run_validation(
            RecordingModel(), [], torch.device("cpu"), 0.1, 0.01,
        )


def test_empty_detections_keep_frame_coverage_and_gt_recall(monkeypatch):
    acc = VideoPredictionAccumulator((1, 1, 1))
    acc.add_window(RecordingModel(), torch.zeros(1, 2, 1, 1, 1, 1),
                   [np.empty((0, 4)), np.empty((0, 4))], [0, 1], (3, 1, 1, 1))
    gt = build_graph(np.array([[t, 0, 0, 0] for t in range(3)]),
                     [(0, 1, 1., 0.), (1, 2, 1., 0.)])
    monkeypatch.setattr(training, "open_dataset", lambda *a, **k: SimpleNamespace(
        tracks=gt, image_shape=(3, 1, 1, 1), scale=(1., 1., 1.),
    ))
    with pytest.warns(UserWarning):
        result = training.score_tracking_predictions({"movie": acc})
    assert result["edge_fn"] == 1
    assert result["edge_tp"] == 0
    assert result["num_pred_nodes"] == 0
    assert result["node_recall"] == 0
