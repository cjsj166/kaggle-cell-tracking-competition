"""Regression coverage for assembling predictions from temporal windows."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import predict_unet_transformer as prediction
from tracking_cellmot.models import UNetNodeTransformer
from tracking_cellmot.prediction import VideoPredictionAccumulator


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


def test_accumulator_keeps_first_coordinates_but_uses_current_features():
    model = RecordingModel()
    acc = VideoPredictionAccumulator((1, 1, 2))
    first = [np.array([[0, 0, 0, 0]]), np.array([[1, 0, 0, 1]])]
    second = [np.array([[1, 0, 0, 2]]), np.array([[2, 0, 0, 2]])]
    acc.add_window(model, torch.zeros(1, 2, 1, 1, 1, 3), first, [0, 1], (3, 1, 1, 3))
    current = torch.tensor([10., 20., 30., 40., 50., 60.]).reshape(1, 2, 1, 1, 1, 3)
    acc.add_window(model, current, second, [1, 2], (3, 1, 1, 3))
    # Coordinate x=1 from the previous window, scaled to x=2 for edge prediction;
    # feature 20 from the current window, not the old feature or new peak at x=2.
    assert model.pairs[1][0][0, 0, 2] == 2
    assert model.pairs[1][1][0, 0, 0] == 20
    acc.add_window(model, current, second, [1, 2], (3, 1, 1, 3))
    coords, edges = acc.result()
    np.testing.assert_array_equal(coords[:, 3], [0, 2, 4])
    assert [(src, dst) for src, dst, _, _ in edges] == [(0, 1), (1, 2)]
    assert len(model.pairs) == 2
    assert model.encodes == 0


@pytest.mark.parametrize("window_size", [2, 3])
def test_predict_video_and_shared_window_assembly_agree(monkeypatch, window_size):
    frames = torch.full((4, 1, 1, 3), -1.)
    frames[..., 1] = 1.
    monkeypatch.setattr(prediction, "open_dataset", lambda *a, **k: SimpleNamespace(
        quantiles={"0.001": 0., "0.999": 1.}, zarr_path="unused",
        image_shape=(4, 1, 1, 3), scale=(1., 1., 1.),
    ))
    monkeypatch.setattr(prediction.zarr, "open_group", lambda *a, **k: {"0": None})
    monkeypatch.setattr(prediction, "_load_frame", lambda arr, t, *a: frames[t])
    cfg = prediction.PredictConfig(det_tta=False, pool_kernel_um=1.)
    predicted = prediction.predict_video(
        RecordingModel(), "unused", torch.device("cpu"), cfg,
        window_size=window_size, downsample=(1, 1, 1),
    )
    acc = VideoPredictionAccumulator((1, 1, 1))
    model = RecordingModel()
    for start in range(4 - window_size + 1):
        imgs = (frames[start:start + window_size] / (1. + 1e-6)).clamp(0.).unsqueeze(0)
        features, logits = model.encode(imgs)
        coords = [prediction._detect_cells_pooled(logits[i][0], start + i, .5, (1, 1, 1))
                  for i in range(window_size)]
        acc.add_window(model, features, coords, list(range(start, start + window_size)),
                       (4, 1, 1, 3), max_parents_per_node=1, max_children_per_node=2)
    collected = acc.result()
    np.testing.assert_array_equal(predicted[0], collected[0])
    assert predicted[1] == collected[1]
    assert len(collected[0]) == 4
    assert len(collected[1]) == 3
