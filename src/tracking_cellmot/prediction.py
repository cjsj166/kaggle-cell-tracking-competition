"""Reusable graph assembly for temporal-window predictions."""

import numpy as np
import polars as pl
import torch

import tracksdata as td

from tracking_cellmot.models import UNetNodeTransformer, extract_pos_features

__all__ = ["VideoPredictionAccumulator", "build_graph"]


def build_graph(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> td.graph.InMemoryGraph:
    """Build a tracksdata graph from detected coordinates and predicted edges."""
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])
    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": prob,
                "edge_dist": dist,
            }
            for src, tgt, prob, dist in edges
        ])
    return graph


class VideoPredictionAccumulator:
    """Assemble windows using first-seen coordinates and current-window features."""

    def __init__(self, downsample: tuple[int, ...]):
        self.downsample = np.asarray(downsample, dtype=np.float32)
        self.seen_frames: set[int] = set()
        self.seen_pairs: set[tuple[int, int]] = set()
        self.coord_offset: dict[int, tuple[int, int]] = {}
        self.coord_lists: list[np.ndarray] = []
        self.global_node_count = 0
        self.all_edges: list[tuple[int, int, float, float]] = []

    @torch.no_grad()
    def add_window(
        self,
        model: UNetNodeTransformer,
        unet_out: torch.Tensor,
        frame_coords: list[np.ndarray | None],
        frame_indices: list[int],
        image_shape: tuple[int, ...],
        *,
        edge_activation: str = "softmax",
        threshold: float = 0.5,
        max_parents_per_node: int | None = None,
        max_children_per_node: int | None = None,
    ) -> None:
        """Add one temporal window in increasing frame order."""
        device = unet_out.device
        window_size = len(frame_indices)
        downsample = torch.as_tensor(self.downsample, device=device)

        for frame_offset, frame_index in enumerate(frame_indices):
            if frame_index not in self.seen_frames:
                coords = frame_coords[frame_offset]
                assert coords is not None
                self.coord_offset[frame_index] = (
                    self.global_node_count,
                    self.global_node_count + len(coords),
                )
                self.global_node_count += len(coords)
                self.coord_lists.append(coords)
                self.seen_frames.add(frame_index)

        coords_so_far = (
            np.concatenate(self.coord_lists)
            if self.coord_lists
            else np.empty((0, 4), dtype=np.int16)
        )

        for frame_offset in range(window_size - 1):
            source_time = frame_indices[frame_offset]
            target_time = frame_indices[frame_offset + 1]
            pair = (source_time, target_time)
            if pair in self.seen_pairs:
                continue
            self.seen_pairs.add(pair)

            source_start, source_end = self.coord_offset[source_time]
            target_start, target_end = self.coord_offset[target_time]
            if source_end == source_start or target_end == target_start:  # No node detected.
                continue

            source_coords = coords_so_far[source_start:source_end]
            target_coords = coords_so_far[target_start:target_end]
            n_source, n_target = len(source_coords), len(target_coords)
            source_indices = np.arange(source_start, source_end, dtype=np.int64)
            target_indices = np.arange(target_start, target_end, dtype=np.int64)

            source_xyz = torch.from_numpy(source_coords[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            target_xyz = torch.from_numpy(target_coords[:, 1:].astype(np.float32)).unsqueeze(0).to(device)

            window_shape = (window_size,) + image_shape[1:]
            source_relative = source_coords.copy()
            # Convert absolute time to the frame's offset within this window.
            source_relative[:, 0] = frame_offset
            target_relative = target_coords.copy()
            target_relative[:, 0] = frame_offset + 1
            source_pos = torch.from_numpy(
                extract_pos_features(source_relative, window_shape),
            ).unsqueeze(0).to(device)
            target_pos = torch.from_numpy(
                extract_pos_features(target_relative, window_shape),
            ).unsqueeze(0).to(device)
            source_mask = torch.ones(1, n_source, dtype=torch.bool, device=device)
            target_mask = torch.ones(1, n_target, dtype=torch.bool, device=device)

            source_features = model._index_features(
                unet_out[:, frame_offset], source_xyz, source_mask,
            )
            target_features = model._index_features(
                unet_out[:, frame_offset + 1], target_xyz, target_mask,
            )
            logits = model.predict_edges(
                source_features,
                target_features,
                source_xyz * downsample,  # Recover coordinates before downsampling.
                target_xyz * downsample,
                source_pos,
                target_pos,
                source_mask,
                target_mask,
            )[0]

            if edge_activation == "softmax":
                probabilities = torch.softmax(logits, dim=0).cpu().numpy()
            else:
                probabilities = torch.sigmoid(logits).cpu().numpy()

            candidates = sorted(
                [
                    (probabilities[i, j], i, j)
                    for i in range(n_source)
                    for j in range(n_target)
                    if probabilities[i, j] > threshold
                ],
                reverse=True,
            )
            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}
            for probability, source, target in candidates:
                n_children = children_count.get(source, 0)
                n_parents = parents_count.get(target, 0)
                if max_children_per_node is not None and n_children >= max_children_per_node:
                    continue
                if max_parents_per_node is not None and n_parents >= max_parents_per_node:
                    continue
                global_source = int(source_indices[source])
                global_target = int(target_indices[target])
                distance = float(np.linalg.norm(
                    coords_so_far[global_source, 1:].astype(np.float32)
                    - coords_so_far[global_target, 1:].astype(np.float32),
                ))
                self.all_edges.append((global_source, global_target, float(probability), distance))
                children_count[source] = n_children + 1
                parents_count[target] = n_parents + 1

    def result(self) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
        """Return coordinates in original resolution and accumulated edges."""
        coords = (
            np.concatenate(self.coord_lists)
            if self.coord_lists
            else np.empty((0, 4), dtype=np.int16)
        ).astype(np.float32)
        coords[:, 1:] *= self.downsample
        return coords.astype(np.int16), list(self.all_edges)
