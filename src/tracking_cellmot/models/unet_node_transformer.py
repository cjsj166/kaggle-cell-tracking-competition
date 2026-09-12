"""Combined temporal UNet and node-transformer model."""

import numpy as np
import torch
import torch.nn as nn

from tracking_cellmot.models.simple_node_transformer import SimpleNodeTransformer

POS_EMBED_DIM = 8


def extract_pos_features(
    coords: np.ndarray,
    image_shape: tuple[int, ...],
    pos_embed_dim: int = POS_EMBED_DIM,
) -> np.ndarray:
    """Create sinusoidal embeddings for ``[t, z, y, x]`` coordinates."""
    t, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    norms = [c / max(s, 1) for c, s in zip([t, z, y, x], image_shape)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(pos_embed_dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


class UNetNodeTransformer(nn.Module):
    """Temporal UNet encoder with detection and transformer edge heads."""

    def __init__(
        self,
        unet: nn.Module,
        unet_out_channels: int,
        pos_feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.unet = unet
        self.unet_out_channels = unet_out_channels
        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)
        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def _index_features(
        self,
        feat_maps: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Index feature maps at integer node positions; pad masked slots with zeros."""
        batch_size, channels = feat_maps.shape[:2]
        spatial = feat_maps.shape[2:]
        max_nodes = coords.shape[1]
        out = torch.zeros(
            batch_size,
            max_nodes,
            channels,
            device=feat_maps.device,
            dtype=feat_maps.dtype,
        )
        for batch_index in range(batch_size):
            n_nodes = int(mask[batch_index].sum().item())
            if n_nodes == 0:
                continue
            z = coords[batch_index, :n_nodes, 0].long().clamp(0, spatial[0] - 1)
            y = coords[batch_index, :n_nodes, 1].long().clamp(0, spatial[1] - 1)
            x = coords[batch_index, :n_nodes, 2].long().clamp(0, spatial[2] - 1)
            out[batch_index, :n_nodes] = feat_maps[batch_index, :, z, y, x].T
        return out

    def detect(self, frame: torch.Tensor) -> torch.Tensor:
        """Run detection on one pre-downsampled frame."""
        pair = torch.stack([frame, frame], dim=0).unsqueeze(0).unsqueeze(2)
        unet_out = self.unet(pair)
        det = self.detect_head(unet_out[0, 0:1])
        return det[0, 0]

    def encode(
        self,
        imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a temporal window and return features plus per-frame logits."""
        unet_out = self.unet(imgs.unsqueeze(2))
        det_logits = [self.detect_head(unet_out[:, i]) for i in range(unet_out.shape[1])]
        return unet_out, det_logits

    def predict_edges(
        self,
        unet_feat_src: torch.Tensor,
        unet_feat_tgt: torch.Tensor,
        coords_src: torch.Tensor,
        coords_tgt: torch.Tensor,
        pos_feat_src: torch.Tensor,
        pos_feat_tgt: torch.Tensor,
        mask_src: torch.Tensor,
        mask_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer edge predictor on pre-indexed UNet features."""
        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        return self.transformer(
            feat_src,
            feat_tgt,
            coords_src,
            coords_tgt,
            mask_src,
            mask_tgt,
        )
