# src/models/categorical_embedding.py
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class CategoricalEmbeddingEncoder(nn.Module):
    """
    Learned per-pixel embeddings for categorical features.

    Input:
      x_cat: [B, T, C_cat, H, W], integer codes
        - values >= 0: class index
        - values < 0: missing (mapped to padding index)

    Output:
      x_emb: [B, T, C_emb_total, H, W], float32
        where C_emb_total = sum(emb_dims[fid] for fid in feature_ids)
    """

    def __init__(
        self,
        feature_ids: List[str],
        num_classes: Dict[str, int],
        emb_dims: Dict[str, int],
    ):
        super().__init__()
        self.feature_ids = feature_ids
        self.num_classes = num_classes
        self.emb_dims = emb_dims

        embs = {}
        missing_index = {}

        for fid in feature_ids:
            K = num_classes[fid]      # real classes 0..K-1
            D = emb_dims[fid]        # embedding dim

            pad_idx = K  # extra index for "missing"
            embs[fid] = nn.Embedding(
                num_embeddings=K + 1,
                embedding_dim=D,
                padding_idx=pad_idx,
            )
            missing_index[fid] = pad_idx

        self.embs = nn.ModuleDict(embs)
        self.missing_index = missing_index

    @property
    def out_channels(self) -> int:
        return sum(self.emb_dims[fid] for fid in self.feature_ids)

    def forward(self, x_cat: torch.Tensor | None) -> torch.Tensor | None:
        if x_cat is None:
            return None

        if x_cat.dim() != 5:
            raise ValueError(f"Expected x_cat [B,T,C,H,W], got {tuple(x_cat.shape)}")

        B, T, C, H, W = x_cat.shape
        if C != len(self.feature_ids):
            raise ValueError(
                f"Channel count {C} does not match feature_ids length {len(self.feature_ids)}"
            )

        outs = []

        for c_idx, fid in enumerate(self.feature_ids):
            codes = x_cat[:, :, c_idx, ...].long()  # [B, T, H, W]
            pad_idx = self.missing_index[fid]

            codes = codes.clone()
            codes[codes < 0] = pad_idx  # map -1 etc. to padding index

            emb = self.embs[fid](codes)           # [B, T, H, W, D]
            emb = emb.permute(0, 1, 4, 2, 3)      # [B, T, D, H, W]
            outs.append(emb)

        return torch.cat(outs, dim=2)             # [B, T, C_emb_total, H, W]
