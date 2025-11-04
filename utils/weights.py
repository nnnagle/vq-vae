#!/usr/bin/env python3
# =============================================================================
# utils/weights.py — Class weight utilities for imbalanced categorical heads
#
# Purpose
#   Build per-feature weight vectors for CrossEntropyLoss over categorical heads.
#   Weights are aligned with dense IDs produced by the schema (MISS=0, UNK=1).
#
# Exposed API
#   cat_class_weights(schema_feature_entry, mode="sqrt_inv", eps=1e-6) -> torch.Tensor[float32]
#
# Notes
#   - schema_feature_entry is the per-feature dict under schema["categorical"][name]
#     and should include "num_ids" and "counts_per_id" (aligned to IDs).
#   - We zero MISS/UNK weights by default to avoid dominating the loss.
# =============================================================================

from __future__ import annotations

from typing import Literal, Dict, Any

import numpy as np
import torch

MISS_ID, UNK_ID = 0, 1


def cat_class_weights(schema_feature_entry: Dict[str, Any],
                      mode: Literal["inv", "sqrt_inv", "uniform"] = "sqrt_inv",
                      eps: float = 1e-6) -> torch.Tensor:
    """
    Construct class weights aligned to dense IDs for a single categorical feature.

    Inputs
      schema_feature_entry:
        {
          "num_ids": int,
          "counts_per_id": [float,...],  # length = num_ids; 0 for MISS/UNK
          ...
        }
      mode:
        - "uniform"   : weight 1 for kept classes (0 for MISS/UNK)
        - "inv"       : 1 / (freq + eps)
        - "sqrt_inv"  : 1 / sqrt(freq + eps)   [safer than pure inverse]
      eps: numerical stability

    Returns
      torch.FloatTensor of shape [num_ids], normalized to mean(weight[>0]) == 1
      with weight[0] (MISS) = weight[1] (UNK) = 0 by default.
    """
    num_ids = int(schema_feature_entry["num_ids"])
    counts = np.asarray(schema_feature_entry.get("counts_per_id", [0.0] * num_ids),
                        dtype=np.float64)
    counts = counts[:num_ids] if counts.size >= num_ids else np.pad(counts, (0, num_ids - counts.size))

    w = np.ones(num_ids, dtype=np.float64)

    if mode == "uniform":
        w[:] = 1.0
    elif mode == "inv":
        w = 1.0 / (counts + eps)
    elif mode == "sqrt_inv":
        w = 1.0 / np.sqrt(counts + eps)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Do not train the loss on MISS/UNK by default
    w[MISS_ID] = 0.0
    w[UNK_ID] = 0.0

    # Normalize so average nonzero weight is 1 (keeps CE scales comparable)
    nz = w[w > 0]
    if nz.size > 0:
        w = w / (nz.mean() + eps)

    return torch.tensor(w, dtype=torch.float32)
