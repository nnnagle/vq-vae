"""
Phase pair construction for soft neighborhood loss.

Two-stage pair selection:
  1. kNN in spectral feature space → candidate pairs per anchor
  2. Filter by ysfc temporal overlap ≥ min_overlap

Pair weights are computed from spectral distance:
  w_ij = exp(-||spec_i - spec_j||₂ / sigma)
Self-pairs (i == i) always get weight = self_pair_weight.

Usage::

    pairs, weights, stats = build_phase_pairs(
        spec_features=spec_at_anchors,   # [N, C]
        ysfc=ysfc_at_anchors,            # [N, T]
        k=16,
        min_overlap=3,
        min_pairs=5,
        include_self=True,
        sigma=5.0,
        self_pair_weight=1.0,
    )
"""

from __future__ import annotations

import torch

from losses.phase_neighborhood import build_ysfc_overlap


def build_phase_pairs(
    spec_features: torch.Tensor,
    ysfc: torch.Tensor,
    k: int = 16,
    min_overlap: int = 3,
    min_pairs: int = 5,
    include_self: bool = True,
    sigma: float = 5.0,
    self_pair_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build pairs and weights for the phase soft neighborhood loss.

    Parameters
    ----------
    spec_features : Tensor ``[N, C]``
        Spectral distance features (Mahalanobis-whitened) at anchor pixels.
        Used for both kNN candidate selection and pair weighting.
    ysfc : Tensor ``[N, T]``
        Per-pixel ysfc time series at anchor pixels.  Single channel,
        integer-valued (e.g. 0..44).
    k : int
        Number of nearest spectral neighbors per anchor (stage 1).
    min_overlap : int
        Minimum shared ysfc values for a pair to survive (stage 2).
    min_pairs : int
        Drop an anchor entirely if fewer than this many cross-pixel
        pairs survive filtering.
    include_self : bool
        If True, add self-pairs ``(i, i)`` for every surviving anchor.
    sigma : float
        Scale for Gaussian pair weighting:
        ``w = exp(-||spec_i - spec_j||₂ / sigma)``.
    self_pair_weight : float
        Fixed weight for self-pairs.

    Returns
    -------
    pair_indices : LongTensor ``[P, 2]``
        Pixel pair indices into the N anchors.  Empty ``[0, 2]`` when no
        pairs survive.
    pair_weights : Tensor ``[P]``
        Per-pair weights (positive, ≤ 1).
    stats : dict
        Diagnostics for logging:

        - ``n_anchors``: total anchors provided
        - ``n_anchors_surviving``: anchors with ≥ min_pairs cross pairs
        - ``n_candidates``: total candidate pairs from kNN (before filter)
        - ``n_after_overlap``: pairs surviving ysfc overlap filter
        - ``n_self_pairs``: self-pairs added
        - ``n_total_pairs``: final pair count
        - ``overlap_mean``: mean ysfc overlap across surviving pairs
        - ``overlap_min``: min ysfc overlap across surviving pairs
        - ``weight_mean``: mean pair weight (cross-pixel only)
        - ``weight_std``: std of pair weights
    """
    device = spec_features.device
    N = spec_features.shape[0]

    empty_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)
    empty_weights = torch.zeros(0, device=device)
    empty_stats = {
        'n_anchors': N,
        'n_anchors_surviving': 0,
        'n_candidates': 0,
        'n_after_overlap': 0,
        'n_self_pairs': 0,
        'n_total_pairs': 0,
        'overlap_mean': 0.0,
        'overlap_min': 0,
        'weight_mean': 0.0,
        'weight_std': 0.0,
    }

    if N < 2:
        return empty_pairs, empty_weights, empty_stats

    # --- Stage 1: kNN in spectral space ---
    # Pairwise L2 distances [N, N]
    spec_dists = torch.cdist(spec_features, spec_features)

    # For each anchor, find k nearest neighbors (excluding self)
    # Set diagonal to inf so self isn't selected as a neighbor
    spec_dists_no_self = spec_dists.clone()
    spec_dists_no_self.fill_diagonal_(float('inf'))

    actual_k = min(k, N - 1)
    if actual_k == 0:
        return empty_pairs, empty_weights, empty_stats

    # topk on negative distances = k smallest distances
    _, knn_indices = spec_dists_no_self.topk(actual_k, dim=1, largest=False)
    # knn_indices: [N, actual_k]

    # Build candidate pairs: (anchor_idx, neighbor_idx)
    anchor_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, actual_k)
    candidate_pairs = torch.stack(
        [anchor_idx.reshape(-1), knn_indices.reshape(-1)], dim=1
    )  # [N * actual_k, 2]

    n_candidates = candidate_pairs.shape[0]

    # --- Stage 2: filter by ysfc overlap ---
    overlaps = torch.zeros(n_candidates, dtype=torch.long, device=device)

    for p in range(n_candidates):
        i_idx = candidate_pairs[p, 0].item()
        j_idx = candidate_pairs[p, 1].item()
        shared, _, _ = build_ysfc_overlap(ysfc[i_idx], ysfc[j_idx])
        overlaps[p] = shared.shape[0]

    # Keep pairs with sufficient overlap
    overlap_mask = overlaps >= min_overlap
    surviving_pairs = candidate_pairs[overlap_mask]
    surviving_overlaps = overlaps[overlap_mask]

    n_after_overlap = surviving_pairs.shape[0]

    # --- Drop anchors with too few surviving pairs ---
    if n_after_overlap > 0:
        # Count surviving cross-pixel pairs per anchor
        anchor_counts = torch.zeros(N, dtype=torch.long, device=device)
        anchor_counts.scatter_add_(
            0,
            surviving_pairs[:, 0],
            torch.ones(n_after_overlap, dtype=torch.long, device=device),
        )
        anchors_ok = anchor_counts >= min_pairs  # [N] bool

        # Filter pairs to only those from surviving anchors
        pair_anchor_ok = anchors_ok[surviving_pairs[:, 0]]
        surviving_pairs = surviving_pairs[pair_anchor_ok]
        surviving_overlaps = surviving_overlaps[pair_anchor_ok]
    else:
        anchors_ok = torch.zeros(N, dtype=torch.bool, device=device)

    n_surviving_anchors = anchors_ok.sum().item()
    n_cross_pairs = surviving_pairs.shape[0]

    if n_cross_pairs == 0:
        empty_stats['n_candidates'] = n_candidates
        return empty_pairs, empty_weights, empty_stats

    # --- Compute pair weights from spectral distance ---
    cross_dists = spec_dists[surviving_pairs[:, 0], surviving_pairs[:, 1]]
    cross_weights = torch.exp(-cross_dists / sigma)

    # --- Add self-pairs ---
    n_self = 0
    if include_self and n_surviving_anchors > 0:
        self_anchor_indices = anchors_ok.nonzero(as_tuple=False).squeeze(1)
        self_pairs = self_anchor_indices.unsqueeze(1).expand(-1, 2)  # [M, 2]
        self_weights = torch.full(
            (self_pairs.shape[0],), self_pair_weight, device=device
        )
        n_self = self_pairs.shape[0]

        # Concatenate
        all_pairs = torch.cat([surviving_pairs, self_pairs], dim=0)
        all_weights = torch.cat([cross_weights, self_weights], dim=0)
    else:
        all_pairs = surviving_pairs
        all_weights = cross_weights

    # --- Stats ---
    stats = {
        'n_anchors': N,
        'n_anchors_surviving': n_surviving_anchors,
        'n_candidates': n_candidates,
        'n_after_overlap': n_after_overlap,
        'n_self_pairs': n_self,
        'n_total_pairs': all_pairs.shape[0],
        'overlap_mean': surviving_overlaps.float().mean().item() if n_cross_pairs > 0 else 0.0,
        'overlap_min': surviving_overlaps.min().item() if n_cross_pairs > 0 else 0,
        'weight_mean': cross_weights.mean().item() if n_cross_pairs > 0 else 0.0,
        'weight_std': cross_weights.std().item() if n_cross_pairs > 1 else 0.0,
    }

    return all_pairs, all_weights, stats
