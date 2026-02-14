"""
Phase pair construction for soft neighborhood loss.

Simplified pair selection:
  1. kNN in spectral feature space → candidate pairs per anchor
  2. Hard threshold prune by spectral distance

Pair weights are computed from spectral distance:
  w_ij = exp(-||spec_i - spec_j||₂ / sigma)
Self-pairs (i == i) always get weight = self_pair_weight.

Usage::

    pairs, weights, stats = build_phase_pairs(
        spec_features=spec_at_anchors,   # [N, C]
        k=16,
        spectral_threshold=10.0,
        include_self=True,
        sigma=5.0,
        self_pair_weight=1.0,
    )
"""

from __future__ import annotations

import torch


def build_phase_pairs(
    spec_features: torch.Tensor,
    k: int = 16,
    spectral_threshold: float | None = None,
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
    k : int
        Number of nearest spectral neighbors per anchor.
    spectral_threshold : float or None
        Hard prune pairs with spectral distance above this value.
        If None, no threshold is applied (all kNN pairs kept).
    include_self : bool
        If True, add self-pairs ``(i, i)`` for every anchor that has
        at least one surviving cross-pixel pair.
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
        - ``n_anchors_with_pairs``: anchors with ≥ 1 surviving cross pair
        - ``n_candidates``: total candidate pairs from kNN
        - ``n_after_threshold``: pairs surviving spectral threshold
        - ``n_self_pairs``: self-pairs added
        - ``n_total_pairs``: final pair count
        - ``pairs_per_anchor_mean``: mean cross pairs per anchor (surviving only)
        - ``pairs_per_anchor_min``: min cross pairs per anchor
        - ``pairs_per_anchor_max``: max cross pairs per anchor
        - ``weight_mean``: mean pair weight (cross-pixel only)
        - ``weight_std``: std of pair weights
        - ``dist_*``: spectral distance distribution stats
    """
    device = spec_features.device
    N = spec_features.shape[0]

    empty_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)
    empty_weights = torch.zeros(0, device=device)
    empty_stats = {
        'n_anchors': N,
        'n_anchors_with_pairs': 0,
        'n_candidates': 0,
        'n_after_threshold': 0,
        'n_self_pairs': 0,
        'n_total_pairs': 0,
        'pairs_per_anchor_mean': 0.0,
        'pairs_per_anchor_min': 0,
        'pairs_per_anchor_max': 0,
        'cand_dist_mean': 0.0, 'cand_dist_std': 0.0,
        'cand_dist_q25': 0.0, 'cand_dist_q50': 0.0,
        'cand_dist_q75': 0.0, 'cand_dist_q95': 0.0,
        'cand_dist_min': 0.0, 'cand_dist_max': 0.0,
        'retained_dist_mean': 0.0, 'retained_dist_std': 0.0,
        'retained_dist_q25': 0.0, 'retained_dist_q50': 0.0,
        'retained_dist_q75': 0.0,
        'retained_dist_min': 0.0, 'retained_dist_max': 0.0,
        'weight_mean': 0.0, 'weight_std': 0.0,
        'weight_q25': 0.0, 'weight_q50': 0.0, 'weight_q75': 0.0,
        'weight_min': 0.0, 'weight_max': 0.0,
    }

    if N < 2:
        return empty_pairs, empty_weights, empty_stats

    # --- kNN in spectral space ---
    spec_dists = torch.cdist(spec_features, spec_features)  # [N, N]

    # Exclude self from kNN candidates
    spec_dists_no_self = spec_dists.clone()
    spec_dists_no_self.fill_diagonal_(float('inf'))

    actual_k = min(k, N - 1)
    if actual_k == 0:
        return empty_pairs, empty_weights, empty_stats

    _, knn_indices = spec_dists_no_self.topk(actual_k, dim=1, largest=False)
    # knn_indices: [N, actual_k]

    # Build candidate pairs: (anchor_idx, neighbor_idx)
    anchor_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, actual_k)
    candidate_pairs = torch.stack(
        [anchor_idx.reshape(-1), knn_indices.reshape(-1)], dim=1
    )  # [N * actual_k, 2]

    n_candidates = candidate_pairs.shape[0]

    # Get distances for candidate pairs
    candidate_dists = spec_dists[candidate_pairs[:, 0], candidate_pairs[:, 1]]

    # --- Pre-threshold candidate distance stats (for tuning threshold) ---
    cand_dist_stats = {
        'cand_dist_mean': candidate_dists.mean().item(),
        'cand_dist_std': candidate_dists.std().item() if n_candidates > 1 else 0.0,
        'cand_dist_q25': torch.quantile(candidate_dists, 0.25).item(),
        'cand_dist_q50': torch.quantile(candidate_dists, 0.50).item(),
        'cand_dist_q75': torch.quantile(candidate_dists, 0.75).item(),
        'cand_dist_q95': torch.quantile(candidate_dists, 0.95).item(),
        'cand_dist_min': candidate_dists.min().item(),
        'cand_dist_max': candidate_dists.max().item(),
    }

    # --- Hard threshold prune ---
    if spectral_threshold is not None:
        threshold_mask = candidate_dists <= spectral_threshold
        surviving_pairs = candidate_pairs[threshold_mask]
        surviving_dists = candidate_dists[threshold_mask]
    else:
        surviving_pairs = candidate_pairs
        surviving_dists = candidate_dists

    n_after_threshold = surviving_pairs.shape[0]

    if n_after_threshold == 0:
        empty_stats['n_candidates'] = n_candidates
        empty_stats.update(cand_dist_stats)
        return empty_pairs, empty_weights, empty_stats

    # --- Compute pair weights from spectral distance ---
    cross_weights = torch.exp(-surviving_dists / sigma)

    # --- Per-anchor pair counts (for logging) ---
    anchor_counts = torch.zeros(N, dtype=torch.long, device=device)
    anchor_counts.scatter_add_(
        0,
        surviving_pairs[:, 0],
        torch.ones(n_after_threshold, dtype=torch.long, device=device),
    )
    anchors_with_pairs = anchor_counts > 0  # [N] bool
    n_anchors_with_pairs = anchors_with_pairs.sum().item()
    counts_nonzero = anchor_counts[anchors_with_pairs]

    # --- Add self-pairs ---
    n_self = 0
    if include_self and n_anchors_with_pairs > 0:
        self_anchor_indices = anchors_with_pairs.nonzero(as_tuple=False).squeeze(1)
        self_pairs = self_anchor_indices.unsqueeze(1).expand(-1, 2)  # [M, 2]
        self_weights = torch.full(
            (self_pairs.shape[0],), self_pair_weight, device=device
        )
        n_self = self_pairs.shape[0]

        all_pairs = torch.cat([surviving_pairs, self_pairs], dim=0)
        all_weights = torch.cat([cross_weights, self_weights], dim=0)
    else:
        all_pairs = surviving_pairs
        all_weights = cross_weights

    # --- Stats ---
    stats = {
        'n_anchors': N,
        'n_anchors_with_pairs': n_anchors_with_pairs,
        'n_candidates': n_candidates,
        'n_after_threshold': n_after_threshold,
        'n_self_pairs': n_self,
        'n_total_pairs': all_pairs.shape[0],
        'pairs_per_anchor_mean': counts_nonzero.float().mean().item() if n_anchors_with_pairs > 0 else 0.0,
        'pairs_per_anchor_min': counts_nonzero.min().item() if n_anchors_with_pairs > 0 else 0,
        'pairs_per_anchor_max': counts_nonzero.max().item() if n_anchors_with_pairs > 0 else 0,
        # Pre-threshold candidate distances (all kNN pairs)
        **cand_dist_stats,
        # Post-threshold retained distances
        'retained_dist_mean': surviving_dists.mean().item(),
        'retained_dist_std': surviving_dists.std().item() if n_after_threshold > 1 else 0.0,
        'retained_dist_q25': torch.quantile(surviving_dists, 0.25).item(),
        'retained_dist_q50': torch.quantile(surviving_dists, 0.50).item(),
        'retained_dist_q75': torch.quantile(surviving_dists, 0.75).item(),
        'retained_dist_min': surviving_dists.min().item(),
        'retained_dist_max': surviving_dists.max().item(),
        # Retained pair weights
        'weight_mean': cross_weights.mean().item(),
        'weight_std': cross_weights.std().item() if n_after_threshold > 1 else 0.0,
        'weight_q25': torch.quantile(cross_weights, 0.25).item(),
        'weight_q50': torch.quantile(cross_weights, 0.50).item(),
        'weight_q75': torch.quantile(cross_weights, 0.75).item(),
        'weight_min': cross_weights.min().item(),
        'weight_max': cross_weights.max().item(),
    }

    return all_pairs, all_weights, stats
