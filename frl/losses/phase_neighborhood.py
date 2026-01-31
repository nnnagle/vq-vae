"""
Forest phase neighborhood matching.

Wrapper around :func:`soft_neighborhood_matching_loss` that handles the
domain-specific logic for comparing forest pixel trajectories:

1.  **ysfc-based temporal alignment** — two pixels are compared at
    recovery stages (ysfc values) that both have observed.  When a ysfc
    value appears at multiple time steps (e.g. two disturbances), the
    spectral features and embeddings are averaged across those time steps
    to produce one representative per ysfc value.

2.  **Self-similarity and cross-pixel matching** — for each pair the
    wrapper computes both:

    * *Self-similarity* distances: ``d(r_j(y), r_j(y'))`` and
      ``d(z_i(y), z_i(y'))`` — teaches trajectory shape.
    * *Cross-pixel* distances: ``d(r_i(y), r_j(y'))`` and
      ``d(z_i(y), z_j(y'))`` — anchors embeddings in a shared space.

3.  **Overlap masking** — entries outside the overlap set (or on the
    diagonal for self-similarity) are masked.

4.  **Type-similarity weighting** — each pixel pair's loss contribution
    is weighted by how similar their type embeddings are.

Usage
-----
::

    from losses.phase_neighborhood import (
        build_ysfc_overlap,
        phase_neighborhood_loss,
    )

    # In the training loop:
    loss, stats = phase_neighborhood_loss(
        spectral_features=r,   # [N, T, C]  Mahalanobis-whitened LS8
        phase_embeddings=z,    # [N, T, D]  from TCN encoder
        ysfc=ysfc,             # [N, T]     years-since-fast-change
        pair_indices=pairs,    # [B, 2]     (i, j) pixel pairs
        pair_weights=weights,  # [B]        from type similarity
        tau_ref=0.1,
        tau_learned=0.1,
    )
"""

from __future__ import annotations

import torch

from losses.soft_neighborhood import soft_neighborhood_matching_loss


# ---------------------------------------------------------------------------
# ysfc overlap utilities
# ---------------------------------------------------------------------------

def build_ysfc_overlap(
    ysfc_i: torch.Tensor,
    ysfc_j: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Find shared ysfc values and the time indices for each.

    Parameters
    ----------
    ysfc_i, ysfc_j : Tensor ``[T]``
        Per-year ysfc values at two pixels (integer-valued floats).

    Returns
    -------
    shared_values : Tensor ``[K]``
        Sorted unique ysfc values present at both pixels.
    groups_i : list of K LongTensors
        ``groups_i[k]`` contains the time indices at pixel *i* where
        ``ysfc_i == shared_values[k]``.
    groups_j : list of K LongTensors
        Same for pixel *j*.
    """
    device = ysfc_i.device

    unique_i = ysfc_i.unique()
    unique_j = ysfc_j.unique()
    shared_values = unique_i[torch.isin(unique_i, unique_j)]

    if shared_values.numel() == 0:
        return (
            torch.empty(0, dtype=ysfc_i.dtype, device=device),
            [],
            [],
        )

    # Sort for deterministic ordering.
    shared_values = shared_values.sort().values

    groups_i = []
    groups_j = []
    for val in shared_values:
        groups_i.append((ysfc_i == val).nonzero(as_tuple=False).squeeze(1))
        groups_j.append((ysfc_j == val).nonzero(as_tuple=False).squeeze(1))

    return shared_values, groups_i, groups_j


def average_features_by_ysfc(
    features: torch.Tensor,
    groups: list[torch.Tensor],
    K: int,
) -> torch.Tensor:
    """Average features within each ysfc group.

    Parameters
    ----------
    features : Tensor ``[T, C]``
        Per-time-step feature vectors at one pixel.
    groups : list of K LongTensors
        Each entry contains time indices belonging to one ysfc value.
    K : int
        Number of groups (= number of shared ysfc values).

    Returns
    -------
    averaged : Tensor ``[K, C]``
        One averaged feature vector per ysfc value.  Gradients flow
        through the averaging operation.
    """
    parts = []
    for k in range(K):
        idx = groups[k]
        parts.append(features[idx].mean(dim=0))  # [C]
    return torch.stack(parts, dim=0)  # [K, C]


# ---------------------------------------------------------------------------
# Aligned distance computation
# ---------------------------------------------------------------------------

def compute_aligned_distances(
    features_i: torch.Tensor,
    features_j: torch.Tensor,
    ysfc_i: torch.Tensor,
    ysfc_j: torch.Tensor,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute ysfc-aligned self-similarity and cross-pixel distance matrices.

    For each shared ysfc value, features are averaged across any
    duplicate time steps.  Then two pairs of distance matrices are
    produced:

    * **Self-similarity reference** at pixel *j*: ``d(r_j(y), r_j(y'))``
    * **Self-similarity learned** at pixel *i*: ``d(z_i(y), z_i(y'))``
    * **Cross-pixel reference**: ``d(r_i(y), r_j(y'))``
    * **Cross-pixel learned**: ``d(z_i(y), z_j(y'))``

    Both use the same mask (valid within the overlap, diagonal excluded
    for self-similarity, all valid for cross-pixel).

    Parameters
    ----------
    features_i : Tensor ``[T, C]``
        Features at pixel *i* (spectral or embedding).
    features_j : Tensor ``[T, C]``
        Features at pixel *j*.
    ysfc_i, ysfc_j : Tensor ``[T]``
        ysfc time series.
    M : int
        Padded output size.

    Returns
    -------
    d_self : Tensor ``[M, M]``
        Self-similarity distances at pixel *j* (reference) or *i*
        (learned), depending on which features are passed in.
    d_cross : Tensor ``[M, M]``
        Cross-pixel distances between *i* and *j*.
    mask_self : BoolTensor ``[M, M]``
        Mask for self-similarity (diagonal excluded, padding excluded).
    mask_cross : BoolTensor ``[M, M]``
        Mask for cross-pixel (padding excluded, diagonal included).
    K : int
        Number of shared ysfc values (overlap size).
    """
    device = features_i.device
    dtype = features_i.dtype

    shared_values, groups_i, groups_j = build_ysfc_overlap(ysfc_i, ysfc_j)
    K = shared_values.shape[0]

    d_self = torch.zeros(M, M, device=device, dtype=dtype)
    d_cross = torch.zeros(M, M, device=device, dtype=dtype)
    mask_self = torch.zeros(M, M, device=device, dtype=torch.bool)
    mask_cross = torch.zeros(M, M, device=device, dtype=torch.bool)

    if K == 0:
        return d_self, d_cross, mask_self, mask_cross, K

    # Average features within ysfc groups.
    avg_i = average_features_by_ysfc(features_i, groups_i, K)  # [K, C]
    avg_j = average_features_by_ysfc(features_j, groups_j, K)  # [K, C]

    # Self-similarity at pixel j: d(r_j(y), r_j(y'))
    d_self_kk = torch.cdist(avg_j.unsqueeze(0), avg_j.unsqueeze(0)).squeeze(0)
    d_self[:K, :K] = d_self_kk

    # Cross-pixel: d(r_i(y), r_j(y'))
    d_cross_kk = torch.cdist(avg_i.unsqueeze(0), avg_j.unsqueeze(0)).squeeze(0)
    d_cross[:K, :K] = d_cross_kk

    # Masks.
    mask_self[:K, :K] = True
    mask_self[range(K), range(K)] = False  # exclude diagonal for self-similarity

    mask_cross[:K, :K] = True  # diagonal is meaningful for cross-pixel

    return d_self, d_cross, mask_self, mask_cross, K


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def build_phase_neighborhood_batch(
    spectral_features: torch.Tensor,
    phase_embeddings: torch.Tensor,
    ysfc: torch.Tensor,
    pair_indices: torch.Tensor,
    min_overlap: int = 3,
) -> dict[str, torch.Tensor | int]:
    """Prepare aligned distance matrices for a batch of pixel pairs.

    For each pair ``(i, j)``:

    1. Find shared ysfc values; average features within duplicates.
    2. Compute self-similarity distances (reference at *j*, learned at *i*).
    3. Compute cross-pixel distances (reference and learned between
       *i* and *j*).
    4. Pad all matrices to a common size M and build masks.

    Parameters
    ----------
    spectral_features : Tensor ``[N, T, C]``
        Mahalanobis-whitened spectral features at N anchor pixels.
    phase_embeddings : Tensor ``[N, T, D]``
        Phase encoder embeddings at the same N anchors.
    ysfc : Tensor ``[N, T]``
        Per-pixel ysfc time series (integer-valued).
    pair_indices : LongTensor ``[B, 2]``
        Pairs of pixel indices ``(i, j)`` into the N anchors.
    min_overlap : int
        Minimum overlap size for a pair to be included.

    Returns
    -------
    dict with keys:

    - ``d_ref_self`` : Tensor ``[B_valid, M, M]``
        Self-similarity reference distances (at pixel *j*).
    - ``d_learned_self`` : Tensor ``[B_valid, M, M]``
        Self-similarity learned distances (at pixel *i*).
    - ``mask_self`` : BoolTensor ``[B_valid, M, M]``
        Mask for self-similarity (diagonal excluded).
    - ``d_ref_cross`` : Tensor ``[B_valid, M, M]``
        Cross-pixel reference distances.
    - ``d_learned_cross`` : Tensor ``[B_valid, M, M]``
        Cross-pixel learned distances.
    - ``mask_cross`` : BoolTensor ``[B_valid, M, M]``
        Mask for cross-pixel (diagonal included).
    - ``valid_pair_mask`` : BoolTensor ``[B]``
        Which input pairs had sufficient overlap.
    - ``M`` : int
        Padded matrix size.
    """
    B = pair_indices.shape[0]
    T = spectral_features.shape[1]
    device = spectral_features.device
    dtype = spectral_features.dtype

    # --- First pass: compute overlap sizes ---------------------------------
    overlap_data = []
    overlap_sizes = []

    for b in range(B):
        i_idx = pair_indices[b, 0].item()
        j_idx = pair_indices[b, 1].item()
        shared_values, groups_i, groups_j = build_ysfc_overlap(
            ysfc[i_idx], ysfc[j_idx]
        )
        K = shared_values.shape[0]
        overlap_data.append((i_idx, j_idx, shared_values, groups_i, groups_j, K))
        overlap_sizes.append(K)

    overlap_sizes_t = torch.tensor(overlap_sizes, device=device)
    valid_pair_mask = overlap_sizes_t >= min_overlap

    empty_result = {
        "d_ref_self": torch.zeros(0, T, T, device=device, dtype=dtype),
        "d_learned_self": torch.zeros(0, T, T, device=device, dtype=dtype),
        "mask_self": torch.zeros(0, T, T, device=device, dtype=torch.bool),
        "d_ref_cross": torch.zeros(0, T, T, device=device, dtype=dtype),
        "d_learned_cross": torch.zeros(0, T, T, device=device, dtype=dtype),
        "mask_cross": torch.zeros(0, T, T, device=device, dtype=torch.bool),
        "valid_pair_mask": valid_pair_mask,
        "M": T,
    }

    if not valid_pair_mask.any():
        return empty_result

    M = overlap_sizes_t[valid_pair_mask].max().item()

    # --- Second pass: build aligned matrices -------------------------------
    valid_indices = valid_pair_mask.nonzero(as_tuple=False).squeeze(1)
    B_valid = valid_indices.shape[0]

    d_ref_self = torch.zeros(B_valid, M, M, device=device, dtype=dtype)
    d_learned_self = torch.zeros(B_valid, M, M, device=device, dtype=dtype)
    mask_self_batch = torch.zeros(B_valid, M, M, device=device, dtype=torch.bool)

    d_ref_cross = torch.zeros(B_valid, M, M, device=device, dtype=dtype)
    d_learned_cross = torch.zeros(B_valid, M, M, device=device, dtype=dtype)
    mask_cross_batch = torch.zeros(B_valid, M, M, device=device, dtype=torch.bool)

    for out_b, in_b in enumerate(valid_indices.tolist()):
        i_idx, j_idx, shared_values, groups_i, groups_j, K = overlap_data[in_b]

        # Average spectral features within ysfc groups.
        r_i_avg = average_features_by_ysfc(
            spectral_features[i_idx], groups_i, K
        )  # [K, C]
        r_j_avg = average_features_by_ysfc(
            spectral_features[j_idx], groups_j, K
        )  # [K, C]

        # Average phase embeddings within ysfc groups.
        z_i_avg = average_features_by_ysfc(
            phase_embeddings[i_idx], groups_i, K
        )  # [K, D]
        z_j_avg = average_features_by_ysfc(
            phase_embeddings[j_idx], groups_j, K
        )  # [K, D]

        # Self-similarity: reference at j, learned at i.
        d_ref_self[out_b, :K, :K] = torch.cdist(
            r_j_avg.unsqueeze(0), r_j_avg.unsqueeze(0)
        ).squeeze(0)
        d_learned_self[out_b, :K, :K] = torch.cdist(
            z_i_avg.unsqueeze(0), z_i_avg.unsqueeze(0)
        ).squeeze(0)

        mask_self_batch[out_b, :K, :K] = True
        mask_self_batch[out_b, range(K), range(K)] = False

        # Cross-pixel: i vs j.
        d_ref_cross[out_b, :K, :K] = torch.cdist(
            r_i_avg.unsqueeze(0), r_j_avg.unsqueeze(0)
        ).squeeze(0)
        d_learned_cross[out_b, :K, :K] = torch.cdist(
            z_i_avg.unsqueeze(0), z_j_avg.unsqueeze(0)
        ).squeeze(0)

        mask_cross_batch[out_b, :K, :K] = True

    return {
        "d_ref_self": d_ref_self,
        "d_learned_self": d_learned_self,
        "mask_self": mask_self_batch,
        "d_ref_cross": d_ref_cross,
        "d_learned_cross": d_learned_cross,
        "mask_cross": mask_cross_batch,
        "valid_pair_mask": valid_pair_mask,
        "M": M,
    }


# ---------------------------------------------------------------------------
# High-level loss
# ---------------------------------------------------------------------------

def phase_neighborhood_loss(
    spectral_features: torch.Tensor,
    phase_embeddings: torch.Tensor,
    ysfc: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_weights: torch.Tensor | None = None,
    tau_ref: float = 0.1,
    tau_learned: float = 0.1,
    min_overlap: int = 3,
    min_valid_per_row: int = 2,
    self_similarity_weight: float = 1.0,
    cross_pixel_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Phase neighborhood matching loss for forest recovery trajectories.

    Computes two complementary losses:

    * **Self-similarity**: pixel *i*'s embedding self-distances across
      time should match pixel *j*'s spectral self-distances across time,
      at shared ysfc values.  Teaches trajectory shape.
    * **Cross-pixel**: embedding distances between *i* and *j* should
      match spectral distances between *i* and *j*, at shared ysfc
      values.  Anchors embeddings in a shared metric space.

    When a ysfc value appears at multiple time steps (e.g. two
    disturbances), features are averaged across those time steps to
    produce one representative per ysfc value.

    Parameters
    ----------
    spectral_features : Tensor ``[N, T, C]``
        Mahalanobis-whitened spectral features at N anchor pixels.
    phase_embeddings : Tensor ``[N, T, D]``
        Phase encoder output at the same N anchors.
    ysfc : Tensor ``[N, T]``
        Per-pixel ysfc time series (integer-valued float).
    pair_indices : LongTensor ``[B, 2]``
        Pixel pair indices ``(i, j)``.  Include ``(i, i)`` self-pairs
        for within-pixel temporal structure.
    pair_weights : Tensor ``[B]``, optional
        Per-pair weights (e.g. from type similarity).
    tau_ref : float
        Temperature for the reference (spectral) distribution.
    tau_learned : float
        Temperature for the learned (embedding) distribution.
    min_overlap : int
        Minimum ysfc overlap size for a pair to contribute.
    min_valid_per_row : int
        Minimum valid entries per row for that row to contribute.
    self_similarity_weight : float
        Weight for the self-similarity loss term.
    cross_pixel_weight : float
        Weight for the cross-pixel loss term.

    Returns
    -------
    loss : scalar Tensor
        Weighted combination of self-similarity and cross-pixel losses.
    stats : dict
        Diagnostics including:

        - ``n_pairs_input``: total pairs provided
        - ``n_pairs_sufficient_overlap``: pairs with enough ysfc overlap
        - ``loss_self``: self-similarity loss value
        - ``loss_cross``: cross-pixel loss value
        - ``self_*``: stats from the self-similarity loss
        - ``cross_*``: stats from the cross-pixel loss
    """
    device = spectral_features.device
    dtype = spectral_features.dtype

    # --- Build aligned batch -----------------------------------------------
    batch = build_phase_neighborhood_batch(
        spectral_features=spectral_features,
        phase_embeddings=phase_embeddings,
        ysfc=ysfc,
        pair_indices=pair_indices,
        min_overlap=min_overlap,
    )

    n_input = pair_indices.shape[0]
    valid_pair_mask = batch["valid_pair_mask"]
    n_valid = valid_pair_mask.sum().item()

    zero_stats = {
        "n_pairs_input": n_input,
        "n_pairs_sufficient_overlap": 0,
        "loss_self": 0.0,
        "loss_cross": 0.0,
    }

    if n_valid == 0:
        return (
            torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True),
            zero_stats,
        )

    # --- Filter pair weights -----------------------------------------------
    if pair_weights is not None:
        valid_weights = pair_weights[valid_pair_mask]
    else:
        valid_weights = None

    # --- Self-similarity loss ----------------------------------------------
    loss_self, stats_self = soft_neighborhood_matching_loss(
        d_reference=batch["d_ref_self"],
        d_learned=batch["d_learned_self"],
        mask=batch["mask_self"],
        tau_ref=tau_ref,
        tau_learned=tau_learned,
        pair_weights=valid_weights,
        min_valid_per_row=min_valid_per_row,
    )

    # --- Cross-pixel loss --------------------------------------------------
    loss_cross, stats_cross = soft_neighborhood_matching_loss(
        d_reference=batch["d_ref_cross"],
        d_learned=batch["d_learned_cross"],
        mask=batch["mask_cross"],
        tau_ref=tau_ref,
        tau_learned=tau_learned,
        pair_weights=valid_weights,
        min_valid_per_row=min_valid_per_row,
    )

    # --- Combined loss -----------------------------------------------------
    loss = self_similarity_weight * loss_self + cross_pixel_weight * loss_cross

    # --- Stats -------------------------------------------------------------
    stats = {
        "n_pairs_input": n_input,
        "n_pairs_sufficient_overlap": n_valid,
        "loss_self": loss_self.detach().item(),
        "loss_cross": loss_cross.detach().item(),
    }
    for k, v in stats_self.items():
        stats[f"self_{k}"] = v
    for k, v in stats_cross.items():
        stats[f"cross_{k}"] = v

    return loss, stats
