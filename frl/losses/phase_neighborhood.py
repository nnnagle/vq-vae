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

    .. note:: Not used internally — ``build_phase_neighborhood_batch``
       now computes overlap in vectorized form via indicator matrices.
       Retained for debugging and external callers.

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

    .. note:: Not used internally — ``build_phase_neighborhood_batch``
       now averages features via batched matmul over indicator matrices.
       Retained for debugging and external callers.

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

    .. note:: Not used internally — ``build_phase_neighborhood_batch``
       now computes aligned distances for all pairs at once via batched
       ``cdist``.  Retained for debugging and external callers.

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

    Vectorized implementation — no Python loops over pairs.  The strategy:

    1. Build per-pixel indicator matrices ``[N, V, T]`` mapping each ysfc
       value to its timestep(s).
    2. Average features per ysfc value via batched matmul.
    3. Compute per-pair overlap from presence vectors.
    4. Align shared ysfc values to padded positions via a mapping matrix
       and batched matmul (autograd-safe).
    5. Compute all distance matrices with four batched ``cdist`` calls.

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
    N, T, C = spectral_features.shape
    D = phase_embeddings.shape[2]
    device = spectral_features.device
    dtype = spectral_features.dtype

    # --- Remap ysfc to contiguous 0..V-1 ---
    ysfc_long = ysfc.long()
    unique_vals, ysfc_remapped = torch.unique(ysfc_long, return_inverse=True)
    ysfc_remapped = ysfc_remapped.reshape(N, T)
    V = unique_vals.shape[0]

    # --- Build indicator matrix [N, V, T] ---
    # indicator[n, v, t] = 1.0 if ysfc_remapped[n, t] == v
    indicator = torch.zeros(N, V, T, device=device, dtype=dtype)
    n_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(ysfc_remapped)
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand_as(ysfc_remapped)
    indicator[n_idx, ysfc_remapped, t_idx] = 1.0

    # --- Counts and presence per (pixel, ysfc value) ---
    counts = indicator.sum(dim=2)  # [N, V]
    presence = counts > 0          # [N, V]

    # --- Average features per ysfc value via batched matmul ---
    counts_safe = counts.clamp(min=1).unsqueeze(-1)  # [N, V, 1]
    # [N, V, T] @ [N, T, C] -> [N, V, C]
    avg_spec = torch.bmm(indicator, spectral_features) / counts_safe
    # [N, V, T] @ [N, T, D] -> [N, V, D]
    avg_phase = torch.bmm(indicator, phase_embeddings) / counts_safe

    # --- Per-pair overlap ---
    idx_i = pair_indices[:, 0]  # [B]
    idx_j = pair_indices[:, 1]  # [B]
    shared = presence[idx_i] & presence[idx_j]  # [B, V]
    K_per_pair = shared.sum(dim=1)  # [B]

    # --- Filter by min_overlap ---
    valid_pair_mask = K_per_pair >= min_overlap

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

    # --- Restrict to valid pairs ---
    valid_idx = valid_pair_mask.nonzero(as_tuple=False).squeeze(1)
    B_valid = valid_idx.shape[0]
    idx_i_v = idx_i[valid_idx]   # [B_valid]
    idx_j_v = idx_j[valid_idx]   # [B_valid]
    shared_v = shared[valid_idx]  # [B_valid, V]
    K_valid = K_per_pair[valid_idx]  # [B_valid]
    M = K_valid.max().item()

    # --- Build mapping matrix [B_valid, M, V] ---
    # Maps each shared ysfc value to a compressed position 0..K-1.
    # mapping[b, pos, v] = 1.0 iff ysfc value v is the pos-th shared
    # value for pair b.  Used via bmm to align features.
    positions = shared_v.long().cumsum(dim=1) - 1  # [B_valid, V]
    b_nz, v_nz = shared_v.nonzero(as_tuple=True)
    pos_nz = positions[b_nz, v_nz]

    mapping = torch.zeros(B_valid, M, V, device=device, dtype=dtype)
    mapping[b_nz, pos_nz, v_nz] = 1.0

    # --- Gather per-pair averaged features and align via bmm ---
    avg_spec_i = avg_spec[idx_i_v]    # [B_valid, V, C]
    avg_spec_j = avg_spec[idx_j_v]    # [B_valid, V, C]
    avg_phase_i = avg_phase[idx_i_v]  # [B_valid, V, D]
    avg_phase_j = avg_phase[idx_j_v]  # [B_valid, V, D]

    aligned_i_spec = torch.bmm(mapping, avg_spec_i)    # [B_valid, M, C]
    aligned_j_spec = torch.bmm(mapping, avg_spec_j)    # [B_valid, M, C]
    aligned_i_phase = torch.bmm(mapping, avg_phase_i)  # [B_valid, M, D]
    aligned_j_phase = torch.bmm(mapping, avg_phase_j)  # [B_valid, M, D]

    # --- Batched distance matrices ---
    d_ref_self = torch.cdist(aligned_j_spec, aligned_j_spec)        # [B_valid, M, M]
    d_learned_self = torch.cdist(aligned_i_phase, aligned_i_phase)   # [B_valid, M, M]
    d_ref_cross = torch.cdist(aligned_i_spec, aligned_j_spec)       # [B_valid, M, M]
    d_learned_cross = torch.cdist(aligned_i_phase, aligned_j_phase)  # [B_valid, M, M]

    # --- Masks ---
    range_M = torch.arange(M, device=device)
    valid_pos = range_M.unsqueeze(0) < K_valid.unsqueeze(1)  # [B_valid, M]
    mask_cross_batch = valid_pos.unsqueeze(2) & valid_pos.unsqueeze(1)  # [B_valid, M, M]
    diag = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0)
    mask_self_batch = mask_cross_batch & ~diag

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
