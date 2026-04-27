"""
Forest phase neighborhood matching.

Wrapper around :func:`soft_neighborhood_matching_loss` that handles the
domain-specific logic for comparing forest pixel trajectories:

1.  **ysfc-based temporal alignment** — two pixels are compared within
    their best shared consecutive recovery region.  Shared ysfc values
    are split into consecutive runs (values differing by 1); the run with
    the lowest starting value is selected (tie: longest run).  This
    avoids mixing post-disturbance recovery stages with pre-disturbance
    years in the same distance matrix.

2.  **No averaging of repeated ysfc values** — if a pixel has ysfc=0 at
    two separate time steps (e.g. two disturbances), both are kept as
    distinct data points rather than averaged into one representative.

3.  **Self-similarity and cross-pixel matching** — for each pair the
    wrapper computes both:

    * *Self-similarity* distances: ``d(r_j(y), r_j(y'))`` and
      ``d(z_i(y), z_i(y'))`` — teaches trajectory shape.  Restricted to
      ysfc values where each pixel has exactly one occurrence so positions
      align unambiguously.
    * *Cross-pixel* distances: ``d(r_i(y), r_j(y'))`` and
      ``d(z_i(y), z_j(y'))`` — anchors embeddings in a shared space.
      Uses all timesteps in the selected region (K_i × K_j).

4.  **Overlap masking** — entries outside the selected region (or on the
    diagonal for self-similarity) are masked.

5.  **Type-similarity weighting** — each pixel pair's loss contribution
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
import torch.nn.functional as F

from losses.soft_neighborhood import soft_neighborhood_matching_loss


# ---------------------------------------------------------------------------
# Consecutive-region utilities
# ---------------------------------------------------------------------------

def _split_consecutive_regions(sorted_vals: list[int]) -> list[list[int]]:
    """Split a sorted list of integers into maximal consecutive runs.

    Two adjacent values belong to the same run iff they differ by exactly 1.

    Parameters
    ----------
    sorted_vals : list of int
        Sorted unique integer ysfc values.

    Returns
    -------
    list of list of int
        Each sub-list is one consecutive run, also sorted.
    """
    if not sorted_vals:
        return []
    regions: list[list[int]] = []
    cur: list[int] = [sorted_vals[0]]
    for v in sorted_vals[1:]:
        if v == cur[-1] + 1:
            cur.append(v)
        else:
            regions.append(cur)
            cur = [v]
    regions.append(cur)
    return regions


def _select_best_region(regions: list[list[int]]) -> list[int]:
    """Pick the region with the lowest starting value; tie → longest.

    Parameters
    ----------
    regions : list of list of int
        Non-empty list of consecutive runs from :func:`_split_consecutive_regions`.

    Returns
    -------
    list of int
        The selected run.
    """
    if not regions:
        return []
    best = regions[0]
    for r in regions[1:]:
        if r[0] < best[0] or (r[0] == best[0] and len(r) > len(best)):
            best = r
    return best


# ---------------------------------------------------------------------------
# ysfc overlap utilities
# ---------------------------------------------------------------------------

def build_ysfc_overlap(
    ysfc_i: torch.Tensor,
    ysfc_j: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Find the best consecutive ysfc region shared by two pixels.

    Shared unique ysfc values are split into consecutive runs; the run
    with the lowest starting value is selected (tie: longest run).  This
    avoids mixing pre-disturbance and post-disturbance stages when both
    pixels share values from two disconnected recovery epochs.

    Parameters
    ----------
    ysfc_i, ysfc_j : Tensor ``[T]``
        Per-year ysfc values at two pixels (integer-valued floats).

    Returns
    -------
    region_values : Tensor ``[K]``
        Sorted ysfc values in the selected consecutive region.
    groups_i : list of K LongTensors
        ``groups_i[k]`` contains all time indices at pixel *i* where
        ``ysfc_i == region_values[k]``.
    groups_j : list of K LongTensors
        Same for pixel *j*.
    """
    device = ysfc_i.device

    unique_i = ysfc_i.unique()
    unique_j = ysfc_j.unique()
    shared_all = unique_i[torch.isin(unique_i, unique_j)]

    if shared_all.numel() == 0:
        return (
            torch.empty(0, dtype=ysfc_i.dtype, device=device),
            [],
            [],
        )

    shared_sorted = shared_all.sort().values

    # Select the best consecutive region.
    regions = _split_consecutive_regions(shared_sorted.tolist())
    region = _select_best_region(regions)
    region_vals = torch.tensor(region, dtype=ysfc_i.dtype, device=device)

    groups_i = [(ysfc_i == v).nonzero(as_tuple=False).squeeze(1) for v in region_vals]
    groups_j = [(ysfc_j == v).nonzero(as_tuple=False).squeeze(1) for v in region_vals]

    return region_vals, groups_i, groups_j


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

    Fully vectorized (no Python loop over pairs).  For each pair (i, j):

    1. **Region selection** — shared unique ysfc values are partitioned
       into consecutive runs; the run with the lowest starting value is
       selected.  The ``gap_mask → run_id`` cumsum trick identifies the
       first run for every pair simultaneously.

    2. **Duplicate filter** — if either pixel has the same ysfc value at
       more than one time step within the selected region (e.g. two
       separate disturbances both starting at ysfc=0), the pair is
       excluded.  With duplicates gone, each ysfc position maps to
       exactly one time step, so averaging and direct indexing agree.

    3. **Distance matrices** — a mapping matrix aligns the selected region
       to padded positions; four batched ``cdist`` calls produce all
       self-similarity and cross-pixel distances.

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
        Minimum selected-region size for a pair to be included.

    Returns
    -------
    dict with keys:

    - ``d_ref_self`` : Tensor ``[B_valid, M, M]``
        Self-similarity reference distances at pixel *j* (spectral).
    - ``d_learned_self`` : Tensor ``[B_valid, M, M]``
        Self-similarity learned distances at pixel *i* (embedding).
    - ``d_learned_self_j`` : Tensor ``[B_valid, M, M]``
        Self-similarity learned distances at pixel *j* (embedding).
    - ``mask_self`` : BoolTensor ``[B_valid, M, M]``
        Valid K × K block, diagonal excluded.
    - ``d_ref_cross`` : Tensor ``[B_valid, M, M]``
        Cross-pixel reference distances (K × K valid block).
    - ``d_learned_cross`` : Tensor ``[B_valid, M, M]``
        Cross-pixel learned distances (K × K valid block).
    - ``mask_cross`` : BoolTensor ``[B_valid, M, M]``
        Valid K × K block (diagonal included).
    - ``valid_pair_mask`` : BoolTensor ``[B]``
        Which input pairs survived region selection and duplicate filter.
    - ``M`` : int
        Padded matrix size = max selected-region size across valid pairs.
    """
    B = pair_indices.shape[0]
    N, T, C = spectral_features.shape
    D = phase_embeddings.shape[2]
    device = spectral_features.device
    dtype = spectral_features.dtype

    # --- Indicator matrix [N, V, T] ---
    # indicator[n, v, t] = 1.0  iff  ysfc[n, t] == unique_vals[v]
    ysfc_long = ysfc.long()
    unique_vals, ysfc_remapped = torch.unique(ysfc_long, return_inverse=True)
    ysfc_remapped = ysfc_remapped.reshape(N, T)
    V = unique_vals.shape[0]

    indicator = torch.zeros(N, V, T, device=device, dtype=dtype)
    n_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(ysfc_remapped)
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand_as(ysfc_remapped)
    indicator[n_idx, ysfc_remapped, t_idx] = 1.0

    counts  = indicator.sum(dim=2)   # [N, V]  occurrences of ysfc v in pixel n
    presence = counts > 0             # [N, V]  boolean presence

    # --- Vectorized region selection ---
    idx_i = pair_indices[:, 0]  # [B]
    idx_j = pair_indices[:, 1]  # [B]
    shared_mask = presence[idx_i] & presence[idx_j]   # [B, V]

    # gap_mask[v] = True where unique_vals[v+1] - unique_vals[v] > 1,
    # i.e. there is a gap between consecutive unique ysfc values.
    gap_mask = (unique_vals[1:] - unique_vals[:-1]) > 1   # [V-1]

    # is_break[b, k] = True if there is a run boundary between positions
    # k and k+1 for pair b (either position k is not shared, or a gap
    # in ysfc values makes positions k and k+1 non-consecutive).
    is_break = ~shared_mask[:, :-1] | gap_mask          # [B, V-1]

    # run_start[b, v] = True at the first shared value of each consecutive run.
    run_start = torch.cat([
        shared_mask[:, :1],                              # v=0: start iff shared
        shared_mask[:, 1:] & is_break,                  # v>0: start iff shared AND break before v
    ], dim=1)                                            # [B, V]

    # run_id counts runs seen so far; the first run has id=1.
    run_id = run_start.long().cumsum(dim=1)              # [B, V]

    # Best run = first consecutive run (lowest starting ysfc value).
    in_best_run = shared_mask & (run_id == 1)            # [B, V]
    region_size = in_best_run.sum(dim=1)                 # [B]

    # --- Duplicate filter ---
    # Exclude pairs where either pixel has a ysfc value appearing more than
    # once within the selected region (two disturbances → two ysfc=0 steps).
    # With no duplicates, averaging == direct indexing, keeping the pipeline exact.
    counts_i = counts[idx_i]   # [B, V]
    counts_j = counts[idx_j]   # [B, V]
    max_count_i = (counts_i * in_best_run).max(dim=1).values   # [B]
    max_count_j = (counts_j * in_best_run).max(dim=1).values   # [B]
    has_duplicates = (max_count_i > 1) | (max_count_j > 1)     # [B]

    valid_pair_mask = (region_size >= min_overlap) & ~has_duplicates   # [B]

    # --- Early exit ---
    empty_result = {
        "d_ref_self": torch.zeros(0, T, T, device=device, dtype=dtype),
        "d_learned_self": torch.zeros(0, T, T, device=device, dtype=dtype),
        "d_learned_self_j": torch.zeros(0, T, T, device=device, dtype=dtype),
        "mask_self": torch.zeros(0, T, T, device=device, dtype=torch.bool),
        "d_ref_cross": torch.zeros(0, T, T, device=device, dtype=dtype),
        "d_learned_cross": torch.zeros(0, T, T, device=device, dtype=dtype),
        "mask_cross": torch.zeros(0, T, T, device=device, dtype=torch.bool),
        "valid_pair_mask": valid_pair_mask,
        "M": T,
    }

    if not valid_pair_mask.any():
        return empty_result

    # --- Filter to valid pairs ---
    valid_idx = valid_pair_mask.nonzero(as_tuple=False).squeeze(1)
    B_valid = valid_idx.shape[0]
    idx_i_v = idx_i[valid_idx]              # [B_valid]
    idx_j_v = idx_j[valid_idx]              # [B_valid]
    in_best_run_v = in_best_run[valid_idx]  # [B_valid, V]
    K_valid = region_size[valid_idx]        # [B_valid]
    M = int(K_valid.max())

    # --- Build mapping matrix [B_valid, M, V] ---
    # mapping[b, k, v] = 1 iff ysfc value v is the k-th aligned position
    # for pair b (ordered by ysfc value within the selected region).
    positions = in_best_run_v.long().cumsum(dim=1) - 1   # [B_valid, V]
    b_nz, v_nz = in_best_run_v.nonzero(as_tuple=True)
    pos_nz = positions[b_nz, v_nz]

    mapping = torch.zeros(B_valid, M, V, device=device, dtype=dtype)
    mapping[b_nz, pos_nz, v_nz] = 1.0

    # --- Average features per ysfc value (equivalent to direct indexing
    #     since duplicates have been filtered out) ---
    counts_safe = counts.clamp(min=1).unsqueeze(-1)           # [N, V, 1]
    avg_spec  = torch.bmm(indicator, spectral_features) / counts_safe   # [N, V, C]
    avg_phase = torch.bmm(indicator, phase_embeddings)  / counts_safe   # [N, V, D]

    # --- Align to selected region via mapping ---
    aligned_i_spec  = torch.bmm(mapping, avg_spec[idx_i_v])    # [B_valid, M, C]
    aligned_j_spec  = torch.bmm(mapping, avg_spec[idx_j_v])    # [B_valid, M, C]
    aligned_i_phase = torch.bmm(mapping, avg_phase[idx_i_v])   # [B_valid, M, D]
    aligned_j_phase = torch.bmm(mapping, avg_phase[idx_j_v])   # [B_valid, M, D]

    # --- Per-pair demeaning (removes static spectral level so distances
    #     reflect trajectory shape only) ---
    aligned_i_spec = aligned_i_spec - aligned_i_spec.mean(dim=1, keepdim=True)
    aligned_j_spec = aligned_j_spec - aligned_j_spec.mean(dim=1, keepdim=True)

    # --- Batched distance matrices ---
    d_ref_self       = torch.cdist(aligned_j_spec,  aligned_j_spec)    # [B_valid, M, M]
    d_learned_self   = torch.cdist(aligned_i_phase, aligned_i_phase)   # [B_valid, M, M]
    d_learned_self_j = torch.cdist(aligned_j_phase, aligned_j_phase)   # [B_valid, M, M]
    d_ref_cross      = torch.cdist(aligned_i_spec,  aligned_j_spec)    # [B_valid, M, M]
    d_learned_cross  = torch.cdist(aligned_i_phase, aligned_j_phase)   # [B_valid, M, M]

    # --- Masks ---
    range_M = torch.arange(M, device=device)
    valid_pos  = range_M.unsqueeze(0) < K_valid.unsqueeze(1)          # [B_valid, M]
    mask_cross = valid_pos.unsqueeze(2) & valid_pos.unsqueeze(1)      # [B_valid, M, M]
    diag       = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0)
    mask_self  = mask_cross & ~diag                                    # [B_valid, M, M]

    return {
        "d_ref_self": d_ref_self,
        "d_learned_self": d_learned_self,
        "d_learned_self_j": d_learned_self_j,
        "mask_self": mask_self,
        "d_ref_cross": d_ref_cross,
        "d_learned_cross": d_learned_cross,
        "mask_cross": mask_cross,
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
    _batch: dict | None = None,
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

    # --- Build aligned batch (or use pre-built) ----------------------------
    if _batch is not None:
        batch = _batch
    else:
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

    # --- Distance distribution stats (for tau calibration) -----------------
    with torch.no_grad():
        def _dist_stats(d: torch.Tensor, mask: torch.Tensor) -> dict:
            vals = d[mask]
            if vals.numel() == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0}
            finite_vals = vals[torch.isfinite(vals)]
            return {
                "mean": vals.mean().item(),
                "std": vals.std().item(),
                "min": finite_vals.min().item() if finite_vals.numel() > 0 else 0.0,
                "max": finite_vals.max().item() if finite_vals.numel() > 0 else 0.0,
                "q25": torch.quantile(vals, 0.25).item(),
                "q50": torch.quantile(vals, 0.50).item(),
                "q75": torch.quantile(vals, 0.75).item(),
            }
        d_ref_self_stats = _dist_stats(batch["d_ref_self"], batch["mask_self"])
        d_ref_cross_stats = _dist_stats(batch["d_ref_cross"], batch["mask_cross"])

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
    for k, v in d_ref_self_stats.items():
        stats[f"d_ref_self_{k}"] = v
    for k, v in d_ref_cross_stats.items():
        stats[f"d_ref_cross_{k}"] = v

    return loss, stats


# ---------------------------------------------------------------------------
# Phase spread ranking loss
# ---------------------------------------------------------------------------

def compute_phase_spread_ranking(
    batch_result: dict,
    idx_i_valid: torch.Tensor,
    idx_j_valid: torch.Tensor,
    dynamism_ref: torch.Tensor,
    margin: float = 0.1,
    delta: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Ranking loss that orders phase spread by inter-annual spectral dynamism.

    For each valid pair (i, j), computes the spread of each pixel's phase
    embeddings across shared ysfc stages (mean off-diagonal distance in the
    ysfc-aligned self-distance matrix).  Applies a soft-margin ranking
    constraint: the more-dynamic pixel (higher ``dynamism_ref``) should have
    greater spread in phase space.

    Parameters
    ----------
    batch_result : dict
        Output of :func:`build_phase_neighborhood_batch`.  Must contain
        ``d_learned_self`` ``[B_valid, M, M]``, ``d_learned_self_j``
        ``[B_valid, M, M]``, and ``mask_self`` ``[B_valid, M, M]``.
    idx_i_valid : LongTensor ``[B_valid]``
        Pixel indices for the i-side of each valid pair.
    idx_j_valid : LongTensor ``[B_valid]``
        Pixel indices for the j-side of each valid pair.
    dynamism_ref : Tensor ``[N]``
        Per-anchor scalar dynamism score (e.g. mean of Mahalanobis-whitened
        variance channels).  Larger = more dynamic.
    margin : float
        Softplus margin.  Constraint fires when
        ``spread_less_dynamic >= spread_more_dynamic - margin``.
    delta : float
        Minimum dynamism difference to trigger the constraint.  Pairs
        with ``|dynamism_ref[i] - dynamism_ref[j]| <= delta`` are skipped.

    Returns
    -------
    loss : scalar Tensor
    stats : dict
        - ``n_pairs``: valid pairs considered
        - ``n_constrained_i``: pairs where i is more dynamic (constraint active)
        - ``n_constrained_j``: pairs where j is more dynamic (constraint active)
        - ``frac_satisfied``: fraction of constrained pairs already satisfied
        - ``mean_spread_i``: mean phase spread for i across all valid pairs
        - ``mean_spread_j``: mean phase spread for j across all valid pairs
        - ``mean_ref_diff``: mean |dynamism_ref[i] - dynamism_ref[j]|
    """
    device = batch_result["d_learned_self"].device
    dtype = batch_result["d_learned_self"].dtype

    d_self_i = batch_result["d_learned_self"]    # [B_valid, M, M]
    d_self_j = batch_result["d_learned_self_j"]  # [B_valid, M, M]
    mask_self = batch_result["mask_self"]         # [B_valid, M, M] off-diag valid

    B_valid = d_self_i.shape[0]

    if B_valid == 0:
        zero = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        return zero, {
            "n_pairs": 0, "n_constrained_i": 0, "n_constrained_j": 0,
            "frac_satisfied": 1.0, "mean_spread_i": 0.0,
            "mean_spread_j": 0.0, "mean_ref_diff": 0.0,
        }

    # Spread = mean off-diagonal valid distance per pair.
    n_valid = mask_self.float().sum(dim=(1, 2)).clamp(min=1)  # [B_valid]
    spread_i = (d_self_i * mask_self).sum(dim=(1, 2)) / n_valid  # [B_valid]
    spread_j = (d_self_j * mask_self).sum(dim=(1, 2)) / n_valid  # [B_valid]

    # Signed dynamism difference: positive means i is more dynamic.
    ref_diff = dynamism_ref[idx_i_valid] - dynamism_ref[idx_j_valid]  # [B_valid]

    i_more_dynamic = (ref_diff >  delta).float()   # [B_valid]
    j_more_dynamic = (ref_diff < -delta).float()   # [B_valid]

    # Softplus ranking constraint (follows triplet_phase.py pattern).
    # When i is more dynamic: enforce spread_i > spread_j + margin.
    # When j is more dynamic: enforce spread_j > spread_i + margin.
    loss_i = F.softplus(spread_j - spread_i + margin) * i_more_dynamic
    loss_j = F.softplus(spread_i - spread_j + margin) * j_more_dynamic
    loss = (loss_i + loss_j).mean()

    with torch.no_grad():
        n_ci = int(i_more_dynamic.sum().item())
        n_cj = int(j_more_dynamic.sum().item())
        n_constrained = n_ci + n_cj

        if n_constrained > 0:
            satisfied_i = ((spread_i - spread_j) > margin) * i_more_dynamic
            satisfied_j = ((spread_j - spread_i) > margin) * j_more_dynamic
            frac_sat = (satisfied_i + satisfied_j).sum().item() / n_constrained
        else:
            frac_sat = 1.0

    return loss, {
        "n_pairs": B_valid,
        "n_constrained_i": n_ci,
        "n_constrained_j": n_cj,
        "frac_satisfied": frac_sat,
        "mean_spread_i": spread_i.mean().item(),
        "mean_spread_j": spread_j.mean().item(),
        "mean_ref_diff": ref_diff.abs().mean().item(),
    }
