"""
Forest phase neighborhood matching.

Wrapper around :func:`soft_neighborhood_matching_loss` that handles the
domain-specific logic for comparing forest pixel trajectories:

1.  **ysfc-based temporal alignment** — two pixels are compared at
    recovery stages (ysfc values) that both have observed.  Their
    ``[T, T]`` distance matrices are reindexed to a shared ``[M, M]``
    matrix aligned by ysfc value.

2.  **Overlap masking** — entries outside the overlap set (or on the
    diagonal) are masked so the generic loss ignores them.

3.  **Type-similarity weighting** — each pixel pair's loss contribution
    is weighted by how similar their type embeddings are.

Usage
-----
::

    from losses.phase_neighborhood import (
        build_ysfc_overlap_mask,
        align_distance_matrices,
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

def build_ysfc_overlap_mask(
    ysfc_i: torch.Tensor,
    ysfc_j: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the ysfc overlap between two pixels and build index mappings.

    For each ysfc value present at *both* pixels, record the time indices
    where that value occurs.  When a value occurs at multiple time steps
    at a pixel (e.g. ysfc stuttering ``[..., 0, 0, 1, ...]``), all
    occurrences are included.

    Parameters
    ----------
    ysfc_i, ysfc_j : Tensor ``[T]``
        Per-year ysfc values at two pixels (integer-valued floats).

    Returns
    -------
    indices_i : LongTensor ``[K]``
        Time indices at pixel *i* for each overlap entry.
    indices_j : LongTensor ``[K]``
        Corresponding time indices at pixel *j*.
    overlap_values : Tensor ``[K]``
        The ysfc value for each overlap entry.

    Notes
    -----
    The overlap is constructed by iterating over shared ysfc values.
    For a value *y* that appears ``n_i`` times at pixel *i* and ``n_j``
    times at pixel *j*, the cross-product of those time indices is
    included (``n_i * n_j`` entries).  For typical ysfc sequences where
    each value appears once or twice, this is a small expansion.
    """
    device = ysfc_i.device

    # Find unique values present at both pixels.
    unique_i = ysfc_i.unique()
    unique_j = ysfc_j.unique()

    # Intersection via broadcasting.
    # shared_mask[k] = True if unique_i[k] is in unique_j.
    shared_values = unique_i[torch.isin(unique_i, unique_j)]

    if shared_values.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=ysfc_i.dtype, device=device),
        )

    idx_i_list = []
    idx_j_list = []
    val_list = []

    for val in shared_values:
        times_i = (ysfc_i == val).nonzero(as_tuple=False).squeeze(1)
        times_j = (ysfc_j == val).nonzero(as_tuple=False).squeeze(1)

        # Cross-product of time indices for this ysfc value.
        grid_i, grid_j = torch.meshgrid(times_i, times_j, indexing="ij")
        idx_i_list.append(grid_i.reshape(-1))
        idx_j_list.append(grid_j.reshape(-1))
        val_list.append(val.expand(grid_i.numel()))

    indices_i = torch.cat(idx_i_list)
    indices_j = torch.cat(idx_j_list)
    overlap_values = torch.cat(val_list)

    return indices_i, indices_j, overlap_values


def align_distance_matrices(
    d_full_ref: torch.Tensor,
    d_full_learned: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_learned: torch.Tensor,
    M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and align distance sub-matrices for the overlap set.

    Given full ``[T, T]`` distance matrices and index mappings from
    :func:`build_ysfc_overlap_mask`, produce aligned ``[M, M]``
    matrices padded with zeros and a boolean mask.

    Parameters
    ----------
    d_full_ref : Tensor ``[T, T]``
        Full pairwise distance matrix in reference space (at pixel *j*).
    d_full_learned : Tensor ``[T, T]``
        Full pairwise distance matrix in learned space (at pixel *i*).
    indices_ref : LongTensor ``[K]``
        Time indices at the reference pixel for each overlap entry.
    indices_learned : LongTensor ``[K]``
        Time indices at the learned pixel for each overlap entry.
    M : int
        Size of the output matrices (padded to this dimension).
        Must be >= K.

    Returns
    -------
    d_ref_aligned : Tensor ``[M, M]``
        Aligned reference distances, zero-padded.
    d_learned_aligned : Tensor ``[M, M]``
        Aligned learned distances, zero-padded.
    mask : BoolTensor ``[M, M]``
        ``True`` for valid entries, ``False`` for padding and diagonal.
    """
    K = indices_ref.shape[0]
    device = d_full_ref.device
    dtype = d_full_ref.dtype

    d_ref_aligned = torch.zeros(M, M, device=device, dtype=dtype)
    d_learned_aligned = torch.zeros(M, M, device=device, dtype=dtype)
    mask = torch.zeros(M, M, device=device, dtype=torch.bool)

    if K == 0:
        return d_ref_aligned, d_learned_aligned, mask

    # Gather distances at overlap indices.
    # d_ref_aligned[a, b] = d_full_ref[indices_ref[a], indices_ref[b]]
    d_ref_aligned[:K, :K] = d_full_ref[indices_ref][:, indices_ref]
    d_learned_aligned[:K, :K] = d_full_learned[indices_learned][:, indices_learned]

    # Mask: valid within the KxK block, excluding diagonal.
    mask[:K, :K] = True
    mask[range(K), range(K)] = False

    return d_ref_aligned, d_learned_aligned, mask


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def build_phase_neighborhood_batch(
    spectral_features: torch.Tensor,
    phase_embeddings: torch.Tensor,
    ysfc: torch.Tensor,
    pair_indices: torch.Tensor,
    min_overlap: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare aligned distance matrices for a batch of pixel pairs.

    For each pair ``(i, j)``:

    1. Compute ysfc overlap between pixels *i* and *j*.
    2. Build the ``[T, T]`` spectral distance matrix at pixel *j*
       (reference) and embedding distance matrix at pixel *i* (learned).
    3. Align both to the overlap set and pad to a common size ``M``.
    4. Build the overlap mask.

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
    d_reference : Tensor ``[B_valid, M, M]``
        Aligned reference distance matrices.
    d_learned : Tensor ``[B_valid, M, M]``
        Aligned learned distance matrices.
    mask : BoolTensor ``[B_valid, M, M]``
        Overlap mask with diagonal excluded.
    valid_pair_mask : BoolTensor ``[B]``
        Which input pairs had sufficient overlap.
    M : int
        Padded matrix size (max overlap across valid pairs).
    """
    B = pair_indices.shape[0]
    T = spectral_features.shape[1]
    device = spectral_features.device

    # --- First pass: compute overlaps and find max M -------------------
    overlaps = []
    overlap_sizes = []

    for b in range(B):
        i_idx = pair_indices[b, 0].item()
        j_idx = pair_indices[b, 1].item()

        idx_i, idx_j, _ = build_ysfc_overlap_mask(
            ysfc[i_idx], ysfc[j_idx]
        )
        overlaps.append((idx_i, idx_j))
        overlap_sizes.append(idx_i.shape[0])

    overlap_sizes_t = torch.tensor(overlap_sizes, device=device)
    valid_pair_mask = overlap_sizes_t >= min_overlap

    if not valid_pair_mask.any():
        M = T  # fallback
        return (
            torch.zeros(0, M, M, device=device),
            torch.zeros(0, M, M, device=device),
            torch.zeros(0, M, M, device=device, dtype=torch.bool),
            valid_pair_mask,
            M,
        )

    M = overlap_sizes_t[valid_pair_mask].max().item()

    # --- Second pass: build aligned matrices ---------------------------
    valid_indices = valid_pair_mask.nonzero(as_tuple=False).squeeze(1)
    B_valid = valid_indices.shape[0]

    d_ref_batch = torch.zeros(B_valid, M, M, device=device)
    d_learned_batch = torch.zeros(B_valid, M, M, device=device)
    mask_batch = torch.zeros(B_valid, M, M, device=device, dtype=torch.bool)

    for out_b, in_b in enumerate(valid_indices.tolist()):
        i_idx = pair_indices[in_b, 0].item()
        j_idx = pair_indices[in_b, 1].item()
        idx_learned, idx_ref = overlaps[in_b]

        # Full [T, T] distance matrices.
        # Reference: spectral distances at pixel j.
        r_j = spectral_features[j_idx]  # [T, C]
        d_full_ref = torch.cdist(r_j.unsqueeze(0), r_j.unsqueeze(0)).squeeze(0)

        # Learned: embedding distances at pixel i.
        z_i = phase_embeddings[i_idx]  # [T, D]
        d_full_learned = torch.cdist(
            z_i.unsqueeze(0), z_i.unsqueeze(0)
        ).squeeze(0)

        # Align to overlap.
        d_ref_a, d_learn_a, mask_a = align_distance_matrices(
            d_full_ref, d_full_learned,
            indices_ref=idx_ref,
            indices_learned=idx_learned,
            M=M,
        )

        d_ref_batch[out_b] = d_ref_a
        d_learned_batch[out_b] = d_learn_a
        mask_batch[out_b] = mask_a

    return d_ref_batch, d_learned_batch, mask_batch, valid_pair_mask, M


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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Phase neighborhood matching loss for forest recovery trajectories.

    End-to-end loss that:

    1. Aligns pixel pairs by ysfc overlap.
    2. Builds reference distributions from spectral self-similarity.
    3. Builds learned distributions from phase embedding self-similarity.
    4. Minimises KL divergence between the two, weighted by type similarity.

    Parameters
    ----------
    spectral_features : Tensor ``[N, T, C]``
        Mahalanobis-whitened spectral features at N anchor pixels.
        These define the reference (target) neighborhood structure.
    phase_embeddings : Tensor ``[N, T, D]``
        Phase encoder output at the same N anchors.
        These are trained to match the reference structure.
    ysfc : Tensor ``[N, T]``
        Per-pixel ysfc time series (integer-valued float).
        Used for temporal alignment between pixel pairs.
    pair_indices : LongTensor ``[B, 2]``
        Pixel pair indices ``(i, j)``.  Include ``(i, i)`` self-pairs
        to enforce within-pixel temporal structure.
    pair_weights : Tensor ``[B]``, optional
        Per-pair weights (e.g. from type similarity).  Self-pairs
        typically get weight 1.0; cross-pixel pairs are weighted by
        type similarity.  If ``None``, all pairs weighted equally.
    tau_ref : float
        Temperature for the reference (spectral) distribution.
    tau_learned : float
        Temperature for the learned (embedding) distribution.
    min_overlap : int
        Minimum ysfc overlap size for a pair to contribute.
    min_valid_per_row : int
        Minimum valid entries per row for that row to contribute.

    Returns
    -------
    loss : scalar Tensor
        Weighted mean KL divergence across valid pairs.
    stats : dict
        Diagnostics from the generic loss, plus:

        - ``n_pairs_input``: total pairs provided
        - ``n_pairs_sufficient_overlap``: pairs with enough ysfc overlap
    """
    device = spectral_features.device
    dtype = spectral_features.dtype

    # --- Build aligned batch -----------------------------------------------
    d_ref, d_learned, mask, valid_pair_mask, M = build_phase_neighborhood_batch(
        spectral_features=spectral_features,
        phase_embeddings=phase_embeddings,
        ysfc=ysfc,
        pair_indices=pair_indices,
        min_overlap=min_overlap,
    )

    n_input = pair_indices.shape[0]
    n_valid = valid_pair_mask.sum().item()

    if d_ref.shape[0] == 0:
        return (
            torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True),
            {
                "n_pairs_input": n_input,
                "n_pairs_sufficient_overlap": 0,
                "n_pairs": 0,
                "n_pairs_active": 0,
                "n_rows_total": 0,
                "n_rows_valid": 0,
                "mean_kl": 0.0,
                "mean_overlap": 0.0,
            },
        )

    # --- Filter pair weights to valid pairs --------------------------------
    if pair_weights is not None:
        valid_weights = pair_weights[valid_pair_mask]
    else:
        valid_weights = None

    # --- Compute loss ------------------------------------------------------
    loss, stats = soft_neighborhood_matching_loss(
        d_reference=d_ref,
        d_learned=d_learned,
        mask=mask,
        tau_ref=tau_ref,
        tau_learned=tau_learned,
        pair_weights=valid_weights,
        min_valid_per_row=min_valid_per_row,
    )

    stats["n_pairs_input"] = n_input
    stats["n_pairs_sufficient_overlap"] = n_valid

    return loss, stats
