"""
Pair generation functions for contrastive learning.

This module provides functions to generate positive and negative pair indices
from distance matrices, for use with contrastive loss functions.

Strategies:
    - KNN: k-nearest neighbors for each anchor
    - Mutual KNN: pairs where both points are in each other's top-k
    - Quantile: pairs with distances in a quantile range
    - Radius: pairs with distances in an absolute range

All functions return [P, 2] tensors of (anchor_id, target_id) pairs, 0-based.

Example:
    >>> from frl.losses import pairs_knn, pairs_quantile, contrastive_loss
    >>> embeddings = torch.randn(100, 128)
    >>> distances = torch.cdist(embeddings, embeddings)  # [100, 100]
    >>> pos_pairs = pairs_knn(distances, k=5)  # 5 nearest neighbors as positives
    >>> neg_pairs = pairs_quantile(distances, low=0.5, high=0.75)  # mid-range as negatives
    >>> loss = contrastive_loss(embeddings, pos_pairs, neg_pairs)
"""

from __future__ import annotations

import torch


def _prepare_distances(
    distances: torch.Tensor,
    anchor_cols: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare distance matrix by masking invalid entries.

    Returns:
        distances: Distance matrix with inf for invalid entries
        anchor_cols: Anchor column indices (default: diagonal for square)
        valid_mask: Validity mask (default: all valid except inf/nan)
    """
    N, M = distances.shape
    device = distances.device
    dtype = distances.dtype

    # Default anchor_cols for square matrices
    if anchor_cols is None:
        if N != M:
            raise ValueError(
                f"anchor_cols is required for rectangular matrices (got {N}x{M})"
            )
        anchor_cols = torch.arange(N, device=device)
    else:
        anchor_cols = anchor_cols.to(device)

    # Default valid_mask
    if valid_mask is None:
        valid_mask = torch.ones(M, device=device, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device).bool()

    # Create working copy of distances
    dist = distances.clone()

    # Mask inf/nan as invalid
    invalid = ~torch.isfinite(dist)
    dist = torch.where(invalid, torch.tensor(float("inf"), device=device, dtype=dtype), dist)

    # Mask invalid columns (targets)
    dist[:, ~valid_mask] = float("inf")

    # Mask invalid anchors (rows where anchor_cols points to invalid)
    invalid_anchors = ~valid_mask[anchor_cols]
    dist[invalid_anchors, :] = float("inf")

    # Mask self-pairs
    row_indices = torch.arange(N, device=device)
    dist[row_indices, anchor_cols] = float("inf")

    return dist, anchor_cols, valid_mask


def _sample_pairs(
    pairs: torch.Tensor,
    max_pairs: int | None,
) -> torch.Tensor:
    """Randomly sample pairs if exceeding max_pairs."""
    if max_pairs is None or pairs.shape[0] <= max_pairs:
        return pairs

    indices = torch.randperm(pairs.shape[0], device=pairs.device)[:max_pairs]
    return pairs[indices]


def _add_symmetric_pairs(
    pairs: torch.Tensor,
    anchor_cols: torch.Tensor,
) -> torch.Tensor:
    """
    Add symmetric pairs (j, i) for each (i, j).

    Only adds reverse pairs where j is a valid anchor (exists in anchor_cols).
    """
    if pairs.numel() == 0:
        return pairs

    # Create set of valid anchor IDs
    anchor_set = set(anchor_cols.tolist())

    # Find pairs where target is also a valid anchor
    targets = pairs[:, 1]
    can_reverse = torch.tensor(
        [t.item() in anchor_set for t in targets],
        device=pairs.device,
        dtype=torch.bool,
    )

    # Create reverse pairs
    reversible = pairs[can_reverse]
    if reversible.numel() == 0:
        return pairs

    reversed_pairs = torch.stack([reversible[:, 1], reversible[:, 0]], dim=1)

    return torch.cat([pairs, reversed_pairs], dim=0)


def pairs_knn(
    distances: torch.Tensor,
    k: int,
    symmetric: bool = False,
    anchor_cols: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> torch.Tensor:
    """
    Generate pairs from k-nearest neighbors.

    For each anchor (row), selects the k closest targets as pairs.

    Args:
        distances: Distance matrix of shape [N, M]. For square matrices (N=M),
            rows and columns share the same index space. For rectangular (N<M),
            N anchors are compared against M candidates.
        k: Number of nearest neighbors to select per anchor.
        symmetric: If True and matrix is square, also add reverse pairs (j, i)
            for each (i, j) where j is a valid anchor. Default: False.
        anchor_cols: Tensor of shape [N] mapping each row to its corresponding
            column index for self-exclusion, and determining anchor IDs in
            returned pairs. Required for rectangular matrices. For square
            matrices, defaults to [0, 1, ..., N-1] (diagonal).
        valid_mask: Optional tensor of shape [M] where 1=valid, 0=invalid.
            Pairs involving invalid targets or anchors are excluded.
            inf/nan distances are also treated as invalid.
        max_pairs: Maximum number of pairs to return. If exceeded, pairs are
            randomly sampled. Default: None (no limit).

    Returns:
        Tensor of shape [P, 2] containing (anchor_id, target_id) pairs.
        anchor_id comes from anchor_cols[row], target_id is the column index.

    Example:
        >>> distances = torch.randn(100, 100).abs()  # 100x100 distance matrix
        >>> pairs = pairs_knn(distances, k=5)  # 5-NN for each anchor
        >>> pairs.shape
        torch.Size([500, 2])  # 100 anchors * 5 neighbors

        >>> # Rectangular case: 10 anchors, 100 candidates
        >>> distances = torch.randn(10, 100).abs()
        >>> anchor_cols = torch.arange(10)  # anchors are first 10 columns
        >>> pairs = pairs_knn(distances, k=5, anchor_cols=anchor_cols)
    """
    N, M = distances.shape

    if symmetric and N != M:
        raise ValueError("symmetric=True is only valid for square matrices")

    if symmetric and anchor_cols is not None:
        raise ValueError("symmetric=True cannot be used with anchor_cols")

    dist, anchor_cols, _ = _prepare_distances(distances, anchor_cols, valid_mask)

    # Count valid candidates per row
    valid_per_row = (dist < float("inf")).sum(dim=1)

    # Clamp k to available valid candidates per row
    k_per_row = torch.clamp(valid_per_row, max=k)
    max_k = k_per_row.max().item()

    if max_k == 0:
        return torch.empty((0, 2), dtype=torch.long, device=distances.device)

    # Get top-k (smallest distances) per row
    # Use max_k to handle rows with fewer valid candidates
    _, indices = torch.topk(dist, k=max_k, dim=1, largest=False)

    # Build pairs
    pairs_list = []
    for row in range(N):
        row_k = k_per_row[row].item()
        if row_k > 0:
            anchor_id = anchor_cols[row]
            target_ids = indices[row, :row_k]
            row_pairs = torch.stack(
                [anchor_id.expand(row_k), target_ids], dim=1
            )
            pairs_list.append(row_pairs)

    if not pairs_list:
        return torch.empty((0, 2), dtype=torch.long, device=distances.device)

    pairs = torch.cat(pairs_list, dim=0)

    if symmetric:
        pairs = _add_symmetric_pairs(pairs, anchor_cols)

    return _sample_pairs(pairs, max_pairs)


def pairs_mutual_knn(
    distances: torch.Tensor,
    k: int,
    valid_mask: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> torch.Tensor:
    """
    Generate pairs from mutual k-nearest neighbors.

    Only includes pairs (i, j) where j is in i's k-NN AND i is in j's k-NN.
    This is inherently symmetric - both (i, j) and (j, i) are included.

    Args:
        distances: Distance matrix of shape [N, N]. Must be square for
            mutual KNN to be well-defined.
        k: Number of nearest neighbors to consider for mutual membership.
        valid_mask: Optional tensor of shape [N] where 1=valid, 0=invalid.
            Pairs involving invalid targets or anchors are excluded.
        max_pairs: Maximum number of pairs to return. If exceeded, pairs are
            randomly sampled. Default: None (no limit).

    Returns:
        Tensor of shape [P, 2] containing (anchor_id, target_id) pairs.

    Example:
        >>> distances = torch.randn(100, 100).abs()
        >>> pairs = pairs_mutual_knn(distances, k=10)
        >>> # Only pairs where both points are in each other's top-10
    """
    N, M = distances.shape

    if N != M:
        raise ValueError("Mutual KNN requires a square distance matrix")

    dist, anchor_cols, _ = _prepare_distances(distances, None, valid_mask)

    # Get k-NN for each row
    valid_per_row = (dist < float("inf")).sum(dim=1)
    k_clamped = torch.clamp(valid_per_row, max=k)
    max_k = k_clamped.max().item()

    if max_k == 0:
        return torch.empty((0, 2), dtype=torch.long, device=distances.device)

    _, knn_indices = torch.topk(dist, k=max_k, dim=1, largest=False)

    # Build k-NN membership matrix: knn_matrix[i, j] = True if j is in i's k-NN
    knn_matrix = torch.zeros((N, M), dtype=torch.bool, device=distances.device)
    for row in range(N):
        row_k = k_clamped[row].item()
        if row_k > 0:
            knn_matrix[row, knn_indices[row, :row_k]] = True

    # Mutual k-NN: both directions must be true
    # Map columns back to rows using anchor_cols
    # For default square case, anchor_cols[i] = i, so this is just knn_matrix & knn_matrix.T
    mutual_matrix = knn_matrix & knn_matrix.T

    # Extract pairs from mutual matrix
    row_idx, col_idx = torch.where(mutual_matrix)

    if row_idx.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=distances.device)

    # Map row indices to anchor IDs
    anchor_ids = anchor_cols[row_idx]
    pairs = torch.stack([anchor_ids, col_idx], dim=1)

    return _sample_pairs(pairs, max_pairs)


def pairs_quantile(
    distances: torch.Tensor,
    low: float = 0.0,
    high: float = 0.1,
    symmetric: bool = False,
    anchor_cols: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> torch.Tensor:
    """
    Generate pairs from a quantile range of distances.

    Selects pairs whose distances fall within [low, high) quantiles of the
    valid distance distribution.

    Args:
        distances: Distance matrix of shape [N, M].
        low: Lower quantile bound (inclusive), in [0, 1]. Default: 0.0.
        high: Upper quantile bound (exclusive), in [0, 1]. Default: 0.1.
        symmetric: If True and matrix is square, also add reverse pairs (j, i)
            for each (i, j) where j is a valid anchor. Default: False.
        anchor_cols: Tensor of shape [N] mapping each row to its corresponding
            column index. Required for rectangular matrices.
        valid_mask: Optional tensor of shape [M] where 1=valid, 0=invalid.
        max_pairs: Maximum number of pairs to return. If exceeded, pairs are
            randomly sampled. Default: None (no limit).

    Returns:
        Tensor of shape [P, 2] containing (anchor_id, target_id) pairs.

    Example:
        >>> distances = torch.cdist(embeddings, embeddings)
        >>> # Closest 10% as positives
        >>> pos_pairs = pairs_quantile(distances, low=0.0, high=0.1)
        >>> # 50th-75th percentile as negatives
        >>> neg_pairs = pairs_quantile(distances, low=0.5, high=0.75)
    """
    N, M = distances.shape

    if symmetric and N != M:
        raise ValueError("symmetric=True is only valid for square matrices")

    if symmetric and anchor_cols is not None:
        raise ValueError("symmetric=True cannot be used with anchor_cols")

    if not (0 <= low < high <= 1):
        raise ValueError(f"Require 0 <= low < high <= 1, got low={low}, high={high}")

    dist, anchor_cols, _ = _prepare_distances(distances, anchor_cols, valid_mask)

    # Get valid distances for quantile computation
    valid_distances = dist[dist < float("inf")]

    if valid_distances.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=distances.device)

    # Compute quantile thresholds
    q_low = torch.quantile(valid_distances, low)
    q_high = torch.quantile(valid_distances, high)

    # Select pairs in range [q_low, q_high)
    in_range = (dist >= q_low) & (dist < q_high)
    row_idx, col_idx = torch.where(in_range)

    if row_idx.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=distances.device)

    # Map row indices to anchor IDs
    anchor_ids = anchor_cols[row_idx]
    pairs = torch.stack([anchor_ids, col_idx], dim=1)

    if symmetric:
        pairs = _add_symmetric_pairs(pairs, anchor_cols)

    return _sample_pairs(pairs, max_pairs)


def pairs_radius(
    distances: torch.Tensor,
    min_dist: float = 0.0,
    max_dist: float = float("inf"),
    symmetric: bool = False,
    anchor_cols: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> torch.Tensor:
    """
    Generate pairs from an absolute distance range.

    Selects pairs whose distances fall within [min_dist, max_dist).

    Args:
        distances: Distance matrix of shape [N, M].
        min_dist: Minimum distance (inclusive). Default: 0.0.
        max_dist: Maximum distance (exclusive). Default: inf.
        symmetric: If True and matrix is square, also add reverse pairs (j, i)
            for each (i, j) where j is a valid anchor. Default: False.
        anchor_cols: Tensor of shape [N] mapping each row to its corresponding
            column index. Required for rectangular matrices.
        valid_mask: Optional tensor of shape [M] where 1=valid, 0=invalid.
        max_pairs: Maximum number of pairs to return. If exceeded, pairs are
            randomly sampled. Default: None (no limit).

    Returns:
        Tensor of shape [P, 2] containing (anchor_id, target_id) pairs.

    Example:
        >>> distances = torch.cdist(embeddings, embeddings)
        >>> # Pairs within distance 0.5
        >>> close_pairs = pairs_radius(distances, min_dist=0.0, max_dist=0.5)
        >>> # Pairs between distance 2.0 and 5.0
        >>> mid_pairs = pairs_radius(distances, min_dist=2.0, max_dist=5.0)
    """
    N, M = distances.shape

    if symmetric and N != M:
        raise ValueError("symmetric=True is only valid for square matrices")

    if symmetric and anchor_cols is not None:
        raise ValueError("symmetric=True cannot be used with anchor_cols")

    if min_dist >= max_dist:
        raise ValueError(f"Require min_dist < max_dist, got {min_dist} >= {max_dist}")

    dist, anchor_cols, _ = _prepare_distances(distances, anchor_cols, valid_mask)

    # Select pairs in range [min_dist, max_dist)
    in_range = (dist >= min_dist) & (dist < max_dist)
    row_idx, col_idx = torch.where(in_range)

    if row_idx.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=distances.device)

    # Map row indices to anchor IDs
    anchor_ids = anchor_cols[row_idx]
    pairs = torch.stack([anchor_ids, col_idx], dim=1)

    if symmetric:
        pairs = _add_symmetric_pairs(pairs, anchor_cols)

    return _sample_pairs(pairs, max_pairs)


def apply_spatial_constraint(
    distances: torch.Tensor,
    spatial_distances: torch.Tensor,
    min_spatial_distance: float,
) -> torch.Tensor:
    """
    Apply spatial distance constraint to a distance matrix.

    Sets distances to infinity where spatial distance is below threshold,
    effectively excluding those pairs from selection.

    Args:
        distances: Distance matrix [N, N] in feature space
        spatial_distances: Distance matrix [N, N] in pixel space
        min_spatial_distance: Minimum spatial distance required

    Returns:
        Modified distance matrix with inf where spatial constraint violated
    """
    masked = distances.clone()
    masked[spatial_distances < min_spatial_distance] = float('inf')
    return masked


def pairs_with_spatial_constraint(
    feature_distances: torch.Tensor,
    spatial_distances: torch.Tensor,
    positive_strategy: str = 'mutual-knn',
    positive_k: int = 16,
    positive_min_spatial: float = 4.0,
    negative_strategy: str = 'quantile',
    negative_quantile_low: float = 0.5,
    negative_quantile_high: float = 0.75,
    negative_min_spatial: float = 8.0,
    max_pairs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate positive and negative pairs with spatial distance constraints.

    This combines feature-based pair selection with spatial constraints to
    prevent selecting trivially similar spatial neighbors.

    Args:
        feature_distances: Distance matrix [N, N] in feature space
        spatial_distances: Distance matrix [N, N] in pixel space (L2)
        positive_strategy: Strategy for positive selection ('mutual-knn' or 'knn')
        positive_k: K for KNN-based positive selection
        positive_min_spatial: Minimum spatial distance for positives
        negative_strategy: Strategy for negative selection ('quantile')
        negative_quantile_low: Lower quantile for negative selection
        negative_quantile_high: Upper quantile for negative selection
        negative_min_spatial: Minimum spatial distance for negatives
        max_pairs: Maximum pairs to return (per positive/negative)

    Returns:
        Tuple of (pos_pairs, neg_pairs), each [P, 2] with (anchor, target) indices

    Example:
        >>> feature_dist = torch.cdist(features, features)
        >>> spatial_dist = torch.cdist(coords.float(), coords.float())
        >>> pos, neg = pairs_with_spatial_constraint(
        ...     feature_dist, spatial_dist,
        ...     positive_k=16, positive_min_spatial=4.0,
        ...     negative_quantile_low=0.5, negative_quantile_high=0.75,
        ...     negative_min_spatial=8.0
        ... )
    """
    # Apply spatial constraints
    pos_dist = apply_spatial_constraint(
        feature_distances, spatial_distances, positive_min_spatial
    )
    neg_dist = apply_spatial_constraint(
        feature_distances, spatial_distances, negative_min_spatial
    )

    # Generate positive pairs
    if positive_strategy == 'mutual-knn':
        pos_pairs = pairs_mutual_knn(pos_dist, k=positive_k, max_pairs=max_pairs)
    elif positive_strategy == 'knn':
        pos_pairs = pairs_knn(pos_dist, k=positive_k, max_pairs=max_pairs)
    else:
        raise ValueError(f"Unknown positive strategy: {positive_strategy}")

    # Generate negative pairs
    if negative_strategy == 'quantile':
        neg_pairs = pairs_quantile(
            neg_dist,
            low=negative_quantile_low,
            high=negative_quantile_high,
            max_pairs=max_pairs,
        )
    else:
        raise ValueError(f"Unknown negative strategy: {negative_strategy}")

    return pos_pairs, neg_pairs
