"""
Spatial utilities for working with pixel coordinates and features.

This module provides common operations for extracting features at pixel
locations and computing spatial relationships.
"""

from __future__ import annotations

import torch


def compute_spatial_distances(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise L2 spatial distances between coordinates.

    Args:
        coords: Tensor of shape [N, 2] with (row, col) coordinates

    Returns:
        Distance matrix [N, N] where entry (i, j) is the L2 distance
        between coords[i] and coords[j]

    Example:
        >>> coords = torch.tensor([[0, 0], [0, 3], [4, 0]])
        >>> dists = compute_spatial_distances(coords)
        >>> dists[0, 1]  # distance from (0,0) to (0,3)
        tensor(3.)
        >>> dists[0, 2]  # distance from (0,0) to (4,0)
        tensor(4.)
    """
    coords_float = coords.float()
    return torch.cdist(coords_float, coords_float)


def compute_spatial_distances_rectangular(
    coords_a: torch.Tensor,
    coords_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise L2 spatial distances between two sets of coordinates.

    Args:
        coords_a: Tensor of shape [N, 2] with (row, col) coordinates (anchors)
        coords_b: Tensor of shape [M, 2] with (row, col) coordinates (candidates)

    Returns:
        Distance matrix [N, M] where entry (i, j) is the L2 distance
        between coords_a[i] and coords_b[j]

    Example:
        >>> anchors = torch.tensor([[0, 0], [10, 10]])
        >>> candidates = torch.tensor([[0, 3], [4, 0], [10, 15]])
        >>> dists = compute_spatial_distances_rectangular(anchors, candidates)
        >>> dists.shape
        torch.Size([2, 3])
    """
    coords_a_float = coords_a.float()
    coords_b_float = coords_b.float()
    return torch.cdist(coords_a_float, coords_b_float)


def get_valid_pixel_coords(mask: torch.Tensor) -> torch.Tensor:
    """
    Get coordinates of all valid (True) pixels in a mask.

    Args:
        mask: Boolean mask tensor [H, W]

    Returns:
        Coordinates [N, 2] as (row, col) for all True pixels

    Example:
        >>> mask = torch.zeros(5, 5, dtype=torch.bool)
        >>> mask[1, 2] = True
        >>> mask[3, 4] = True
        >>> coords = get_valid_pixel_coords(mask)
        >>> coords
        tensor([[1, 2], [3, 4]])
    """
    rows, cols = torch.where(mask)
    return torch.stack([rows, cols], dim=1)


def find_anchor_indices_in_candidates(
    anchor_coords: torch.Tensor,
    candidate_coords: torch.Tensor,
) -> torch.Tensor:
    """
    Find the index of each anchor in the candidate coordinate array.

    This is used to create anchor_cols for rectangular distance matrices,
    mapping each anchor row to its corresponding column for self-exclusion.

    Args:
        anchor_coords: Tensor of shape [N, 2] with anchor (row, col) coordinates
        candidate_coords: Tensor of shape [M, 2] with candidate coordinates

    Returns:
        Tensor of shape [N] where entry i is the index j such that
        candidate_coords[j] == anchor_coords[i]. Returns -1 if not found.

    Example:
        >>> anchors = torch.tensor([[5, 10], [20, 30]])
        >>> candidates = torch.tensor([[0, 0], [5, 10], [10, 20], [20, 30]])
        >>> indices = find_anchor_indices_in_candidates(anchors, candidates)
        >>> indices
        tensor([1, 3])
    """
    n_anchors = anchor_coords.shape[0]
    n_candidates = candidate_coords.shape[0]
    device = anchor_coords.device

    # Expand for broadcasting comparison
    # anchor_coords: [N, 1, 2], candidate_coords: [1, M, 2]
    anchor_exp = anchor_coords.unsqueeze(1)  # [N, 1, 2]
    candidate_exp = candidate_coords.unsqueeze(0)  # [1, M, 2]

    # Check equality for both coordinates
    matches = (anchor_exp == candidate_exp).all(dim=2)  # [N, M]

    # Get index of first match for each anchor (-1 if not found)
    indices = torch.full((n_anchors,), -1, dtype=torch.long, device=device)
    for i in range(n_anchors):
        match_indices = torch.where(matches[i])[0]
        if match_indices.numel() > 0:
            indices[i] = match_indices[0]

    return indices


def extract_at_locations(
    feature: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """
    Extract feature vectors at given pixel locations.

    Args:
        feature: Feature tensor [C, H, W]
        coords: Coordinates [N, 2] as (row, col)

    Returns:
        Extracted features [N, C]

    Example:
        >>> feature = torch.randn(64, 256, 256)  # 64 channels, 256x256
        >>> coords = torch.tensor([[10, 20], [30, 40], [50, 60]])
        >>> extracted = extract_at_locations(feature, coords)
        >>> extracted.shape
        torch.Size([3, 64])
    """
    rows, cols = coords[:, 0], coords[:, 1]
    # feature[:, rows, cols] gives [C, N], transpose to [N, C]
    return feature[:, rows, cols].T


def extract_temporal_at_locations(
    feature: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Extract temporal feature vectors at given pixel locations.

    Args:
        feature: Feature tensor [C, T, H, W]
        coords: Coordinates [N, 2] as (row, col)

    Returns:
        Extracted features [N, T, C]
    """
    rows, cols = coords[:, 0], coords[:, 1]
    # feature[:, :, rows, cols] gives [C, T, N], permute to [N, T, C]
    return feature[:, :, rows, cols].permute(2, 1, 0)


def spatial_knn_pairs(
    anchor_coords: torch.Tensor,
    mask: torch.Tensor,
    k: int = 4,
    max_radius: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest spatial neighbors for each anchor using offset grid.

    This is an efficient implementation that avoids computing a full distance
    matrix. Instead, it precomputes neighbor offsets within max_radius and
    applies them to all anchors simultaneously.

    Args:
        anchor_coords: Tensor [N, 2] of (row, col) anchor coordinates
        mask: Boolean mask [H, W] indicating valid pixels
        k: Number of nearest neighbors per anchor
        max_radius: Maximum distance to consider for neighbors

    Returns:
        Tuple of:
            - anchor_indices: Tensor [M] of anchor indices (into anchor_coords)
            - neighbor_coords: Tensor [M, 2] of neighbor (row, col) coordinates
        Only returns pairs where the neighbor is valid in the mask.
    """
    device = anchor_coords.device
    n_anchors = anchor_coords.shape[0]
    H, W = mask.shape

    # Create offset grid: all (dr, dc) within max_radius
    r = max_radius
    dr = torch.arange(-r, r + 1, device=device)
    dc = torch.arange(-r, r + 1, device=device)
    grid_r, grid_c = torch.meshgrid(dr, dc, indexing='ij')
    offsets = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)  # [(2r+1)^2, 2]

    # Compute distance from center for each offset
    offset_dists = torch.sqrt((offsets[:, 0].float() ** 2) + (offsets[:, 1].float() ** 2))

    # Exclude self (distance 0) and offsets beyond max_radius
    valid_offset_mask = (offset_dists > 0) & (offset_dists <= max_radius)
    valid_offsets = offsets[valid_offset_mask]
    valid_dists = offset_dists[valid_offset_mask]

    # Sort by distance and take top k
    sorted_indices = torch.argsort(valid_dists)
    k_actual = min(k, len(sorted_indices))
    top_k_indices = sorted_indices[:k_actual]
    neighbor_offsets = valid_offsets[top_k_indices]  # [k, 2]

    # Apply offsets to all anchors: [N, 1, 2] + [1, k, 2] -> [N, k, 2]
    anchor_exp = anchor_coords.unsqueeze(1)  # [N, 1, 2]
    offset_exp = neighbor_offsets.unsqueeze(0)  # [1, k, 2]
    neighbor_coords = anchor_exp + offset_exp  # [N, k, 2]

    # Check bounds
    in_bounds = (
        (neighbor_coords[:, :, 0] >= 0) &
        (neighbor_coords[:, :, 0] < H) &
        (neighbor_coords[:, :, 1] >= 0) &
        (neighbor_coords[:, :, 1] < W)
    )  # [N, k]

    # Check mask validity (only for in-bounds coordinates)
    neighbor_r = neighbor_coords[:, :, 0].clamp(0, H - 1).long()
    neighbor_c = neighbor_coords[:, :, 1].clamp(0, W - 1).long()
    mask_valid = mask[neighbor_r, neighbor_c] & in_bounds  # [N, k]

    # Create anchor index tensor
    anchor_indices = torch.arange(n_anchors, device=device).unsqueeze(1).expand(-1, k_actual)

    # Get valid pairs
    valid_pairs_mask = mask_valid.flatten()
    anchor_flat = anchor_indices.flatten()[valid_pairs_mask]
    neighbor_coords_flat = neighbor_coords.reshape(-1, 2)[valid_pairs_mask].long()

    return anchor_flat, neighbor_coords_flat


def spatial_negative_pairs(
    anchor_coords: torch.Tensor,
    mask: torch.Tensor,
    min_distance: float = 16.0,
    max_distance: float | None = None,
    n_per_anchor: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample spatially distant negative pairs for each anchor.

    For each anchor, randomly samples valid pixels within a distance range.
    This avoids computing quantiles over large distance matrices.

    Args:
        anchor_coords: Tensor [N, 2] of (row, col) anchor coordinates
        mask: Boolean mask [H, W] indicating valid pixels
        min_distance: Minimum distance for negatives
        max_distance: Maximum distance for negatives (None = no limit)
        n_per_anchor: Number of negatives to sample per anchor

    Returns:
        Tuple of:
            - anchor_indices: Tensor [M] of anchor indices (into anchor_coords)
            - neighbor_coords: Tensor [M, 2] of neighbor (row, col) coordinates
    """
    device = anchor_coords.device
    n_anchors = anchor_coords.shape[0]
    H, W = mask.shape

    # Get all valid pixel coordinates
    valid_coords = get_valid_pixel_coords(mask)  # [V, 2]
    n_valid = valid_coords.shape[0]

    if n_valid == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
        )

    # Compute distances from each anchor to all valid pixels
    # This is [N, V] but we process per-anchor to avoid huge matrix
    all_anchor_indices = []
    all_neighbor_coords = []

    for i in range(n_anchors):
        anchor = anchor_coords[i].float()  # [2]

        # Compute distances to all valid pixels
        dists = torch.sqrt(
            ((valid_coords.float() - anchor) ** 2).sum(dim=1)
        )  # [V]

        # Filter by distance range
        dist_mask = dists >= min_distance
        if max_distance is not None:
            dist_mask = dist_mask & (dists <= max_distance)

        candidate_indices = torch.where(dist_mask)[0]

        if candidate_indices.numel() == 0:
            continue

        # Random sample
        n_sample = min(n_per_anchor, candidate_indices.numel())
        perm = torch.randperm(candidate_indices.numel(), device=device)[:n_sample]
        sampled_indices = candidate_indices[perm]

        # Collect results
        all_anchor_indices.append(torch.full((n_sample,), i, dtype=torch.long, device=device))
        all_neighbor_coords.append(valid_coords[sampled_indices])

    if len(all_anchor_indices) == 0:
        return (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros((0, 2), dtype=torch.long, device=device),
        )

    return torch.cat(all_anchor_indices), torch.cat(all_neighbor_coords)
