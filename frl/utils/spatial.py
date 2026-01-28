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
