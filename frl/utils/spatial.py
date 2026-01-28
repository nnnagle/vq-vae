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
