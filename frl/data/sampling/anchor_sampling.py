"""
Anchor sampling strategies for contrastive learning.

This module provides functions to sample anchor pixel locations within a patch
for use in contrastive learning. Anchors are the reference points for which
positive and negative pairs are generated.

Strategies:
    - grid: Regular grid of points with optional jitter
    - grid-plus-supplement: Grid points plus random supplement samples

Example:
    >>> from data.sampling import build_anchor_sampler
    >>>
    >>> # From config
    >>> sampler = build_anchor_sampler(bindings_config, 'grid-plus-supplement')
    >>> anchors = sampler(mask, training=True)
    >>>
    >>> # Direct usage
    >>> anchors = sample_anchors_grid_plus_supplement(
    ...     mask, stride=16, border=16, jitter_radius=4, supplement_n=104
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class GridConfig:
    """Configuration for grid-based anchor sampling."""
    stride: int = 16
    border: int = 16
    jitter_radius: int = 4


@dataclass
class GridPlusSupplementConfig:
    """Configuration for grid-plus-supplement anchor sampling."""
    grid: GridConfig
    supplement_n: int = 104


def sample_anchors_grid(
    mask: torch.Tensor,
    stride: int = 16,
    border: int = 16,
    jitter_radius: int = 0,
) -> torch.Tensor:
    """
    Sample anchor pixel locations on a regular grid.

    Args:
        mask: Valid pixel mask [H, W], True = valid
        stride: Grid stride in pixels
        border: Border to exclude from grid
        jitter_radius: Random jitter applied to grid origin (0 for deterministic)

    Returns:
        Tensor of shape [N, 2] with (row, col) coordinates of anchors
    """
    H, W = mask.shape
    device = mask.device

    # Apply random jitter to grid origin
    if jitter_radius > 0:
        jitter_r = torch.randint(-jitter_radius, jitter_radius + 1, (1,), device=device).item()
        jitter_c = torch.randint(-jitter_radius, jitter_radius + 1, (1,), device=device).item()
    else:
        jitter_r, jitter_c = 0, 0

    # Generate grid points
    grid_rows = torch.arange(border + jitter_r, H - border, stride, device=device)
    grid_cols = torch.arange(border + jitter_c, W - border, stride, device=device)

    if len(grid_rows) == 0 or len(grid_cols) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    # Create meshgrid of grid points
    row_grid, col_grid = torch.meshgrid(grid_rows, grid_cols, indexing='ij')
    grid_coords = torch.stack([row_grid.flatten(), col_grid.flatten()], dim=1)

    # Filter to valid grid points
    if grid_coords.shape[0] > 0:
        grid_valid = mask[grid_coords[:, 0], grid_coords[:, 1]]
        grid_coords = grid_coords[grid_valid]

    return grid_coords.long()


def sample_anchors_grid_plus_supplement(
    mask: torch.Tensor,
    stride: int = 16,
    border: int = 16,
    jitter_radius: int = 0,
    supplement_n: int = 104,
) -> torch.Tensor:
    """
    Sample anchor pixel locations using grid-plus-supplement strategy.

    Combines a regular grid of anchors with additional randomly sampled
    points from valid pixels.

    Args:
        mask: Valid pixel mask [H, W], True = valid
        stride: Grid stride in pixels
        border: Border to exclude from grid
        jitter_radius: Random jitter applied to grid origin (0 for deterministic)
        supplement_n: Number of additional random samples

    Returns:
        Tensor of shape [N, 2] with (row, col) coordinates of anchors
    """
    H, W = mask.shape
    device = mask.device

    # Get grid anchors
    grid_coords = sample_anchors_grid(mask, stride, border, jitter_radius)

    # Sample supplement points from valid pixels
    valid_indices = torch.where(mask.flatten())[0]

    if len(valid_indices) > 0 and supplement_n > 0:
        # Random sample from valid pixels
        n_supplement = min(supplement_n, len(valid_indices))
        perm = torch.randperm(len(valid_indices), device=device)[:n_supplement]
        supplement_flat = valid_indices[perm]

        # Convert flat indices to (row, col)
        supplement_rows = supplement_flat // W
        supplement_cols = supplement_flat % W
        supplement_coords = torch.stack([supplement_rows, supplement_cols], dim=1)
    else:
        supplement_coords = torch.empty((0, 2), dtype=torch.long, device=device)

    # Combine grid and supplement
    if grid_coords.shape[0] > 0 and supplement_coords.shape[0] > 0:
        anchors = torch.cat([grid_coords, supplement_coords], dim=0)
    elif grid_coords.shape[0] > 0:
        anchors = grid_coords
    else:
        anchors = supplement_coords

    return anchors.long()


class AnchorSampler:
    """
    Callable anchor sampler configured from bindings config.

    Example:
        >>> sampler = AnchorSampler(config, 'grid-plus-supplement')
        >>> anchors = sampler(mask, training=True)
    """

    def __init__(
        self,
        strategy: str,
        stride: int = 16,
        border: int = 16,
        jitter_radius: int = 4,
        supplement_n: int = 104,
    ):
        """
        Initialize anchor sampler.

        Args:
            strategy: Sampling strategy ('grid' or 'grid-plus-supplement')
            stride: Grid stride
            border: Border to exclude
            jitter_radius: Jitter radius for training (0 for validation)
            supplement_n: Number of supplement samples (for grid-plus-supplement)
        """
        self.strategy = strategy
        self.stride = stride
        self.border = border
        self.jitter_radius = jitter_radius
        self.supplement_n = supplement_n

    def __call__(
        self,
        mask: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Sample anchors from mask.

        Args:
            mask: Valid pixel mask [H, W]
            training: If True, apply jitter; if False, deterministic

        Returns:
            Anchor coordinates [N, 2]
        """
        jitter = self.jitter_radius if training else 0

        if self.strategy == 'grid':
            return sample_anchors_grid(
                mask,
                stride=self.stride,
                border=self.border,
                jitter_radius=jitter,
            )
        elif self.strategy == 'grid-plus-supplement':
            return sample_anchors_grid_plus_supplement(
                mask,
                stride=self.stride,
                border=self.border,
                jitter_radius=jitter,
                supplement_n=self.supplement_n,
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")


def build_anchor_sampler(
    bindings_config,
    strategy_name: str,
) -> AnchorSampler:
    """
    Build an anchor sampler from bindings configuration.

    Args:
        bindings_config: Parsed bindings configuration
        strategy_name: Name of strategy (e.g., 'grid-plus-supplement')

    Returns:
        Configured AnchorSampler instance
    """
    # Get sampling strategy config from bindings
    # The config has a 'sampling_strategy' dict with strategy definitions
    if not hasattr(bindings_config, 'sampling_strategy'):
        # Fall back to defaults
        return AnchorSampler(
            strategy=strategy_name,
            stride=16,
            border=16,
            jitter_radius=4,
            supplement_n=104,
        )

    strategies = bindings_config.sampling_strategy

    if strategy_name == 'grid':
        grid_config = strategies.get('grid', {})
        return AnchorSampler(
            strategy='grid',
            stride=grid_config.get('stride', 16),
            border=grid_config.get('exclude_border', 16),
            jitter_radius=grid_config.get('jitter', {}).get('radius', 4),
            supplement_n=0,
        )

    elif strategy_name == 'grid-plus-supplement':
        gps_config = strategies.get('grid-plus-supplement', {})
        grid_config = gps_config.get('grid', {})
        supplement_config = gps_config.get('supplement', {})

        return AnchorSampler(
            strategy='grid-plus-supplement',
            stride=grid_config.get('stride', 16),
            border=grid_config.get('exclude_border', 16),
            jitter_radius=grid_config.get('jitter', {}).get('radius', 4),
            supplement_n=supplement_config.get('n', 104),
        )

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy_name}")
