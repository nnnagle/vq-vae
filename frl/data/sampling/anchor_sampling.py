"""
Anchor sampling strategies for contrastive learning.

This module provides functions to sample anchor pixel locations within a patch
for use in contrastive learning. Anchors are the reference points for which
positive and negative pairs are generated.

Strategies:
    - grid: Regular grid of points with optional jitter
    - grid-plus-supplement: Grid points plus random supplement samples
      (uniform or weighted)

Example:
    >>> from data.sampling import build_anchor_sampler
    >>>
    >>> # From config
    >>> sampler = build_anchor_sampler(bindings_config, 'grid-plus-supplement')
    >>> anchors = sampler(mask, training=True)
    >>>
    >>> # With weighted sampling (pass sample dict for weight resolution)
    >>> sampler = build_anchor_sampler(bindings_config, 'grid-plus-supplement-ysfc')
    >>> anchors = sampler(mask, training=True, sample=batch)
    >>>
    >>> # Direct usage
    >>> anchors = sample_anchors_grid_plus_supplement(
    ...     mask, stride=16, border=16, jitter_radius=4, supplement_n=104
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch


@dataclass
class GridConfig:
    """Configuration for grid-based anchor sampling."""
    stride: int = 16
    border: int = 16
    jitter_radius: int = 4


@dataclass
class WeightSpec:
    """Specification for a single weight source.

    Simple mask references (e.g. ``static_mask.aoi``) have only ``channel``
    set.  Continuous channels that need a transform (e.g. inverse of
    ``static.ysfc_min``) additionally carry a ``transform`` string.
    """
    channel: str            # e.g. "static_mask.aoi" or "static.ysfc_min"
    transform: Optional[str] = None  # e.g. "inverse"


@dataclass
class GridPlusSupplementConfig:
    """Configuration for grid-plus-supplement anchor sampling."""
    grid: GridConfig
    supplement_n: int = 104
    weight_specs: List[WeightSpec] = field(default_factory=list)


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
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample anchor pixel locations using grid-plus-supplement strategy.

    Combines a regular grid of anchors with additional randomly sampled
    points from valid pixels.  When *weights* is provided the supplement
    points are drawn proportionally to the weight at each valid pixel;
    otherwise uniform random sampling is used.

    Args:
        mask: Valid pixel mask [H, W], True = valid
        stride: Grid stride in pixels
        border: Border to exclude from grid
        jitter_radius: Random jitter applied to grid origin (0 for deterministic)
        supplement_n: Number of additional random samples
        weights: Optional per-pixel sampling weight [H, W].  Only values at
            valid (masked) pixels matter.  Need not be normalised.

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
        n_supplement = min(supplement_n, len(valid_indices))

        if weights is not None:
            # Weighted sampling: gather weights at valid pixel locations
            flat_weights = weights.flatten()[valid_indices].float()
            # Clamp negatives and ensure non-zero sum
            flat_weights = flat_weights.clamp(min=0.0)
            total = flat_weights.sum()
            if total > 0:
                probs = flat_weights / total
            else:
                probs = torch.ones_like(flat_weights) / len(flat_weights)
            chosen = torch.multinomial(probs, n_supplement, replacement=False)
            supplement_flat = valid_indices[chosen]
        else:
            # Uniform sampling
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


# ---------------------------------------------------------------------------
# Weight resolution helpers
# ---------------------------------------------------------------------------

def _resolve_channel_from_sample(
    sample: Dict[str, Any],
    channel_ref: str,
) -> torch.Tensor:
    """Resolve a ``dataset_group.channel_name`` reference into a [H, W] tensor.

    Args:
        sample: Batch dict as returned by the dataset ``__getitem__``.
            Keys are group names (``static_mask``, ``static``, …) mapping to
            tensors, plus ``metadata`` with ``channel_names``.
        channel_ref: Dot-separated reference, e.g. ``static_mask.aoi`` or
            ``static.ysfc_min``.

    Returns:
        2-D tensor [H, W] with the channel data (float32).
    """
    parts = channel_ref.split('.')
    if len(parts) != 2:
        raise ValueError(
            f"weight_by channel reference must be 'group.channel', got '{channel_ref}'"
        )
    group_name, channel_name = parts

    group_data = sample[group_name]  # [C, H, W] or [C, T, H, W]
    channel_names = sample['metadata']['channel_names'][group_name]
    if channel_name not in channel_names:
        raise ValueError(
            f"Channel '{channel_name}' not found in group '{group_name}'. "
            f"Available: {channel_names}"
        )
    idx = channel_names.index(channel_name)
    data = group_data[idx]  # [H, W] or [T, H, W]
    if data.ndim == 3:
        # Temporal – shouldn't normally appear in weights, but take first slice
        data = data[0]
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    return data.float()


def _apply_inverse_frequency(
    data: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute inverse-frequency weights over the valid region of a channel.

    Each valid pixel gets weight ``1 / count(v)`` where ``count(v)`` is the
    number of **valid** pixels that share value ``v``.  This upweights pixels
    whose discrete value covers a small fraction of the valid area so that
    supplement samples are drawn more uniformly across all levels.

    Designed for discrete-valued channels (e.g. ~40 unique ``ysfc_min``
    levels stored as float16).  Pixels outside *valid_mask* get weight 0.

    Args:
        data: [H, W] channel tensor (float32).
        valid_mask: [H, W] boolean tensor – True where preceding binary
            masks are all 1 and the channel value is finite.

    Returns:
        Weight tensor [H, W] (float32, ≥ 0).
    """
    eligible = valid_mask & data.isfinite()
    vals = data[eligible]
    if vals.numel() == 0:
        return torch.ones_like(data) * valid_mask.float()

    # Count pixels per unique value within the valid region
    unique_vals, inverse_idx = vals.unique(return_inverse=True)
    counts = torch.zeros(unique_vals.shape[0], device=data.device)
    counts.scatter_add_(0, inverse_idx, torch.ones_like(inverse_idx, dtype=counts.dtype))

    # Weight = 1 / count for each pixel's value
    per_pixel_weight = 1.0 / counts[inverse_idx]

    w = torch.zeros_like(data)
    w[eligible] = per_pixel_weight
    return w


def resolve_supplement_weights(
    sample: Dict[str, Any],
    weight_specs: List[WeightSpec],
) -> torch.Tensor:
    """Build a combined [H, W] weight map from weight specifications.

    Processing order:
        1. All specs **without** a transform (binary masks) are resolved
           first and multiplied together to form a valid-pixel mask.
        2. Specs **with** a transform (e.g. ``inverse-frequency``) are then
           evaluated using only the pixels that survived step 1, so that
           frequency counts reflect the masked region rather than the whole
           patch.
        3. The final weight map is the product of (1) and (2).

    Args:
        sample: Dataset sample dict (from ``__getitem__``).
        weight_specs: List of weight specifications.

    Returns:
        Combined weight tensor [H, W] (float32, ≥ 0).
    """
    # Separate plain masks from transformed specs
    mask_specs = [s for s in weight_specs if s.transform is None]
    transform_specs = [s for s in weight_specs if s.transform is not None]

    # Step 1: multiply binary masks
    combined: Optional[torch.Tensor] = None
    for spec in mask_specs:
        data = _resolve_channel_from_sample(sample, spec.channel).float()
        if combined is None:
            combined = data
        else:
            combined = combined * data

    # Step 2: apply transforms using the mask from step 1
    for spec in transform_specs:
        data = _resolve_channel_from_sample(sample, spec.channel)
        if spec.transform == 'inverse-frequency':
            valid_mask = (combined > 0) if combined is not None else data.isfinite()
            w = _apply_inverse_frequency(data, valid_mask)
        else:
            raise ValueError(f"Unknown weight transform: '{spec.transform}'")

        if combined is None:
            combined = w
        else:
            combined = combined * w

    return combined


# ---------------------------------------------------------------------------
# Config-driven sampler
# ---------------------------------------------------------------------------

class AnchorSampler:
    """
    Callable anchor sampler configured from bindings config.

    Example:
        >>> sampler = AnchorSampler(config, 'grid-plus-supplement')
        >>> anchors = sampler(mask, training=True)

        >>> # Weighted variant – pass sample dict
        >>> sampler = AnchorSampler(config, 'grid-plus-supplement',
        ...                         weight_specs=[...])
        >>> anchors = sampler(mask, training=True, sample=batch)
    """

    def __init__(
        self,
        strategy: str,
        stride: int = 16,
        border: int = 16,
        jitter_radius: int = 4,
        supplement_n: int = 104,
        weight_specs: Optional[List[WeightSpec]] = None,
    ):
        """
        Initialize anchor sampler.

        Args:
            strategy: Sampling strategy ('grid' or 'grid-plus-supplement')
            stride: Grid stride
            border: Border to exclude
            jitter_radius: Jitter radius for training (0 for validation)
            supplement_n: Number of supplement samples (for grid-plus-supplement)
            weight_specs: Optional list of weight specifications for weighted
                supplement sampling.  When provided, the ``sample`` kwarg must
                be passed to ``__call__``.
        """
        self.strategy = strategy
        self.stride = stride
        self.border = border
        self.jitter_radius = jitter_radius
        self.supplement_n = supplement_n
        self.weight_specs = weight_specs or []

    def __call__(
        self,
        mask: torch.Tensor,
        training: bool = True,
        sample: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Sample anchors from mask.

        Args:
            mask: Valid pixel mask [H, W]
            training: If True, apply jitter; if False, deterministic
            sample: Dataset sample dict, required when weight_specs is non-empty

        Returns:
            Anchor coordinates [N, 2]
        """
        jitter = self.jitter_radius if training else 0

        # Resolve weights if configured
        weights = None
        if self.weight_specs and sample is not None:
            weights = resolve_supplement_weights(sample, self.weight_specs)

        if self.strategy == 'grid':
            return sample_anchors_grid(
                mask,
                stride=self.stride,
                border=self.border,
                jitter_radius=jitter,
            )
        elif self.strategy in ('grid-plus-supplement', 'grid-plus-supplement-ysfc'):
            return sample_anchors_grid_plus_supplement(
                mask,
                stride=self.stride,
                border=self.border,
                jitter_radius=jitter,
                supplement_n=self.supplement_n,
                weights=weights,
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")


def _parse_weight_by(weight_by_list: list) -> List[WeightSpec]:
    """Parse the ``weight_by`` list from YAML into :class:`WeightSpec` objects.

    Handles two formats:
        - Simple string: ``"static_mask.aoi"`` → ``WeightSpec(channel=..., transform=None)``
        - Dict with transform: ``{channel: "static.ysfc_min", transform: "inverse"}``

    Args:
        weight_by_list: Raw list from YAML config.

    Returns:
        List of WeightSpec.
    """
    specs = []
    for item in weight_by_list:
        if isinstance(item, str):
            specs.append(WeightSpec(channel=item))
        elif isinstance(item, dict):
            channel = item.get('channel')
            if channel is None:
                raise ValueError(
                    f"weight_by dict entry must have 'channel' key, got: {item}"
                )
            transform = item.get('transform')
            specs.append(WeightSpec(channel=channel, transform=transform))
        else:
            raise ValueError(
                f"weight_by entry must be a string or dict, got: {type(item)}"
            )
    return specs


def build_anchor_sampler(
    bindings_config,
    strategy_name: str,
) -> AnchorSampler:
    """
    Build an anchor sampler from bindings configuration.

    Supports both the parsed ``BindingsConfig`` (with typed
    ``sampling_strategies`` dict) and legacy raw-dict configs.

    Args:
        bindings_config: Parsed bindings configuration
        strategy_name: Name of strategy (e.g., 'grid-plus-supplement')

    Returns:
        Configured AnchorSampler instance
    """
    # Try parsed config first (typed dataclass objects)
    strat_cfg = None
    if hasattr(bindings_config, 'sampling_strategies') and bindings_config.sampling_strategies:
        strat_cfg = bindings_config.sampling_strategies.get(strategy_name)

    if strat_cfg is not None:
        # Use parsed SamplingStrategyConfig
        grid = strat_cfg.grid
        supplement = strat_cfg.supplement

        # Convert WeightByEntry objects to WeightSpec objects
        weight_specs: List[WeightSpec] = []
        if supplement and supplement.weight_by:
            for entry in supplement.weight_by:
                if entry.mask_ref:
                    weight_specs.append(WeightSpec(channel=entry.mask_ref))
                elif entry.channel:
                    weight_specs.append(WeightSpec(
                        channel=entry.channel,
                        transform=entry.transform,
                    ))

        if strategy_name == 'grid':
            return AnchorSampler(
                strategy='grid',
                stride=grid.stride if grid else 16,
                border=grid.exclude_border if grid else 16,
                jitter_radius=grid.jitter.radius if grid and grid.jitter else 4,
                supplement_n=0,
            )
        else:
            return AnchorSampler(
                strategy=strategy_name,
                stride=grid.stride if grid else 16,
                border=grid.exclude_border if grid else 16,
                jitter_radius=grid.jitter.radius if grid and grid.jitter else 4,
                supplement_n=supplement.n if supplement else 104,
                weight_specs=weight_specs,
            )

    # Legacy raw-dict fallback
    if hasattr(bindings_config, 'sampling_strategy') and bindings_config.sampling_strategy:
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
        elif strategy_name in ('grid-plus-supplement', 'grid-plus-supplement-ysfc'):
            gps_config = strategies.get(strategy_name, {})
            grid_config = gps_config.get('grid', {})
            supplement_config = gps_config.get('supplement', {})
            sampling_config = supplement_config.get('sampling', {})

            weight_by_raw = sampling_config.get('weight_by', [])
            weight_specs = _parse_weight_by(weight_by_raw)

            return AnchorSampler(
                strategy=strategy_name,
                stride=grid_config.get('stride', 16),
                border=grid_config.get('exclude_border', 16),
                jitter_radius=grid_config.get('jitter', {}).get('radius', 4),
                supplement_n=supplement_config.get('n', 104),
                weight_specs=weight_specs,
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy_name}")

    # No config found — use defaults
    return AnchorSampler(
        strategy=strategy_name,
        stride=16,
        border=16,
        jitter_radius=4,
        supplement_n=104,
    )
