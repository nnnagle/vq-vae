"""
Data configuration classes for the refactored dataset loader.

This module defines the structure of dataset bindings configuration,
focusing on simple, typed data structures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import numpy as np


@dataclass
class ChannelConfig:
    """Configuration for a single channel within a dataset group.

    A channel can be either:
    - Loaded from a zarr source (has `source`)
    - Computed from a formula (has `formula`)
    - Loaded and thresholded (has `source` and `ok_if`)
    """
    name: str

    # Source loading (mutually exclusive with formula)
    source: Optional[str] = None

    # Formula-based derivation (mutually exclusive with source)
    formula: Optional[str] = None

    # Optional year extraction for temporal sources → static output
    year: Optional[int] = None
    time: Optional[Dict[str, int]] = None  # Alternative: {use: 2024}

    # Thresholding logic
    ok_if: Optional[Dict[str, Any]] = None  # {op: ">=", value: 0.25}

    # Fill value handling
    fill_value: Optional[float] = None

    # Temporal reduction (e.g. 'min', 'max', 'mean', 'median')
    # Collapses a temporal source [T, H, W] → [H, W] using a nan-safe reducer
    reducer: Optional[str] = None

    # Supported reducer names mapped to numpy functions
    _REDUCERS = {
        'min': 'nanmin',
        'max': 'nanmax',
        'mean': 'nanmean',
        'median': 'nanmedian',
        'std': 'nanstd',
        'sum': 'nansum',
    }

    def __post_init__(self):
        """Validate channel configuration."""
        # Normalize time specification
        if self.time is not None and 'use' in self.time:
            self.year = self.time['use']

        # Ensure exactly one of source or formula
        has_source = self.source is not None
        has_formula = self.formula is not None

        if has_source == has_formula:
            raise ValueError(
                f"Channel '{self.name}' must have exactly one of 'source' or 'formula', "
                f"got source={self.source}, formula={self.formula}"
            )

        # Validate reducer
        if self.reducer is not None:
            if self.reducer not in self._REDUCERS:
                raise ValueError(
                    f"Channel '{self.name}' has unsupported reducer '{self.reducer}', "
                    f"expected one of {list(self._REDUCERS.keys())}"
                )
            if not has_source:
                raise ValueError(
                    f"Channel '{self.name}' has reducer but no source"
                )
            if self.year is not None:
                raise ValueError(
                    f"Channel '{self.name}' cannot have both 'reducer' and 'year'"
                )

    def is_formula_based(self) -> bool:
        """Check if this channel is computed from a formula."""
        return self.formula is not None

    def is_source_based(self) -> bool:
        """Check if this channel is loaded from zarr."""
        return self.source is not None

    def requires_year_extraction(self) -> bool:
        """Check if this channel needs a specific year extracted."""
        return self.year is not None

    def requires_thresholding(self) -> bool:
        """Check if this channel needs thresholding."""
        return self.ok_if is not None

    def requires_reduction(self) -> bool:
        """Check if this channel needs temporal reduction."""
        return self.reducer is not None

    def get_reducer_func(self):
        """Get the numpy nan-safe reduction function for this channel."""
        if self.reducer is None:
            return None
        return getattr(np, self._REDUCERS[self.reducer])


@dataclass
class DatasetGroupConfig:
    """Configuration for a dataset group (e.g., static_mask, annual, static).

    Each group defines:
    - name: Group identifier (e.g., 'static_mask')
    - type: Data type for the tensor (e.g., 'uint8', 'float16')
    - dim: Expected dimensions [C,H,W] or [C,T,H,W]
    - channels: List of channels to load/compute
    """
    name: str
    dtype: str  # 'uint8', 'float16', etc.
    dim: List[str]  # ['C', 'H', 'W'] or ['C', 'T', 'H', 'W']
    channels: List[ChannelConfig]

    def __post_init__(self):
        """Validate group configuration."""
        valid_dims = [['C', 'H', 'W'], ['C', 'T', 'H', 'W']]
        if self.dim not in valid_dims:
            raise ValueError(
                f"Group '{self.name}' has invalid dim={self.dim}, "
                f"expected one of {valid_dims}"
            )

        # Convert dtype string to numpy dtype
        self._np_dtype = np.dtype(self.dtype)

    def is_temporal(self) -> bool:
        """Check if this group has a temporal dimension."""
        return 'T' in self.dim

    def is_static(self) -> bool:
        """Check if this group is static (no temporal dimension)."""
        return 'T' not in self.dim

    def get_numpy_dtype(self) -> np.dtype:
        """Get the numpy dtype for this group."""
        return self._np_dtype

    def get_channel_names(self) -> List[str]:
        """Get list of channel names in order."""
        return [ch.name for ch in self.channels]

    def get_channel_by_name(self, name: str) -> Optional[ChannelConfig]:
        """Retrieve a channel configuration by name."""
        for ch in self.channels:
            if ch.name == name:
                return ch
        return None


@dataclass
class TimeWindowConfig:
    """Configuration for temporal window."""
    start: int  # Start year (inclusive)
    end: int    # End year (inclusive)

    def __post_init__(self):
        """Validate time window."""
        if self.start > self.end:
            raise ValueError(
                f"Invalid time_window: start={self.start} > end={self.end}"
            )

    @property
    def n_years(self) -> int:
        """Number of years in the window."""
        return self.end - self.start + 1

    @property
    def years(self) -> List[int]:
        """List of years in the window."""
        return list(range(self.start, self.end + 1))

    def to_zarr_slice(self, zarr_start_year: int) -> slice:
        """Convert to a slice for indexing zarr arrays.

        Args:
            zarr_start_year: The first year in the zarr array

        Returns:
            slice object for temporal indexing
        """
        offset = self.start - zarr_start_year
        return slice(offset, offset + self.n_years)


@dataclass
class ZarrConfig:
    """Configuration for zarr dataset."""
    path: str
    structure: str = "hierarchical"

    def __post_init__(self):
        """Validate zarr configuration."""
        if self.structure not in ["hierarchical", "flat"]:
            raise ValueError(
                f"Invalid zarr structure: {self.structure}, "
                f"expected 'hierarchical' or 'flat'"
            )


@dataclass
class StatsConfig:
    """Configuration for statistics computation."""
    compute: str  # 'always', 'if-not-exists', 'never'
    type: str  # 'json'
    file: str  # Path to stats file
    stats: List[str]  # List of stat names: ['mean', 'sd', 'q02', ...]
    covariance: bool  # Whether to compute covariance
    samples: Dict[str, int]  # {'n': 16}
    mask: List[str]  # List of mask references: ['static_mask.aoi', 'static_mask.forest']


@dataclass
class NormalizationPresetConfig:
    """Configuration for a normalization preset."""
    name: str
    type: str  # 'zscore', 'robust_iqr', 'clamp', 'none', 'linear_rescale'
    stats_source: Optional[str] = None  # 'zarr', 'fixed', None
    fields: Optional[Dict[str, str]] = None  # {'mean': 'mean', 'std': 'sd'}
    clamp: Optional[Dict[str, Any]] = None  # {'enabled': true, 'min': -6.0, 'max': 6.0}
    # For linear_rescale
    in_min: Optional[float] = None
    in_max: Optional[float] = None
    out_min: Optional[float] = None
    out_max: Optional[float] = None


@dataclass
class FeatureChannelConfig:
    """Configuration for a single channel within a feature."""
    dataset_group: str  # 'static', 'annual', etc.
    channel_name: str  # 'elevation', 'evi2_summer_p95', etc.
    mask: Optional[str] = None  # 'static_mask.dem_mask'
    quality: Optional[str] = None  # Quality mask (not used yet)
    norm: Optional[str] = None  # Normalization preset name


@dataclass
class CovarianceConfig:
    """Configuration for covariance computation."""
    dim: List[str]  # ['C', 'C']
    calculate: bool  # Whether to compute
    stat_domain: str  # 'patch' or 'global'


@dataclass
class FeatureConfig:
    """Configuration for a feature (collection of channels with processing)."""
    name: str
    dim: List[str]  # ['C', 'H', 'W'] or ['C', 'T', 'H', 'W']
    channels: Dict[str, FeatureChannelConfig]  # channel_ref -> config
    masks: Optional[List[str]] = None  # Feature-level masks
    covariance: Optional[CovarianceConfig] = None

    def get_channel_list(self) -> List[str]:
        """Get ordered list of channel references."""
        return list(self.channels.keys())


# ---------------------------------------------------------------------------
# Sampling strategy configs
# ---------------------------------------------------------------------------

@dataclass
class JitterConfig:
    """Jitter configuration for grid sampling."""
    radius: int = 0


@dataclass
class GridConfig:
    """Grid sampling configuration."""
    stride: int = 16
    exclude_border: int = 16
    jitter: Optional[JitterConfig] = None


@dataclass
class WeightByEntry:
    """A single entry in a supplement weight_by list.

    Can be a simple mask reference (e.g. 'static_mask.aoi') or a
    channel-based weight with a transform.
    """
    # Simple mask reference (e.g. 'static_mask.aoi')
    mask_ref: Optional[str] = None
    # Channel-based weight
    channel: Optional[str] = None
    transform: Optional[str] = None  # e.g. 'inverse-frequency'


@dataclass
class SupplementConfig:
    """Supplement sampling configuration."""
    n: int = 0
    sampling_type: str = 'weighted'
    weight_by: List[WeightByEntry] = field(default_factory=list)


@dataclass
class SamplingStrategyConfig:
    """Configuration for a named sampling strategy.

    Strategies can be one of:
    - patch-only (just interior-only + border_width)
    - grid-only (stride, exclude_border, jitter)
    - grid-plus-supplement (grid + supplemental weighted sampling)
    """
    name: str
    # Patch-level settings
    interior_only: Optional[bool] = None
    border_width: Optional[int] = None
    # Grid settings (for grid and grid-plus-supplement)
    grid: Optional[GridConfig] = None
    # Supplement settings (for grid-plus-supplement)
    supplement: Optional[SupplementConfig] = None


# ---------------------------------------------------------------------------
# Loss configs
# ---------------------------------------------------------------------------

@dataclass
class AuxiliaryDistanceConfig:
    """Configuration for auxiliary distance used in pair selection."""
    feature: str  # e.g. 'features.infonce_type_spectral'
    metric: str = 'l2'
    covariance: bool = False


@dataclass
class SelectionConfig:
    """Configuration for a pair selection step."""
    type: str  # 'mutual-knn', 'quantile', 'spatial-knn', 'spatial-range'
    k: Optional[int] = None
    min_distance: Optional[float] = None
    max_distance: Optional[float] = None
    range: Optional[List[float]] = None  # [low, high] for quantile
    n_per_anchor: Optional[int] = None  # for spatial-range: negatives per anchor


@dataclass
class PairEndpointStrategyConfig:
    """Configuration for positive or negative pair endpoint selection."""
    candidates: str = 'anchors'  # 'anchors' or 'patch'
    distance: Optional[str] = None  # 'auxiliary' or None for spatial
    selection: Optional[SelectionConfig] = None


@dataclass
class TypeSimilarityConfig:
    """Stage 1 of knn-with-ysfc-overlap pair strategy."""
    feature: str  # e.g. 'features.infonce_type_spectral'
    k: int = 16


@dataclass
class YsfcOverlapConfig:
    """Stage 2 of knn-with-ysfc-overlap pair strategy."""
    channel: str  # e.g. 'annual.ysfc'
    min_overlap: int = 3


@dataclass
class PairStrategyConfig:
    """Configuration for pair selection in soft neighborhood losses."""
    type: str  # e.g. 'knn-with-ysfc-overlap'
    include_self: bool = True
    type_similarity: Optional[TypeSimilarityConfig] = None
    ysfc_overlap: Optional[YsfcOverlapConfig] = None
    min_pairs: int = 5


@dataclass
class PairWeightsConfig:
    """Configuration for continuous pair weighting."""
    source: str  # 'type_embedding'
    sigma: float = 5.0
    self_pair_weight: float = 1.0


@dataclass
class SpectralWeightingConfig:
    """Configuration for weighting pairs by spectral similarity."""
    feature: str  # e.g. 'features.infonce_type_spectral'
    tau: float = 1.0  # temperature for exp(-d/tau)
    min_weight: float = 0.01  # floor for pair weights


@dataclass
class CurriculumConfig:
    """Configuration for loss curriculum (ramp-in schedule)."""
    start_epoch: int = 0
    ramp_epochs: int = 0


@dataclass
class LossConfig:
    """Configuration for a named loss function.

    Loss types:
    - 'infonce': InfoNCE contrastive loss with positive/negative strategies
    - 'soft_neighborhood': Soft neighborhood KL matching loss
    """
    name: str
    weight: float = 1.0
    type: str = 'infonce'
    mask: Optional[List[str]] = None
    temperature: Optional[float] = None  # InfoNCE temperature

    # --- infonce-specific ---
    auxiliary_distance: Optional[AuxiliaryDistanceConfig] = None
    anchor_population: Optional[str] = None  # reference to sampling-strategy name
    positive_strategy: Optional[PairEndpointStrategyConfig] = None
    negative_strategy: Optional[PairEndpointStrategyConfig] = None
    spectral_weighting: Optional[SpectralWeightingConfig] = None

    # --- soft_neighborhood-specific ---
    neighborhood_target: Optional[str] = None  # e.g. 'features.soft_neighborhood_phase'
    pair_strategy: Optional[PairStrategyConfig] = None
    pair_weights: Optional[PairWeightsConfig] = None
    tau_ref: Optional[float] = None
    tau_learned: Optional[float] = None
    min_valid_per_row: Optional[int] = None
    self_similarity_weight: Optional[float] = None
    cross_pixel_weight: Optional[float] = None
    curriculum: Optional[CurriculumConfig] = None


@dataclass
class BindingsConfig:
    """Top-level bindings configuration.

    This represents the parsed YAML configuration for the dataset.
    """
    version: str
    name: str
    zarr: ZarrConfig
    time_window: TimeWindowConfig
    dataset_groups: Dict[str, DatasetGroupConfig]

    # Optional sections
    stats: Optional[StatsConfig] = None
    normalization_presets: Optional[Dict[str, NormalizationPresetConfig]] = None
    features: Optional[Dict[str, FeatureConfig]] = None
    sampling_strategies: Optional[Dict[str, SamplingStrategyConfig]] = None
    losses: Optional[Dict[str, LossConfig]] = None

    def get_group(self, name: str) -> Optional[DatasetGroupConfig]:
        """Get a dataset group by name."""
        return self.dataset_groups.get(name)

    def get_feature(self, name: str) -> Optional[FeatureConfig]:
        """Get a feature by name."""
        if self.features is None:
            return None
        return self.features.get(name)

    def get_normalization_preset(self, name: str) -> Optional[NormalizationPresetConfig]:
        """Get a normalization preset by name."""
        if self.normalization_presets is None:
            return None
        return self.normalization_presets.get(name)

    def get_sampling_strategy(self, name: str) -> Optional[SamplingStrategyConfig]:
        """Get a sampling strategy by name."""
        if self.sampling_strategies is None:
            return None
        return self.sampling_strategies.get(name)

    def get_loss(self, name: str) -> Optional[LossConfig]:
        """Get a loss configuration by name."""
        if self.losses is None:
            return None
        return self.losses.get(name)

    def get_all_source_paths(self) -> List[str]:
        """Get all unique zarr source paths referenced in the config."""
        paths = set()
        for group in self.dataset_groups.values():
            for channel in group.channels:
                if channel.is_source_based():
                    paths.add(channel.source)
        return sorted(paths)
