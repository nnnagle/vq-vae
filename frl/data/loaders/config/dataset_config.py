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

    def get_all_source_paths(self) -> List[str]:
        """Get all unique zarr source paths referenced in the config."""
        paths = set()
        for group in self.dataset_groups.values():
            for channel in group.channels:
                if channel.is_source_based():
                    paths.add(channel.source)
        return sorted(paths)
