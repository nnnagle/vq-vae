"""
Data Bundle System for Forest Representation Model

Provides TrainingBundle and WindowData dataclasses for organizing multi-window
training data, plus BundleBuilder for constructing bundles from raw Zarr data.

A TrainingBundle contains:
- 3 temporal windows (t0, t2, t4) each with their own temporal/snapshot/irregular data
- Shared static data
- Masks and quality weights for all groups
- Derived features

Each window's data is organized per-group (not per-band), matching the output
structure of DataReader and MaskBuilder.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

from data.loaders import (
  DataReader, 
  GroupReadResult,
  MaskBuilder, 
  MaskResult, 
  QualityResult, 
  SpatialWindow, 
  TemporalWindow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WindowData:
    """
    Data for one temporal window (t0, t2, or t4).
    
    All data is organized per-group (not per-band). Each Dict[str, ...] maps
    group_name -> result, where the result contains all bands for that group.
    
    Shapes:
        temporal: [C_group, T, H, W] for each group
        snapshot: [C_group, H, W] for each group
        irregular: [C_group, T_obs, H, W] for each group
        derived: Variable shapes depending on derived feature type
        masks/quality: Match corresponding data shapes exactly
    """
    # Data groups (per-group dictionaries)
    temporal: Dict[str, GroupReadResult] = field(default_factory=dict)
    snapshot: Dict[str, GroupReadResult] = field(default_factory=dict)
    irregular: Dict[str, GroupReadResult] = field(default_factory=dict)
    derived: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Masks (per-group, shapes match corresponding data)
    temporal_masks: Dict[str, MaskResult] = field(default_factory=dict)
    snapshot_masks: Dict[str, MaskResult] = field(default_factory=dict)
    irregular_masks: Dict[str, MaskResult] = field(default_factory=dict)
    
    # Quality weights (per-group, shapes match corresponding data)
    temporal_quality: Dict[str, QualityResult] = field(default_factory=dict)
    snapshot_quality: Dict[str, QualityResult] = field(default_factory=dict)
    irregular_quality: Dict[str, QualityResult] = field(default_factory=dict)
    
    # Metadata
    year: int = 0
    window_label: str = ""  # 't0', 't2', 't4'
    
    def __post_init__(self):
        """Validate that all group names are consistent across data/masks/quality."""
        # Check temporal
        temp_groups = set(self.temporal.keys())
        if self.temporal_masks and set(self.temporal_masks.keys()) != temp_groups:
            logger.warning(
                f"Temporal mask groups {set(self.temporal_masks.keys())} "
                f"don't match data groups {temp_groups}"
            )
        if self.temporal_quality and set(self.temporal_quality.keys()) != temp_groups:
            logger.warning(
                f"Temporal quality groups {set(self.temporal_quality.keys())} "
                f"don't match data groups {temp_groups}"
            )
        
        # Check snapshot
        snap_groups = set(self.snapshot.keys())
        if self.snapshot_masks and set(self.snapshot_masks.keys()) != snap_groups:
            logger.warning(
                f"Snapshot mask groups {set(self.snapshot_masks.keys())} "
                f"don't match data groups {snap_groups}"
            )
        if self.snapshot_quality and set(self.snapshot_quality.keys()) != snap_groups:
            logger.warning(
                f"Snapshot quality groups {set(self.snapshot_quality.keys())} "
                f"don't match data groups {snap_groups}"
            )
        
        # Check irregular
        irreg_groups = set(self.irregular.keys())
        if self.irregular_masks and set(self.irregular_masks.keys()) != irreg_groups:
            logger.warning(
                f"Irregular mask groups {set(self.irregular_masks.keys())} "
                f"don't match data groups {irreg_groups}"
            )
        if self.irregular_quality and set(self.irregular_quality.keys()) != irreg_groups:
            logger.warning(
                f"Irregular quality groups {set(self.irregular_quality.keys())} "
                f"don't match data groups {irreg_groups}"
            )


@dataclass
class TrainingBundle:
    """
    Complete data bundle for training (3 temporal windows + static data).
    
    Contains all data, masks, and quality weights needed for one training sample.
    Data is organized per-group to match DataReader/MaskBuilder output structure.
    
    Memory: ~600MB per bundle (256x256 patches, ~300 channels across 3 windows).
    Should be built fresh for each sample, not cached in RAM.
    """
    # Window-specific data (one entry per window: 't0', 't2', 't4')
    windows: Dict[str, WindowData] = field(default_factory=dict)
    
    # Static data (shared across all windows)
    static: Dict[str, GroupReadResult] = field(default_factory=dict)
    static_masks: Dict[str, MaskResult] = field(default_factory=dict)
    static_quality: Dict[str, QualityResult] = field(default_factory=dict)
    
    # Bundle metadata
    anchor_id: str = ""  # 't0', 't2', or 't4' - which window is primary reference
    spatial_window: Optional[SpatialWindow] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_window(self, window_id: str) -> WindowData:
        """Get WindowData for specific window (t0, t2, or t4)."""
        if window_id not in self.windows:
            raise KeyError(f"Window '{window_id}' not found. Available: {list(self.windows.keys())}")
        return self.windows[window_id]
    
    def get_anchor_window(self) -> WindowData:
        """Get the anchor window's data."""
        return self.get_window(self.anchor_id)
    
    def __post_init__(self):
        """Validate bundle structure."""
        # Check that we have the expected windows
        expected_windows = {'t0', 't2', 't4'}
        actual_windows = set(self.windows.keys())
        if actual_windows != expected_windows:
            logger.warning(
                f"Expected windows {expected_windows}, got {actual_windows}"
            )
        
        # Check anchor_id is valid
        if self.anchor_id not in self.windows:
            logger.warning(
                f"Anchor '{self.anchor_id}' not in windows {list(self.windows.keys())}"
            )


class BundleBuilder:
    """
    Constructs TrainingBundle objects from raw Zarr data.
    
    Uses DataReader and MaskBuilder to load and organize data for multi-window
    training. Builds bundles fresh for each call (no caching) to avoid massive
    RAM usage.
    
    Usage:
        builder = BundleBuilder(config, data_reader, mask_builder)
        bundle = builder.build_bundle(
            spatial_window,
            anchor_end=2024,
            anchor_id='t0',
            offsets=[0, -2, -4]
        )
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_reader: DataReader,
        mask_builder: MaskBuilder
    ):
        """
        Initialize BundleBuilder.
        
        Args:
            config: Parsed bindings configuration
            data_reader: DataReader instance for loading raw data
            mask_builder: MaskBuilder instance for loading masks/quality
        """
        self.config = config
        self.reader = data_reader
        self.masker = mask_builder
        
        # Validate required groups exist in config
        self._validate_required_groups()
    
    def _validate_required_groups(self):
        """
        Validate that required input groups exist in configuration.
        
        Raises:
            ValueError: If required groups are missing
        """
        # Define required groups (adjust based on your model requirements)
        required_groups = {
            'temporal': ['ls8day'],  # At minimum, need ls8day
            'static': ['topo'],      # At minimum, need topo
            'snapshot': ['ccdc_snapshot'],  # At minimum, need ccdc_snapshot
        }
        
        for category, group_names in required_groups.items():
            if category not in self.config.get('inputs', {}):
                raise ValueError(f"Config missing 'inputs.{category}' section")
            
            for group_name in group_names:
                if group_name not in self.config['inputs'][category]:
                    raise ValueError(
                        f"Required group missing: inputs.{category}.{group_name}"
                    )
        
        logger.info("✓ All required groups present in configuration")
    
    def build_bundle(
        self,
        spatial_window: SpatialWindow,
        anchor_year: int,
        endpoint_years: Optional[List[int]] = None
    ) -> TrainingBundle:
        """
        Build complete training bundle for fixed endpoint years.
        
        Builds fresh data for each call - no caching to avoid RAM issues.
        
        The endpoint years are FIXED (e.g., [2020, 2022, 2024]) and come from
        the snapshot group configuration. What varies is which endpoint is the
        anchor for this sample.
        
        Args:
            spatial_window: Spatial region to read
            anchor_year: Which endpoint year is the anchor (e.g., 2022)
            endpoint_years: Fixed list of endpoint years (if None, reads from config)
        
        Returns:
            TrainingBundle with all data, masks, and quality weights
        
        Raises:
            ValueError: If anchor_year is not in endpoint_years
        
        Example:
            >>> builder = BundleBuilder(config, reader, masker)
            >>> # Endpoint years are [2020, 2022, 2024] from config
            >>> # Sampler randomly picks anchor_year = 2022
            >>> bundle = builder.build_bundle(
            ...     spatial_window,
            ...     anchor_year=2022
            ... )
            >>> # Bundle contains windows for ALL endpoints:
            >>> # - windows['t0'] = 2024 (most recent)
            >>> # - windows['t2'] = 2022 (anchor)
            >>> # - windows['t4'] = 2020 (oldest)
        """
        # Get endpoint years from config if not provided
        if endpoint_years is None:
            endpoint_years = self._get_endpoint_years_from_config()
        
        # Validate anchor_year is in endpoint_years
        if anchor_year not in endpoint_years:
            raise ValueError(
                f"anchor_year {anchor_year} not in endpoint_years {endpoint_years}"
            )
        
        # Sort endpoint years (oldest to newest)
        sorted_years = sorted(endpoint_years)
        
        # Determine anchor_id based on which endpoint is the anchor
        anchor_idx = sorted_years.index(anchor_year)
        # Map index to label: 0->t4 (oldest), 1->t2 (middle), 2->t0 (newest)
        idx_to_label = {0: 't4', 1: 't2', 2: 't0'}
        anchor_id = idx_to_label.get(anchor_idx, f't{2*(len(sorted_years)-1-anchor_idx)}')
        
        logger.debug(
            f"Building bundle: anchor_year={anchor_year} ({anchor_id}), "
            f"endpoint_years={sorted_years}, spatial={spatial_window}"
        )
        
        # Build static data (shared across all windows) - read once
        logger.debug("Reading static groups...")
        static = self.reader.read_all_static_groups(spatial_window)
        static_masks = self._read_static_masks(spatial_window)
        static_quality = self._read_static_quality(spatial_window)
        
        # Build each temporal window for each endpoint year
        windows = {}
        for i, year in enumerate(sorted_years):
            # Map year position to label
            # Position 0 (oldest) -> t4, Position 1 (middle) -> t2, Position 2 (newest) -> t0
            offset_from_newest = len(sorted_years) - 1 - i
            label = f"t{2 * offset_from_newest}"
            
            logger.debug(f"Building window {label} (year={year})...")
            windows[label] = self._build_window_data(
                spatial_window, year, label
            )
        
        bundle = TrainingBundle(
            windows=windows,
            static=static,
            static_masks=static_masks,
            static_quality=static_quality,
            anchor_id=anchor_id,
            spatial_window=spatial_window,
            metadata={
                'anchor_year': anchor_year,
                'endpoint_years': sorted_years
            }
        )
        
        logger.debug(f"Bundle built successfully: {len(windows)} windows, "
                    f"{len(static)} static groups")
        
        return bundle
    
    def _get_endpoint_years_from_config(self) -> List[int]:
        """
        Get endpoint years from snapshot group configuration.
        
        Reads from the first snapshot group's 'years' field.
        
        Returns:
            List of endpoint years (e.g., [2020, 2022, 2024])
        
        Raises:
            ValueError: If no snapshot groups or years not configured
        """
        if 'snapshot' not in self.config['inputs']:
            raise ValueError("No snapshot groups in config")
        
        snapshot_groups = self.config['inputs']['snapshot']
        if not snapshot_groups:
            raise ValueError("No snapshot groups configured")
        
        # Get first snapshot group
        first_group_name = next(iter(snapshot_groups.keys()))
        first_group = snapshot_groups[first_group_name]
        
        if not hasattr(first_group, 'years') or not first_group.years:
            raise ValueError(
                f"Snapshot group '{first_group_name}' missing 'years' field"
            )
        
        return sorted(first_group.years)
    
    def _build_window_data(
        self,
        spatial_window: SpatialWindow,
        year: int,
        label: str
    ) -> WindowData:
        """
        Build data for one temporal window.
        
        Args:
            spatial_window: Spatial region to read
            year: End year for this window
            label: Window label ('t0', 't2', 't4')
        
        Returns:
            WindowData with all data, masks, and quality for this window
        """
        temporal_window = TemporalWindow(end_year=year, window_length=10)
        
        # Read all data groups (returns per-group dicts)
        temporal = self.reader.read_all_temporal_groups(
            spatial_window, 
            temporal_window,
            return_full_temporal=True  # Get full [C, T, H, W]
        )
        
        snapshot = self.reader.read_all_snapshot_groups(spatial_window, year)
        
        irregular = self.reader.read_all_irregular_groups(
            spatial_window, 
            temporal_window
        )
        
        # Read masks for each category (per-group dicts)
        # MaskBuilder guarantees these always return valid results (never None)
        temporal_masks = self._read_temporal_masks(spatial_window, temporal_window)
        snapshot_masks = self._read_snapshot_masks(spatial_window, year)
        irregular_masks = self._read_irregular_masks(spatial_window, temporal_window)
        
        # Read quality weights for each category (per-group dicts)
        temporal_quality = self._read_temporal_quality(spatial_window, temporal_window)
        snapshot_quality = self._read_snapshot_quality(spatial_window, year)
        irregular_quality = self._read_irregular_quality(spatial_window, temporal_window)
        
        # Compute derived features (if enabled in config)
        derived = self._compute_derived_features(
            temporal, snapshot, spatial_window, temporal_window
        )
        
        return WindowData(
            temporal=temporal,
            snapshot=snapshot,
            irregular=irregular,
            derived=derived,
            temporal_masks=temporal_masks,
            snapshot_masks=snapshot_masks,
            irregular_masks=irregular_masks,
            temporal_quality=temporal_quality,
            snapshot_quality=snapshot_quality,
            irregular_quality=irregular_quality,
            year=year,
            window_label=label
        )
    
    # ========================================================================
    # Mask reading methods
    # ========================================================================
    
    def _read_temporal_masks(
        self,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> Dict[str, MaskResult]:
        """
        Read masks for all temporal groups.
        
        Returns per-group dict where each mask matches the corresponding
        data group's shape exactly.
        """
        masks = {}
        
        for group_name in self.config['inputs']['temporal'].keys():
            try:
                mask = self.masker.read_mask_for_group(
                    'temporal',
                    group_name,
                    spatial_window,
                    temporal_window
                )
                # MaskBuilder contract: this should never be None
                assert mask is not None, f"MaskBuilder returned None for {group_name}"
                masks[group_name] = mask
            except Exception as e:
                logger.error(f"Error reading mask for temporal.{group_name}: {e}")
                raise
        
        return masks
    
    def _read_snapshot_masks(
        self,
        spatial_window: SpatialWindow,
        year: int
    ) -> Dict[str, MaskResult]:
        """Read masks for all snapshot groups for specific year."""
        masks = {}
        
        for group_name in self.config['inputs']['snapshot'].keys():
            try:
                mask = self.masker.read_mask_for_snapshot_group(
                    group_name,
                    spatial_window,
                    year
                )
                assert mask is not None, f"MaskBuilder returned None for {group_name}"
                masks[group_name] = mask
            except Exception as e:
                logger.error(f"Error reading mask for snapshot.{group_name}: {e}")
                raise
        
        return masks
    
    def _read_irregular_masks(
        self,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> Dict[str, MaskResult]:
        """Read masks for all irregular groups."""
        masks = {}
        
        if 'irregular' not in self.config['inputs']:
            return masks
        
        for group_name in self.config['inputs']['irregular'].keys():
            try:
                mask = self.masker.read_mask_for_irregular_group(
                    group_name,
                    spatial_window,
                    temporal_window
                )
                assert mask is not None, f"MaskBuilder returned None for {group_name}"
                masks[group_name] = mask
            except Exception as e:
                logger.error(f"Error reading mask for irregular.{group_name}: {e}")
                raise
        
        return masks
    
    def _read_static_masks(
        self,
        spatial_window: SpatialWindow
    ) -> Dict[str, MaskResult]:
        """Read masks for all static groups."""
        masks = {}
        
        for group_name in self.config['inputs']['static'].keys():
            try:
                mask = self.masker.read_mask_for_group(
                    'static',
                    group_name,
                    spatial_window,
                    temporal_window=None
                )
                assert mask is not None, f"MaskBuilder returned None for {group_name}"
                masks[group_name] = mask
            except Exception as e:
                logger.error(f"Error reading mask for static.{group_name}: {e}")
                raise
        
        return masks
    
    # ========================================================================
    # Quality reading methods
    # ========================================================================
    
    def _read_temporal_quality(
        self,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> Dict[str, QualityResult]:
        """Read quality weights for all temporal groups."""
        quality = {}
        
        for group_name in self.config['inputs']['temporal'].keys():
            try:
                q = self.masker.read_quality_for_group(
                    'temporal',
                    group_name,
                    spatial_window,
                    temporal_window
                )
                assert q is not None, f"MaskBuilder returned None quality for {group_name}"
                quality[group_name] = q
            except Exception as e:
                logger.error(f"Error reading quality for temporal.{group_name}: {e}")
                raise
        
        return quality
    
    def _read_snapshot_quality(
        self,
        spatial_window: SpatialWindow,
        year: int
    ) -> Dict[str, QualityResult]:
        """Read quality weights for all snapshot groups for specific year."""
        quality = {}
        
        for group_name in self.config['inputs']['snapshot'].keys():
            try:
                q = self.masker.read_quality_for_snapshot_group(
                    group_name,
                    spatial_window,
                    year
                )
                assert q is not None, f"MaskBuilder returned None quality for {group_name}"
                quality[group_name] = q
            except Exception as e:
                logger.error(f"Error reading quality for snapshot.{group_name}: {e}")
                raise
        
        return quality
    
    def _read_irregular_quality(
        self,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> Dict[str, QualityResult]:
        """Read quality weights for all irregular groups."""
        quality = {}
        
        if 'irregular' not in self.config['inputs']:
            return quality
        
        for group_name in self.config['inputs']['irregular'].keys():
            try:
                q = self.masker.read_quality_for_irregular_group(
                    group_name,
                    spatial_window,
                    temporal_window
                )
                assert q is not None, f"MaskBuilder returned None quality for {group_name}"
                quality[group_name] = q
            except Exception as e:
                logger.error(f"Error reading quality for irregular.{group_name}: {e}")
                raise
        
        return quality
    
    def _read_static_quality(
        self,
        spatial_window: SpatialWindow
    ) -> Dict[str, QualityResult]:
        """Read quality weights for all static groups."""
        quality = {}
        
        for group_name in self.config['inputs']['static'].keys():
            try:
                q = self.masker.read_quality_for_group(
                    'static',
                    group_name,
                    spatial_window,
                    temporal_window=None
                )
                assert q is not None, f"MaskBuilder returned None quality for {group_name}"
                quality[group_name] = q
            except Exception as e:
                logger.error(f"Error reading quality for static.{group_name}: {e}")
                raise
        
        return quality
    
    # ========================================================================
    # Derived features
    # ========================================================================
    
    def _compute_derived_features(
        self,
        temporal_data: Dict[str, GroupReadResult],
        snapshot_data: Dict[str, GroupReadResult],
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> Dict[str, np.ndarray]:
        """
        Compute derived features (e.g., temporal_position, ls8_delta).
        
        This is a stub - actual implementation depends on which derived
        features are enabled in config.
        
        Args:
            temporal_data: Temporal groups data
            snapshot_data: Snapshot groups data
            spatial_window: Spatial window
            temporal_window: Temporal window
        
        Returns:
            Dict mapping derived feature name -> array
        """
        derived = {}
        
        # Check if derived features are enabled in config
        if 'derived' not in self.config:
            return derived
        
        # Temporal position encoding (if enabled)
        if self.config['derived'].get('temporal_position', {}).get('enabled', False):
            T = temporal_window.window_length
            H = spatial_window.height
            W = spatial_window.width
            
            # Create position channels: p_t in [0,1] and c_t in [-1,1]
            t_indices = np.arange(T, dtype=np.float32)
            p_t = t_indices / (T - 1) if T > 1 else np.zeros(T, dtype=np.float32)
            c_t = (t_indices - (T - 1) / 2) / ((T - 1) / 2) if T > 1 else np.zeros(T, dtype=np.float32)
            
            # Broadcast to [2, T, H, W]
            temporal_position = np.stack([
                np.broadcast_to(p_t[:, None, None], (T, H, W)),
                np.broadcast_to(c_t[:, None, None], (T, H, W))
            ], axis=0)
            
            derived['temporal_position'] = temporal_position
        
        # LS8 delta (first differences) - if enabled
        if self.config['derived'].get('ls8_delta', {}).get('enabled', False):
            if 'ls8day' in temporal_data:
                ls8_data = temporal_data['ls8day'].data  # [C, T, H, W]
                
                # Compute first differences along time: x[t] - x[t-1]
                delta = np.diff(ls8_data, axis=1)  # [C, T-1, H, W]
                
                # Prepend zeros to get back to [C, T, H, W]
                zero_pad = np.zeros((ls8_data.shape[0], 1, ls8_data.shape[2], ls8_data.shape[3]))
                ls8_delta = np.concatenate([zero_pad, delta], axis=1)
                
                derived['ls8_delta'] = ls8_delta
        
        # Add more derived features here as needed
        # - Gradients (spectral, topographic)
        # - Other transformations
        
        return derived
