"""
Data Bundle System for Forest Representation Model - Stacked Array Version

Provides TrainingBundle dataclass with window-stacked arrays optimized for
efficient batching and distributed training.

A TrainingBundle contains:
- Temporal groups: (Win, C, T, H, W) arrays
- Snapshot groups: (Win, C, H, W) arrays
- Static groups: (C, H, W) arrays
- Anchor mask: (Win,) binary array indicating which window is the anchor

All data is stacked across windows for efficient batch processing with gradient masking.
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
class TrainingBundle:
    """
    Complete data bundle for training with window-stacked arrays.

    All temporal/snapshot data is stacked across windows (Win=3) for efficient
    processing. Static data is shared across windows.

    Shapes:
        temporal[group_name]: (Win, C, T, H, W) - stacked across windows
        snapshot[group_name]: (Win, C, H, W) - stacked across windows
        static[group_name]: (C, H, W) - shared, not windowed

        anchor_mask: (Win,) - binary, 1.0 for anchor window, 0.0 for others

        Masks and quality follow same shapes as corresponding data.

    The anchor_mask enables gradient masking: only the anchor window gets gradients
    during training, while other windows provide context without gradient updates.

    Example:
        >>> bundle = dataset[0]
        >>> ls8 = bundle.temporal['ls8day']  # (3, 7, 10, 256, 256)
        >>> anchor_mask = bundle.anchor_mask  # [0., 1., 0.] if t2 is anchor
        >>>
        >>> # For batching
        >>> batched_ls8 = np.stack([b.temporal['ls8day'] for b in batch])  # (B, Win, C, T, H, W)
    """
    # Window-stacked data (Win dimension present)
    temporal: Dict[str, np.ndarray] = field(default_factory=dict)  # (Win, C, T, H, W)
    snapshot: Dict[str, np.ndarray] = field(default_factory=dict)  # (Win, C, H, W)
    derived: Dict[str, np.ndarray] = field(default_factory=dict)    # (Win, C, ...) varies

    # Static data (no Win dimension - shared across windows)
    static: Dict[str, np.ndarray] = field(default_factory=dict)    # (C, H, W)

    # Masks (match data shapes)
    temporal_masks: Dict[str, np.ndarray] = field(default_factory=dict)   # (Win, C, T, H, W)
    snapshot_masks: Dict[str, np.ndarray] = field(default_factory=dict)   # (Win, C, H, W)
    static_masks: Dict[str, np.ndarray] = field(default_factory=dict)     # (C, H, W)

    # Quality weights (match data shapes)
    temporal_quality: Dict[str, np.ndarray] = field(default_factory=dict)  # (Win, C, T, H, W)
    snapshot_quality: Dict[str, np.ndarray] = field(default_factory=dict)  # (Win, C, H, W)
    static_quality: Dict[str, np.ndarray] = field(default_factory=dict)    # (C, H, W)

    # Anchor information
    anchor_mask: np.ndarray = field(default_factory=lambda: np.array([]))  # (Win,) binary
    anchor_id: str = ""  # 't0', 't2', or 't4' for reference

    # Metadata
    spatial_window: Optional[SpatialWindow] = None
    window_labels: List[str] = field(default_factory=list)  # ['t0', 't2', 't4']
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate bundle structure."""
        Win = len(self.window_labels)

        # Validate anchor_mask shape
        if self.anchor_mask.size > 0 and self.anchor_mask.shape != (Win,):
            logger.warning(f"anchor_mask shape {self.anchor_mask.shape} != ({Win},)")

        # Validate temporal shapes
        for name, arr in self.temporal.items():
            if arr.ndim != 5:
                logger.warning(f"temporal[{name}] has {arr.ndim} dims, expected 5 (Win,C,T,H,W)")
            if arr.shape[0] != Win:
                logger.warning(f"temporal[{name}] Win={arr.shape[0]}, expected {Win}")

        # Validate snapshot shapes
        for name, arr in self.snapshot.items():
            if arr.ndim != 4:
                logger.warning(f"snapshot[{name}] has {arr.ndim} dims, expected 4 (Win,C,H,W)")
            if arr.shape[0] != Win:
                logger.warning(f"snapshot[{name}] Win={arr.shape[0]}, expected {Win}")

        # Validate static shapes (no Win dimension)
        for name, arr in self.static.items():
            if arr.ndim != 3:
                logger.warning(f"static[{name}] has {arr.ndim} dims, expected 3 (C,H,W)")


class BundleBuilder:
    """
    Constructs TrainingBundle objects with window-stacked arrays.

    Reads data for multiple temporal windows and stacks them into arrays with
    Win dimension, optimized for efficient batching and gradient masking.

    Usage:
        builder = BundleBuilder(config, data_reader, mask_builder)
        bundle = builder.build_bundle(spatial_window, anchor_year=2024)
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
        Build complete training bundle with window-stacked arrays.

        Args:
            spatial_window: Spatial region to read
            anchor_year: Which endpoint year is the anchor (e.g., 2022)
            endpoint_years: Fixed list of endpoint years (if None, reads from config)

        Returns:
            TrainingBundle with stacked arrays and anchor mask

        Raises:
            ValueError: If anchor_year is not in endpoint_years

        Example:
            >>> builder = BundleBuilder(config, reader, masker)
            >>> bundle = builder.build_bundle(spatial_window, anchor_year=2022)
            >>> bundle.temporal['ls8day'].shape  # (3, 7, 10, 256, 256)
            >>> bundle.anchor_mask  # [0., 1., 0.] if 2022 is middle year
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
        Win = len(sorted_years)

        # Determine anchor index
        anchor_idx = sorted_years.index(anchor_year)

        # Map index to label: 0->t4 (oldest), 1->t2 (middle), 2->t0 (newest)
        # For Win=3: [0,1,2] -> ['t4','t2','t0']
        window_labels = [f"t{2*(Win-1-i)}" for i in range(Win)]
        anchor_id = window_labels[anchor_idx]

        # Create anchor mask: (Win,) with 1.0 at anchor position
        anchor_mask = np.zeros(Win, dtype=np.float32)
        anchor_mask[anchor_idx] = 1.0

        logger.debug(
            f"Building bundle: anchor_year={anchor_year} ({anchor_id}), "
            f"endpoint_years={sorted_years}, spatial={spatial_window}"
        )

        # Build static data (shared across all windows) - read once
        logger.debug("Reading static groups...")
        static = self._read_all_static_groups(spatial_window)
        static_masks = self._read_all_static_masks(spatial_window)
        static_quality = self._read_all_static_quality(spatial_window)

        # Build data for each window, then stack
        logger.debug("Reading and stacking window data...")
        temporal_by_window = []
        snapshot_by_window = []
        derived_by_window = []
        temporal_masks_by_window = []
        snapshot_masks_by_window = []
        temporal_quality_by_window = []
        snapshot_quality_by_window = []

        for i, year in enumerate(sorted_years):
            label = window_labels[i]
            logger.debug(f"Building window {label} (year={year})...")

            temporal_window = TemporalWindow(end_year=year, window_length=10)

            # Read temporal groups for this window
            temporal = self.reader.read_all_temporal_groups(
                spatial_window,
                temporal_window,
                return_full_temporal=True
            )
            temporal_by_window.append(temporal)

            # Read snapshot groups for this window
            snapshot = self.reader.read_all_snapshot_groups(spatial_window, year)
            snapshot_by_window.append(snapshot)

            # Read masks
            temporal_masks = self._read_temporal_masks(spatial_window, temporal_window)
            temporal_masks_by_window.append(temporal_masks)

            snapshot_masks = self._read_snapshot_masks(spatial_window, year)
            snapshot_masks_by_window.append(snapshot_masks)

            # Read quality
            temporal_quality = self._read_temporal_quality(spatial_window, temporal_window)
            temporal_quality_by_window.append(temporal_quality)

            snapshot_quality = self._read_snapshot_quality(spatial_window, year)
            snapshot_quality_by_window.append(snapshot_quality)

            # Compute derived features for this window
            derived = self._compute_derived_features(
                temporal, snapshot, spatial_window, temporal_window
            )
            derived_by_window.append(derived)

        # Stack data across windows
        logger.debug("Stacking arrays across windows...")
        temporal_stacked = self._stack_groups(temporal_by_window)  # Dict[name, (Win,C,T,H,W)]
        snapshot_stacked = self._stack_groups(snapshot_by_window)  # Dict[name, (Win,C,H,W)]
        derived_stacked = self._stack_derived(derived_by_window)   # Dict[name, (Win,...)]

        temporal_masks_stacked = self._stack_masks(temporal_masks_by_window)
        snapshot_masks_stacked = self._stack_masks(snapshot_masks_by_window)

        temporal_quality_stacked = self._stack_quality(temporal_quality_by_window)
        snapshot_quality_stacked = self._stack_quality(snapshot_quality_by_window)

        bundle = TrainingBundle(
            temporal=temporal_stacked,
            snapshot=snapshot_stacked,
            derived=derived_stacked,
            static=static,
            temporal_masks=temporal_masks_stacked,
            snapshot_masks=snapshot_masks_stacked,
            static_masks=static_masks,
            temporal_quality=temporal_quality_stacked,
            snapshot_quality=snapshot_quality_stacked,
            static_quality=static_quality,
            anchor_mask=anchor_mask,
            anchor_id=anchor_id,
            spatial_window=spatial_window,
            window_labels=window_labels,
            metadata={
                'anchor_year': anchor_year,
                'endpoint_years': sorted_years
            }
        )

        logger.debug(
            f"Bundle built: {len(temporal_stacked)} temporal groups, "
            f"{len(snapshot_stacked)} snapshot groups, {len(static)} static groups"
        )

        return bundle

    def _get_endpoint_years_from_config(self) -> List[int]:
        """
        Get endpoint years from snapshot group configuration.

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

    def _read_all_static_groups(self, spatial_window: SpatialWindow) -> Dict[str, np.ndarray]:
        """
        Read all static groups and extract data arrays.

        Returns:
            Dict[group_name, array] where arrays are (C, H, W)
        """
        result_dict = self.reader.read_all_static_groups(spatial_window)
        # Extract .data from GroupReadResult objects
        return {name: result.data for name, result in result_dict.items()}

    def _stack_groups(self, groups_by_window: List[Dict[str, GroupReadResult]]) -> Dict[str, np.ndarray]:
        """
        Stack GroupReadResult dicts across windows.

        Args:
            groups_by_window: List of length Win, each containing Dict[name, GroupReadResult]

        Returns:
            Dict[name, array] where arrays are (Win, C, T, H, W) or (Win, C, H, W)
        """
        if not groups_by_window:
            return {}

        # Get group names from first window
        group_names = list(groups_by_window[0].keys())

        stacked = {}
        for name in group_names:
            # Extract data arrays from each window
            arrays = [window_dict[name].data for window_dict in groups_by_window]
            # Stack along new Win dimension
            stacked[name] = np.stack(arrays, axis=0)

        return stacked

    def _stack_derived(self, derived_by_window: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Stack derived feature dicts across windows.

        Args:
            derived_by_window: List of length Win, each containing Dict[name, array]

        Returns:
            Dict[name, array] where arrays are (Win, ...)
        """
        if not derived_by_window or not derived_by_window[0]:
            return {}

        # Get feature names from first window
        feature_names = list(derived_by_window[0].keys())

        stacked = {}
        for name in feature_names:
            # Extract arrays from each window
            arrays = [window_dict[name] for window_dict in derived_by_window]
            # Stack along new Win dimension
            stacked[name] = np.stack(arrays, axis=0)

        return stacked

    def _stack_masks(self, masks_by_window: List[Dict[str, MaskResult]]) -> Dict[str, np.ndarray]:
        """Stack mask dicts across windows."""
        if not masks_by_window or not masks_by_window[0]:
            return {}

        mask_names = list(masks_by_window[0].keys())

        stacked = {}
        for name in mask_names:
            arrays = [window_dict[name].data for window_dict in masks_by_window]
            stacked[name] = np.stack(arrays, axis=0)

        return stacked

    def _stack_quality(self, quality_by_window: List[Dict[str, QualityResult]]) -> Dict[str, np.ndarray]:
        """Stack quality dicts across windows."""
        if not quality_by_window or not quality_by_window[0]:
            return {}

        quality_names = list(quality_by_window[0].keys())

        stacked = {}
        for name in quality_names:
            arrays = [window_dict[name].data for window_dict in quality_by_window]
            stacked[name] = np.stack(arrays, axis=0)

        return stacked

    # ========================================================================
    # Mask reading methods
    # ========================================================================

    def _read_temporal_masks(
        self,
        spatial_window: SpatialWindow,
        temporal_window: TemporalWindow
    ) -> Dict[str, MaskResult]:
        """Read masks for all temporal groups."""
        masks = {}

        for group_name in self.config['inputs']['temporal'].keys():
            try:
                mask = self.masker.read_mask_for_group(
                    'temporal',
                    group_name,
                    spatial_window,
                    temporal_window
                )
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

    def _read_all_static_masks(
        self,
        spatial_window: SpatialWindow
    ) -> Dict[str, np.ndarray]:
        """Read masks for all static groups and extract data."""
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
                masks[group_name] = mask.data  # Extract numpy array
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

    def _read_all_static_quality(
        self,
        spatial_window: SpatialWindow
    ) -> Dict[str, np.ndarray]:
        """Read quality weights for all static groups and extract data."""
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
                quality[group_name] = q.data  # Extract numpy array
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

        Args:
            temporal_data: Temporal groups data for this window
            snapshot_data: Snapshot groups data for this window
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

        return derived
