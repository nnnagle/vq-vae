"""
Refactored forest dataset loader.

This module provides a simplified PyTorch dataset that loads data according
to the new bindings configuration format.
"""

import numpy as np
import zarr
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import Dataset

from ..config.dataset_config import (
    BindingsConfig,
    DatasetGroupConfig,
    ChannelConfig,
)
from ..config.dataset_bindings_parser import DatasetBindingsParser
from .forest_patch_sampler import ForestPatchSampler, SamplerConfig
from ..readers.windows import SpatialWindow


class ForestDatasetV2(Dataset):
    """PyTorch dataset for loading forest data from zarr.

    This dataset:
    - Loads data according to bindings configuration
    - Reuses spatial sampling logic (train/val/test splits)
    - Handles temporal slicing and padding
    - Computes formula-based channels
    - Applies thresholding and fill value handling
    - Returns raw data (no normalization)

    Example usage:
        config = DatasetBindingsParser('config/bindings.yaml').parse()
        dataset = ForestDatasetV2(config, split='train')
        batch = dataset[0]
        # batch = {'static': array, 'annual': array, 'metadata': {...}}
    """

    def __init__(
        self,
        config: BindingsConfig,
        split: Optional[str] = None,
        patch_size: int = 256,
        min_aoi_fraction: float = 0.3,
        epoch_mode: str = 'full',
        sample_frac: Optional[float] = None,
        sample_number: Optional[int] = None,
        debug_window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ):
        """Initialize dataset.

        Args:
            config: Parsed bindings configuration
            split: 'train', 'val', 'test', or None for all data
            patch_size: Size of spatial patches (default: 256)
            min_aoi_fraction: Minimum fraction of valid AOI pixels per patch
            epoch_mode: 'full', 'frac', or 'number'
            sample_frac: Fraction of patches to sample per epoch (for 'frac' mode)
            sample_number: Number of patches per epoch (for 'number' mode)
            debug_window: Optional debug window ((row, col), (height, width))
        """
        self.config = config
        self.split = split
        self.patch_size = patch_size

        # Open zarr dataset
        self.zarr_path = Path(config.zarr.path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr dataset not found: {self.zarr_path}")

        self.zarr_root = zarr.open(str(self.zarr_path), mode='r')

        # Validate all source paths exist
        self._validate_sources()

        # Determine zarr temporal metadata (if any temporal arrays exist)
        self._infer_zarr_temporal_metadata()

        # Initialize spatial sampler (reuses existing logic)
        sampler_config = SamplerConfig(
            zarr_path=str(self.zarr_path),
            aoi_zarr_group='',  # AOI at root level
            aoi_zarr_array='aoi',
            patch_size=patch_size,
            block_height=4,
            block_width=4,
            use_debug_window=debug_window is not None,
            window_origin=debug_window[0] if debug_window else None,
            window_size=debug_window[1] if debug_window else None,
            endpoint_years=[config.time_window.end],  # Single endpoint
            anchor_weights={config.time_window.end: 1.0},
            epoch_mode=epoch_mode,
            sample_frac=sample_frac,
            sample_number=sample_number,
            min_aoi_fraction=min_aoi_fraction,
        )

        self.sampler = ForestPatchSampler(sampler_config, split=split)

    def _validate_sources(self):
        """Validate that all source paths exist in zarr.

        Raises:
            ValueError: If any source path is not found in zarr
        """
        missing = []
        for group in self.config.dataset_groups.values():
            for channel in group.channels:
                if channel.is_source_based():
                    if not self._zarr_path_exists(channel.source):
                        missing.append(channel.source)

        if missing:
            raise ValueError(
                f"Missing {len(missing)} zarr arrays:\n" +
                "\n".join(f"  - {path}" for path in missing)
            )

    def _zarr_path_exists(self, path: str) -> bool:
        """Check if a zarr path exists.

        Args:
            path: Zarr path (e.g., 'static/topo/data/elevation')

        Returns:
            True if path exists in zarr
        """
        try:
            node = self.zarr_root
            for part in path.split('/'):
                node = node[part]
            return True
        except (KeyError, ValueError):
            return False

    def _infer_zarr_temporal_metadata(self):
        """Infer temporal metadata from zarr arrays.

        This finds a temporal array and determines the start year and
        number of timesteps available.
        """
        # Find a temporal group
        temporal_group = None
        for group in self.config.dataset_groups.values():
            if group.is_temporal():
                temporal_group = group
                break

        if temporal_group is None:
            # No temporal data, skip
            self.zarr_start_year = None
            self.zarr_n_timesteps = None
            return

        # Find a source-based channel to inspect
        for channel in temporal_group.channels:
            if channel.is_source_based():
                arr = self._get_zarr_array(channel.source)
                self.zarr_n_timesteps = arr.shape[0]  # First dim is time
                # Assume zarr starts at some base year (we'll need to get this from metadata)
                # For now, assume it covers the config time window
                self.zarr_start_year = self.config.time_window.start
                return

        # No source-based channels found
        self.zarr_start_year = None
        self.zarr_n_timesteps = None

    def _get_zarr_array(self, path: str) -> zarr.Array:
        """Get a zarr array by path.

        Args:
            path: Zarr path (e.g., 'static/topo/data/elevation')

        Returns:
            Zarr array
        """
        node = self.zarr_root
        for part in path.split('/'):
            node = node[part]
        return node

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sampler)

    def on_epoch_start(self):
        """Call this at the start of each epoch to reshuffle samples."""
        self.sampler.on_epoch_start()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single patch.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - Keys: group names ('static_mask', 'annual', etc.)
                - Values: numpy arrays with loaded data
                - 'metadata': dict with spatial_window, channel_names, etc.
        """
        # Get spatial window from sampler
        spatial_window, _ = self.sampler[idx]  # Returns (SpatialWindow, anchor_year)

        # Load all dataset groups
        result = {}
        metadata = {
            'spatial_window': spatial_window,
            'channel_names': {},
            'patch_idx': idx,
        }

        for group_name, group_config in self.config.dataset_groups.items():
            data, channel_names = self._load_group(group_config, spatial_window)
            result[group_name] = data
            metadata['channel_names'][group_name] = channel_names

        result['metadata'] = metadata
        return result

    def _load_group(
        self,
        group_config: DatasetGroupConfig,
        spatial_window: SpatialWindow,
    ) -> Tuple[np.ndarray, List[str]]:
        """Load all channels for a dataset group.

        Args:
            group_config: Configuration for this group
            spatial_window: Spatial region to load

        Returns:
            Tuple of (data_array, channel_names)
            - data_array: [C, H, W] or [C, T, H, W]
            - channel_names: List of channel names in order
        """
        channels_data = []
        channel_names = []

        for channel in group_config.channels:
            channel_array = self._load_channel(
                channel,
                spatial_window,
                is_temporal=group_config.is_temporal(),
            )
            channels_data.append(channel_array)
            channel_names.append(channel.name)

        # Stack along channel dimension
        if group_config.is_temporal():
            # Shape: [C, T, H, W]
            data = np.stack(channels_data, axis=0)
        else:
            # Shape: [C, H, W]
            data = np.stack(channels_data, axis=0)

        # Convert to target dtype
        data = data.astype(group_config.get_numpy_dtype())

        return data, channel_names

    def _load_channel(
        self,
        channel: ChannelConfig,
        spatial_window: SpatialWindow,
        is_temporal: bool,
    ) -> np.ndarray:
        """Load or compute a single channel.

        Args:
            channel: Channel configuration
            spatial_window: Spatial region to load
            is_temporal: Whether this channel is part of a temporal group

        Returns:
            Channel data array:
            - [H, W] for static channels
            - [T, H, W] for temporal channels
        """
        if channel.is_formula_based():
            return self._compute_formula_channel(channel, spatial_window, is_temporal)
        else:
            return self._load_source_channel(channel, spatial_window, is_temporal)

    def _load_source_channel(
        self,
        channel: ChannelConfig,
        spatial_window: SpatialWindow,
        is_temporal: bool,
    ) -> np.ndarray:
        """Load a channel from zarr source.

        Args:
            channel: Channel configuration
            spatial_window: Spatial region to load
            is_temporal: Whether output should be temporal

        Returns:
            Channel data array
        """
        arr = self._get_zarr_array(channel.source)

        # Determine if source array is temporal
        source_is_temporal = len(arr.shape) == 3  # [T, H, W]

        # Extract spatial slice
        row_slice, col_slice = spatial_window.to_slice()

        if source_is_temporal:
            # Source is temporal
            if channel.requires_year_extraction():
                # Extract specific year and output as static [H, W]
                year_idx = self._year_to_index(channel.year)
                if year_idx < 0 or year_idx >= arr.shape[0]:
                    raise ValueError(
                        f"Channel '{channel.name}' requests year {channel.year} "
                        f"but zarr array only has {arr.shape[0]} timesteps "
                        f"(starting at year {self.zarr_start_year})"
                    )
                data = arr[year_idx, row_slice, col_slice]
            else:
                # Load full temporal range with padding if needed
                data = self._load_temporal_with_padding(arr, spatial_window)
        else:
            # Source is static [H, W]
            data = arr[row_slice, col_slice]

        # Handle fill values
        if channel.fill_value is not None:
            data = data.astype(np.float32)  # Ensure float for NaN
            data[data == channel.fill_value] = np.nan

        # Apply thresholding if needed
        if channel.requires_thresholding():
            data = self._apply_threshold(data, channel.ok_if)

        return data

    def _load_temporal_with_padding(
        self,
        arr: zarr.Array,
        spatial_window: SpatialWindow,
    ) -> np.ndarray:
        """Load temporal array with padding to match time_window.

        Args:
            arr: Zarr array [T, H, W]
            spatial_window: Spatial region to load

        Returns:
            Array [T_window, H, W] potentially padded with NaN
        """
        row_slice, col_slice = spatial_window.to_slice()

        # Determine time slice
        zarr_years = list(range(
            self.zarr_start_year,
            self.zarr_start_year + self.zarr_n_timesteps
        ))
        config_years = self.config.time_window.years

        # Find overlap
        overlap_start = max(zarr_years[0], config_years[0])
        overlap_end = min(zarr_years[-1], config_years[-1])

        if overlap_start > overlap_end:
            # No overlap - return all NaN
            return np.full(
                (len(config_years), spatial_window.height, spatial_window.width),
                np.nan,
                dtype=np.float32,
            )

        # Compute indices
        zarr_start_idx = overlap_start - zarr_years[0]
        zarr_end_idx = zarr_start_idx + (overlap_end - overlap_start + 1)

        config_start_idx = overlap_start - config_years[0]
        config_end_idx = config_start_idx + (overlap_end - overlap_start + 1)

        # Load overlapping data
        overlap_data = arr[zarr_start_idx:zarr_end_idx, row_slice, col_slice]

        # Create output with padding
        output = np.full(
            (len(config_years), spatial_window.height, spatial_window.width),
            np.nan,
            dtype=np.float32,
        )
        output[config_start_idx:config_end_idx] = overlap_data

        return output

    def _compute_formula_channel(
        self,
        channel: ChannelConfig,
        spatial_window: SpatialWindow,
        is_temporal: bool,
    ) -> np.ndarray:
        """Compute a formula-based channel.

        Args:
            channel: Channel configuration with formula
            spatial_window: Spatial region
            is_temporal: Whether output should be temporal

        Returns:
            Computed channel data
        """
        formula = channel.formula

        # Currently only support temporal_position formula
        if 't / (T - 1)' in formula:
            if not is_temporal:
                raise ValueError(
                    f"Formula channel '{channel.name}' uses temporal formula "
                    f"but is in a static group"
                )

            T = self.config.time_window.n_years
            t = np.arange(T, dtype=np.float32)
            temporal_pos = t / (T - 1)  # [T]

            # Broadcast to [T, H, W]
            H, W = spatial_window.height, spatial_window.width
            temporal_pos = np.broadcast_to(
                temporal_pos[:, None, None],
                (T, H, W)
            ).copy()

            return temporal_pos
        else:
            raise NotImplementedError(
                f"Formula '{formula}' not supported for channel '{channel.name}'"
            )

    def _apply_threshold(self, data: np.ndarray, ok_if: Dict[str, Any]) -> np.ndarray:
        """Apply threshold operation to data.

        Args:
            data: Input data
            ok_if: Threshold specification {op: ">=", value: 0.25}

        Returns:
            Binary mask (0 or 1)
        """
        op = ok_if['op']
        value = ok_if['value']

        if op == '>=':
            mask = (data >= value).astype(np.uint8)
        elif op == '>':
            mask = (data > value).astype(np.uint8)
        elif op == '<=':
            mask = (data <= value).astype(np.uint8)
        elif op == '<':
            mask = (data < value).astype(np.uint8)
        elif op == '==':
            mask = (data == value).astype(np.uint8)
        elif op == '!=':
            mask = (data != value).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported threshold operator: {op}")

        return mask

    def _year_to_index(self, year: int) -> int:
        """Convert year to zarr array index.

        Args:
            year: Year value

        Returns:
            Index in zarr temporal dimension
        """
        if self.zarr_start_year is None:
            raise ValueError("No temporal data available")

        return year - self.zarr_start_year


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched dictionary with numpy arrays converted to tensors
    """
    if len(batch) == 0:
        return {}

    # Get all group names (excluding metadata)
    group_names = [k for k in batch[0].keys() if k != 'metadata']

    result = {}

    # Stack each group
    for group_name in group_names:
        arrays = [sample[group_name] for sample in batch]
        stacked = np.stack(arrays, axis=0)  # [B, C, H, W] or [B, C, T, H, W]
        result[group_name] = torch.from_numpy(stacked)

    # Collect metadata
    result['metadata'] = [sample['metadata'] for sample in batch]

    return result
