"""
Derived Statistics Computer

Computes statistics and covariance matrices for derived features in a single pass.
Stores results in Zarr dataset for use during training.

Usage:
    computer = DerivedStatsComputer(bindings_config, zarr_path)
    computer.compute_and_save()
"""

import numpy as np
import zarr
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass

from data.loaders.readers import SpatialWindow, TemporalWindow
from data.loaders.readers import DataReader
from data.loaders.readers import MaskBuilder
from data.loaders.config import BindingsRegistry
from data.loaders.builders import DerivedFeatureBuilder
from .online_covariance_computer import (
    OnlineCovarianceComputer,
    OnlineStatsComputer,
    compute_masked_covariance_online
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DerivedStatsConfig:
    """Configuration for derived statistics computation"""
    zarr_location: str
    patch_size: Tuple[int, int]
    n_patches: int
    mask_ref: Optional[str]
    splits: str  # 'all' or list of splits
    seed: int
    features: Dict[str, Dict[str, Any]]
    covariance_matrices: Dict[str, Dict[str, Any]]


class DerivedStatsComputer:
    """
    Compute statistics and covariances for derived features.
    
    Performs single-pass computation over sampled patches, computing:
    - Per-channel statistics (mean, sd, quantiles, etc.) for derived features
    - Covariance matrices (global or per-patch) for specified array groups
    
    Results are stored in the Zarr dataset for use during training.
    
    Usage:
        computer = DerivedStatsComputer(bindings_config, zarr_path)
        computer.compute_and_save()
    """
    
    def __init__(
        self,
        bindings_config: Dict[str, Any],
        zarr_path: Optional[str] = None
    ):
        """
        Initialize DerivedStatsComputer.
        
        Args:
            bindings_config: Parsed bindings configuration
            zarr_path: Path to Zarr dataset (overrides config if provided)
        """
        self.config = bindings_config
        
        # Parse derived statistics config
        if 'derived_statistics' not in self.config:
            raise ValueError("No 'derived_statistics' section in bindings config")
        
        self.stats_config = self._parse_stats_config()
        
        # Initialize readers and builders
        self.zarr_path = zarr_path or self.config['zarr']['path']
        self.zarr_root = zarr.open(self.zarr_path, mode='r+')
        
        self.reader = DataReader(self.config, self.zarr_path)
        
        registry = BindingsRegistry(self.config)
        self.mask_builder = MaskBuilder(registry, self.config, self.zarr_path)
        
        self.feature_builder = DerivedFeatureBuilder(
            self.config,
            self.reader,
            self.mask_builder
        )
        
        logger.info(f"Initialized DerivedStatsComputer for {self.zarr_path}")
        logger.info(f"Will compute stats for {len(self.stats_config.features)} features")
        logger.info(f"Will compute {len(self.stats_config.covariance_matrices)} covariance matrices")
    
    def _parse_stats_config(self) -> DerivedStatsConfig:
        """Parse derived_statistics section from config"""
        ds_config = self.config['derived_statistics']
        
        # Parse sampling config
        sampling = ds_config.get('sampling', {})
        patch_size = tuple(sampling.get('patch_size', [512, 512]))
        n_patches = sampling.get('n_patches', 100)
        mask_ref = sampling.get('mask', 'shared.masks.aoi')
        splits = sampling.get('splits', 'all')
        seed = sampling.get('seed', 42)
        
        # Parse feature configs
        features = ds_config.get('features', {})
        
        # Parse covariance matrix configs
        cov_matrices = ds_config.get('covariance_matrices', {})
        
        zarr_location = ds_config.get('zarr_location', 'derived_stats')
        
        return DerivedStatsConfig(
            zarr_location=zarr_location,
            patch_size=patch_size,
            n_patches=n_patches,
            mask_ref=mask_ref,
            splits=splits,
            seed=seed,
            features=features,
            covariance_matrices=cov_matrices
        )
    
    def compute_and_save(self, verbose: bool = True):
        """
        Compute all statistics and save to Zarr.
        
        Args:
            verbose: Print progress messages
        """
        logger.info("=" * 70)
        logger.info("Starting derived statistics computation")
        logger.info("=" * 70)
        
        # Sample patch locations
        patch_specs = self._sample_patches(verbose=verbose)
        
        # Compute feature statistics
        if self.stats_config.features:
            logger.info("\nComputing feature statistics...")
            feature_stats = self._compute_feature_stats(patch_specs, verbose=verbose)
            self._save_feature_stats(feature_stats, verbose=verbose)
        
        # Compute covariance matrices
        if self.stats_config.covariance_matrices:
            logger.info("\nComputing covariance matrices...")
            cov_results = self._compute_covariances(patch_specs, verbose=verbose)
            self._save_covariances(cov_results, verbose=verbose)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Derived statistics computation complete!")
        logger.info("=" * 70)
    
    def _sample_patches(self, verbose: bool = True) -> List[Tuple[SpatialWindow, TemporalWindow]]:
        """
        Sample random patch locations from the dataset.
        
        Returns:
            List of (SpatialWindow, TemporalWindow) tuples
        """
        logger.info(f"\nSampling {self.stats_config.n_patches} patches...")
        
        np.random.seed(self.stats_config.seed)
        
        # Get dataset dimensions from a reference array
        # Use elevation from topo as reference
        ref_array = self.zarr_root['static/topo/data/elevation']
        H, W = ref_array.shape
        
        # Get temporal range from config
        # Use the temporal window from a temporal input group
        temporal_groups = self.config['inputs']['temporal']
        first_temporal = list(temporal_groups.values())[0]
        window_length = first_temporal.time_window_years or 10
        
        # Determine valid year range
        # Use end year range from training config if available
        training_config = self.config.get('training', {})
        windowing = training_config.get('windowing', {})
        end_year_range = windowing.get('end_year_range', {'min': 2020, 'max': 2024})
        
        valid_years = list(range(end_year_range['min'], end_year_range['max'] + 1))
        
        patch_h, patch_w = self.stats_config.patch_size
        
        patches = []
        for i in range(self.stats_config.n_patches):
            # Random spatial location (ensure patch fits in dataset)
            row = np.random.randint(0, max(1, H - patch_h + 1))
            col = np.random.randint(0, max(1, W - patch_w + 1))
            
            spatial = SpatialWindow.from_upper_left_and_hw(
                (row, col),
                (patch_h, patch_w)
            )
            
            # Random temporal window
            end_year = np.random.choice(valid_years)
            temporal = TemporalWindow(end_year, window_length)
            
            patches.append((spatial, temporal))
            
            if verbose and (i + 1) % 20 == 0:
                logger.info(f"  Sampled {i + 1}/{self.stats_config.n_patches} patches")
        
        logger.info(f"✓ Sampled {len(patches)} patches")
        return patches
    
    def _compute_feature_stats(
        self,
        patch_specs: List[Tuple[SpatialWindow, TemporalWindow]],
        verbose: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute statistics for all enabled derived features.
        
        Returns:
            Dict mapping feature_name -> {stat_name: array}
        """
        feature_stats = {}
        
        for feature_name, feature_config in self.stats_config.features.items():
            if not feature_config.get('enabled', True):
                continue
            
            logger.info(f"\nComputing stats for '{feature_name}'...")
            
            # Get list of stats to compute
            stat_names = feature_config.get('stats', ['mean', 'sd', 'min', 'max'])
            
            # Parse quantiles from stat names
            quantiles = []
            for stat in stat_names:
                if stat.startswith('q'):
                    # q02 -> 0.02, q25 -> 0.25, etc.
                    q_val = int(stat[1:]) / 100.0
                    quantiles.append(q_val)
            
            # Initialize online stats computer (will determine n_features from first patch)
            stats_computer = None
            
            # Load mask if specified
            mask_ref = self.stats_config.mask_ref
            
            # Process patches
            for i, (spatial, temporal) in enumerate(patch_specs):
                try:
                    import time
                    patch_start = time.time()
                    
                    # Build derived feature
                    result = self.feature_builder.build_derived_feature(
                        feature_name,
                        spatial,
                        temporal
                    )
                    build_time = time.time() - patch_start
                    
                    # Get data
                    data = result.data  # Shape varies: [C, H, W] or [C, T, H, W]
                    
                    # We want to combine TWO masks:
                    # 1. Spatial AOI mask (exclude pixels outside study area)
                    # 2. Derived feature's mask (exclude invalid data points)
                    
                    # Start with derived feature's mask if available
                    feature_mask = result.mask  # [C, T, H, W] or [C, H, W] or None
                    
                    # Get spatial AOI mask
                    if mask_ref:
                        mask_result = self.mask_builder.read_mask(
                            mask_ref.split('.')[-1],  # Extract mask name (e.g., 'aoi')
                            spatial,
                            None  # No temporal window - we want spatial mask only
                        )
                        aoi_mask = mask_result.data  # Should be [H, W]
                    else:
                        aoi_mask = None
                    
                    # Combine masks
                    if feature_mask is not None and aoi_mask is not None:
                        # Broadcast AOI mask to match feature mask dimensions
                        if feature_mask.ndim == 4:
                            # Feature mask is [C, T, H, W], AOI is [H, W]
                            # Broadcast AOI to [C, T, H, W]
                            C, T, H, W = feature_mask.shape
                            aoi_broadcast = np.broadcast_to(
                                aoi_mask[None, None, :, :],
                                (C, T, H, W)
                            )
                        elif feature_mask.ndim == 3:
                            # Feature mask is [C, H, W], AOI is [H, W]
                            # Broadcast AOI to [C, H, W]
                            C, H, W = feature_mask.shape
                            aoi_broadcast = np.broadcast_to(
                                aoi_mask[None, :, :],
                                (C, H, W)
                            )
                        else:
                            aoi_broadcast = aoi_mask
                        
                        # Combine: pixel is valid only if BOTH masks say it's valid
                        combined_mask = feature_mask & aoi_broadcast
                    elif feature_mask is not None:
                        combined_mask = feature_mask
                    elif aoi_mask is not None:
                        # Broadcast AOI mask to match data dimensions
                        if data.ndim == 4:
                            C, T, H, W = data.shape
                            combined_mask = np.broadcast_to(
                                aoi_mask[None, None, :, :],
                                (C, T, H, W)
                            )
                        elif data.ndim == 3:
                            C, H, W = data.shape
                            combined_mask = np.broadcast_to(
                                aoi_mask[None, :, :],
                                (C, H, W)
                            )
                        else:
                            combined_mask = aoi_mask
                    else:
                        combined_mask = None
                    
                    # Initialize stats computer on first patch
                    if stats_computer is None:
                        n_features = data.shape[0]  # First dimension is channels
                        stats_computer = OnlineStatsComputer(
                            n_features=n_features,
                            quantiles=quantiles,
                            max_samples_for_quantiles=100000  # Limit to 100k samples
                        )
                        logger.info(f"  Feature has {n_features} channels")
                        logger.info(f"  Data shape: {data.shape}")
                        if combined_mask is not None:
                            logger.info(f"  Combined mask shape: {combined_mask.shape}")
                            logger.info(f"  Valid fraction: {combined_mask.mean():.2%}")
                    
                    # Flatten for stats computation
                    # data: [C, ...] -> [C, N] where N = product of non-channel dims
                    data_flat = data.reshape(data.shape[0], -1)  # [C, N]
                    
                    # Apply mask if present
                    if combined_mask is not None:
                        # Flatten mask: [C, ...] -> [C, N]
                        mask_flat = combined_mask.reshape(combined_mask.shape[0], -1)  # [C, N]
                        
                        # For each channel, extract only valid pixels
                        # We'll process each channel separately since each has different valid pixels
                        valid_data_per_channel = []
                        for c in range(data.shape[0]):
                            valid_pixels = data_flat[c, mask_flat[c]]  # [N_valid_c]
                            if len(valid_pixels) > 0:
                                valid_data_per_channel.append(valid_pixels)
                            else:
                                # No valid pixels for this channel in this patch
                                valid_data_per_channel.append(np.array([]))
                        
                        # Stack back to [C, max_valid]
                        # But OnlineStatsComputer expects [N_samples, C] format
                        # So we need to transpose and handle variable lengths
                        
                        # Actually, let's use a different approach:
                        # Collect all valid (channel, value) pairs across all spatial/temporal locations
                        all_valid_samples = []
                        for c in range(data.shape[0]):
                            for val in valid_data_per_channel[c]:
                                sample = np.zeros(data.shape[0])
                                sample[c] = val
                                # But this doesn't make sense - we want per-channel stats
                        
                        # Better approach: just flatten everything and use per-channel mask
                        # Update stats per-channel
                        for c in range(data.shape[0]):
                            valid_pixels = data_flat[c, mask_flat[c]]  # [N_valid_c]
                            if len(valid_pixels) > 0:
                                # Create a "sample" with just this channel's data
                                # OnlineStatsComputer expects [N_samples, n_features]
                                # We'll give it [N_valid, 1] and update only this channel
                                
                                # Actually, we need to rethink this...
                                # OnlineStatsComputer computes stats across ALL features together
                                # But we have per-channel masks
                                
                                # Simplest solution: update with all channels at once,
                                # but only for locations where ALL channels are valid
                                pass
                        
                        # OK, let's use the simplest approach:
                        # Take intersection of all channel masks (pixel valid only if ALL channels valid)
                        combined_mask_all = combined_mask.all(axis=0)  # [T, H, W] or [H, W]
                        mask_flat_all = combined_mask_all.ravel()  # [N]
                        data_masked = data_flat[:, mask_flat_all]  # [C, N_valid]
                    else:
                        data_masked = data_flat
                    
                    # Update stats (transpose to [N_valid, C] for OnlineStatsComputer)
                    if data_masked.shape[1] > 0:
                        # PERFORMANCE: Subsample if we have too many pixels
                        # No need to use ALL 2.4M pixels - 10k is plenty for good stats
                        max_pixels_per_patch = 10000
                        
                        if data_masked.shape[1] > max_pixels_per_patch:
                            # Randomly subsample pixels
                            indices = np.random.choice(
                                data_masked.shape[1],
                                max_pixels_per_patch,
                                replace=False
                            )
                            data_subsampled = data_masked[:, indices]  # [C, 10k]
                        else:
                            data_subsampled = data_masked
                        
                        stats_computer.update(data_subsampled.T)  # [N, C]
                    
                    total_time = time.time() - patch_start
                    
                    if verbose and (i + 1) % 10 == 0:
                        logger.info(
                            f"    Processed {i + 1}/{len(patch_specs)} patches "
                            f"(build: {build_time:.2f}s, total: {total_time:.2f}s, "
                            f"valid pixels: {data_masked.shape[1]})"
                        )
                
                except Exception as e:
                    logger.warning(f"    Error processing patch {i}: {e}")
                    continue
            
            # Get final statistics
            if stats_computer and stats_computer.n_samples > 0:
                stats = stats_computer.get_stats()
                feature_stats[feature_name] = stats
                
                logger.info(f"  ✓ Computed {len(stats)} statistics over {stats_computer.n_samples} samples")
                logger.info(f"    Available stats: {list(stats.keys())}")
            else:
                logger.warning(f"  ✗ No valid samples for '{feature_name}'")
        
        return feature_stats
    
    def _compute_covariances(
        self,
        patch_specs: List[Tuple[SpatialWindow, TemporalWindow]],
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute covariance matrices for specified array groups.
        
        Returns:
            Dict mapping cov_name -> {
                'covariance': array,
                'inverse': array (if computed),
                'mean': array,
                'n_samples': int
            }
        """
        cov_results = {}
        
        for cov_name, cov_config in self.stats_config.covariance_matrices.items():
            logger.info(f"\nComputing covariance matrix '{cov_name}'...")
            
            # Parse config
            arrays = cov_config.get('arrays', [])
            domain = cov_config.get('domain', 'global')
            inverse_config = cov_config.get('inverse', {})
            compute_inverse = inverse_config.get('compute', False)
            regularization = inverse_config.get('regularization', 1e-6)
            
            n_features = len(arrays)
            logger.info(f"  Arrays: {n_features} features")
            logger.info(f"  Domain: {domain}")
            
            # Initialize covariance computer
            if domain == 'global':
                cov_computer = OnlineCovarianceComputer(n_features)
            else:
                # For per-patch, collect covariance matrices
                patch_covs = []
                patch_weights = []
                global_mean = None
                total_samples = 0
            
            # Load mask
            mask_ref = self.stats_config.mask_ref
            
            # Process patches
            for i, (spatial, temporal) in enumerate(patch_specs):
                try:
                    # Read all arrays for this patch
                    patch_data = []
                    for array_path in arrays:
                        data = self._read_array(array_path, spatial, temporal)
                        patch_data.append(data)
                    
                    # Stack to [n_features, H, W] or [n_features, T, H, W]
                    patch_data = np.stack(patch_data, axis=0)
                    
                    # Read mask
                    if mask_ref:
                        mask_result = self.mask_builder.read_mask(
                            mask_ref.split('.')[-1],
                            spatial,
                            temporal
                        )
                        mask = mask_result.data
                        
                        # Handle temporal dimension if present
                        if patch_data.ndim == 4 and mask.ndim == 2:
                            # Data is [n_features, T, H, W], mask is [H, W]
                            # Broadcast mask to all timesteps
                            mask = np.broadcast_to(mask[None, :, :], patch_data.shape[1:])
                        elif patch_data.ndim == 4 and mask.ndim == 3:
                            # Data is [n_features, T, H, W], mask is [T, H, W]
                            pass  # Already matches
                        # else: both are spatial only [n_features, H, W] and [H, W]
                    else:
                        # No mask - all valid
                        mask = np.ones(patch_data.shape[1:], dtype=bool)
                    
                    # Flatten spatial (and temporal) dimensions
                    # patch_data: [n_features, ...] -> [n_features, N]
                    data_flat = patch_data.reshape(n_features, -1)
                    mask_flat = mask.ravel()
                    
                    # Extract valid pixels
                    valid_data = data_flat[:, mask_flat]  # [n_features, n_valid]
                    
                    if valid_data.shape[1] == 0:
                        continue  # Skip patches with no valid data
                    
                    # Update covariance
                    if domain == 'global':
                        cov_computer.update(valid_data)
                    else:
                        # Per-patch: compute covariance for this patch
                        if valid_data.shape[1] > 1:
                            patch_cov = np.cov(valid_data, ddof=1)
                            patch_covs.append(patch_cov)
                            patch_weights.append(valid_data.shape[1])
                            
                            # Track mean
                            if global_mean is None:
                                global_mean = valid_data.mean(axis=1)
                            else:
                                n_old = total_samples
                                n_new = valid_data.shape[1]
                                global_mean = (
                                    n_old * global_mean + n_new * valid_data.mean(axis=1)
                                ) / (n_old + n_new)
                            
                            total_samples += valid_data.shape[1]
                    
                    if verbose and (i + 1) % 20 == 0:
                        logger.info(f"    Processed {i + 1}/{len(patch_specs)} patches")
                
                except Exception as e:
                    logger.warning(f"    Error processing patch {i}: {e}")
                    continue
            
            # Finalize covariance computation
            if domain == 'global':
                if cov_computer.n_samples > 0:
                    cov = cov_computer.get_covariance()
                    mean = cov_computer.get_mean()
                    n_samples = cov_computer.n_samples
                else:
                    logger.warning(f"  ✗ No valid samples for '{cov_name}'")
                    continue
            else:
                # Per-patch: average covariance matrices
                if patch_covs:
                    patch_covs = np.array(patch_covs)
                    patch_weights = np.array(patch_weights)
                    cov = np.average(patch_covs, axis=0, weights=patch_weights)
                    mean = global_mean
                    n_samples = total_samples
                else:
                    logger.warning(f"  ✗ No valid patches for '{cov_name}'")
                    continue
            
            # Compute inverse if requested
            inv = None
            if compute_inverse:
                try:
                    if regularization > 0:
                        cov_reg = cov + regularization * np.eye(n_features)
                        inv = np.linalg.inv(cov_reg)
                    else:
                        inv = np.linalg.inv(cov)
                    logger.info(f"  ✓ Computed inverse (regularization={regularization})")
                except np.linalg.LinAlgError as e:
                    logger.warning(f"  ✗ Could not compute inverse: {e}")
            
            # Store results
            cov_results[cov_name] = {
                'covariance': cov,
                'inverse': inv,
                'mean': mean,
                'n_samples': n_samples,
                'domain': domain,
                'regularization': regularization if compute_inverse else None
            }
            
            logger.info(f"  ✓ Covariance shape: {cov.shape}, n_samples: {n_samples}")
        
        return cov_results
    
    def _read_array(
        self,
        array_path: str,
        spatial: SpatialWindow,
        temporal: Optional[TemporalWindow]
    ) -> np.ndarray:
        """
        Read a single array from Zarr.
        
        Supports both direct Zarr paths and reference paths.
        
        Args:
            array_path: Either 'static/topo/data/elevation' or 'inputs.static.topo.elevation'
            spatial: Spatial window
            temporal: Temporal window (for temporal arrays)
        
        Returns:
            Array data with shape [H, W] or [T, H, W]
        """
        # Check if it's a reference path
        if array_path.startswith('inputs.'):
            # Parse reference: inputs.static.topo.elevation
            parts = array_path.split('.')
            if len(parts) < 4:
                raise ValueError(f"Invalid reference path: {array_path}")
            
            category = parts[1]  # 'static', 'temporal', etc.
            group_name = parts[2]  # 'topo'
            band_name = parts[3]  # 'elevation'
            
            # Read using DataReader
            if category == 'temporal':
                result = self.reader.read_temporal_group(
                    group_name, spatial, temporal, return_full_temporal=True
                )
            elif category == 'static':
                result = self.reader.read_static_group(group_name, spatial)
            else:
                raise ValueError(f"Unsupported category for covariance: {category}")
            
            # Find the band
            try:
                band_idx = result.band_names.index(band_name)
                data = result.data[band_idx]  # [H, W] or [T, H, W]
            except ValueError:
                raise ValueError(
                    f"Band '{band_name}' not found in group '{group_name}'. "
                    f"Available: {result.band_names}"
                )
        
        else:
            # Direct Zarr path: static/topo/data/elevation
            array = self.zarr_root[array_path]
            
            # Read with spatial slicing
            row_slice = slice(spatial.row_start, spatial.row_start + spatial.height)
            col_slice = slice(spatial.col_start, spatial.col_start + spatial.width)
            
            # Handle temporal dimension if present
            if array.ndim == 3:
                # Temporal array [T, H, W]
                if temporal is None:
                    raise ValueError(f"Temporal window required for array: {array_path}")
                
                years = temporal.get_years()
                # Assume zarr array has years as dimension
                # This is simplified - real implementation needs year mapping
                data = array[:, row_slice, col_slice]
            elif array.ndim == 2:
                # Static array [H, W]
                data = array[row_slice, col_slice]
            else:
                raise ValueError(f"Unexpected array dimensions: {array.ndim}")
        
        return data
    
    def _save_feature_stats(
        self,
        feature_stats: Dict[str, Dict[str, np.ndarray]],
        verbose: bool = True
    ):
        """Save feature statistics to Zarr"""
        logger.info("\nSaving feature statistics to Zarr...")
        
        # Create zarr group for derived stats
        zarr_location = self.stats_config.zarr_location
        
        # Delete existing group if present
        if zarr_location in self.zarr_root:
            logger.info(f"  Deleting existing '{zarr_location}' group...")
            del self.zarr_root[zarr_location]
        
        stats_group = self.zarr_root.create_group(zarr_location)
        features_group = stats_group.create_group('features')
        
        for feature_name, stats in feature_stats.items():
            # Create group for this feature
            feature_group = features_group.create_group(feature_name)
            
            # Save each statistic as an array
            for stat_name, stat_values in stats.items():
                feature_group.create_dataset(
                    stat_name,
                    data=stat_values,
                    dtype=np.float32,
                    chunks=None
                )
            
            # Save metadata
            feature_group.attrs['feature_name'] = feature_name
            feature_group.attrs['n_channels'] = len(stats['mean'])
            feature_group.attrs['stats'] = list(stats.keys())
            
            logger.info(f"  ✓ Saved '{feature_name}': {list(stats.keys())}")
        
        logger.info("✓ Feature statistics saved")
    
    def _save_covariances(
        self,
        cov_results: Dict[str, Dict[str, Any]],
        verbose: bool = True
    ):
        """Save covariance matrices to Zarr"""
        logger.info("\nSaving covariance matrices to Zarr...")
        
        zarr_location = self.stats_config.zarr_location
        stats_group = self.zarr_root[zarr_location]
        
        cov_group = stats_group.create_group('covariance_matrices')
        
        for cov_name, cov_data in cov_results.items():
            # Create group for this covariance matrix
            cov_matrix_group = cov_group.create_group(cov_name)
            
            # Save covariance matrix
            cov_matrix_group.create_dataset(
                'covariance',
                data=cov_data['covariance'],
                dtype=np.float32,
                chunks=None
            )
            
            # Save mean
            cov_matrix_group.create_dataset(
                'mean',
                data=cov_data['mean'],
                dtype=np.float32,
                chunks=None
            )
            
            # Save inverse if computed
            if cov_data['inverse'] is not None:
                inverse_id = self.stats_config.covariance_matrices[cov_name]['inverse']['id']
                cov_matrix_group.create_dataset(
                    inverse_id,
                    data=cov_data['inverse'],
                    dtype=np.float32,
                    chunks=None
                )
            
            # Save metadata
            cov_matrix_group.attrs['n_features'] = cov_data['covariance'].shape[0]
            cov_matrix_group.attrs['n_samples'] = int(cov_data['n_samples'])
            cov_matrix_group.attrs['domain'] = cov_data['domain']
            if cov_data['regularization'] is not None:
                cov_matrix_group.attrs['regularization'] = float(cov_data['regularization'])
            
            logger.info(
                f"  ✓ Saved '{cov_name}': {cov_data['covariance'].shape} "
                f"({cov_data['n_samples']} samples)"
            )
        
        logger.info("✓ Covariance matrices saved")


if __name__ == '__main__':
    # Example usage
    import sys
    from data.loaders.config import BindingsParser
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config/frl_bindings_v0.yaml'
    
    print("Example: Computing Derived Statistics")
    print("=" * 70)
    
    # Parse config
    print("\n1. Parsing configuration...")
    parser = BindingsParser(config_path)
    config = parser.parse()
    print(f"   ✓ Loaded config: {config['name']}")
    
    # Check if derived_statistics section exists
    if 'derived_statistics' not in config:
        print("\n   ✗ No 'derived_statistics' section found in config")
        print("   Add this section to your YAML to enable stats computation")
        sys.exit(0)
    
    # Initialize computer
    print("\n2. Initializing DerivedStatsComputer...")
    computer = DerivedStatsComputer(config)
    print(f"   ✓ Will compute stats for {len(computer.stats_config.features)} features")
    print(f"   ✓ Will compute {len(computer.stats_config.covariance_matrices)} covariance matrices")
    print(f"   ✓ Sampling {computer.stats_config.n_patches} patches of size {computer.stats_config.patch_size}")
    
    # Compute and save
    print("\n3. Computing statistics (this may take a while)...")
    try:
        computer.compute_and_save(verbose=True)
        print("\n✓ SUCCESS: All statistics computed and saved!")
        
        # Show where results are stored
        print(f"\nResults stored in Zarr at: {computer.stats_config.zarr_location}")
        print(f"  - Feature stats: {computer.stats_config.zarr_location}/features/")
        print(f"  - Covariances: {computer.stats_config.zarr_location}/covariance_matrices/")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
