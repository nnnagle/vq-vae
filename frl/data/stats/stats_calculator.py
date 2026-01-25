"""
Statistics calculator for forest dataset features.

This module computes statistics (mean, std, quantiles, covariance) for features
by sampling patches from the dataset.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..loaders.config.dataset_config import (
    BindingsConfig,
    FeatureConfig,
    FeatureChannelConfig,
)
from ..loaders.dataset.forest_dataset_v2 import ForestDatasetV2

logger = logging.getLogger(__name__)


class StatsCalculator:
    """Calculates statistics for features from sampled dataset patches.

    This class:
    - Samples patches from the dataset (all splits)
    - Extracts channels for each feature
    - Applies masks and handles NaNs/fill values
    - Computes univariate statistics per channel
    - Optionally computes covariance matrices
    - Saves results to JSON
    """

    def __init__(self, config: BindingsConfig):
        """Initialize stats calculator.

        Args:
            config: Parsed bindings configuration
        """
        self.config = config
        self.stats_config = config.stats

        if not self.stats_config:
            raise ValueError("Config must have 'stats' section")

        if not config.features:
            raise ValueError("Config must have 'features' section")

    def compute_and_save(self, force: bool = False) -> None:
        """Compute statistics and save to JSON file.

        Args:
            force: If True, always compute. If False, respect 'compute' setting.
        """
        stats_path = Path(self.stats_config.file)

        # Check if we should compute
        if not force:
            if self.stats_config.compute == 'never':
                logger.info("Stats compute mode is 'never', skipping")
                return
            elif self.stats_config.compute == 'if-not-exists' and stats_path.exists():
                logger.info(f"Stats file exists: {stats_path}, skipping")
                return

        logger.info("Computing statistics...")

        # Compute stats
        stats_dict = self.compute_stats()

        # Save to JSON
        self._save_stats(stats_dict, stats_path)

        logger.info(f"Statistics saved to {stats_path}")

    def compute_stats(self) -> Dict[str, Any]:
        """Compute statistics for all features.

        Returns:
            Dictionary with stats for each feature
        """
        # Create dataset (all splits, no filtering)
        dataset = self._create_dataset()

        # Sample patches
        sample_indices = self._sample_patches(dataset)

        logger.info(f"Computing stats from {len(sample_indices)} patches")

        # Compute stats for each feature
        stats_dict = {}
        for feature_name, feature_config in self.config.features.items():
            logger.info(f"Computing stats for feature: {feature_name}")
            feature_stats = self._compute_feature_stats(
                dataset, sample_indices, feature_name, feature_config
            )
            stats_dict[feature_name] = feature_stats

        return stats_dict

    def _create_dataset(self) -> ForestDatasetV2:
        """Create dataset for stats computation.

        Returns:
            ForestDatasetV2 instance with split=None (all splits)
        """
        return ForestDatasetV2(
            config=self.config,
            split=None,  # Use all splits
            patch_size=256,
            epoch_mode='full',
            min_aoi_fraction=0.5,
        )

    def _sample_patches(self, dataset: ForestDatasetV2) -> List[int]:
        """Sample patch indices for stats computation.

        Args:
            dataset: Dataset to sample from

        Returns:
            List of sampled patch indices
        """
        n_samples = self.stats_config.samples.get('n', 16)
        total_patches = len(dataset)

        if n_samples >= total_patches:
            logger.warning(
                f"Requested {n_samples} samples but only {total_patches} patches available, "
                f"using all patches"
            )
            return list(range(total_patches))

        # Random sampling
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        return rng.choice(total_patches, size=n_samples, replace=False).tolist()

    def _compute_feature_stats(
        self,
        dataset: ForestDatasetV2,
        sample_indices: List[int],
        feature_name: str,
        feature_config: FeatureConfig,
    ) -> Dict[str, Any]:
        """Compute statistics for a single feature.

        Args:
            dataset: Dataset to sample from
            sample_indices: Patch indices to use
            feature_name: Name of the feature
            feature_config: Feature configuration

        Returns:
            Dictionary with stats for this feature
        """
        # Collect data from all sampled patches
        patch_data_list = []

        for idx in sample_indices:
            # Get patch sample
            sample = dataset[idx]

            # Extract feature channels and apply masks
            feature_data, valid_mask = self._extract_feature_data(
                sample, feature_config
            )

            if feature_data is not None:
                patch_data_list.append((feature_data, valid_mask))

        if not patch_data_list:
            logger.warning(f"No valid data for feature {feature_name}")
            return {}

        # Compute univariate stats
        channel_stats = self._compute_univariate_stats(
            patch_data_list, feature_config
        )

        # Compute covariance if requested
        result = channel_stats
        if feature_config.covariance and feature_config.covariance.calculate:
            cov_matrix = self._compute_covariance(
                patch_data_list, feature_config
            )
            result['covariance'] = cov_matrix.tolist()

        return result

    def _extract_feature_data(
        self,
        sample: Dict[str, Any],
        feature_config: FeatureConfig,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract and stack feature channels from a patch sample.

        Args:
            sample: Dataset sample (dict with groups and metadata)
            feature_config: Feature configuration

        Returns:
            Tuple of (feature_data, valid_mask):
                - feature_data: [C, H, W] or [C, T, H, W] array
                - valid_mask: [H, W] or [T, H, W] boolean mask (pixels to include)
        """
        channel_arrays = []

        # Extract each channel
        for channel_ref, channel_config in feature_config.channels.items():
            # Parse reference (format: dataset_group.channel_name)
            dataset_group, channel_name = channel_ref.split('.')

            # Get group data
            if dataset_group not in sample:
                logger.warning(f"Dataset group {dataset_group} not in sample")
                return None, None

            group_data = sample[dataset_group]

            # Get channel index
            channel_names = sample['metadata']['channel_names'][dataset_group]
            if channel_name not in channel_names:
                logger.warning(f"Channel {channel_name} not in {dataset_group}")
                return None, None

            channel_idx = channel_names.index(channel_name)

            # Extract channel data
            channel_data = group_data[channel_idx].astype(np.float32)  # [H,W] or [T,H,W]
            channel_arrays.append(channel_data)

        # Stack channels: [C, H, W] or [C, T, H, W]
        feature_data = np.stack(channel_arrays, axis=0)

        # Build valid mask from global masks
        valid_mask = self._build_valid_mask(sample, feature_data)

        return feature_data, valid_mask

    def _build_valid_mask(
        self,
        sample: Dict[str, Any],
        feature_data: np.ndarray,
    ) -> np.ndarray:
        """Build valid pixel mask from global masks and NaN handling.

        Args:
            sample: Dataset sample
            feature_data: Feature data array [C, H, W] or [C, T, H, W]

        Returns:
            Boolean mask [H, W] or [T, H, W] (True = valid pixel)
        """
        # Start with all True
        if feature_data.ndim == 3:
            # Static: [C, H, W] -> mask is [H, W]
            spatial_shape = feature_data.shape[1:]
            valid_mask = np.ones(spatial_shape, dtype=bool)
        else:
            # Temporal: [C, T, H, W] -> mask is [T, H, W]
            spatial_shape = feature_data.shape[2:]
            valid_mask = np.ones((feature_data.shape[1],) + spatial_shape, dtype=bool)

        # Apply global masks from stats config
        for mask_ref in self.stats_config.mask:
            dataset_group, mask_name = mask_ref.split('.')

            if dataset_group in sample:
                channel_names = sample['metadata']['channel_names'][dataset_group]
                if mask_name in channel_names:
                    mask_idx = channel_names.index(mask_name)
                    mask_data = sample[dataset_group][mask_idx]  # [H,W] or [T,H,W]

                    # If mask is static [H,W] but data is temporal, broadcast
                    if mask_data.ndim == 2 and valid_mask.ndim == 3:
                        mask_data = mask_data[np.newaxis, :, :]  # [1, H, W]

                    # AND with existing mask
                    valid_mask = valid_mask & (mask_data > 0)

        # Mask out NaN pixels in feature data
        # For temporal data: any NaN across channels invalidates that pixel
        if feature_data.ndim == 3:
            # [C, H, W]
            nan_mask = np.any(np.isnan(feature_data), axis=0)  # [H, W]
            valid_mask = valid_mask & ~nan_mask
        else:
            # [C, T, H, W]
            nan_mask = np.any(np.isnan(feature_data), axis=0)  # [T, H, W]
            valid_mask = valid_mask & ~nan_mask

        return valid_mask

    def _compute_univariate_stats(
        self,
        patch_data_list: List[Tuple[np.ndarray, np.ndarray]],
        feature_config: FeatureConfig,
    ) -> Dict[str, Dict[str, float]]:
        """Compute univariate statistics for each channel.

        Args:
            patch_data_list: List of (feature_data, valid_mask) tuples
            feature_config: Feature configuration

        Returns:
            Dictionary mapping channel names to their stats
        """
        channel_names = list(feature_config.channels.keys())
        n_channels = len(channel_names)

        # Collect all valid values for each channel
        channel_values = [[] for _ in range(n_channels)]

        for feature_data, valid_mask in patch_data_list:
            # feature_data: [C, H, W] or [C, T, H, W]
            # valid_mask: [H, W] or [T, H, W]

            if feature_data.ndim == 3:
                # Static data [C, H, W]
                for c in range(n_channels):
                    valid_values = feature_data[c][valid_mask]
                    channel_values[c].extend(valid_values.flatten())
            else:
                # Temporal data [C, T, H, W] - collapse time dimension
                for c in range(n_channels):
                    # Average across time for each valid pixel, then collect
                    channel_temporal = feature_data[c]  # [T, H, W]
                    # Only average where valid
                    valid_temporal = channel_temporal[:, valid_mask[0]]  # Assume mask same across time
                    # Actually, valid_mask is [T, H, W], so we need to apply it properly
                    if valid_mask.ndim == 3:
                        # Mask is temporal
                        masked_data = np.where(valid_mask, channel_temporal, np.nan)
                        # Compute mean across time, ignoring NaNs
                        mean_across_time = np.nanmean(masked_data, axis=0)  # [H, W]
                        valid_means = mean_across_time[~np.isnan(mean_across_time)]
                        channel_values[c].extend(valid_means.flatten())
                    else:
                        # Mask is static, broadcast across time
                        valid_values = channel_temporal[:, valid_mask]  # [T, N_valid_pixels]
                        mean_across_time = np.mean(valid_values, axis=0)  # [N_valid_pixels]
                        channel_values[c].extend(mean_across_time.flatten())

        # Compute stats for each channel
        stats_dict = {}
        for channel_name, values in zip(channel_names, channel_values):
            if len(values) == 0:
                logger.warning(f"No valid values for channel {channel_name}")
                continue

            values = np.array(values)

            # Compute requested stats
            channel_stats = {}
            if 'mean' in self.stats_config.stats:
                channel_stats['mean'] = float(np.mean(values))
            if 'sd' in self.stats_config.stats:
                channel_stats['sd'] = float(np.std(values))

            # Quantiles
            quantiles_map = {
                'q02': 2, 'q05': 5, 'q25': 25, 'q50': 50,
                'q75': 75, 'q95': 95, 'q98': 98
            }
            quantile_requests = [
                (name, q) for name, q in quantiles_map.items()
                if name in self.stats_config.stats
            ]
            if quantile_requests:
                percentiles = [q for _, q in quantile_requests]
                quantile_values = np.percentile(values, percentiles)
                for (name, _), val in zip(quantile_requests, quantile_values):
                    channel_stats[name] = float(val)

            stats_dict[channel_name] = channel_stats

        return stats_dict

    def _compute_covariance(
        self,
        patch_data_list: List[Tuple[np.ndarray, np.ndarray]],
        feature_config: FeatureConfig,
    ) -> np.ndarray:
        """Compute covariance matrix for feature channels.

        For stat_domain='patch', computes covariance using patch-local means,
        then averages covariance matrices across patches.

        Args:
            patch_data_list: List of (feature_data, valid_mask) tuples
            feature_config: Feature configuration

        Returns:
            Covariance matrix [C, C]
        """
        n_channels = len(feature_config.channels)
        stat_domain = feature_config.covariance.stat_domain

        if stat_domain == 'patch':
            # Compute per-patch covariance, then average
            cov_matrices = []

            for feature_data, valid_mask in patch_data_list:
                # Extract valid values for each channel
                channel_values = []

                if feature_data.ndim == 3:
                    # Static [C, H, W]
                    for c in range(n_channels):
                        valid_vals = feature_data[c][valid_mask]
                        channel_values.append(valid_vals.flatten())
                else:
                    # Temporal [C, T, H, W] - average across time first
                    for c in range(n_channels):
                        channel_temporal = feature_data[c]  # [T, H, W]
                        if valid_mask.ndim == 3:
                            masked_data = np.where(valid_mask, channel_temporal, np.nan)
                            mean_across_time = np.nanmean(masked_data, axis=0)  # [H, W]
                            valid_means = mean_across_time[~np.isnan(mean_across_time)]
                            channel_values.append(valid_means.flatten())
                        else:
                            valid_values = channel_temporal[:, valid_mask]
                            mean_across_time = np.mean(valid_values, axis=0)
                            channel_values.append(mean_across_time.flatten())

                # Stack into [C, N] matrix
                if all(len(v) > 0 for v in channel_values):
                    # Pad to same length (take minimum length)
                    min_len = min(len(v) for v in channel_values)
                    data_matrix = np.stack([v[:min_len] for v in channel_values], axis=0)

                    # Compute covariance for this patch
                    patch_cov = np.cov(data_matrix)  # [C, C]
                    cov_matrices.append(patch_cov)

            # Average covariance matrices
            if cov_matrices:
                avg_cov = np.mean(cov_matrices, axis=0)
                return avg_cov
            else:
                logger.warning("No valid patches for covariance computation")
                return np.eye(n_channels)

        else:
            # Global covariance (not implemented yet)
            raise NotImplementedError(f"stat_domain='{stat_domain}' not supported yet")

    def _save_stats(self, stats_dict: Dict[str, Any], output_path: Path) -> None:
        """Save statistics to JSON file.

        Args:
            stats_dict: Statistics dictionary
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)


def compute_stats_from_config(
    bindings_path: str,
    force: bool = False
) -> None:
    """Convenience function to compute stats from a bindings YAML file.

    Args:
        bindings_path: Path to bindings YAML file
        force: If True, always compute even if file exists
    """
    from ..loaders.config.dataset_bindings_parser import DatasetBindingsParser

    # Parse config
    parser = DatasetBindingsParser(bindings_path)
    config = parser.parse()

    # Compute stats
    calculator = StatsCalculator(config)
    calculator.compute_and_save(force=force)
