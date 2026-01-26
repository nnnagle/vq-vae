"""
Feature Builder for Forest Representation Model

Builds normalized features from raw dataset tensors according to the bindings configuration.

The FeatureBuilder:
- Loads precomputed statistics (mean, sd, quantiles, covariance) from JSON
- Extracts and stacks channels for each feature
- Applies masks (global feature masks, channel-level masks, NaN masks) combined with AND
- Broadcasts spatial-only masks across time for temporal features
- Applies normalization according to preset types
- For covariance features: centers data and applies Mahalanobis transform

Usage:
    from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
    from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2
    from data.loaders.builders.feature_builder import FeatureBuilder

    # Parse config and create dataset
    config = DatasetBindingsParser('config/frl_binding_v1.yaml').parse()
    dataset = ForestDatasetV2(config, split='train')

    # Create feature builder
    builder = FeatureBuilder(config)

    # Get a sample and build features
    sample = dataset[0]
    data, mask = builder.build_feature('infonce_type_spectral', sample)
    # data: [C, H, W] normalized tensor
    # mask: [H, W] boolean tensor (True = valid pixel)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import logging

from ..config.dataset_config import (
    BindingsConfig,
    FeatureConfig,
    FeatureChannelConfig,
    NormalizationPresetConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """Result from building a feature."""
    data: np.ndarray        # Normalized feature data [C, H, W] or [C, T, H, W]
    mask: np.ndarray        # Boolean mask [H, W] or [T, H, W], True = valid
    feature_name: str       # Name of the feature
    channel_names: List[str]  # Names of channels in order
    is_temporal: bool       # Whether this is a temporal feature


class FeatureBuilder:
    """
    Build normalized features from raw dataset tensors.

    Handles:
    - Loading precomputed statistics from JSON
    - Extracting and stacking channels for each feature
    - Applying masks (global, channel-level, NaN)
    - Broadcasting spatial masks across time for temporal features
    - Applying normalization (zscore, robust_iqr, clamp, etc.)
    - Mahalanobis transform for features with covariance
    """

    def __init__(
        self,
        config: BindingsConfig,
        stats_path: Optional[str] = None,
    ):
        """
        Initialize FeatureBuilder.

        Args:
            config: Parsed bindings configuration
            stats_path: Optional path to stats JSON (overrides config.stats.file)
        """
        self.config = config

        if not config.features:
            raise ValueError("Config must have 'features' section")

        # Load precomputed statistics
        self.stats_path = stats_path or (config.stats.file if config.stats else None)
        self.stats = self._load_stats()

        # Cache for precomputed transforms (e.g., whitening matrices)
        self._transform_cache: Dict[str, np.ndarray] = {}

        logger.info(f"Initialized FeatureBuilder with {len(config.features)} features")
        if self.stats:
            logger.info(f"  Loaded stats for features: {list(self.stats.keys())}")

    def _load_stats(self) -> Dict[str, Any]:
        """Load precomputed statistics from JSON file."""
        if not self.stats_path:
            logger.warning("No stats path configured, normalization will use defaults")
            return {}

        stats_path = Path(self.stats_path)
        if not stats_path.exists():
            logger.warning(f"Stats file not found: {stats_path}")
            return {}

        with open(stats_path, 'r') as f:
            stats = json.load(f)

        logger.info(f"Loaded stats from {stats_path}")
        return stats

    def build_feature(
        self,
        feature_name: str,
        sample: Dict[str, Any],
        apply_normalization: bool = True,
        apply_mahalanobis: bool = True,
    ) -> FeatureResult:
        """
        Build a single feature from a dataset sample.

        Args:
            feature_name: Name of the feature to build
            sample: Dataset sample dictionary from ForestDatasetV2
            apply_normalization: Whether to apply normalization
            apply_mahalanobis: Whether to apply Mahalanobis transform for covariance features

        Returns:
            FeatureResult with normalized data and mask
        """
        feature_config = self.config.get_feature(feature_name)
        if feature_config is None:
            raise ValueError(f"Feature '{feature_name}' not found in config")

        # Extract and stack channels
        channel_data, channel_names = self._extract_channels(sample, feature_config)

        # Determine if temporal
        is_temporal = len(feature_config.dim) == 4  # [C, T, H, W]

        # Build combined mask
        mask = self._build_combined_mask(sample, feature_config, channel_data)

        # Apply normalization per channel
        if apply_normalization:
            channel_data = self._apply_normalization(
                channel_data, feature_name, feature_config, mask
            )

        # Apply Mahalanobis transform if feature has covariance
        if apply_mahalanobis and feature_config.covariance and feature_config.covariance.calculate:
            channel_data = self._apply_mahalanobis_transform(
                channel_data, feature_name, feature_config, mask
            )

        # Zero out masked values
        channel_data = self._apply_mask_to_data(channel_data, mask)

        return FeatureResult(
            data=channel_data,
            mask=mask,
            feature_name=feature_name,
            channel_names=channel_names,
            is_temporal=is_temporal,
        )

    def _extract_channels(
        self,
        sample: Dict[str, Any],
        feature_config: FeatureConfig,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract and stack channels for a feature.

        Args:
            sample: Dataset sample dictionary
            feature_config: Feature configuration

        Returns:
            Tuple of (stacked_data, channel_names)
            - stacked_data: [C, H, W] or [C, T, H, W]
            - channel_names: List of channel reference names
        """
        channel_arrays = []
        channel_names = list(feature_config.channels.keys())

        for channel_ref in channel_names:
            channel_config = feature_config.channels[channel_ref]

            # Parse channel reference (format: dataset_group.channel_name)
            dataset_group = channel_config.dataset_group
            channel_name = channel_config.channel_name

            # Get group data
            if dataset_group not in sample:
                raise ValueError(f"Dataset group '{dataset_group}' not in sample")

            group_data = sample[dataset_group]

            # Get channel index
            group_channel_names = sample['metadata']['channel_names'][dataset_group]
            if channel_name not in group_channel_names:
                raise ValueError(
                    f"Channel '{channel_name}' not found in group '{dataset_group}'. "
                    f"Available: {group_channel_names}"
                )

            channel_idx = group_channel_names.index(channel_name)

            # Extract channel data and convert to float32
            channel_data = group_data[channel_idx].astype(np.float32)
            channel_arrays.append(channel_data)

        # Stack channels: [C, H, W] or [C, T, H, W]
        stacked_data = np.stack(channel_arrays, axis=0)

        return stacked_data, channel_names

    def _build_combined_mask(
        self,
        sample: Dict[str, Any],
        feature_config: FeatureConfig,
        feature_data: np.ndarray,
    ) -> np.ndarray:
        """
        Build combined mask from all sources (global, channel, NaN).

        Mask sources:
        1. Feature-level global masks (feature_config.masks)
        2. Channel-level masks (channel_config.mask for each channel)
        3. NaN values in the data

        All masks are combined with AND. Spatial-only masks are broadcast
        across time for temporal features.

        Args:
            sample: Dataset sample dictionary
            feature_config: Feature configuration
            feature_data: Extracted feature data [C, H, W] or [C, T, H, W]

        Returns:
            Boolean mask [H, W] or [T, H, W] (True = valid pixel)
        """
        # Determine output shape based on feature data
        if feature_data.ndim == 3:
            # Static: [C, H, W] -> mask is [H, W]
            spatial_shape = feature_data.shape[1:]
            mask = np.ones(spatial_shape, dtype=bool)
            is_temporal = False
        else:
            # Temporal: [C, T, H, W] -> mask is [T, H, W]
            n_timesteps = feature_data.shape[1]
            spatial_shape = feature_data.shape[2:]
            mask = np.ones((n_timesteps,) + spatial_shape, dtype=bool)
            is_temporal = True

        # 1. Apply feature-level global masks
        if feature_config.masks:
            for mask_ref in feature_config.masks:
                mask = self._apply_mask_ref(sample, mask_ref, mask, is_temporal)

        # 2. Apply channel-level masks
        for channel_ref, channel_config in feature_config.channels.items():
            if channel_config.mask:
                mask = self._apply_mask_ref(
                    sample, channel_config.mask, mask, is_temporal
                )

        # 3. Apply NaN mask - any NaN across channels invalidates the pixel
        nan_mask = np.any(np.isnan(feature_data), axis=0)  # [H, W] or [T, H, W]
        mask = mask & ~nan_mask

        return mask

    def _apply_mask_ref(
        self,
        sample: Dict[str, Any],
        mask_ref: str,
        current_mask: np.ndarray,
        target_is_temporal: bool,
    ) -> np.ndarray:
        """
        Apply a mask reference to the current mask.

        Handles broadcasting of spatial masks to temporal targets.

        Args:
            sample: Dataset sample dictionary
            mask_ref: Mask reference string (e.g., "static_mask.aoi")
            current_mask: Current combined mask
            target_is_temporal: Whether the target mask is temporal

        Returns:
            Updated mask after AND with the referenced mask
        """
        # Parse mask reference
        parts = mask_ref.split('.')
        if len(parts) != 2:
            logger.warning(f"Invalid mask reference: {mask_ref}")
            return current_mask

        dataset_group, mask_name = parts

        # Get mask data from sample
        if dataset_group not in sample:
            logger.warning(f"Mask group '{dataset_group}' not in sample")
            return current_mask

        group_data = sample[dataset_group]
        channel_names = sample['metadata']['channel_names'].get(dataset_group, [])

        if mask_name not in channel_names:
            logger.warning(
                f"Mask '{mask_name}' not found in group '{dataset_group}'. "
                f"Available: {channel_names}"
            )
            return current_mask

        mask_idx = channel_names.index(mask_name)
        mask_data = group_data[mask_idx]  # [H, W] or [T, H, W]

        # Convert to boolean (mask is valid where value > 0)
        mask_bool = mask_data > 0

        # Handle broadcasting: spatial mask to temporal target
        if mask_bool.ndim == 2 and target_is_temporal:
            # Mask is [H, W], target is [T, H, W] - broadcast across time
            mask_bool = mask_bool[np.newaxis, :, :]  # [1, H, W]
            # Broadcasting will happen automatically in AND operation

        # AND with current mask
        return current_mask & mask_bool

    def _apply_normalization(
        self,
        data: np.ndarray,
        feature_name: str,
        feature_config: FeatureConfig,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply normalization to each channel based on preset configuration.

        Args:
            data: Feature data [C, H, W] or [C, T, H, W]
            feature_name: Name of the feature
            feature_config: Feature configuration
            mask: Validity mask [H, W] or [T, H, W]

        Returns:
            Normalized data with same shape
        """
        normalized_data = data.copy()
        channel_names = list(feature_config.channels.keys())

        for c_idx, channel_ref in enumerate(channel_names):
            channel_config = feature_config.channels[channel_ref]
            norm_preset_name = channel_config.norm

            if not norm_preset_name or norm_preset_name == 'identity':
                continue

            # Get normalization preset
            norm_preset = self.config.get_normalization_preset(norm_preset_name)
            if norm_preset is None:
                logger.warning(
                    f"Normalization preset '{norm_preset_name}' not found for "
                    f"channel '{channel_ref}', skipping normalization"
                )
                continue

            # Get channel stats
            channel_stats = self._get_channel_stats(feature_name, channel_ref)

            # Apply normalization based on type
            if data.ndim == 3:
                # Static: [C, H, W]
                normalized_data[c_idx] = self._normalize_array(
                    data[c_idx], norm_preset, channel_stats
                )
            else:
                # Temporal: [C, T, H, W]
                normalized_data[c_idx] = self._normalize_array(
                    data[c_idx], norm_preset, channel_stats
                )

        return normalized_data

    def _get_channel_stats(
        self,
        feature_name: str,
        channel_ref: str,
    ) -> Dict[str, float]:
        """
        Get statistics for a specific channel from precomputed stats.

        Args:
            feature_name: Name of the feature
            channel_ref: Channel reference (e.g., "static.elevation")

        Returns:
            Dictionary with stats (mean, sd, q25, q50, q75, etc.)
        """
        if not self.stats:
            return {}

        feature_stats = self.stats.get(feature_name, {})
        channel_stats = feature_stats.get(channel_ref, {})

        return channel_stats

    def _normalize_array(
        self,
        data: np.ndarray,
        preset: NormalizationPresetConfig,
        stats: Dict[str, float],
    ) -> np.ndarray:
        """
        Apply normalization to an array.

        Args:
            data: Array to normalize [H, W] or [T, H, W]
            preset: Normalization preset configuration
            stats: Channel statistics

        Returns:
            Normalized array
        """
        normalized = data.copy()

        if preset.type == 'zscore':
            mean = stats.get('mean', 0.0)
            sd = stats.get('sd', 1.0)
            if sd < 1e-8:
                sd = 1.0
            normalized = (data - mean) / sd

        elif preset.type == 'robust_iqr':
            q25 = stats.get('q25', 0.0)
            q50 = stats.get('q50', 0.0)
            q75 = stats.get('q75', 1.0)
            iqr = q75 - q25
            if iqr < 1e-8:
                iqr = 1.0
            normalized = (data - q50) / iqr

        elif preset.type == 'linear_rescale':
            in_min = preset.in_min if preset.in_min is not None else 0.0
            in_max = preset.in_max if preset.in_max is not None else 1.0
            out_min = preset.out_min if preset.out_min is not None else 0.0
            out_max = preset.out_max if preset.out_max is not None else 1.0

            in_range = in_max - in_min
            if in_range < 1e-8:
                in_range = 1.0
            out_range = out_max - out_min

            normalized = ((data - in_min) / in_range) * out_range + out_min

        elif preset.type == 'clamp':
            pass  # Just apply clamping below

        elif preset.type == 'none':
            pass  # No normalization

        # Apply clamping if configured
        if preset.clamp and preset.clamp.get('enabled', False):
            clip_min = preset.clamp.get('min')
            clip_max = preset.clamp.get('max')
            if clip_min is not None or clip_max is not None:
                normalized = np.clip(normalized, clip_min, clip_max)

        return normalized

    def _apply_mahalanobis_transform(
        self,
        data: np.ndarray,
        feature_name: str,
        feature_config: FeatureConfig,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Mahalanobis transform to feature data.

        For features with covariance, the data is:
        1. Centered (subtract mean)
        2. Transformed by the whitening matrix (Cholesky of inverse covariance)

        This transforms the data such that Euclidean distance in the
        transformed space equals Mahalanobis distance in the original space.

        Args:
            data: Feature data [C, H, W] or [C, T, H, W]
            feature_name: Name of the feature
            feature_config: Feature configuration
            mask: Validity mask [H, W] or [T, H, W]

        Returns:
            Transformed data with same shape
        """
        # Get whitening matrix (cached)
        whitening_matrix = self._get_whitening_matrix(feature_name)
        if whitening_matrix is None:
            logger.warning(
                f"No covariance found for feature '{feature_name}', "
                f"skipping Mahalanobis transform"
            )
            return data

        # Get channel means for centering
        channel_means = self._get_channel_means(feature_name, feature_config)

        # Center the data
        centered_data = data.copy()
        for c_idx, mean in enumerate(channel_means):
            centered_data[c_idx] -= mean

        # Apply whitening transform
        # For [C, H, W]: reshape to [C, H*W], apply [C, C] transform, reshape back
        # For [C, T, H, W]: reshape to [C, T*H*W], apply, reshape back

        original_shape = data.shape
        n_channels = original_shape[0]

        # Reshape to [C, N] where N = all other dimensions
        centered_flat = centered_data.reshape(n_channels, -1)

        # Apply whitening: W @ X where W is [C, C] and X is [C, N]
        transformed_flat = whitening_matrix @ centered_flat

        # Reshape back
        transformed_data = transformed_flat.reshape(original_shape)

        return transformed_data

    def _get_whitening_matrix(self, feature_name: str) -> Optional[np.ndarray]:
        """
        Get or compute the whitening matrix for a feature.

        The whitening matrix W satisfies: W @ W.T = Sigma^{-1}
        where Sigma is the covariance matrix.

        This is the Cholesky decomposition of the inverse covariance.

        Args:
            feature_name: Name of the feature

        Returns:
            Whitening matrix [C, C] or None if not available
        """
        cache_key = f"{feature_name}_whitening"

        if cache_key in self._transform_cache:
            return self._transform_cache[cache_key]

        # Get covariance matrix from stats
        if not self.stats or feature_name not in self.stats:
            return None

        feature_stats = self.stats[feature_name]
        covariance = feature_stats.get('covariance')

        if covariance is None:
            return None

        covariance = np.array(covariance)

        # Regularize covariance for numerical stability
        n_channels = covariance.shape[0]
        regularization = 1e-6 * np.eye(n_channels)
        covariance_reg = covariance + regularization

        try:
            # Compute inverse covariance
            cov_inv = np.linalg.inv(covariance_reg)

            # Compute Cholesky decomposition of inverse covariance
            # The whitening matrix W such that W @ W.T = cov_inv
            whitening_matrix = np.linalg.cholesky(cov_inv)

        except np.linalg.LinAlgError as e:
            logger.warning(
                f"Failed to compute whitening matrix for '{feature_name}': {e}. "
                f"Using identity matrix."
            )
            whitening_matrix = np.eye(n_channels)

        self._transform_cache[cache_key] = whitening_matrix
        return whitening_matrix

    def _get_channel_means(
        self,
        feature_name: str,
        feature_config: FeatureConfig,
    ) -> List[float]:
        """
        Get mean values for each channel in a feature.

        Args:
            feature_name: Name of the feature
            feature_config: Feature configuration

        Returns:
            List of mean values, one per channel
        """
        channel_names = list(feature_config.channels.keys())
        means = []

        for channel_ref in channel_names:
            stats = self._get_channel_stats(feature_name, channel_ref)
            mean = stats.get('mean', 0.0)
            means.append(mean)

        return means

    def _apply_mask_to_data(
        self,
        data: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Zero out masked (invalid) values in the data.

        Args:
            data: Feature data [C, H, W] or [C, T, H, W]
            mask: Validity mask [H, W] or [T, H, W] (True = valid)

        Returns:
            Data with invalid values set to 0
        """
        masked_data = data.copy()

        # Broadcast mask to match data shape
        if data.ndim == 3:
            # [C, H, W] data, [H, W] mask
            mask_broadcast = mask[np.newaxis, :, :]  # [1, H, W]
        else:
            # [C, T, H, W] data, [T, H, W] mask
            mask_broadcast = mask[np.newaxis, :, :, :]  # [1, T, H, W]

        # Set invalid values to 0
        masked_data = np.where(mask_broadcast, masked_data, 0.0)

        return masked_data

    def build_all_features(
        self,
        sample: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
        apply_normalization: bool = True,
        apply_mahalanobis: bool = True,
    ) -> Dict[str, FeatureResult]:
        """
        Build multiple features from a sample.

        Args:
            sample: Dataset sample dictionary
            feature_names: List of features to build (None = all features)
            apply_normalization: Whether to apply normalization
            apply_mahalanobis: Whether to apply Mahalanobis transform

        Returns:
            Dictionary mapping feature names to FeatureResults
        """
        if feature_names is None:
            feature_names = list(self.config.features.keys())

        results = {}
        for feature_name in feature_names:
            try:
                result = self.build_feature(
                    feature_name, sample,
                    apply_normalization=apply_normalization,
                    apply_mahalanobis=apply_mahalanobis,
                )
                results[feature_name] = result
            except Exception as e:
                logger.warning(f"Failed to build feature '{feature_name}': {e}")

        return results

    def get_feature_info(self, feature_name: str) -> Dict[str, Any]:
        """
        Get information about a feature configuration.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with feature information
        """
        feature_config = self.config.get_feature(feature_name)
        if feature_config is None:
            raise ValueError(f"Feature '{feature_name}' not found")

        channel_info = []
        for channel_ref, channel_config in feature_config.channels.items():
            channel_info.append({
                'reference': channel_ref,
                'dataset_group': channel_config.dataset_group,
                'channel_name': channel_config.channel_name,
                'mask': channel_config.mask,
                'normalization': channel_config.norm,
            })

        return {
            'name': feature_name,
            'dim': feature_config.dim,
            'n_channels': len(feature_config.channels),
            'channels': channel_info,
            'global_masks': feature_config.masks,
            'has_covariance': (
                feature_config.covariance is not None and
                feature_config.covariance.calculate
            ),
        }


def create_feature_builder_from_config(
    config_path: str,
    stats_path: Optional[str] = None,
) -> FeatureBuilder:
    """
    Convenience function to create a FeatureBuilder from a config file.

    Args:
        config_path: Path to bindings YAML file
        stats_path: Optional path to stats JSON (overrides config)

    Returns:
        Configured FeatureBuilder instance
    """
    from ..config.dataset_bindings_parser import DatasetBindingsParser

    parser = DatasetBindingsParser(config_path)
    config = parser.parse()

    return FeatureBuilder(config, stats_path=stats_path)
