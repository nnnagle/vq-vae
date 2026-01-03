"""
Forest Dataset - PyTorch Dataset for Multi-Window Forest Training

Combines ForestPatchSampler, DataReader, MaskBuilder, and BundleBuilder to
provide a complete PyTorch Dataset for training the forest representation model.

Returns TrainingBundle objects containing all data for 3 temporal windows plus
static data. Includes a custom collate function for batching complex nested
structures.

Usage:
    dataset = ForestDataset(bindings_config, training_config, split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=forest_collate_fn,
        num_workers=4
    )
    
    for epoch in range(num_epochs):
        dataset.on_epoch_start()  # Regenerate sample indices
        for batch in dataloader:
            # batch is a dict with batched tensors
            ...
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from data.loaders.config import BindingsParser, BindingsRegistry
from data.loaders import DataReader, MaskBuilder
from data.loaders.builders import BundleBuilder, TrainingBundle
from data.stats import DerivedStatsLoader
from data.normalization import ZScoreNormalizer, NormalizationConfig
from .forest_patch_sampler import ForestPatchSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForestDataset(Dataset):
    """
    PyTorch Dataset for forest representation learning.

    Generates training samples as (spatial_window, anchor_year) pairs using
    ForestPatchSampler, then builds complete TrainingBundle objects containing
    data for all 3 temporal windows plus static data.

    Optionally normalizes derived features using pre-computed statistics from Zarr.

    Attributes:
        sampler: ForestPatchSampler for generating sample specifications
        reader: DataReader for loading raw Zarr data
        masker: MaskBuilder for loading masks and quality weights
        builder: BundleBuilder for constructing TrainingBundle objects
        stats_loader: Optional DerivedStatsLoader for normalizing derived features
        normalize_derived: Whether to normalize derived features
    """

    def __init__(
        self,
        bindings_config: Dict[str, Any],
        training_config: Dict[str, Any],
        split: Optional[str] = None,
        normalize_derived: bool = False
    ):
        """
        Initialize ForestDataset.

        Args:
            bindings_config: Parsed bindings configuration
            training_config: Parsed training configuration
            split: Optional split ('train', 'val', 'test', or None for all)
            normalize_derived: If True, normalize derived features using
                pre-computed statistics from Zarr (default: False)

        Example:
            >>> # Parse configs
            >>> bindings_parser = BindingsParser('config/frl_bindings_v0.yaml')
            >>> bindings_config = bindings_parser.parse()
            >>>
            >>> import yaml
            >>> with open('config/frl_training_v1.yaml') as f:
            ...     training_config = yaml.safe_load(f)
            >>>
            >>> # Create dataset with normalization
            >>> dataset = ForestDataset(
            ...     bindings_config,
            ...     training_config,
            ...     split='train',
            ...     normalize_derived=True
            ... )
            >>>
            >>> # Use with DataLoader
            >>> from torch.utils.data import DataLoader
            >>> dataloader = DataLoader(
            ...     dataset,
            ...     batch_size=4,
            ...     collate_fn=forest_collate_fn,
            ...     num_workers=4
            ... )
        """
        self.bindings_config = bindings_config
        self.training_config = training_config
        self.split = split
        self.normalize_derived = normalize_derived

        logger.info(f"Initializing ForestDataset with split='{split}', normalize_derived={normalize_derived}")
        
        # Initialize sampler
        logger.info("Creating ForestPatchSampler...")
        self.sampler = ForestPatchSampler(bindings_config, training_config, split)
        
        # Initialize data reader
        logger.info("Creating DataReader...")
        self.reader = DataReader(bindings_config)
        
        # Initialize mask builder
        logger.info("Creating MaskBuilder...")
        registry = BindingsRegistry(bindings_config)
        self.masker = MaskBuilder(registry, bindings_config)
        
        # Initialize bundle builder
        logger.info("Creating BundleBuilder...")
        self.builder = BundleBuilder(bindings_config, self.reader, self.masker)

        # Initialize stats loader if normalization is enabled
        self.stats_loader = None
        if self.normalize_derived:
            logger.info("Creating DerivedStatsLoader for normalization...")
            try:
                self.stats_loader = DerivedStatsLoader(bindings_config)
                logger.info("✓ DerivedStatsLoader initialized successfully")
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Could not initialize DerivedStatsLoader: {e}\n"
                    f"Derived features will NOT be normalized. "
                    f"Run DerivedStatsComputer.compute_and_save() first."
                )
                self.normalize_derived = False

        logger.info(
            f"ForestDataset initialized: {len(self)} samples for split='{split}'"
        )
    
    def __len__(self) -> int:
        """
        Return number of samples for current epoch.
        
        Returns:
            Number of samples (depends on epoch mode: full/frac/number)
        """
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> TrainingBundle:
        """
        Get a training sample.
        
        Args:
            idx: Sample index (0 to len(self)-1)
        
        Returns:
            TrainingBundle containing:
                - windows: Dict[str, WindowData] for t0, t2, t4
                - static: Dict[str, GroupReadResult] for static groups
                - static_masks/quality: Masks and quality for static
                - anchor_id: Which window is the anchor
                - spatial_window: SpatialWindow for this sample
                - metadata: Additional info
        
        Example:
            >>> bundle = dataset[0]
            >>> 
            >>> # Access data
            >>> ls8_t0 = bundle.windows['t0'].temporal['ls8day'].data  # [7, 10, 256, 256]
            >>> ls8_mask = bundle.windows['t0'].temporal_masks['ls8day'].data
            >>> 
            >>> # Access static
            >>> topo = bundle.static['topo'].data  # [8, 256, 256]
            >>> 
            >>> # Check anchor
            >>> print(bundle.anchor_id)  # 't0', 't2', or 't4'
        """
        # Get sample specification from sampler
        spatial_window, anchor_year = self.sampler[idx]

        # Build complete bundle
        try:
            bundle = self.builder.build_bundle(spatial_window, anchor_year)
        except Exception as e:
            logger.error(
                f"Error building bundle for idx={idx}, "
                f"spatial_window={spatial_window}, anchor_year={anchor_year}: {e}"
            )
            raise

        # Normalize derived features if enabled
        if self.normalize_derived:
            bundle = self._normalize_derived_features(bundle)

        return bundle

    def _normalize_derived_features(self, bundle: TrainingBundle) -> TrainingBundle:
        """
        Normalize derived features in a TrainingBundle using pre-computed statistics.

        Uses the existing normalization framework (ZScoreNormalizer) with statistics
        loaded from the derived_stats group in Zarr via DerivedStatsLoader.

        Args:
            bundle: TrainingBundle with raw derived features

        Returns:
            TrainingBundle with normalized derived features

        Note:
            - Modifies the bundle in-place for efficiency
            - Features without available statistics are left unchanged
            - Uses existing ZScoreNormalizer for consistent behavior
        """
        if not self.normalize_derived or self.stats_loader is None:
            return bundle

        # Get normalization config from YAML
        norm_presets = self.bindings_config.get('normalization', {}).get('presets', {})
        zscore_preset = norm_presets.get('zscore', {})

        # Create normalization config using existing framework
        norm_config = NormalizationConfig(
            type='zscore',
            stats_source='zarr',  # Stats come from Zarr (via DerivedStatsLoader)
            fields=zscore_preset.get('fields', {'mean': 'mean', 'std': 'sd'}),
            clamp=zscore_preset.get('clamp', {'enabled': True, 'min': -6.0, 'max': 6.0}),
            missing=zscore_preset.get('missing', {'fill': 0.0})
        )

        # Normalize derived features in each window
        for window_label, window_data in bundle.windows.items():
            if not window_data.derived:
                continue

            for feature_name, feature_data in window_data.derived.items():
                try:
                    # Get statistics for this feature from DerivedStatsLoader
                    stats = self.stats_loader.get_feature_stats(feature_name)

                    # Validate channel dimensions
                    if len(stats['mean']) != feature_data.shape[0]:
                        logger.warning(
                            f"Feature '{feature_name}' has {feature_data.shape[0]} channels "
                            f"but stats have {len(stats['mean'])} channels. Skipping normalization."
                        )
                        continue

                    # Prepare stats for broadcasting
                    # ZScoreNormalizer expects stats to broadcast to data shape
                    shape = feature_data.shape
                    n_channels = shape[0]
                    broadcast_shape = [n_channels] + [1] * (len(shape) - 1)

                    broadcast_stats = {
                        'mean': stats['mean'].reshape(broadcast_shape),
                        'sd': stats['sd'].reshape(broadcast_shape)
                    }

                    # Create normalizer using existing framework
                    normalizer = ZScoreNormalizer(norm_config, broadcast_stats)

                    # Apply normalization
                    normalized = normalizer.normalize(feature_data, mask=None)

                    # Update in place
                    window_data.derived[feature_name] = normalized

                    logger.debug(
                        f"Normalized {window_label}.{feature_name} using ZScoreNormalizer: "
                        f"shape={feature_data.shape}"
                    )

                except KeyError as e:
                    # Stats not available for this feature - skip normalization
                    logger.debug(
                        f"No statistics available for '{feature_name}', "
                        f"leaving unnormalized: {e}"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Error normalizing feature '{feature_name}' "
                        f"in window '{window_label}': {e}"
                    )
                    # Leave feature unnormalized on error
                    continue

        return bundle

    def on_epoch_start(self):
        """
        Call at the start of each epoch.
        
        Regenerates sample indices in the sampler. This ensures:
        - 'full' mode: patches are reshuffled each epoch
        - 'frac'/'number' modes: new random samples are drawn each epoch
        
        Example:
            >>> for epoch in range(num_epochs):
            ...     dataset.on_epoch_start()  # Regenerate samples
            ...     for batch in dataloader:
            ...         train_step(batch)
        """
        logger.debug(f"Starting new epoch for split='{self.split}'")
        self.sampler.new_epoch()


def forest_collate_fn(batch: List[TrainingBundle]) -> Dict[str, Any]:
    """
    Collate function for batching TrainingBundle objects.
    
    Converts a list of TrainingBundle objects into a single batched dictionary
    with all arrays stacked along batch dimension.
    
    Args:
        batch: List of TrainingBundle objects from Dataset.__getitem__
    
    Returns:
        Dictionary with batched structure:
        {
            'windows': {
                't0': {
                    'temporal': {'ls8day': Tensor[B,C,T,H,W], ...},
                    'snapshot': {'ccdc_snapshot': Tensor[B,C,H,W], ...},
                    'irregular': {'naip': Tensor[B,C,T_obs,H,W], ...},
                    'derived': {'temporal_position': Tensor[B,C,T,H,W], ...},
                    'temporal_masks': {'ls8day': Tensor[B,C,T,H,W], ...},
                    'snapshot_masks': {...},
                    'irregular_masks': {...},
                    'temporal_quality': {...},
                    'snapshot_quality': {...},
                    'irregular_quality': {...},
                },
                't2': {...},
                't4': {...}
            },
            'static': {'topo': Tensor[B,C,H,W], ...},
            'static_masks': {'topo': Tensor[B,C,H,W], ...},
            'static_quality': {'topo': Tensor[B,C,H,W], ...},
            'anchor_ids': ['t0', 't2', 't0', ...],  # List[str], length B
            'spatial_windows': [...],  # List[SpatialWindow], length B
            'metadata': [...]  # List[Dict], length B
        }
    
    Example:
        >>> dataloader = DataLoader(dataset, batch_size=4, collate_fn=forest_collate_fn)
        >>> batch = next(iter(dataloader))
        >>> 
        >>> # Access batched data
        >>> ls8_batch = batch['windows']['t0']['temporal']['ls8day']
        >>> print(ls8_batch.shape)  # [4, 7, 10, 256, 256]
        >>> 
        >>> # Get anchor for first sample in batch
        >>> print(batch['anchor_ids'][0])  # 't0'
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")
    
    batch_size = len(batch)
    
    # Collect window labels from first bundle (should be same for all)
    window_labels = list(batch[0].windows.keys())  # ['t0', 't2', 't4']
    
    # Initialize batched structure
    batched = {
        'windows': {label: {} for label in window_labels},
        'static': {},
        'static_masks': {},
        'static_quality': {},
        'anchor_ids': [],
        'spatial_windows': [],
        'metadata': []
    }
    
    # Collate each window
    for window_label in window_labels:
        window_batch = batched['windows'][window_label]
        
        # Get all WindowData objects for this window
        window_data_list = [bundle.windows[window_label] for bundle in batch]
        
        # Collate temporal groups
        window_batch['temporal'] = _collate_group_dicts(
            [wd.temporal for wd in window_data_list]
        )
        
        # Collate snapshot groups
        window_batch['snapshot'] = _collate_group_dicts(
            [wd.snapshot for wd in window_data_list]
        )
        
        # Collate irregular groups
        window_batch['irregular'] = _collate_group_dicts(
            [wd.irregular for wd in window_data_list]
        )
        
        # Collate derived features
        window_batch['derived'] = _collate_derived_dicts(
            [wd.derived for wd in window_data_list]
        )
        
        # Collate masks
        window_batch['temporal_masks'] = _collate_mask_dicts(
            [wd.temporal_masks for wd in window_data_list]
        )
        window_batch['snapshot_masks'] = _collate_mask_dicts(
            [wd.snapshot_masks for wd in window_data_list]
        )
        window_batch['irregular_masks'] = _collate_mask_dicts(
            [wd.irregular_masks for wd in window_data_list]
        )
        
        # Collate quality
        window_batch['temporal_quality'] = _collate_quality_dicts(
            [wd.temporal_quality for wd in window_data_list]
        )
        window_batch['snapshot_quality'] = _collate_quality_dicts(
            [wd.snapshot_quality for wd in window_data_list]
        )
        window_batch['irregular_quality'] = _collate_quality_dicts(
            [wd.irregular_quality for wd in window_data_list]
        )
    
    # Collate static groups
    batched['static'] = _collate_group_dicts([bundle.static for bundle in batch])
    batched['static_masks'] = _collate_mask_dicts([bundle.static_masks for bundle in batch])
    batched['static_quality'] = _collate_quality_dicts([bundle.static_quality for bundle in batch])
    
    # Collect metadata (not stacked, just lists)
    batched['anchor_ids'] = [bundle.anchor_id for bundle in batch]
    batched['spatial_windows'] = [bundle.spatial_window for bundle in batch]
    batched['metadata'] = [bundle.metadata for bundle in batch]
    
    return batched


def _collate_group_dicts(group_dicts: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate list of group result dicts into batched tensors.
    
    Args:
        group_dicts: List of Dict[group_name, GroupReadResult]
    
    Returns:
        Dict[group_name, Tensor] with batch dimension added
    """
    if not group_dicts or not group_dicts[0]:
        return {}
    
    # Get group names from first dict
    group_names = list(group_dicts[0].keys())
    
    batched = {}
    for group_name in group_names:
        # Stack data arrays for this group across batch
        arrays = [gd[group_name].data for gd in group_dicts]
        batched[group_name] = torch.from_numpy(np.stack(arrays, axis=0))
    
    return batched


def _collate_mask_dicts(mask_dicts: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate list of mask result dicts into batched tensors.
    
    Args:
        mask_dicts: List of Dict[group_name, MaskResult]
    
    Returns:
        Dict[group_name, Tensor] with batch dimension added
    """
    if not mask_dicts or not mask_dicts[0]:
        return {}
    
    group_names = list(mask_dicts[0].keys())
    
    batched = {}
    for group_name in group_names:
        arrays = [md[group_name].data for md in mask_dicts]
        batched[group_name] = torch.from_numpy(np.stack(arrays, axis=0))
    
    return batched


def _collate_quality_dicts(quality_dicts: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate list of quality result dicts into batched tensors.
    
    Args:
        quality_dicts: List of Dict[group_name, QualityResult]
    
    Returns:
        Dict[group_name, Tensor] with batch dimension added
    """
    if not quality_dicts or not quality_dicts[0]:
        return {}
    
    group_names = list(quality_dicts[0].keys())
    
    batched = {}
    for group_name in group_names:
        arrays = [qd[group_name].data for qd in quality_dicts]
        batched[group_name] = torch.from_numpy(np.stack(arrays, axis=0))
    
    return batched


def _collate_derived_dicts(derived_dicts: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate list of derived feature dicts into batched tensors.
    
    Args:
        derived_dicts: List of Dict[feature_name, np.ndarray]
    
    Returns:
        Dict[feature_name, Tensor] with batch dimension added
    """
    if not derived_dicts or not derived_dicts[0]:
        return {}
    
    feature_names = list(derived_dicts[0].keys())
    
    batched = {}
    for feature_name in feature_names:
        arrays = [dd[feature_name] for dd in derived_dicts]
        batched[feature_name] = torch.from_numpy(np.stack(arrays, axis=0))
    
    return batched
