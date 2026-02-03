"""
Parser for dataset bindings YAML configuration.

This module provides a simple parser that loads YAML configuration
and creates typed configuration objects.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union

from .dataset_config import (
    BindingsConfig,
    ZarrConfig,
    TimeWindowConfig,
    DatasetGroupConfig,
    ChannelConfig,
    StatsConfig,
    NormalizationPresetConfig,
    FeatureConfig,
    FeatureChannelConfig,
    CovarianceConfig,
    JitterConfig,
    GridConfig,
    WeightByEntry,
    SupplementConfig,
    SamplingStrategyConfig,
    AuxiliaryDistanceConfig,
    SelectionConfig,
    PairEndpointStrategyConfig,
    TypeSimilarityConfig,
    YsfcOverlapConfig,
    PairStrategyConfig,
    SpectralWeightingConfig,
    PairWeightsConfig,
    CurriculumConfig,
    LossConfig,
)


class BindingsParseError(Exception):
    """Raised when bindings YAML parsing fails."""
    pass


class DatasetBindingsParser:
    """Parser for dataset bindings YAML files.

    This parser focuses on the dataset section of the bindings YAML,
    creating a strongly-typed configuration object.

    Example usage:
        parser = DatasetBindingsParser('config/bindings.yaml')
        config = parser.parse()
        static_group = config.get_group('static')
    """

    def __init__(self, yaml_path: Union[str, Path]):
        """Initialize parser with path to YAML file.

        Args:
            yaml_path: Path to bindings YAML file
        """
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Bindings YAML not found: {self.yaml_path}")

        # Load raw YAML
        with open(self.yaml_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)

    def parse(self) -> BindingsConfig:
        """Parse the YAML file and return structured configuration.

        Returns:
            BindingsConfig object with all parsed configuration

        Raises:
            BindingsParseError: If configuration is invalid
        """
        try:
            # Parse top-level metadata
            version = self._parse_version()
            name = self._parse_name()

            # Parse zarr configuration
            zarr_config = self._parse_zarr()

            # Parse time window
            time_window = self._parse_time_window()

            # Parse dataset groups
            dataset_groups = self._parse_dataset_groups()

            # Parse optional sections
            stats = self._parse_stats() if 'stats' in self.raw_config else None
            normalization_presets = self._parse_normalization() if 'normalization' in self.raw_config else None
            features = self._parse_features() if 'features' in self.raw_config else None
            sampling_strategies = self._parse_sampling_strategies() if 'sampling-strategy' in self.raw_config else None
            losses = self._parse_losses() if 'losses' in self.raw_config else None

            return BindingsConfig(
                version=version,
                name=name,
                zarr=zarr_config,
                time_window=time_window,
                dataset_groups=dataset_groups,
                stats=stats,
                normalization_presets=normalization_presets,
                features=features,
                sampling_strategies=sampling_strategies,
                losses=losses,
            )

        except Exception as e:
            raise BindingsParseError(f"Failed to parse bindings: {e}") from e

    def _parse_version(self) -> str:
        """Parse version string."""
        version = self.raw_config.get('version')
        if not version:
            raise BindingsParseError("Missing required field: 'version'")
        return str(version)

    def _parse_name(self) -> str:
        """Parse dataset name."""
        name = self.raw_config.get('name')
        if not name:
            raise BindingsParseError("Missing required field: 'name'")
        return str(name)

    def _parse_zarr(self) -> ZarrConfig:
        """Parse zarr configuration."""
        zarr_dict = self.raw_config.get('zarr')
        if not zarr_dict:
            raise BindingsParseError("Missing required section: 'zarr'")

        path = zarr_dict.get('path')
        if not path:
            raise BindingsParseError("Missing required field: 'zarr.path'")

        structure = zarr_dict.get('structure', 'hierarchical')

        return ZarrConfig(path=path, structure=structure)

    def _parse_time_window(self) -> TimeWindowConfig:
        """Parse time window configuration."""
        tw_dict = self.raw_config.get('time_window')
        if not tw_dict:
            raise BindingsParseError("Missing required section: 'time_window'")

        start = tw_dict.get('start')
        end = tw_dict.get('end')

        if start is None or end is None:
            raise BindingsParseError(
                "time_window must have 'start' and 'end' fields"
            )

        return TimeWindowConfig(start=int(start), end=int(end))

    def _parse_dataset_groups(self) -> Dict[str, DatasetGroupConfig]:
        """Parse all dataset groups."""
        dataset_dict = self.raw_config.get('dataset')
        if not dataset_dict:
            raise BindingsParseError("Missing required section: 'dataset'")

        groups = {}
        for group_name, group_spec in dataset_dict.items():
            groups[group_name] = self._parse_dataset_group(group_name, group_spec)

        return groups

    def _parse_dataset_group(
        self,
        group_name: str,
        group_spec: Dict[str, Any]
    ) -> DatasetGroupConfig:
        """Parse a single dataset group.

        Args:
            group_name: Name of the group (e.g., 'static_mask')
            group_spec: Dictionary with group specification

        Returns:
            DatasetGroupConfig object
        """
        # Parse dtype
        dtype = group_spec.get('type')
        if not dtype:
            raise BindingsParseError(
                f"Group '{group_name}' missing required field: 'type'"
            )

        # Parse dimensions
        dim = group_spec.get('dim')
        if not dim:
            raise BindingsParseError(
                f"Group '{group_name}' missing required field: 'dim'"
            )
        if not isinstance(dim, list):
            raise BindingsParseError(
                f"Group '{group_name}' field 'dim' must be a list, got {type(dim)}"
            )

        # Parse channels
        channels_list = group_spec.get('channels')
        if not channels_list:
            raise BindingsParseError(
                f"Group '{group_name}' missing required field: 'channels'"
            )
        if not isinstance(channels_list, list):
            raise BindingsParseError(
                f"Group '{group_name}' field 'channels' must be a list, got {type(channels_list)}"
            )

        channels = []
        for i, channel_spec in enumerate(channels_list):
            try:
                channels.append(self._parse_channel(channel_spec))
            except Exception as e:
                raise BindingsParseError(
                    f"Error parsing channel {i} in group '{group_name}': {e}"
                ) from e

        return DatasetGroupConfig(
            name=group_name,
            dtype=dtype,
            dim=dim,
            channels=channels,
        )

    def _parse_channel(self, channel_spec: Dict[str, Any]) -> ChannelConfig:
        """Parse a single channel specification.

        Args:
            channel_spec: Dictionary with channel specification

        Returns:
            ChannelConfig object
        """
        # Channel name is required
        name = channel_spec.get('name')
        if not name:
            raise BindingsParseError(
                f"Channel specification missing required field: 'name'"
            )

        # Extract optional fields
        source = channel_spec.get('source')
        formula = channel_spec.get('formula')
        year = channel_spec.get('year')
        time = channel_spec.get('time')
        ok_if = channel_spec.get('ok_if')
        fill_value = channel_spec.get('fill_value')
        reducer = channel_spec.get('reducer')

        return ChannelConfig(
            name=name,
            source=source,
            formula=formula,
            year=year,
            time=time,
            ok_if=ok_if,
            fill_value=fill_value,
            reducer=reducer,
        )

    def _parse_stats(self) -> StatsConfig:
        """Parse stats configuration."""
        stats_dict = self.raw_config.get('stats')
        if not stats_dict:
            raise BindingsParseError("Missing 'stats' section")

        compute = stats_dict.get('compute', 'if-not-exists')
        type_ = stats_dict.get('type', 'json')
        file = stats_dict.get('file')
        if not file:
            raise BindingsParseError("stats.file is required")

        stats_list = stats_dict.get('stats', [])
        if not isinstance(stats_list, list):
            raise BindingsParseError("stats.stats must be a list")

        covariance = stats_dict.get('covariance', False)
        samples = stats_dict.get('samples', {})
        mask = stats_dict.get('mask', [])

        return StatsConfig(
            compute=compute,
            type=type_,
            file=file,
            stats=stats_list,
            covariance=covariance,
            samples=samples,
            mask=mask,
        )

    def _parse_normalization(self) -> Dict[str, NormalizationPresetConfig]:
        """Parse normalization presets."""
        norm_dict = self.raw_config.get('normalization', {})
        presets_dict = norm_dict.get('presets', {})

        presets = {}
        for preset_name, preset_spec in presets_dict.items():
            presets[preset_name] = NormalizationPresetConfig(
                name=preset_name,
                type=preset_spec.get('type', 'none'),
                stats_source=preset_spec.get('stats_source'),
                fields=preset_spec.get('fields'),
                clamp=preset_spec.get('clamp'),
                in_min=preset_spec.get('in_min'),
                in_max=preset_spec.get('in_max'),
                out_min=preset_spec.get('out_min'),
                out_max=preset_spec.get('out_max'),
            )

        return presets

    def _parse_features(self) -> Dict[str, FeatureConfig]:
        """Parse features configuration."""
        features_dict = self.raw_config.get('features', {})

        features = {}
        for feature_name, feature_spec in features_dict.items():
            features[feature_name] = self._parse_feature(feature_name, feature_spec)

        return features

    def _parse_feature(self, feature_name: str, feature_spec: Dict[str, Any]) -> FeatureConfig:
        """Parse a single feature.

        Args:
            feature_name: Name of the feature
            feature_spec: Dictionary with feature specification

        Returns:
            FeatureConfig object
        """
        # Parse dim
        dim = feature_spec.get('dim')
        if not dim:
            raise BindingsParseError(f"Feature '{feature_name}' missing 'dim'")

        # Parse channels
        channels_spec = feature_spec.get('channels')
        if not channels_spec:
            raise BindingsParseError(f"Feature '{feature_name}' missing 'channels'")

        # Channels can be either a dict or a list
        channels = {}
        if isinstance(channels_spec, dict):
            # Dict format: {channel_ref: {mask: ..., norm: ...}}
            for channel_ref, channel_config in channels_spec.items():
                channels[channel_ref] = self._parse_feature_channel(channel_ref, channel_config)
        elif isinstance(channels_spec, list):
            # List format: [{dataset.channel: {mask: ..., norm: ...}}]
            for item in channels_spec:
                if not isinstance(item, dict):
                    raise BindingsParseError(
                        f"Feature '{feature_name}' channel must be a dict, got {type(item)}"
                    )
                # Each item should have exactly one key
                if len(item) != 1:
                    raise BindingsParseError(
                        f"Feature '{feature_name}' channel dict must have exactly one key"
                    )
                channel_ref = list(item.keys())[0]
                channel_config = item[channel_ref]
                channels[channel_ref] = self._parse_feature_channel(channel_ref, channel_config)
        else:
            raise BindingsParseError(
                f"Feature '{feature_name}' channels must be dict or list, got {type(channels_spec)}"
            )

        # Parse optional masks
        masks = feature_spec.get('masks')

        # Parse optional covariance
        covariance = None
        if 'covariance' in feature_spec:
            cov_spec = feature_spec['covariance']
            covariance = CovarianceConfig(
                dim=cov_spec.get('dim', ['C', 'C']),
                calculate=cov_spec.get('calculate', False),
                stat_domain=cov_spec.get('stat_domain', 'patch'),
            )

        return FeatureConfig(
            name=feature_name,
            dim=dim,
            channels=channels,
            masks=masks,
            covariance=covariance,
        )

    def _parse_feature_channel(
        self,
        channel_ref: str,
        channel_config: Dict[str, Any]
    ) -> FeatureChannelConfig:
        """Parse a single feature channel reference.

        Args:
            channel_ref: Channel reference like 'static.elevation' or 'annual.evi2_summer_p95'
            channel_config: Configuration dict with mask, quality, norm

        Returns:
            FeatureChannelConfig object
        """
        # Parse channel reference (format: dataset_group.channel_name)
        parts = channel_ref.split('.')
        if len(parts) != 2:
            raise BindingsParseError(
                f"Channel reference must be 'dataset_group.channel_name', got '{channel_ref}'"
            )
        dataset_group, channel_name = parts

        # Parse optional fields
        mask = channel_config.get('mask')
        quality = channel_config.get('quality')
        norm = channel_config.get('norm')

        return FeatureChannelConfig(
            dataset_group=dataset_group,
            channel_name=channel_name,
            mask=mask,
            quality=quality,
            norm=norm,
        )

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def _parse_sampling_strategies(self) -> Dict[str, SamplingStrategyConfig]:
        """Parse all sampling strategies."""
        strategies_dict = self.raw_config.get('sampling-strategy', {})
        strategies = {}
        for name, spec in strategies_dict.items():
            strategies[name] = self._parse_sampling_strategy(name, spec)
        return strategies

    def _parse_sampling_strategy(
        self, name: str, spec: Dict[str, Any]
    ) -> SamplingStrategyConfig:
        """Parse a single sampling strategy."""
        # Patch-only strategy (has interior-only / border_width at top level)
        interior_only = spec.get('interior-only')
        border_width = spec.get('border_width')

        # Grid config — either at top level (grid-only) or nested under 'grid'
        grid = None
        grid_spec = spec.get('grid') if 'grid' in spec else None
        if grid_spec is None and 'stride' in spec:
            # Grid-only: top-level keys
            grid_spec = spec
        if grid_spec is not None and isinstance(grid_spec, dict):
            jitter = None
            if 'jitter' in grid_spec:
                jitter = JitterConfig(radius=grid_spec['jitter'].get('radius', 0))
            grid = GridConfig(
                stride=grid_spec.get('stride', 16),
                exclude_border=grid_spec.get('exclude_border', 16),
                jitter=jitter,
            )

        # Supplement config
        supplement = None
        if 'supplement' in spec:
            supplement = self._parse_supplement(spec['supplement'])

        return SamplingStrategyConfig(
            name=name,
            interior_only=interior_only,
            border_width=border_width,
            grid=grid,
            supplement=supplement,
        )

    def _parse_supplement(self, spec: Dict[str, Any]) -> SupplementConfig:
        """Parse supplement sampling configuration."""
        n = spec.get('n', 0)
        sampling = spec.get('sampling', {})
        sampling_type = sampling.get('type', 'weighted')

        weight_by = []
        for entry in sampling.get('weight_by', []):
            if isinstance(entry, str):
                # Simple mask reference
                weight_by.append(WeightByEntry(mask_ref=entry))
            elif isinstance(entry, dict):
                # Channel-based weight with optional transform
                weight_by.append(WeightByEntry(
                    channel=entry.get('channel'),
                    transform=entry.get('transform'),
                ))
            else:
                raise BindingsParseError(
                    f"Invalid weight_by entry: {entry}"
                )

        return SupplementConfig(
            n=n,
            sampling_type=sampling_type,
            weight_by=weight_by,
        )

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _parse_losses(self) -> Dict[str, LossConfig]:
        """Parse all loss configurations."""
        losses_dict = self.raw_config.get('losses', {})
        losses = {}
        for name, spec in losses_dict.items():
            losses[name] = self._parse_loss(name, spec)
        return losses

    def _parse_loss(self, name: str, spec: Dict[str, Any]) -> LossConfig:
        """Parse a single loss configuration."""
        loss_type = spec.get('type', 'infonce')

        # Common fields
        weight = spec.get('weight', 1.0)
        mask = spec.get('mask')

        # Anchor population
        anchor_spec = spec.get('anchor', {})
        anchor_population = anchor_spec.get('population') if anchor_spec else None

        # InfoNCE-specific
        auxiliary_distance = None
        if 'auxiliary_distance' in spec:
            ad = spec['auxiliary_distance']
            auxiliary_distance = AuxiliaryDistanceConfig(
                feature=ad['feature'],
                metric=ad.get('metric', 'l2'),
                covariance=ad.get('covariance', False),
            )

        positive_strategy = None
        if 'positive_strategy' in spec:
            positive_strategy = self._parse_pair_endpoint_strategy(
                spec['positive_strategy']
            )

        negative_strategy = None
        if 'negative_strategy' in spec:
            negative_strategy = self._parse_pair_endpoint_strategy(
                spec['negative_strategy']
            )

        # Spectral weighting (for spatial infonce)
        spectral_weighting = None
        if 'spectral_weighting' in spec:
            sw = spec['spectral_weighting']
            spectral_weighting = SpectralWeightingConfig(
                feature=sw['feature'],
                tau=sw.get('tau', 1.0),
                min_weight=sw.get('min_weight', 0.01),
            )

        # Soft neighborhood-specific
        neighborhood_target = spec.get('neighborhood_target')

        pair_strategy = None
        if 'pair_strategy' in spec:
            pair_strategy = self._parse_pair_strategy(spec['pair_strategy'])

        pair_weights = None
        if 'pair_weights' in spec:
            pw = spec['pair_weights']
            pair_weights = PairWeightsConfig(
                source=pw['source'],
                sigma=pw.get('sigma', 5.0),
                self_pair_weight=pw.get('self_pair_weight', 1.0),
            )

        curriculum = None
        if 'curriculum' in spec:
            cur = spec['curriculum']
            curriculum = CurriculumConfig(
                start_epoch=cur.get('start_epoch', 0),
                ramp_epochs=cur.get('ramp_epochs', 0),
            )

        return LossConfig(
            name=name,
            weight=weight,
            type=loss_type,
            mask=mask,
            temperature=spec.get('temperature'),
            auxiliary_distance=auxiliary_distance,
            anchor_population=anchor_population,
            positive_strategy=positive_strategy,
            negative_strategy=negative_strategy,
            spectral_weighting=spectral_weighting,
            neighborhood_target=neighborhood_target,
            pair_strategy=pair_strategy,
            pair_weights=pair_weights,
            tau_ref=spec.get('tau_ref'),
            tau_learned=spec.get('tau_learned'),
            min_valid_per_row=spec.get('min_valid_per_row'),
            self_similarity_weight=spec.get('self_similarity_weight'),
            cross_pixel_weight=spec.get('cross_pixel_weight'),
            curriculum=curriculum,
        )

    def _parse_pair_endpoint_strategy(
        self, spec: Dict[str, Any]
    ) -> PairEndpointStrategyConfig:
        """Parse a positive or negative pair endpoint strategy."""
        selection = None
        if 'selection' in spec:
            sel = spec['selection']
            selection = SelectionConfig(
                type=sel['type'],
                k=sel.get('k'),
                min_distance=sel.get('min_distance'),
                max_distance=sel.get('max_distance'),
                range=sel.get('range'),
                n_per_anchor=sel.get('n_per_anchor'),
            )

        return PairEndpointStrategyConfig(
            candidates=spec.get('candidates', 'anchors'),
            distance=spec.get('distance'),
            selection=selection,
        )

    def _parse_pair_strategy(self, spec: Dict[str, Any]) -> PairStrategyConfig:
        """Parse pair strategy for soft neighborhood losses."""
        type_similarity = None
        if 'type_similarity' in spec:
            ts = spec['type_similarity']
            type_similarity = TypeSimilarityConfig(
                feature=ts['feature'],
                k=ts.get('k', 16),
            )

        ysfc_overlap = None
        if 'ysfc_overlap' in spec:
            yo = spec['ysfc_overlap']
            ysfc_overlap = YsfcOverlapConfig(
                channel=yo['channel'],
                min_overlap=yo.get('min_overlap', 3),
            )

        return PairStrategyConfig(
            type=spec.get('type', 'knn-with-ysfc-overlap'),
            include_self=spec.get('include_self', True),
            type_similarity=type_similarity,
            ysfc_overlap=ysfc_overlap,
            min_pairs=spec.get('min_pairs', 5),
        )
