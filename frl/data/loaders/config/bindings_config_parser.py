"""
YAML Bindings Parser for Forest Representation Model
Provides:

@dataclass class BandConfig

@dataclass class InputGroup

class BindingsParser

Parses and validates the comprehensive YAML bindings configuration file that defines:
- Normalization presets
- Shared masks, quality metrics, and derived quantities
- Input specifications (temporal, irregular, static)
- Derived features
- Model input mappings
- Training configuration

Key Features:
- Reference validation (e.g., shared.masks.forest)
- Normalization preset resolution
- Window binding template expansion
- Circular dependency detection
- Comprehensive error messages
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BindingsError(Exception):
    """Base exception for configuration errors"""
    pass


class ReferenceError(BindingsError):
    """Exception for invalid references"""
    pass


class ValidationError(BindingsError):
    """Exception for validation failures"""
    pass


@dataclass
class ZarrReference:
    """Represents a reference to a Zarr group/array"""
    group: str
    array: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'ZarrReference':
        return cls(group=d['group'], array=d.get('array'))
    
    def __str__(self):
        if self.array:
            return f"{self.group}:{self.array}"
        return self.group


@dataclass
class BandConfig:
    """Configuration for a single band/channel"""
    name: str
    array: Optional[str] = None
    norm: Optional[str] = None
    mask: List[str] = field(default_factory=list)
    quality_weight: List[str] = field(default_factory=list)
    loss_weight: Optional[str] = None
    num_classes: Optional[int] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BandConfig':
        return cls(
            name=d['name'],
            array=d.get('array'),
            norm=d.get('norm'),
            mask=d.get('mask', []),
            quality_weight=d.get('quality_weight', []),
            loss_weight=d.get('loss_weight'),
            num_classes=d.get('num_classes')
        )


@dataclass
class InputGroup:
    """Configuration for a group of input bands"""
    name: str
    zarr: ZarrReference
    kind: str
    bands: List[BandConfig]
    time_window_years: Optional[int] = None
    years: Optional[List[int]] = None
    window_binding: Optional[Dict[str, str]] = None
    bands_template: Optional[List[Dict[str, Any]]] = None
    quality: Optional[List[Dict[str, Any]]] = None
    missing_policy: Optional[Dict[str, Any]] = None
    
    def get_bands_for_year(self, year: int) -> List[BandConfig]:
        """
        Get bands for a specific year (snapshot inputs only).
        
        Args:
            year: Year to get bands for
            
        Returns:
            List of BandConfig objects for the specified year
            
        Raises:
            ValueError: If not a snapshot input or year not available
            
        Example:
            >>> snapshot = config['inputs']['snapshot']['ccdc_snapshot']
            >>> bands_2024 = snapshot.get_bands_for_year(2024)
        """
        if not self.years:
            raise ValueError(
                f"get_bands_for_year() only works on snapshot inputs. "
                f"Input '{self.name}' has no years."
            )
        
        if year not in self.years:
            raise ValueError(
                f"Year {year} not available for snapshot '{self.name}'. "
                f"Available years: {self.years}"
            )
        
        # Filter bands for this year
        year_suffix = f"_{year}"
        return [b for b in self.bands if b.name.endswith(year_suffix)]
    
    def get_zarr_prefix(self, year: int) -> str:
        """
        Get zarr prefix for a specific year (snapshot inputs only).
        
        Args:
            year: Year to get prefix for
            
        Returns:
            Instantiated zarr prefix (e.g., 'snap_2024_0831')
            
        Raises:
            ValueError: If not a snapshot input or year not available
            AttributeError: If zarr_pattern not set
            
        Example:
            >>> snapshot = config['inputs']['snapshot']['ccdc_snapshot']
            >>> prefix = snapshot.get_zarr_prefix(2024)
            >>> # Returns: 'snap_2024_0831'
        """
        if not self.years:
            raise ValueError(
                f"get_zarr_prefix() only works on snapshot inputs. "
                f"Input '{self.name}' has no years."
            )
        
        if year not in self.years:
            raise ValueError(
                f"Year {year} not available for snapshot '{self.name}'. "
                f"Available years: {self.years}"
            )
        
        if not hasattr(self, 'zarr_pattern'):
            raise AttributeError(
                f"Snapshot '{self.name}' missing zarr_pattern attribute"
            )
        
        return self.zarr_pattern.format(year=year)
    
    @property
    def is_snapshot(self) -> bool:
        """Check if this is a snapshot input."""
        return self.years is not None and len(self.years) > 0


class BindingsParser:
    """
    Main parser for forest representation model data bindings YAML.
    
    Validates and resolves all references, ensuring the configuration is
    internally consistent and ready for use by the dataloader.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize parser and load configuration.
        
        Args:
            config_path: Path to YAML bindings configuration file
        """
        self.config_path = Path(config_path)
        self.raw_config = self._load_yaml()
        self.parsed = {}
        
        # Track references for validation
        self.defined_references: Set[str] = set()
        self.used_references: Set[str] = set()
        self.reference_locations: Dict[str, List[str]] = defaultdict(list)
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Load and parse YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise BindingsError(f"Config file not found: {self.config_path}")
        except Exception as e: # Catches all YAML errors
            raise BindingsError(f"Failed to parse YAML: {e}")
    
    def parse(self) -> Dict[str, Any]:
        """
        Parse and validate the entire configuration.
        
        Returns:
            Validated and enriched bindings configuration dictionary
            
        Raises:
            BindingsError: If configuration is invalid
        """
        logger.info(f"Parsing configuration from {self.config_path}")
        
        # Parse sections in dependency order
        self._parse_metadata()
        self._parse_zarr_config()
        self._parse_normalization_presets()
        self._parse_shared_section()
        self._parse_inputs()
        self._parse_derived_features()
        self._parse_model_inputs()
        self._parse_training_config()
        
        # Validate references
        self._validate_all_references()
        
        # Check for circular dependencies
        self._check_circular_dependencies()
        
        logger.info("Configuration parsed successfully")
        return self.parsed
    
    def _parse_metadata(self):
        """Parse version and name metadata"""
        self.parsed['version'] = self.raw_config.get('version', '1.0')
        self.parsed['name'] = self.raw_config.get('name', 'unnamed')
        
    def _parse_zarr_config(self):
        """Parse Zarr dataset configuration"""
        zarr_config = self.raw_config.get('zarr', {})
        self.parsed['zarr'] = {
            'path': zarr_config.get('path'),
            'structure': zarr_config.get('structure', 'hierarchical')
        }
        
        if not self.parsed['zarr']['path']:
            raise BindingsError("Missing required zarr.path")
    
    def _parse_normalization_presets(self):
        """Parse normalization preset definitions"""
        norm_config = self.raw_config.get('normalization', {})
        presets = norm_config.get('presets', {})
        
        self.parsed['normalization'] = {'presets': {}}
        
        for preset_name, preset_config in presets.items():
            # Validate preset configuration
            self._validate_normalization_preset(preset_name, preset_config)
            self.parsed['normalization']['presets'][preset_name] = preset_config
            
            # Track as defined reference
            self.defined_references.add(f"normalization.presets.{preset_name}")
        
        logger.info(f"Parsed {len(presets)} normalization presets")
    
    def _validate_normalization_preset(self, name: str, config: Dict[str, Any]):
        """Validate a single normalization preset"""
        norm_type = config.get('type')
        if not norm_type:
            raise ValidationError(f"Normalization preset '{name}' missing 'type'")
        
        valid_types = ['zscore', 'robust_iqr', 'minmax', 'linear_rescale', 'clamp', 'none']
        if norm_type not in valid_types:
            raise ValidationError(
                f"Invalid normalization type '{norm_type}' in preset '{name}'. "
                f"Valid types: {valid_types}"
            )
        
        # Type-specific validation
        if norm_type == 'zscore':
            if config.get('stats_source') == 'zarr':
                fields = config.get('fields', {})
                if 'mean' not in fields or 'std' not in fields:
                    raise ValidationError(
                        f"Z-score preset '{name}' requires 'mean' and 'std' fields"
                    )
        
        elif norm_type == 'robust_iqr':
            if config.get('stats_source') == 'zarr':
                fields = config.get('fields', {})
                required = ['q25', 'q50', 'q75']
                missing = [f for f in required if f not in fields]
                if missing:
                    raise ValidationError(
                        f"Robust IQR preset '{name}' missing fields: {missing}"
                    )
        
        elif norm_type == 'minmax':
            stats_source = config.get('stats_source', 'zarr')
            if stats_source == 'fixed':
                if 'min' not in config or 'max' not in config:
                    raise ValidationError(
                        f"Min-max preset '{name}' with fixed stats requires 'min' and 'max'"
                    )
        
        elif norm_type == 'linear_rescale':
            required = ['in_min', 'in_max', 'out_min', 'out_max']
            missing = [f for f in required if f not in config]
            if missing:
                raise ValidationError(
                    f"Linear rescale preset '{name}' missing: {missing}"
                )
    
    def _parse_shared_section(self):
        """Parse shared masks, quality metrics, and derived quantities"""
        shared = self.raw_config.get('shared', {})
        
        self.parsed['shared'] = {
            'masks': {},
            'quality': {}
        }
        
        # Parse masks
        for mask_name, mask_config in shared.get('masks', {}).items():
            self._parse_shared_mask(mask_name, mask_config)
        
        # Parse quality metrics
        for quality_name, quality_config in shared.get('quality', {}).items():
            self._parse_shared_quality(quality_name, quality_config)
        
        logger.info(
            f"Parsed {len(self.parsed['shared']['masks'])} masks, "
            f"{len(self.parsed['shared']['quality'])} quality metrics"
        )
    
    def _parse_shared_mask(self, name: str, config: Dict[str, Any]):
        """Parse a single shared mask definition"""
        mask_type = config.get('type', 'boolean')
        
        # Resolve Zarr reference if present
        if 'zarr' in config:
            zarr_ref = ZarrReference.from_dict(config['zarr'])
            config['zarr_ref'] = zarr_ref
        
        # Handle threshold-based masks
        if mask_type == 'threshold':
            if 'source' not in config:
                raise ValidationError(f"Threshold mask '{name}' missing 'source'")
            
            source = config['source']
            if 'zarr' in source:
                config['source_zarr_ref'] = ZarrReference.from_dict(source['zarr'])
        
        self.parsed['shared']['masks'][name] = config
        self.defined_references.add(f"shared.masks.{name}")
    
    def _parse_shared_quality(self, name: str, config: Dict[str, Any]):
        """Parse a single shared quality metric definition"""
        # Resolve Zarr reference if present
        if 'zarr' in config:
            zarr_ref = ZarrReference.from_dict(config['zarr'])
            config['zarr_ref'] = zarr_ref
        
        # Handle expressions (computed from other metrics)
        if config.get('type') == 'expression':
            expression = config.get('expression', '')
            # Extract references from expression (e.g., "pow(p_forest, 2)")
            refs = re.findall(r'\b([a-z_][a-z0-9_]*)\b', expression)
            for ref in refs:
                if ref not in ['pow']:  # Ignore function names
                    self.used_references.add(f"shared.quality.{ref}")
                    self.reference_locations[f"shared.quality.{ref}"].append(
                        f"shared.quality.{name}"
                    )
        
        self.parsed['shared']['quality'][name] = config
        self.defined_references.add(f"shared.quality.{name}")
    
    def _parse_inputs(self):
        """Parse all input group definitions"""
        inputs = self.raw_config.get('inputs', {})
        
        self.parsed['inputs'] = {
            'temporal': {},
            'irregular': {},
            'static': {},
            'snapshot': {}
        }
        
        # Parse temporal inputs
        for group_name, group_config in inputs.get('temporal', {}).items():
            self._parse_input_group(group_name, group_config, 'temporal')
        
        # Parse irregular inputs
        for group_name, group_config in inputs.get('irregular', {}).items():
            self._parse_input_group(group_name, group_config, 'irregular')
        
        # Parse static inputs
        for group_name, group_config in inputs.get('static', {}).items():
            self._parse_input_group(group_name, group_config, 'static')
        
        # Parse snapshot inputs
        for group_name, group_config in inputs.get('snapshot', {}).items():
            self._parse_snapshot_input(group_name, group_config)
        
        logger.info(
            f"Parsed inputs: "
            f"{len(self.parsed['inputs']['temporal'])} temporal, "
            f"{len(self.parsed['inputs']['irregular'])} irregular, "
            f"{len(self.parsed['inputs']['static'])} static, "
            f"{len(self.parsed['inputs']['snapshot'])} snapshot"
        )
    
    def _parse_input_group(self, name: str, config: Dict[str, Any], category: str):
        """Parse a single input group"""
        # Resolve Zarr reference
        if 'zarr' not in config:
            raise ValidationError(f"Input group '{name}' missing 'zarr' reference")
        
        zarr_ref = ZarrReference.from_dict(config['zarr'])
        
        # Parse bands
        bands = []
        bands_config = config.get('bands', [])
        bands_template = config.get('bands_template')
        
        if bands_template:
            # Handle template-based bands (e.g., ccdc_snapshot)
            window_binding = config.get('window_binding', {})
            bands = self._expand_band_templates(bands_template, window_binding)
        else:
            # Regular band definitions
            for band_dict in bands_config:
                band = BandConfig.from_dict(band_dict)
                bands.append(band)
                
                # Track reference usage
                if band.norm:
                    norm_ref = f"normalization.presets.{band.norm}"
                    self.used_references.add(norm_ref)
                    self.reference_locations[norm_ref].append(
                        f"inputs.{category}.{name}.{band.name}"
                    )
                
                for mask_ref in band.mask:
                    self.used_references.add(mask_ref)
                    self.reference_locations[mask_ref].append(
                        f"inputs.{category}.{name}.{band.name}"
                    )
                
                for quality_ref in band.quality_weight:
                    self.used_references.add(quality_ref)
                    self.reference_locations[quality_ref].append(
                        f"inputs.{category}.{name}.{band.name}"
                    )
                
                if band.loss_weight:
                    self.used_references.add(band.loss_weight)
                    self.reference_locations[band.loss_weight].append(
                        f"inputs.{category}.{name}.{band.name}"
                    )
        
        # Create input group object
        input_group = InputGroup(
            name=name,
            zarr=zarr_ref,
            kind=config.get('kind', 'continuous'),
            bands=bands,
            time_window_years=config.get('time_window_years'),
            years=config.get('years'),
            window_binding=config.get('window_binding'),
            bands_template=bands_template,
            quality=config.get('quality'),
            missing_policy=config.get('missing_policy')
        )
        
        self.parsed['inputs'][category][name] = input_group
        self.defined_references.add(f"inputs.{category}.{name}")
    
    def _parse_snapshot_input(self, name: str, config: Dict[str, Any]):
        """
        Parse a snapshot input group with year-based pattern expansion.
        
        Snapshot inputs have:
        - years: list of available years [2020, 2022, 2024]
        - zarr_pattern: pattern like "snap_{year}_0831"
        - bands_template: list of band definitions with {zarr_pattern} placeholder
        
        This method expands the template for each year.
        """
        # Validate required fields
        if 'zarr' not in config:
            raise ValidationError(f"Snapshot input '{name}' missing 'zarr' reference")
        
        if 'years' not in config:
            raise ValidationError(f"Snapshot input '{name}' missing 'years' list")
        
        if 'zarr_pattern' not in config:
            raise ValidationError(f"Snapshot input '{name}' missing 'zarr_pattern'")
        
        if 'bands_template' not in config:
            raise ValidationError(f"Snapshot input '{name}' missing 'bands_template'")
        
        zarr_ref = ZarrReference.from_dict(config['zarr'])
        years = config['years']
        zarr_pattern = config['zarr_pattern']
        bands_template = config['bands_template']
        
        # Validate years list
        if not isinstance(years, list) or not years:
            raise ValidationError(f"Snapshot input '{name}' years must be a non-empty list")
        
        # Validate zarr_pattern contains {year} placeholder
        if '{year}' not in zarr_pattern:
            raise ValidationError(
                f"Snapshot input '{name}' zarr_pattern must contain '{{year}}' placeholder"
            )
        
        # Expand bands for each year
        bands = self._expand_snapshot_bands(name, years, zarr_pattern, bands_template)
        
        # Create input group object
        input_group = InputGroup(
            name=name,
            zarr=zarr_ref,
            kind=config.get('kind', 'continuous'),
            bands=bands,
            time_window_years=None,  # Snapshots don't have time windows
            years=years,
            window_binding=None,  # No longer needed - derived from years
            bands_template=bands_template,
            quality=config.get('quality'),
            missing_policy=config.get('missing_policy')
        )
        
        # Store additional snapshot-specific metadata
        input_group.zarr_pattern = zarr_pattern
        
        self.parsed['inputs']['snapshot'][name] = input_group
        self.defined_references.add(f"inputs.snapshot.{name}")
        
        logger.info(
            f"Parsed snapshot input '{name}': {len(years)} years, "
            f"{len(bands)} total bands ({len(bands_template)} per year)"
        )
    
    def _expand_snapshot_bands(
        self,
        snapshot_name: str,
        years: List[int],
        zarr_pattern: str,
        templates: List[Dict[str, Any]]
    ) -> List[BandConfig]:
        """
        Expand snapshot band templates for each year.
        
        For each year:
        1. Instantiate zarr_pattern: snap_{year}_0831 -> snap_2024_0831
        2. Replace {zarr_pattern} in band templates with the instantiated pattern
        3. Create unique band names: green_2024, red_2024, etc.
        
        Args:
            snapshot_name: Name of the snapshot input group
            years: List of years (e.g., [2020, 2022, 2024])
            zarr_pattern: Pattern with {year} placeholder
            templates: List of band template definitions
            
        Returns:
            List of expanded BandConfig objects
        """
        expanded_bands = []
        
        for year in years:
            # Instantiate the zarr_pattern for this year
            instantiated_pattern = zarr_pattern.format(year=year)
            
            # Expand each band template for this year
            for template in templates:
                template_name = template.get('name')
                template_array = template.get('array', '')
                
                # Replace {zarr_pattern} placeholder with instantiated pattern
                if '{zarr_pattern}' in template_array:
                    expanded_array = template_array.replace('{zarr_pattern}', instantiated_pattern)
                    expanded_name = f"{template_name}_{year}"
                else:
                    # Band doesn't use the pattern (e.g., shared arrays)
                    expanded_array = template_array
                    expanded_name = f"{template_name}_{year}"
                
                band = BandConfig(
                    name=expanded_name,
                    array=expanded_array,
                    norm=template.get('norm'),
                    mask=template.get('mask', []),
                    quality_weight=template.get('quality_weight', []),
                    loss_weight=template.get('loss_weight')
                )
                expanded_bands.append(band)
                
                # Track reference usage
                if band.norm:
                    norm_ref = f"normalization.presets.{band.norm}"
                    self.used_references.add(norm_ref)
                    self.reference_locations[norm_ref].append(
                        f"inputs.snapshot.{snapshot_name}.{band.name}"
                    )
                
                for mask_ref in band.mask:
                    self.used_references.add(mask_ref)
                    self.reference_locations[mask_ref].append(
                        f"inputs.snapshot.{snapshot_name}.{band.name}"
                    )
                
                for quality_ref in band.quality_weight:
                    self.used_references.add(quality_ref)
                    self.reference_locations[quality_ref].append(
                        f"inputs.snapshot.{snapshot_name}.{band.name}"
                    )
                
                if band.loss_weight:
                    self.used_references.add(band.loss_weight)
                    self.reference_locations[band.loss_weight].append(
                        f"inputs.snapshot.{snapshot_name}.{band.name}"
                    )
        
        return expanded_bands
    
    def _expand_band_templates(
        self, 
        templates: List[Dict[str, Any]], 
        window_binding: Dict[str, str]
    ) -> List[BandConfig]:
        """
        DEPRECATED: Old method for expanding band templates with window bindings.
        
        This is kept for backward compatibility but should not be used for 
        snapshot inputs. Use _expand_snapshot_bands instead.
        """
        expanded_bands = []
        
        for template in templates:
            template_name = template.get('name')
            template_array = template.get('array', '')
            
            # Check if this is a templated array
            if '{snap}' in template_array or '{' in template_array:
                # Expand for each window binding
                for window_key, snapshot_name in window_binding.items():
                    expanded_array = template_array.replace('{snap}', snapshot_name)
                    expanded_name = f"{template_name}_{window_key}"
                    
                    band = BandConfig(
                        name=expanded_name,
                        array=expanded_array,
                        norm=template.get('norm'),
                        mask=template.get('mask', []),
                        quality_weight=template.get('quality_weight', []),
                        loss_weight=template.get('loss_weight')
                    )
                    expanded_bands.append(band)
            else:
                # Regular band (not templated)
                band = BandConfig.from_dict(template)
                expanded_bands.append(band)
        
        return expanded_bands
    
    def _parse_derived_features(self):
        """Parse derived feature definitions"""
        derived = self.raw_config.get('derived', {})
        self.parsed['derived'] = {}
        
        for feature_name, feature_config in derived.items():
            if feature_name.startswith('_'):  # Skip YAML anchors
                continue
            
            self.parsed['derived'][feature_name] = feature_config
            self.defined_references.add(f"derived.{feature_name}")
            
            # Track dependencies
            if 'source' in feature_config:
                source = feature_config['source']
                if isinstance(source, str):
                    self.used_references.add(source)
                    self.reference_locations[source].append(f"derived.{feature_name}")
            
            if 'apply_to' in feature_config.get('injection', {}):
                for target in feature_config['injection']['apply_to']:
                    self.used_references.add(target)
                    self.reference_locations[target].append(f"derived.{feature_name}")
        
        logger.info(f"Parsed {len(self.parsed['derived'])} derived features")
    
    def _parse_model_inputs(self):
        """Parse model input mapping"""
        model_inputs = self.raw_config.get('model_inputs', {})
        self.parsed['model_inputs'] = model_inputs
        
        # Track references
        for encoder_name, encoder_config in model_inputs.items():
            for input_type, input_list in encoder_config.items():
                for input_ref in input_list:
                    self.used_references.add(input_ref)
                    self.reference_locations[input_ref].append(
                        f"model_inputs.{encoder_name}.{input_type}"
                    )
    
    def _parse_training_config(self):
        """Parse training configuration"""
        training = self.raw_config.get('training', {})
        self.parsed['training'] = training
        
        sampling = self.raw_config.get('sampling', {})
        self.parsed['sampling'] = sampling
        
        losses = self.raw_config.get('losses', {})
        self.parsed['losses'] = losses
        
        # Track references in loss configurations
        for loss_name, loss_config in losses.items():
            if 'weight' in loss_config:
                weight = loss_config['weight']
                if isinstance(weight, str) and weight.startswith('shared.'):
                    self.used_references.add(weight)
                    self.reference_locations[weight].append(f"losses.{loss_name}")
    
    def _validate_all_references(self):
        """Validate that all used references are defined"""
        undefined = self.used_references - self.defined_references
        
        if undefined:
            errors = []
            for ref in sorted(undefined):
                locations = self.reference_locations.get(ref, ['unknown'])
                errors.append(
                    f"  - '{ref}' (used in: {', '.join(locations[:3])})"
                )
            
            raise ReferenceError(
                f"Found {len(undefined)} undefined reference(s):\n" +
                "\n".join(errors)
            )
        
        logger.info(
            f"Validated {len(self.used_references)} reference(s): all defined"
        )
    
    def _check_circular_dependencies(self):
        """Check for circular dependencies in derived features"""
        # Build dependency graph
        graph = defaultdict(list)
        
        for feature_name, feature_config in self.parsed.get('derived', {}).items():
            if 'source' in feature_config:
                source = feature_config['source']
                if isinstance(source, str) and source.startswith('derived.'):
                    # Extract source feature name
                    source_feature = source.split('.')[1]
                    graph[feature_name].append(source_feature)
        
        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    raise BindingsError(
                        f"Circular dependency detected: {' -> '.join(cycle)}"
                    )
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                has_cycle(node, [])
        
        logger.info("No circular dependencies detected")
    
    def get_normalization_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Get a normalization preset by name.
        
        Args:
            preset_name: Name of preset (e.g., 'zscore', 'robust_iqr')
            
        Returns:
            Preset bindings configuration dictionary
            
        Raises:
            KeyError: If preset not found
        """
        presets = self.parsed.get('normalization', {}).get('presets', {})
        if preset_name not in presets:
            available = list(presets.keys())
            raise KeyError(
                f"Normalization preset '{preset_name}' not found. "
                f"Available: {available}"
            )
        return presets[preset_name]
    
    def resolve_reference(self, ref_path: str) -> Any:
        """
        Resolve a dotted reference path to its value.
        
        Args:
            ref_path: Reference like 'shared.masks.forest' or 'inputs.temporal.ls8day'
            
        Returns:
            Resolved value
            
        Raises:
            ReferenceError: If reference cannot be resolved
        """
        parts = ref_path.split('.')
        current = self.parsed
        
        for i, part in enumerate(parts):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                partial_path = '.'.join(parts[:i+1])
                raise ReferenceError(
                    f"Cannot resolve reference '{ref_path}' at '{partial_path}'"
                )
        
        return current
    
    def get_input_group(self, category: str, name: str) -> InputGroup:
        """
        Get an input group by category and name.
        
        Args:
            category: 'temporal', 'irregular', 'static', or 'snapshot'
            name: Group name (e.g., 'ls8day', 'topo', 'ccdc_snapshot')
            
        Returns:
            InputGroup object
            
        Raises:
            KeyError: If group not found
        """
        inputs = self.parsed.get('inputs', {})
        if category not in inputs:
            raise KeyError(f"Invalid input category: {category}")
        
        if name not in inputs[category]:
            available = list(inputs[category].keys())
            raise KeyError(
                f"Input group '{name}' not found in '{category}'. "
                f"Available: {available}"
            )
        
        return inputs[category][name]
    
    def get_snapshot_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get snapshot-specific metadata for a snapshot input.
        
        Args:
            name: Snapshot input name (e.g., 'ccdc_snapshot')
            
        Returns:
            Dictionary with 'years' and 'zarr_pattern'
            
        Raises:
            KeyError: If snapshot not found
        """
        snapshot_group = self.get_input_group('snapshot', name)
        return {
            'years': snapshot_group.years,
            'zarr_pattern': getattr(snapshot_group, 'zarr_pattern', None),
            'kind': snapshot_group.kind,
            'missing_policy': snapshot_group.missing_policy
        }
    
    def get_zarr_prefix_for_year(self, snapshot_name: str, year: int) -> str:
        """
        Get the zarr prefix for a specific year in a snapshot input.
        
        Args:
            snapshot_name: Name of snapshot input (e.g., 'ccdc_snapshot')
            year: Year to get prefix for
            
        Returns:
            Instantiated zarr prefix (e.g., 'snap_2024_0831')
            
        Raises:
            KeyError: If snapshot not found
            ValueError: If year not available in snapshot
        """
        metadata = self.get_snapshot_metadata(snapshot_name)
        
        if year not in metadata['years']:
            raise ValueError(
                f"Year {year} not available for snapshot '{snapshot_name}'. "
                f"Available years: {metadata['years']}"
            )
        
        zarr_pattern = metadata['zarr_pattern']
        if not zarr_pattern:
            raise ValueError(f"Snapshot '{snapshot_name}' missing zarr_pattern")
        
        return zarr_pattern.format(year=year)
    
    def get_snapshot_bands(self, snapshot_name: str, year: int) -> List[BandConfig]:
        """
        Get only the bands for a specific year from a snapshot input.
        
        This is a convenience method that filters the expanded bands to return
        only those for the requested year. Useful when loading data for a 
        specific training window.
        
        Args:
            snapshot_name: Name of snapshot input (e.g., 'ccdc_snapshot')
            year: Year to get bands for (e.g., 2024)
            
        Returns:
            List of BandConfig objects for the specified year
            
        Raises:
            KeyError: If snapshot not found
            ValueError: If year not available in snapshot
            
        Example:
            >>> parser = BindingsParser('bindings.yaml')
            >>> config = parser.parse()
            >>> 
            >>> # Get bands for 2024
            >>> bands_2024 = parser.get_snapshot_bands('ccdc_snapshot', 2024)
            >>> print(len(bands_2024))  # 20 (if 20 bands per year)
            >>> 
            >>> # Iterate through bands
            >>> for band in bands_2024:
            ...     print(f"{band.name}: {band.array}")
            # green_2024: snap_2024_0831_green
            # red_2024: snap_2024_0831_red
            # ...
        """
        # Validate year is available
        metadata = self.get_snapshot_metadata(snapshot_name)
        if year not in metadata['years']:
            raise ValueError(
                f"Year {year} not available for snapshot '{snapshot_name}'. "
                f"Available years: {metadata['years']}"
            )
        
        # Get the full snapshot group
        snapshot_group = self.get_input_group('snapshot', snapshot_name)
        
        # Filter bands for this specific year
        # Bands are named like: green_2024, red_2024, etc.
        year_suffix = f"_{year}"
        year_bands = [
            band for band in snapshot_group.bands 
            if band.name.endswith(year_suffix)
        ]
        
        return year_bands
    
    def get_all_bands(self) -> Dict[str, List[BandConfig]]:
        """
        Get all bands organized by input group.
        
        Returns:
            Dictionary mapping group names to lists of bands
        """
        all_bands = {}
        
        for category in ['temporal', 'irregular', 'static', 'snapshot']:
            for group_name, group in self.parsed['inputs'][category].items():
                full_name = f"{category}.{group_name}"
                all_bands[full_name] = group.bands
        
        return all_bands
    
    def validate_with_zarr(self, zarr_root):
        """
        Validate configuration against actual Zarr structure.
        
        Args:
            zarr_root: Opened Zarr group (from zarr.open())
            
        Raises:
            ValidationError: If Zarr structure doesn't match config
        """
        errors = []
        
        # Validate all Zarr references
        for category in ['temporal', 'irregular', 'static', 'snapshot']:
            for group_name, group in self.parsed['inputs'][category].items():
                zarr_ref = group.zarr
                
                # Check if group exists
                try:
                    zarr_group = zarr_root[zarr_ref.group]
                except KeyError:
                    errors.append(
                        f"Zarr group not found: {zarr_ref.group} "
                        f"(input: {category}.{group_name})"
                    )
                    continue
                
                # Check if arrays exist
                for band in group.bands:
                    if band.array:
                        try:
                            _ = zarr_group[band.array]
                        except KeyError:
                            errors.append(
                                f"Zarr array not found: {zarr_ref.group}/{band.array} "
                                f"(band: {category}.{group_name}.{band.name})"
                            )
        
        if errors:
            raise ValidationError(
                f"Zarr validation failed with {len(errors)} error(s):\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
        
        logger.info("Configuration validated against Zarr structure")
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the configuration.
        
        Returns:
            Multi-line summary string
        """
        lines = [
            f"Configuration Summary: {self.parsed['name']} (v{self.parsed['version']})",
            "=" * 70,
            f"\nZarr Dataset: {self.parsed['zarr']['path']}",
            f"\nNormalization Presets: {len(self.parsed['normalization']['presets'])}",
            f"  - {', '.join(self.parsed['normalization']['presets'].keys())}",
            f"\nShared Resources:",
            f"  - Masks: {len(self.parsed['shared']['masks'])}",
            f"  - Quality Metrics: {len(self.parsed['shared']['quality'])}",
            f"\nInputs:",
        ]
        
        for category in ['temporal', 'irregular', 'static', 'snapshot']:
            groups = self.parsed['inputs'][category]
            if groups:
                lines.append(f"  {category.capitalize()}:")
                for name, group in groups.items():
                    lines.append(f"    - {name}: {len(group.bands)} band(s)")
        
        derived_count = len([k for k in self.parsed['derived'].keys() if not k.startswith('_')])
        lines.append(f"\nDerived Features: {derived_count}")
        
        lines.append(f"\nReferences:")
        lines.append(f"  - Defined: {len(self.defined_references)}")
        lines.append(f"  - Used: {len(self.used_references)}")
        
        return "\n".join(lines)


def load_bindings(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load and parse bindings configuration.
    
    Args:
        config_path: Path to YAML bindings configuration file
        
    Returns:
        Parsed bindings configuration dictionary
        
    Raises:
        BindingsError: If configuration is invalid
    """
    parser = BindingsParser(config_path)
    return parser.parse()


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config/frl_bindings_v0.yaml'
    
    try:
        parser = BindingsParser(config_path)
        config = parser.parse()
        print(parser.summary())
        
    except BindingsError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
