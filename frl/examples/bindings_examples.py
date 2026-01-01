"""
Bindings Parser - Usage Guide and Examples (Updated with Snapshot Support)

This guide demonstrates how to use the bindings parser in various scenarios,
from basic parsing to integration with the existing stats loader and normalization system.

Updated to include snapshot input category examples.
"""

from pathlib import Path
from data.loaders.bindings.parser import BindingsParser, load_bindings
from data.loaders.bindings.utils import BindingsQuery, ReferenceResolver, BandSelector
from data.normalization.zarr_stats_loader import ZarrStatsLoader
from data.normalization.normalization import NormalizationManager


# ============================================================================
# Example 1: Basic Bindings Configuration Loading
# ============================================================================

def example_1_basic_loading():
    """
    Basic configuration loading and validation.
    """
    print("=" * 70)
    print("Example 1: Basic Bindings Configuration Loading")
    print("=" * 70)
    
    # Load and parse configuration
    parser = BindingsParser('config/frl_bindings_v0.yaml')
    config = parser.parse()
    
    # Print summary
    print(parser.summary())
    
    # Access specific sections
    print("\nZarr path:", config['zarr']['path'])
    print("Available norm presets:", list(config['normalization']['presets'].keys()))
    print("Shared masks:", list(config['shared']['masks'].keys()))
    
    # Check for snapshot inputs
    if 'snapshot' in config['inputs'] and config['inputs']['snapshot']:
        print("\nSnapshot inputs found:")
        for name in config['inputs']['snapshot'].keys():
            print(f"  - {name}")
    
    return config


# ============================================================================
# Example 2: Reference Resolution
# ============================================================================

def example_2_reference_resolution(config):
    """
    Resolving references to shared resources.
    """
    print("\n" + "=" * 70)
    print("Example 2: Reference Resolution")
    print("=" * 70)
    
    resolver = ReferenceResolver(config)
    
    # Resolve a shared mask
    forest_mask = resolver.resolve('shared.masks.forest')
    print("\nForest mask config:")
    print(f"  Type: {forest_mask.get('type')}")
    print(f"  Source: {forest_mask.get('source')}")
    
    # Resolve a quality metric
    forest_weight = resolver.resolve('shared.quality.forest_weight')
    print("\nForest weight config:")
    print(f"  Type: {forest_weight.get('type')}")
    print(f"  Expression: {forest_weight.get('expression')}")
    
    # Resolve a normalization preset
    zscore_preset = resolver.resolve('normalization.presets.zscore')
    print("\nZ-score preset:")
    print(f"  Type: {zscore_preset.get('type')}")
    print(f"  Stats source: {zscore_preset.get('stats_source')}")
    print(f"  Clamp range: [{zscore_preset['clamp']['min']}, {zscore_preset['clamp']['max']}]")
    
    return resolver


# ============================================================================
# Example 3: Full Band Resolution
# ============================================================================

def example_3_full_band_resolution(config):
    """
    Resolve a band with all its dependencies.
    """
    print("\n" + "=" * 70)
    print("Example 3: Full Band Resolution")
    print("=" * 70)
    
    resolver = ReferenceResolver(config)
    
    # Resolve a temporal band
    resolved = resolver.resolve_band_fully('temporal', 'ls8day', 'NDVI_summer_p95')
    
    print(f"\nBand: {resolved.name}")
    print(f"Array path: {resolved.array_path}")
    print(f"Category: {resolved.metadata['category']}")
    print(f"Time window: {resolved.metadata['time_window_years']} years")
    
    print(f"\nNormalization:")
    if resolved.norm_config:
        print(f"  Type: {resolved.norm_config['type']}")
        print(f"  Stats source: {resolved.norm_config.get('stats_source', 'N/A')}")
    
    print(f"\nMasks ({len(resolved.masks)}):")
    for i, mask in enumerate(resolved.masks):
        print(f"  {i+1}. {mask.get('type', 'unknown')} mask")
    
    print(f"\nQuality weights ({len(resolved.quality_weights)}):")
    for i, qw in enumerate(resolved.quality_weights):
        if 'zarr_ref' in qw:
            print(f"  {i+1}. {qw['zarr_ref'].group}/{qw['zarr_ref'].array}")
    
    if resolved.loss_weight:
        print(f"\nLoss weight: {resolved.loss_weight.get('type', 'unknown')}")
    
    return resolved


# ============================================================================
# Example 4: Querying Configuration
# ============================================================================

def example_4_querying_config(config):
    """
    Use ConfigQuery for high-level queries.
    """
    print("\n" + "=" * 70)
    print("Example 4: Configuration Queries")
    print("=" * 70)
    
    query = BindingsQuery(config)
    
    # Count bands by category
    counts = query.count_bands()
    print("\nBand counts:")
    for category, count in counts.items():
        print(f"  {category}: {count}")
    
    # Total channels (including temporal expansion)
    total = query.count_total_channels()
    print(f"\nTotal channels (with temporal expansion): {total}")
    
    # List resources
    print(f"\nNormalization presets ({len(query.list_normalization_presets())}):")
    for preset in query.list_normalization_presets()[:5]:
        print(f"  - {preset}")
    
    print(f"\nShared masks ({len(query.list_shared_masks())}):")
    for mask in query.list_shared_masks()[:5]:
        print(f"  - {mask}")
    
    print(f"\nDerived features ({len(query.list_derived_features())}):")
    for feature in query.list_derived_features()[:5]:
        print(f"  - {feature}")
    
    # Get training configuration
    windowing = query.get_window_config()
    print(f"\nTraining windowing:")
    print(f"  Window length: {windowing.get('length_years')} years")
    print(f"  End year range: {windowing.get('end_year_range')}")
    print(f"  Bundle size: {windowing.get('multi_window_bundle', {}).get('bundle_size')}")
    
    return query


# ============================================================================
# Example 5: Band Selection and Filtering
# ============================================================================

def example_5_band_selection(config):
    """
    Select and filter bands based on criteria.
    """
    print("\n" + "=" * 70)
    print("Example 5: Band Selection and Filtering")
    print("=" * 70)
    
    selector = BandSelector(config)
    
    # Select all temporal bands
    temporal_bands = selector.select_by_category('temporal')
    print(f"\nTemporal bands: {len(temporal_bands)}")
    for group_name, band in temporal_bands[:3]:
        print(f"  {group_name}.{band.name}")
    
    # Select all snapshot bands
    if 'snapshot' in config['inputs']:
        snapshot_bands = selector.select_by_category('snapshot')
        print(f"\nSnapshot bands: {len(snapshot_bands)}")
        for group_name, band in snapshot_bands[:5]:
            print(f"  {group_name}.{band.name}")
    
    # Select bands using z-score normalization
    zscore_bands = selector.select_by_norm('zscore')
    print(f"\nBands using z-score normalization: {len(zscore_bands)}")
    for category, group_name, band in zscore_bands[:5]:
        print(f"  {category}.{group_name}.{band.name}")
    
    # Select bands with forest mask
    forest_bands = selector.select_by_mask('shared.masks.forest')
    print(f"\nBands with forest mask: {len(forest_bands)}")
    for category, group_name, band in forest_bands[:5]:
        print(f"  {category}.{group_name}.{band.name}")
    
    # Select temporal bands with 10-year window
    window_10_bands = selector.select_temporal_with_window(10)
    print(f"\nTemporal bands with 10-year window: {len(window_10_bands)}")
    for group_name, band in window_10_bands[:5]:
        print(f"  {group_name}.{band.name}")


# ============================================================================
# Example 6: Integration with Stats Loader
# ============================================================================

def example_6_integration_with_stats():
    """
    Integrate configuration parser with existing stats loader.
    """
    print("\n" + "=" * 70)
    print("Example 6: Integration with Stats Loader")
    print("=" * 70)
    
    # Load bindings configuration
    config = load_bindings('config/frl_bindings_v0.yaml')
    
    # Initialize stats loader with path from config
    zarr_path = config['zarr']['path']
    print(f"\nInitializing stats loader for: {zarr_path}")
    
    # Note: This would work with actual Zarr file
    # stats_loader = ZarrStatsLoader(zarr_path)
    
    # Initialize normalization manager
    # norm_manager = NormalizationManager(config, stats_loader)
    
    # Example: Get normalizer for a specific band
    query = BindingsQuery(config)
    resolver = ReferenceResolver(config)
    
    band = resolver.resolve_band_fully('temporal', 'ls8day', 'NDVI_summer_p95')
    
    print(f"\nBand: {band.name}")
    print(f"Array: {band.array_path}")
    print(f"Normalization type: {band.norm_config['type']}")
    
    # With actual stats loader and data, you would do:
    # normalized_data = norm_manager.normalize_band(
    #     data=raw_data,
    #     group_path='annual/ls8day/data',
    #     band_config={'array': 'NDVI_summer_p95', 'norm': 'zscore'}
    # )
    
    print("\nReady for integration with stats loader and normalization manager!")


# ============================================================================
# Example 7: NEW - Working with Snapshot Inputs
# ============================================================================

def example_7_snapshot_inputs(config):
    """
    Working with snapshot inputs - NEW in refactored version.
    """
    print("\n" + "=" * 70)
    print("Example 7: Working with Snapshot Inputs")
    print("=" * 70)
    
    # Check if snapshot inputs exist
    if 'snapshot' not in config['inputs'] or not config['inputs']['snapshot']:
        print("\nNo snapshot inputs in this configuration.")
        return
    
    parser = BindingsParser('config/frl_bindings_v0.yaml')
    parser.parse()
    
    # Get snapshot metadata
    for snapshot_name in config['inputs']['snapshot'].keys():
        print(f"\nSnapshot: {snapshot_name}")
        
        metadata = parser.get_snapshot_metadata(snapshot_name)
        print(f"  Years: {metadata['years']}")
        print(f"  Pattern: {metadata['zarr_pattern']}")
        print(f"  Kind: {metadata['kind']}")
        
        # Get snapshot group
        snapshot = config['inputs']['snapshot'][snapshot_name]
        print(f"  Total bands: {len(snapshot.bands)}")
        print(f"  Bands per year: {len(snapshot.bands_template)}")
        
        # Show zarr prefixes for each year
        print(f"\n  Zarr prefixes:")
        for year in metadata['years']:
            prefix = parser.get_zarr_prefix_for_year(snapshot_name, year)
            print(f"    {year}: {prefix}")
        
        # Show sample expanded bands
        print(f"\n  Sample expanded bands:")
        for band in snapshot.bands[:6]:
            print(f"    {band.name}: {band.array}")
        
        # Group bands by year
        from collections import defaultdict
        by_year = defaultdict(list)
        for band in snapshot.bands:
            # Extract year from band name (e.g., green_2024 -> 2024)
            year = int(band.name.split('_')[-1])
            by_year[year].append(band.name)
        
        print(f"\n  Bands per year:")
        for year in sorted(by_year.keys()):
            print(f"    {year}: {len(by_year[year])} bands")


# ============================================================================
# Example 8: NEW - Loading Snapshot Data for Training Window
# ============================================================================

def example_8_snapshot_for_training_window(config):
    """
    Example of how to load snapshot data for a specific training window.
    """
    print("\n" + "=" * 70)
    print("Example 8: Loading Snapshot Data for Training Window")
    print("=" * 70)
    
    if 'snapshot' not in config['inputs'] or not config['inputs']['snapshot']:
        print("\nNo snapshot inputs in this configuration.")
        return
    
    parser = BindingsParser('config/frl_bindings_v0.yaml')
    parser.parse()
    
    # Simulate training scenario
    snapshot_name = 'ccdc_snapshot'
    window_end_years = [2024, 2022, 2020]  # t0, t2, t4
    
    print(f"\nSnapshot: {snapshot_name}")
    print(f"Training windows: {window_end_years}")
    
    for i, window_end_year in enumerate(window_end_years):
        window_label = f"t{i*2}"  # t0, t2, t4
        
        print(f"\n  Window {window_label} (end year: {window_end_year}):")
        
        # Get zarr prefix for this year
        zarr_prefix = parser.get_zarr_prefix_for_year(snapshot_name, window_end_year)
        print(f"    Zarr prefix: {zarr_prefix}")
        
        # Get snapshot configuration
        snapshot = config['inputs']['snapshot'][snapshot_name]
        
        # Filter bands for this specific year
        year_bands = [b for b in snapshot.bands if f"_{window_end_year}" in b.name]
        print(f"    Bands to load: {len(year_bands)}")
        
        # Show how to construct array paths
        print(f"    Sample array paths:")
        for band in year_bands[:3]:
            array_path = f"{snapshot.zarr.group}/{band.array}"
            print(f"      {band.name}: {array_path}")
    
    # Pseudocode for actual loading
    print("\n  Pseudocode for data loading:")
    print("""
    def load_snapshot_for_window(zarr_root, snapshot_config, window_end_year):
        # Get zarr prefix for this year
        zarr_prefix = get_zarr_prefix_for_year(snapshot_name, window_end_year)
        
        # Load zarr group
        zarr_group = zarr_root[snapshot_config.zarr.group]
        
        # Load bands for this year
        data = {}
        for band in snapshot_config.bands:
            if f"_{window_end_year}" in band.name:
                data[band.name] = zarr_group[band.array][...]
        
        return data
    """)


# ============================================================================
# Example 9: Template Expansion (Updated)
# ============================================================================

def example_9_template_expansion(config):
    """
    Examine template expansion for snapshot inputs.
    Updated to work with new snapshot format.
    """
    print("\n" + "=" * 70)
    print("Example 9: Snapshot Template Expansion")
    print("=" * 70)
    
    if 'snapshot' not in config['inputs'] or 'ccdc_snapshot' not in config['inputs']['snapshot']:
        print("\nNo ccdc_snapshot in this configuration.")
        return
    
    # Get CCDC snapshot group (uses templates)
    ccdc = config['inputs']['snapshot']['ccdc_snapshot']
    
    print(f"\nCCDC Snapshot Group:")
    print(f"  Kind: {ccdc.kind}")
    print(f"  Years: {ccdc.years}")
    print(f"  Pattern: {ccdc.zarr_pattern}")
    print(f"  Total bands (after expansion): {len(ccdc.bands)}")
    print(f"  Template bands: {len(ccdc.bands_template)}")
    
    # Show template expansion example
    print(f"\nTemplate expansion example:")
    template_band = ccdc.bands_template[0]
    print(f"  Template: {template_band}")
    
    expanded_examples = [b for b in ccdc.bands if b.name.startswith(template_band['name'])]
    print(f"\n  Expanded to {len(expanded_examples)} bands:")
    for band in expanded_examples:
        print(f"    {band.name}: {band.array}")
    
    # Group by base feature
    from collections import defaultdict
    by_feature = defaultdict(list)
    for band in ccdc.bands:
        # Extract base feature name (before _YEAR)
        base_name = '_'.join(band.name.split('_')[:-1])
        by_feature[base_name].append(band.name)
    
    print(f"\n  Bands grouped by feature (first 5):")
    for feature, variants in list(by_feature.items())[:5]:
        print(f"    {feature}: {variants}")


# ============================================================================
# Example 10: Model Input Mapping (Updated)
# ============================================================================

def example_10_model_input_mapping(config):
    """
    Examine model input mappings to encoders.
    Updated to show snapshot inputs.
    """
    print("\n" + "=" * 70)
    print("Example 10: Model Input Mapping")
    print("=" * 70)
    
    query = BindingsQuery(config)
    
    # Type encoder inputs
    type_inputs = query.get_model_encoder_inputs('type_encoder')
    print("\nType Encoder Inputs:")
    for input_type, input_refs in type_inputs.items():
        print(f"  {input_type}:")
        for ref in input_refs:
            print(f"    - {ref}")
    
    # Phase encoder inputs
    phase_inputs = query.get_model_encoder_inputs('phase_encoder')
    print("\nPhase Encoder Inputs:")
    for input_type, input_refs in phase_inputs.items():
        print(f"  {input_type}:")
        for ref in input_refs:
            print(f"    - {ref}")
    
    # Highlight snapshot inputs if present
    if 'snapshot' in phase_inputs:
        print("\n  Note: Phase encoder uses snapshot inputs!")
        print("  These will vary by training window end year.")


# ============================================================================
# Example 11: Derived Features
# ============================================================================

def example_11_derived_features(config):
    """
    Examine derived feature configurations.
    """
    print("\n" + "=" * 70)
    print("Example 11: Derived Features")
    print("=" * 70)
    
    query = BindingsQuery(config)
    
    # List all derived features
    derived_features = query.list_derived_features()
    print(f"\nDerived features: {len(derived_features)}")
    
    # Examine temporal position encoding
    if 'temporal_position' in config['derived']:
        temp_pos = config['derived']['temporal_position']
        print("\nTemporal Position Encoding:")
        print(f"  Enabled: {temp_pos.get('enabled')}")
        print(f"  Window length: {temp_pos.get('window_length')}")
        print(f"  Channels: {len(temp_pos.get('channels', []))}")
        
        injection = temp_pos.get('injection', {})
        print(f"  Injection mode: {injection.get('mode')}")
        print(f"  Applied to:")
        for target in injection.get('apply_to', []):
            print(f"    - {target}")
    
    # Examine ls8_delta
    if 'ls8_delta' in config['derived']:
        ls8_delta = config['derived']['ls8_delta']
        print("\nLS8 Delta Computation:")
        print(f"  Enabled: {ls8_delta.get('enabled')}")
        print(f"  Source: {ls8_delta.get('source')}")
        print(f"  Operation: {ls8_delta.get('operation')}")
        print(f"  Bands derived: {len(ls8_delta.get('bands', []))}")


# ============================================================================
# Example 12: Validation and Error Handling
# ============================================================================

def example_12_validation():
    """
    Demonstrate validation and error handling.
    """
    print("\n" + "=" * 70)
    print("Example 12: Validation and Error Handling")
    print("=" * 70)
    
    try:
        # Try to load configuration
        parser = BindingsParser('config/frl_bindings_v0.yaml')
        config = parser.parse()
        print("\n✓ Bindings configuration is valid!")
        
        # Check specific references
        query = BindingsQuery(config)
        
        test_refs = [
            'shared.masks.forest',
            'shared.quality.forest_weight',
            'normalization.presets.zscore',
            'inputs.temporal.ls8day'
        ]
        
        # Add snapshot reference if it exists
        if 'snapshot' in config['inputs'] and config['inputs']['snapshot']:
            snapshot_name = list(config['inputs']['snapshot'].keys())[0]
            test_refs.append(f'inputs.snapshot.{snapshot_name}')
        
        print("\nReference validation:")
        for ref in test_refs:
            exists = query.validate_reference_exists(ref)
            status = "✓" if exists else "✗"
            print(f"  {status} {ref}")
        
        # Test invalid reference
        invalid_ref = 'shared.masks.nonexistent'
        exists = query.validate_reference_exists(invalid_ref)
        print(f"  {'✓' if exists else '✗'} {invalid_ref} (should be ✗)")
        
        # Test snapshot-specific validation
        if 'snapshot' in config['inputs'] and config['inputs']['snapshot']:
            print("\nSnapshot-specific validation:")
            snapshot_name = list(config['inputs']['snapshot'].keys())[0]
            
            # Test valid year
            try:
                prefix = parser.get_zarr_prefix_for_year(snapshot_name, 2024)
                print(f"  ✓ Year 2024 is valid: {prefix}")
            except ValueError as e:
                print(f"  ✗ Year 2024 validation failed: {e}")
            
            # Test invalid year
            try:
                prefix = parser.get_zarr_prefix_for_year(snapshot_name, 2026)
                print(f"  ✗ Year 2026 should have failed")
            except ValueError as e:
                print(f"  ✓ Year 2026 correctly rejected: {e}")
        
    except Exception as e:
        print(f"\n✗ Configuration error: {e}")


# ============================================================================
# Example 13: Working with Zarr Structure (with actual Zarr)
# ============================================================================

def example_13_zarr_validation(config):
    """
    Validate configuration against actual Zarr structure.
    
    Note: This requires an actual Zarr dataset.
    """
    print("\n" + "=" * 70)
    print("Example 13: Zarr Structure Validation")
    print("=" * 70)
    
    zarr_path = config['zarr']['path']
    print(f"\nZarr path: {zarr_path}")
    
    # This would work with actual Zarr file:
    """
    import zarr
    zarr_root = zarr.open(zarr_path, mode='r')
     
    parser = BindingsParser('config/frl_bindings_v0.yaml')
    parser.parse()
    parser.validate_with_zarr(zarr_root)
    
    print("✓ Configuration matches Zarr structure!")
    """
    
    print("\n(Requires actual Zarr dataset to run)")


# ============================================================================
# Main: Run All Examples
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("BINDINGS PARSER - USAGE EXAMPLES (UPDATED)")
    print("=" * 70)
    
    try:
        # Example 1: Basic loading
        config = example_1_basic_loading()
        
        # Example 2: Reference resolution
        resolver = example_2_reference_resolution(config)
        
        # Example 3: Full band resolution
        resolved_band = example_3_full_band_resolution(config)
        
        # Example 4: Querying
        query = example_4_querying_config(config)
        
        # Example 5: Band selection
        example_5_band_selection(config)
        
        # Example 6: Stats loader integration
        example_6_integration_with_stats()
        
        # Example 7: NEW - Snapshot inputs
        example_7_snapshot_inputs(config)
        
        # Example 8: NEW - Loading snapshots for training
        example_8_snapshot_for_training_window(config)
        
        # Example 9: Template expansion (updated)
        example_9_template_expansion(config)
        
        # Example 10: Model input mapping (updated)
        example_10_model_input_mapping(config)
        
        # Example 11: Derived features
        example_11_derived_features(config)
        
        # Example 12: Validation
        example_12_validation()
        
        # Example 13: Zarr validation (requires actual Zarr)
        example_13_zarr_validation(config)
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Run specific example
        example_num = int(sys.argv[1])
        examples = {
            1: example_1_basic_loading,
            2: lambda: example_2_reference_resolution(load_bindings('config/frl_bindings_v0.yaml')),
            3: lambda: example_3_full_band_resolution(load_bindings('config/frl_bindings_v0.yaml')),
            4: lambda: example_4_querying_config(load_bindings('config/frl_bindings_v0.yaml')),
            5: lambda: example_5_band_selection(load_bindings('config/frl_bindings_v0.yaml')),
            6: example_6_integration_with_stats,
            7: lambda: example_7_snapshot_inputs(load_bindings('config/frl_bindings_v0.yaml')),
            8: lambda: example_8_snapshot_for_training_window(load_bindings('config/frl_bindings_v0.yaml')),
            9: lambda: example_9_template_expansion(load_bindings('config/frl_bindings_v0.yaml')),
            10: lambda: example_10_model_input_mapping(load_bindings('config/frl_bindings_v0.yaml')),
            11: lambda: example_11_derived_features(load_bindings('config/frl_bindings_v0.yaml')),
            12: example_12_validation,
            13: lambda: example_13_zarr_validation(load_bindings('config/frl_bindings_v0.yaml')),
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Example {example_num} not found. Available: 1-13")
    else:
        # Run all examples
        main()
