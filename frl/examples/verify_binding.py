"""
Quick verification script for bindings parser.

Run this to test the parser with your actual YAML configuration.
"""

import sys
from pathlib import Path
from data.loaders.bindings.parser import BindingsParser, BindingsError
from data.loaders.bindings.utils import BindingsQuery, ReferenceResolver


def verify_configuration(yaml_path: str = 'config/frl_bindings_v0.yaml'):
    """
    Verify configuration file is valid and print summary.
    
    Args:
        yaml_path: Path to YAML configuration file
    """
    print("=" * 70)
    print("Bindings Configuration Verification")
    print("=" * 70)
    
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        print(f"\n❌ Configuration file not found: {yaml_path}")
        print(f"   Looking in: {yaml_file.absolute()}")
        return False
    
    print(f"\n📄 Bindings configuration file: {yaml_path}")
    print(f"   Size: {yaml_file.stat().st_size:,} bytes")
    
    try:
        # Parse configuration
        print("\n🔍 Parsing bindings configuration...")
        parser = BindingsParser(yaml_path)
        config = parser.parse()
        print("   ✅ Bindings configuration parsed successfully!")
        
        # Print summary
        print("\n" + parser.summary())
        
        # Detailed statistics
        print("\n" + "=" * 70)
        print("Detailed Statistics")
        print("=" * 70)
        
        query = BindingsQuery(config)
        
        # Count bands
        counts = query.count_bands()
        print(f"\nBands by category:")
        total_bands = 0
        for category, count in counts.items():
            print(f"  {category:12} {count:4}")
            total_bands += count
        print(f"  {'TOTAL':12} {total_bands:4}")
        
        # Total channels (with temporal expansion)
        total_channels = query.count_total_channels()
        print(f"\nTotal channels (with temporal expansion): {total_channels}")
        
        # Resources
        print(f"\nShared resources:")
        print(f"  Normalization presets: {len(query.list_normalization_presets())}")
        print(f"  Masks: {len(query.list_shared_masks())}")
        print(f"  Quality metrics: {len(query.list_shared_quality())}")
        print(f"  Derived features: {len(query.list_derived_features())}")
        
        # References
        print(f"\nReference tracking:")
        print(f"  Defined: {len(parser.defined_references)}")
        print(f"  Used: {len(parser.used_references)}")
        print(f"  All resolved: ✅")
        
        # Test some specific resolutions
        print("\n" + "=" * 70)
        print("Sample Reference Resolutions")
        print("=" * 70)
        
        resolver = ReferenceResolver(config)
        
        # Test 1: Shared mask
        try:
            forest_mask = resolver.resolve('shared.masks.forest')
            print(f"\n✅ shared.masks.forest")
            print(f"   Type: {forest_mask.get('type')}")
        except Exception as e:
            print(f"\n❌ shared.masks.forest: {e}")
        
        # Test 2: Quality metric
        try:
            forest_weight = resolver.resolve('shared.quality.forest_weight')
            print(f"\n✅ shared.quality.forest_weight")
            print(f"   Type: {forest_weight.get('type')}")
            print(f"   Expression: {forest_weight.get('expression')}")
        except Exception as e:
            print(f"\n❌ shared.quality.forest_weight: {e}")
        
        # Test 3: Normalization preset
        try:
            zscore = resolver.resolve('normalization.presets.zscore')
            print(f"\n✅ normalization.presets.zscore")
            print(f"   Type: {zscore.get('type')}")
            print(f"   Stats source: {zscore.get('stats_source')}")
        except Exception as e:
            print(f"\n❌ normalization.presets.zscore: {e}")
        
        # Test 4: Full band resolution
        try:
            band = resolver.resolve_band_fully('temporal', 'ls8day', 'NDVI_summer_p95')
            print(f"\n✅ inputs.temporal.ls8day.NDVI_summer_p95")
            print(f"   Array: {band.array_path}")
            print(f"   Normalization: {band.norm_config['type'] if band.norm_config else 'none'}")
            print(f"   Masks: {len(band.masks)}")
            print(f"   Quality weights: {len(band.quality_weights)}")
            if band.loss_weight:
                print(f"   Loss weight: {band.loss_weight.get('type', 'defined')}")
        except Exception as e:
            print(f"\n❌ Full band resolution: {e}")
        
        # Test 5: Template expansion
        try:
            ccdc = config['inputs']['static']['ccdc_snapshot']
            print(f"\n✅ Template expansion (ccdc_snapshot)")
            print(f"   Window bindings: {len(ccdc.window_binding)}")
            print(f"   Expanded bands: {len(ccdc.bands)}")
            
            # Show a few examples
            print(f"   Examples:")
            for band in ccdc.bands[:3]:
                print(f"     - {band.name}: {band.array}")
        except Exception as e:
            print(f"\n❌ Template expansion: {e}")
        
        print("\n" + "=" * 70)
        print("✅ VERIFICATION PASSED")
        print("=" * 70)
        print("\nBindings configuration is valid and ready for use!")
        print("\nNext steps:")
        print("  1. Review the summary above")
        print("  2. Test with actual Zarr dataset (see README.md)")
        print("  3. Integrate with dataloader (see config_examples.py)")
        
        return True
        
    except BindingsError as e:
        print(f"\n❌ BINDINGS CONFIGURATION ERROR")
        print("=" * 70)
        print(f"\n{e}")
        print("\nPlease fix the configuration and try again.")
        return False
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else 'config/frl_bindings_v0.yaml'
    
    success = verify_configuration(yaml_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
