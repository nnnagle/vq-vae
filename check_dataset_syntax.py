"""
Syntax checker for the refactored dataset code.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if modules can be imported without errors."""
    print("=" * 60)
    print("Checking module imports...")
    print("=" * 60)

    errors = []

    # Test dataset_config
    try:
        print("\n1. Checking dataset_config.py...")
        from frl.data.loaders.config import dataset_config
        print("   ✓ dataset_config module imports successfully")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        errors.append(('dataset_config', e))

    # Test dataset_bindings_parser
    try:
        print("\n2. Checking dataset_bindings_parser.py...")
        from frl.data.loaders.config import dataset_bindings_parser
        print("   ✓ dataset_bindings_parser module imports successfully")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        errors.append(('dataset_bindings_parser', e))

    # Test forest_dataset_v2
    try:
        print("\n3. Checking forest_dataset_v2.py...")
        from frl.data.loaders.dataset import forest_dataset_v2
        print("   ✓ forest_dataset_v2 module imports successfully")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        errors.append(('forest_dataset_v2', e))

    # Test can import classes
    try:
        print("\n4. Checking class imports...")
        from frl.data.loaders.config import (
            DatasetBindingsParser,
            BindingsConfig,
            DatasetGroupConfig,
            ChannelConfig,
        )
        from frl.data.loaders.dataset import ForestDatasetV2, collate_fn
        print("   ✓ All classes import successfully")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        errors.append(('class imports', e))

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} import error(s)")
        print("=" * 60)
        for module, error in errors:
            print(f"\n{module}:")
            print(f"  {error}")
        return False
    else:
        print("SUCCESS: All modules import without errors")
        print("=" * 60)
        return True


def check_yaml_syntax():
    """Check if YAML file is valid."""
    print("\n" + "=" * 60)
    print("Checking YAML syntax...")
    print("=" * 60)

    yaml_path = Path("frl/config/forest_repr_model_bindings.yaml")

    if not yaml_path.exists():
        print(f"✗ YAML file not found: {yaml_path}")
        return False

    try:
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        print(f"✓ YAML file is valid")
        print(f"  - Top-level keys: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"✗ YAML parsing error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("DATASET REFACTOR SYNTAX CHECK")
    print("=" * 60)

    # Check YAML
    yaml_ok = check_yaml_syntax()

    # Check imports
    import_ok = check_imports()

    print("\n" + "=" * 60)
    if yaml_ok and import_ok:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗")
        sys.exit(1)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
