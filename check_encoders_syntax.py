"""
Syntax and import check for encoder modules.

This script verifies that all encoder modules can be imported
and their classes are properly defined, without requiring PyTorch
to actually run forward passes.
"""

import sys
import importlib.util


def check_module_exists(module_path):
    """Check if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            print(f"✗ Module '{module_path}' not found")
            return False
        print(f"✓ Module '{module_path}' found")
        return True
    except Exception as e:
        print(f"✗ Error checking '{module_path}': {e}")
        return False


def check_module_imports(module_path, expected_names):
    """Check if a module can be imported and has expected names."""
    try:
        module = __import__(module_path, fromlist=expected_names)
        missing = []
        for name in expected_names:
            if not hasattr(module, name):
                missing.append(name)

        if missing:
            print(f"✗ Module '{module_path}' missing: {missing}")
            return False
        else:
            print(f"✓ Module '{module_path}' has all expected exports: {expected_names}")
            return True
    except Exception as e:
        print(f"✗ Error importing '{module_path}': {e}")
        return False


def main():
    print("="*70)
    print("Encoder Module Syntax and Import Check")
    print("="*70)

    all_passed = True

    # Check individual modules exist
    print("\n1. Checking module files exist...")
    modules = [
        'frl.models.tcn',
        'frl.models.conv2d_encoder',
        'frl.models.spatial',
        'frl.models.conditioning',
        'frl.models.heads',
        'frl.models',
    ]

    for module in modules:
        if not check_module_exists(module):
            all_passed = False

    # Check package exports
    print("\n2. Checking frl.models package exports...")
    expected_exports = [
        'TCNEncoder',
        'GatedResidualBlock',
        'build_tcn_from_config',
        'Conv2DEncoder',
        'build_conv2d_from_config',
        'GatedResidualConv2D',
        'build_gated_residual_conv2d_from_config',
        'FiLMLayer',
        'FiLMConditionedBlock',
        'broadcast_to_time',
        'build_film_from_config',
        'MLPHead',
        'LinearHead',
        'Conv2DHead',
        'build_mlp_from_config',
        'build_linear_from_config',
        'build_conv2d_head_from_config',
    ]

    if not check_module_imports('frl.models', expected_exports):
        all_passed = False

    # Check individual module exports
    print("\n3. Checking individual module exports...")

    module_exports = {
        'frl.models.tcn': ['TCNEncoder', 'GatedResidualBlock', 'build_tcn_from_config'],
        'frl.models.conv2d_encoder': ['Conv2DEncoder', 'build_conv2d_from_config'],
        'frl.models.spatial': ['GatedResidualConv2D', 'build_gated_residual_conv2d_from_config'],
        'frl.models.conditioning': ['FiLMLayer', 'FiLMConditionedBlock', 'broadcast_to_time', 'build_film_from_config'],
        'frl.models.heads': ['MLPHead', 'LinearHead', 'Conv2DHead', 'build_mlp_from_config', 'build_linear_from_config', 'build_conv2d_head_from_config'],
    }

    for module_path, expected in module_exports.items():
        if not check_module_imports(module_path, expected):
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✓ All checks passed! Encoder modules are properly defined.")
    else:
        print("✗ Some checks failed. Please review the errors above.")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
