"""
Simple minimal example of using ForestDatasetV2.

Quick start script for testing the new dataset loader.
"""

from frl.data.loaders.config import DatasetBindingsParser
from frl.data.loaders.dataset import ForestDatasetV2

# Load configuration
config = DatasetBindingsParser('frl/config/forest_repr_model_bindings.yaml').parse()

# Create dataset
dataset = ForestDatasetV2(
    config,
    split='train',
    patch_size=256,
    epoch_mode='number',
    sample_number=5,  # Just 5 samples for quick test
    debug_window=((0, 0), (512, 512)),  # Small spatial region
)

print(f"Dataset created with {len(dataset)} samples")

# Load a sample
sample = dataset[0]

print("\nSample keys:", list(sample.keys()))
print("\nData shapes:")
for key, value in sample.items():
    if key != 'metadata':
        print(f"  {key}: {value.shape}, dtype={value.dtype}")

print("\nChannel names:")
for group, names in sample['metadata']['channel_names'].items():
    print(f"  {group}: {names}")

print("\n✓ Dataset is working!")
