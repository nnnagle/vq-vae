# src/data/dataset.py (minimal version)
"""
Start simple: Load just one band from one year
"""
import zarr
import torch
from torch.utils.data import Dataset
import numpy as np

class ForestReprDatasetV0(Dataset):
    """Minimal dataset - loads ONE band to verify everything works"""
    
    def __init__(self, zarr_path):
        self.store = zarr.open(zarr_path, mode='r')
        
        # Hardcode for now - just get SOMETHING working
        self.ndvi_array = self.store['annual/ls8day/data/NDVI_summer_p95']
        
        # Get spatial dims
        self.n_years, self.height, self.width = self.ndvi_array.shape
        
        print(f"Dataset ready: {self.n_years} years, {self.height}x{self.width} pixels")
    
    def __len__(self):
        # For now, just return 100 samples to test
        return 100
    
    def __getitem__(self, idx):
        # Sample random location and year
        patch_size = 256
        
        year_idx = np.random.randint(0, self.n_years)
        y = np.random.randint(0, self.height - patch_size)
        x = np.random.randint(0, self.width - patch_size)
        
        # Load patch
        patch = self.ndvi_array[year_idx, y:y+patch_size, x:x+patch_size]
        
        # Convert to tensor
        tensor = torch.from_numpy(patch).float()
        
        return {
            'ndvi': tensor.unsqueeze(0),  # [1, H, W]
            'year': year_idx,
            'location': (y, x)
        }

# Test it!
if __name__ == '__main__':
    dataset = ForestReprDatasetV0('/data/VA/zarr/va_vae_dataset_test.zarr')
    
    # Try loading one sample
    sample = dataset[0]
    print(f"Sample shape: {sample['ndvi'].shape}")
    print(f"Sample year: {sample['year']}")
    print(f"Sample location: {sample['location']}")
    
    # Check for NaNs before computing stats
    tensor = sample['ndvi']
    n_nan = torch.isnan(tensor).sum().item()
    n_total = tensor.numel()
    
    print(f"\nData quality:")
    print(f"  Total pixels: {n_total}")
    print(f"  NaN pixels: {n_nan} ({100*n_nan/n_total:.1f}%)")
    
    if n_nan < n_total:  # Have some valid data
        valid = tensor[~torch.isnan(tensor)]
        print(f"  Value range: [{valid.min():.3f}, {valid.max():.3f}]")
        print(f"  Mean: {valid.mean():.3f}")
    else:
        print("  ⚠️  ALL values are NaN!")
    
    # Try creating a DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, num_workers=0)
    
    batch = next(iter(loader))
    print(f"\nBatch shape: {batch['ndvi'].shape}")  # Should be [4, 1, 256, 256]
    print("✅ Basic dataset works!")
