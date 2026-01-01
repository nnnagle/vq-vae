# scripts/data/inspect_zarr.py
"""Quick script to understand your Zarr structure"""
import zarr
import numpy as np

store = zarr.open('/data/VA/zarr/va_vae_dataset_test.zarr', mode='r')

def print_zarr_tree(group, indent=0):
    """Recursively print Zarr structure"""
    for key in group.keys():
        item = group[key]
        prefix = "  " * indent
        
        if isinstance(item, zarr.Group):
            print(f"{prefix}📁 {key}/")
            print_zarr_tree(item, indent + 1)
        else:
            print(f"{prefix}📄 {key}")
            print(f"{prefix}   Shape: {item.shape}")
            print(f"{prefix}   Dtype: {item.dtype}")
            print(f"{prefix}   Chunks: {item.chunks}")
            
            # Check for attributes (stats)
            if item.attrs:
                print(f"{prefix}   Attrs: {list(item.attrs.keys())}")
            print()

print_zarr_tree(store)
