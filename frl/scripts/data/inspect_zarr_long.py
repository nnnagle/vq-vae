#!/usr/bin/env python3
"""
Inspect Zarr structure and write detailed output to a file.
"""
import zarr
import sys
from pathlib import Path

def inspect_zarr(zarr_path, output_file='zarr_structure.txt'):
    """Inspect Zarr store and write structure to file."""
    
    with open(output_file, 'w') as f:
        def write(msg):
            print(msg)  # Also print to console
            f.write(msg + '\n')
        
        write(f"Inspecting Zarr: {zarr_path}")
        write("=" * 80)
        
        try:
            z = zarr.open(zarr_path, mode='r')
            
            # Top-level groups
            write("\n[1] TOP-LEVEL GROUPS:")
            write("-" * 80)
            top_keys = list(z.keys())
            write(f"Found {len(top_keys)} top-level groups: {top_keys}")
            
            # Full tree structure (limit depth to avoid huge output)
            write("\n[2] TREE STRUCTURE (depth=3):")
            write("-" * 80)
            tree_str = str(z.tree(level=3))
            write(tree_str)
            
            # Detailed inspection of key groups
            write("\n[3] DETAILED GROUP INSPECTION:")
            write("-" * 80)
            
            # Check annual group
            if 'annual' in z:
                write("\n--- Annual Group ---")
                annual = z['annual']
                write(f"  Subgroups: {list(annual.keys())}")
                
                # Check ls8day specifically
                if 'ls8day' in annual and 'data' in annual['ls8day']:
                    write("\n  LS8DAY Arrays:")
                    ls8_data = annual['ls8day/data']
                    for arr_name in sorted(ls8_data.keys())[:5]:  # First 5 arrays
                        arr = ls8_data[arr_name]
                        write(f"    {arr_name}:")
                        write(f"      shape: {arr.shape}")
                        write(f"      dtype: {arr.dtype}")
                        write(f"      chunks: {arr.chunks}")
                        
                        # Check for statistics in attributes
                        if arr.attrs:
                            write(f"      attrs: {dict(arr.attrs)}")
                    
                    if len(ls8_data.keys()) > 5:
                        write(f"    ... and {len(ls8_data.keys()) - 5} more arrays")
            
            # Check static group
            if 'static' in z:
                write("\n--- Static Group ---")
                static = z['static']
                write(f"  Subgroups: {list(static.keys())}")
                
                # Check topo
                if 'topo' in static and 'data' in static['topo']:
                    write("\n  TOPO Arrays:")
                    topo_data = static['topo/data']
                    for arr_name in sorted(topo_data.keys())[:3]:  # First 3
                        arr = topo_data[arr_name]
                        write(f"    {arr_name}:")
                        write(f"      shape: {arr.shape}")
                        write(f"      dtype: {arr.dtype}")
                        if arr.attrs:
                            write(f"      attrs: {dict(arr.attrs)}")
                
                # Check CCDC snapshot arrays
                if 'ccdc_metrics_current' in static and 'data' in static['ccdc_metrics_current']:
                    write("\n  CCDC Snapshot Arrays (first 5):")
                    ccdc_data = static['ccdc_metrics_current/data']
                    for arr_name in sorted(ccdc_data.keys())[:5]:
                        arr = ccdc_data[arr_name]
                        write(f"    {arr_name}:")
                        write(f"      shape: {arr.shape}")
                        write(f"      dtype: {arr.dtype}")
                    
                    if len(ccdc_data.keys()) > 5:
                        write(f"    ... and {len(ccdc_data.keys()) - 5} more arrays")
            
            # Check irregular group (NAIP)
            if 'irregular' in z:
                write("\n--- Irregular Group ---")
                irregular = z['irregular']
                write(f"  Subgroups: {list(irregular.keys())}")
                
                if 'naip' in irregular and 'data' in irregular['naip']:
                    write("\n  NAIP Arrays:")
                    naip_data = irregular['naip/data']
                    for arr_name in sorted(naip_data.keys()):
                        arr = naip_data[arr_name]
                        write(f"    {arr_name}:")
                        write(f"      shape: {arr.shape}")
                        write(f"      dtype: {arr.dtype}")
            
            # Check AOI
            if 'aoi' in z:
                write("\n--- AOI Group ---")
                aoi = z['aoi']
                if 'aoi' in aoi:
                    arr = aoi['aoi']
                    write(f"  AOI array:")
                    write(f"    shape: {arr.shape}")
                    write(f"    dtype: {arr.dtype}")
            
            # Sample one array for detailed stats inspection
            write("\n[4] SAMPLE ARRAY - DETAILED INSPECTION:")
            write("-" * 80)
            if 'annual' in z and 'ls8day' in z['annual'] and 'data' in z['annual/ls8day']:
                sample_arr = z['annual/ls8day/data/NDVI_summer_p95']
                write(f"\nArray: annual/ls8day/data/NDVI_summer_p95")
                write(f"  Shape: {sample_arr.shape}")
                write(f"  Dtype: {sample_arr.dtype}")
                write(f"  Chunks: {sample_arr.chunks}")
                write(f"\n  All attributes:")
                for key, val in sample_arr.attrs.items():
                    write(f"    {key}: {val}")
                
                # Try to access stats if they exist
                write(f"\n  Looking for statistics...")
                if 'mean' in sample_arr.attrs:
                    write(f"    ✓ Found stats in array attributes")
                elif 'stats' in z['annual/ls8day']:
                    write(f"    ✓ Found separate stats group")
                    stats_group = z['annual/ls8day/stats']
                    write(f"    Stats arrays: {list(stats_group.keys())}")
                else:
                    write(f"    ✗ Stats location unclear - please check manually")
            
            write("\n[5] DIMENSIONS SUMMARY:")
            write("-" * 80)
            # Infer time dimension
            if 'annual' in z and 'ls8day' in z['annual'] and 'data' in z['annual/ls8day']:
                arr = z['annual/ls8day/data/NDVI_summer_p95']
                if len(arr.shape) == 3:
                    T, H, W = arr.shape
                    write(f"  Temporal arrays: [T={T}, H={H}, W={W}]")
                    write(f"  Assuming years: 1985-{1985+T-1} (if T={T})")
                elif len(arr.shape) == 2:
                    H, W = arr.shape
                    write(f"  Static arrays: [H={H}, W={W}]")
            
            write("\n" + "=" * 80)
            write(f"Inspection complete! Output written to: {output_file}")
            
        except Exception as e:
            write(f"\nERROR: {str(e)}")
            write(f"Full traceback:")
            import traceback
            write(traceback.format_exc())
            return False
        
        return True

if __name__ == "__main__":
    zarr_path = "/data/VA/zarr/va_vae_dataset_test.zarr"
    output_file = "zarr_structure.txt"
    
    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Running inspection...")
    print(f"  Zarr path: {zarr_path}")
    print(f"  Output file: {output_file}")
    print()
    
    success = inspect_zarr(zarr_path, output_file)
    
    if success:
        print(f"\n✓ Success! Check {output_file} for results")
    else:
        print(f"\n✗ Failed - check {output_file} for error details")
        sys.exit(1)
