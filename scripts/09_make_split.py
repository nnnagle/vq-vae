import argparse
import numpy as np
import rasterio


def make_split_raster(
    mask_path: str,
    outfile: str,
    chunk_width: int = 256,
    chunk_height: int = 256,
    block_width: int = 7,
    block_height: int = 7,
) -> None:
    """
    Create a test/validation/train split raster based on a block/chunk tiling pattern.

    Rules:
      - br = r // (block_height * chunk_height)
      - bc = c // (block_width * chunk_width)
      - A = (br // 2 + bc // 2) % 2
      - B = (br + bc) % 4
      - Output:
          3 = test       if A == 0 and B == 0
          2 = validation if A == 1 and B == 0
          1 = train      otherwise
      - Outside mask: NaN
    """
    # Read mask raster
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # assume first band is the mask
        profile = src.profile

    nrows, ncols = mask.shape

    # Build row/col indices
    rows, cols = np.indices((nrows, ncols), dtype=np.int64)

    # Block indices
    br = rows // (block_height * chunk_height)
    bc = cols // (block_width * chunk_width)

    # A and B as per spec
    A = (br // 2 + bc // 2) % 2
    B = (br + bc) % 4

    # Initialize output as "train" = 1
    out = np.ones((nrows, ncols), dtype=np.uint8)

    # Test: A == 0 and B == 0  -> 3
    test_mask = (A == 0) & (B == 0)
    out[test_mask] = 3

    # Validation: A == 1 and B == 0 -> 2
    val_mask = (A == 0) & (B == 2)
    out[val_mask] = 2

    NODATA = 0
    # Apply mask: outside mask -> NaN
    # Here I assume "inside mask" means non-zero and not NaN.
    invalid = (mask == 0) | np.isnan(mask)
    #out[invalid] = np.nan
    out[invalid] = NODATA
    
    # Update profile for float output with NaN
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=NODATA,
    )

    # Write output GeoTIFF
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(out, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate test/validation/train raster from a mask using block/chunk tiling."
    )
    parser.add_argument("--mask", required=True, help="Path to mask raster (template).")
    parser.add_argument("--chunk_width", type=int, default=256)
    parser.add_argument("--chunk_height", type=int, default=256)
    parser.add_argument("--block_width", type=int, default=7)
    parser.add_argument("--block_height", type=int, default=7)
    parser.add_argument("--outfile", required=True, help="Output GeoTIFF path (e.g. outfile.tif).")

    args = parser.parse_args()

    make_split_raster(
        mask_path=args.mask,
        outfile=args.outfile,
        chunk_width=args.chunk_width,
        chunk_height=args.chunk_height,
        block_width=args.block_width,
        block_height=args.block_height,
    )


if __name__ == "__main__":
    main()
