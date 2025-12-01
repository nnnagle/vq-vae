# tests/test_normalizer_from_zarr.py

import xarray as xr
import numpy as np
from src.data.normalization import build_normalizer_from_zarr

def test_build_normalizer_from_zarr(tmp_path):
    # tiny fake dataset with attrs
    data = np.zeros((1, 1, 4, 4), dtype="float32")
    ds = xr.Dataset(
        {
            "continuous": (("time", "feature_continuous", "y", "x"), data)
        },
        coords={
            "time": [2015],
            "feature_continuous": ["lc_p_trees"],
            "y": np.arange(4),
            "x": np.arange(4),
        },
    )

    ds.attrs["feature_stats"] = {
        "lc_p_trees": {"kind": "continuous", "mean": 10.0, "std": 2.0},
    }
    ds.attrs["feature_normalization"] = {
        "lc_p_trees": {"type": "z-score"},
    }

    zarr_path = tmp_path / "mini.zarr"
    ds.to_zarr(zarr_path, mode="w")

    normalizer = build_normalizer_from_zarr(zarr_path)

    # minimal sanity check: it has the band we expect
    assert "lc_p_trees" in normalizer.shift
    assert "lc_p_trees" in normalizer.scale
