#!/usr/bin/env python3
# =============================================================================
# tests/test_loader_utils_continuous.py — Round-trip validation for continuous
# normalization/denormalization helpers
#
# Verifies that utils/loader_utils.py correctly:
#   - clips raw values to [q01, q99] and applies z-score (norm_continuous_row)
#   - inverts the z-score step back to original units (denorm_continuous_row)
#   - preserves NaN signaling via a mask (NaNs -> z=0 with mask=1)
#   - handles degenerate stats (std=0) safely
#
# Invariants tested:
#   - For finite x: z = (clip(x, q01, q99) - mean) / (std + 1e-6)
#   - Inverse: denorm(z) ≈ clip(x, q01, q99) (unclipped reconstruction)
#   - NaN input: z -> 0.0, nan_mask -> 1.0; denorm(0) -> mean
#   - std=0: z -> 0.0; denorm(0) -> mean
# =============================================================================

import numpy as np
import pytest

from utils.loader_utils import (
    extract_cont_stats,
    norm_continuous_row,
    denorm_continuous_row,
)

@pytest.fixture
def feature_meta():
    # Two continuous features with distinct stats
    # f0: well-behaved
    # f1: tiny std to test numerical stability
    return {
        "features": [
            {"name": "f0", "kind": "int",
             "stats": {"mean": 100.0, "std": 10.0, "min": 0.0, "max": 255.0, "q01": 20.0, "q99": 200.0}},
            {"name": "f1", "kind": "int",
             "stats": {"mean": 5.0, "std": 0.0, "min": 0.0, "max": 10.0, "q01": 1.0, "q99": 9.0}},
            # include a categorical in meta to ensure extract_cont_stats filters correctly
            {"name": "cat", "kind": "cat", "classes": [{"code": 1}, {"code": 2}]},
        ]
    }

@pytest.fixture
def names():
    # Order must correspond to how the row is laid out
    return ["f0", "f1", "cat"]

@pytest.fixture
def cont_indices():
    # positions of continuous features in the row
    return [0, 1]

def _expected_z(x, lo, hi, mu, sd):
    # mirrors implementation: clip then z-score with 1e-6 guard in denominator
    xc = np.clip(x, lo, hi) if (lo is not None and hi is not None) else x
    if sd is None or sd == 0:
        return 0.0
    return (xc - mu) / (sd + 1e-6)

def test_norm_clip_and_zscore(feature_meta, names, cont_indices):
    cont_stats = extract_cont_stats(feature_meta)
    # raw row: f0 is above q99 -> should clip; f1 is below q01 -> clip, but std=0
    raw = np.array([250.0, 0.5, 2.0], dtype=np.float32)

    z, nan_mask = norm_continuous_row(raw, cont_indices, names, cont_stats)
    assert z.shape == (2,)
    assert nan_mask.shape == (2,)

    # Expected for f0
    z0_exp = _expected_z(250.0, 20.0, 200.0, 100.0, 10.0)
    np.testing.assert_allclose(z[0], z0_exp, rtol=0, atol=1e-6)
    # f1 has sd=0 -> z should be 0
    assert z[1] == 0.0

    # No NaNs in inputs
    np.testing.assert_array_equal(nan_mask, np.array([0.0, 0.0], dtype=np.float32))

def test_denorm_inverse_matches_clipped(feature_meta, names, cont_indices):
    cont_stats = extract_cont_stats(feature_meta)
    # raw row: f0 below q01, f1 above q99 (std=0)
    raw = np.array([10.0, 15.0, 0.0], dtype=np.float32)

    z, _ = norm_continuous_row(raw, cont_indices, names, cont_stats)
    x_rec = denorm_continuous_row(z, cont_indices, names, cont_stats)

    # Expect reconstruction equals clipped value (unclipped inverse of z-score)
    # f0: clip to 20, then x_rec ≈ 20
    assert np.isclose(x_rec[0], 20.0, atol=1e-4)
    # f1: std=0 -> z=0 -> denorm -> mean
    assert np.isclose(x_rec[1], 5.0, atol=1e-6)

def test_nan_handling_roundtrip(feature_meta, names, cont_indices):
    cont_stats = extract_cont_stats(feature_meta)
    # raw row: f0 NaN, f1 finite
    raw = np.array([np.nan, 7.0, 0.0], dtype=np.float32)

    z, nan_mask = norm_continuous_row(raw, cont_indices, names, cont_stats)
    # NaN -> z=0 and mask=1 for f0
    assert z[0] == 0.0
    assert nan_mask[0] == 1.0
    # f1 finite -> mask=0
    assert nan_mask[1] == 0.0

    # Denorm(0) -> mean for both features
    x_rec = denorm_continuous_row(z, cont_indices, names, cont_stats)
    assert np.isclose(x_rec[0], 100.0, atol=1e-6)  # mean of f0
    assert np.isclose(x_rec[1], 5.0, atol=1e-6)    # mean of f1

def test_vector_shapes_and_types(feature_meta, names, cont_indices):
    cont_stats = extract_cont_stats(feature_meta)
    raw = np.array([150.0, 6.0, 0.0], dtype=np.float32)

    z, mask = norm_continuous_row(raw, cont_indices, names, cont_stats)
    assert z.dtype == np.float32 and mask.dtype == np.float32
    assert z.shape == (len(cont_indices),)
    x_rec = denorm_continuous_row(z, cont_indices, names, cont_stats)
    assert x_rec.dtype == np.float32 and x_rec.shape == (len(cont_indices),)
