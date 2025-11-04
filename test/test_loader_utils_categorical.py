#!/usr/bin/env python3
# =============================================================================
# tests/test_loader_utils_categorical.py — Round-trip validation for categorical
# encoding/decoding helpers
#
# Verifies that utils/loader_utils.py correctly:
#   - maps observed categorical codes → dense IDs (encode)
#   - reconstructs dense IDs → raw codes (decode)
#   - preserves NaN / MISS / UNK conventions
#   - handles per-row and batched decoding equivalently
#
# Each test focuses on functional correctness, not performance:
#   - test_forward_inverse_roundtrip_observed: observed codes round-trip
#   - test_miss_and_unk_handling: MISS (NaN) and UNK (unseen codes) behavior
#   - test_batch_decode_vectorized: [T,C_cat] decoding matches row-wise decode
#   - test_inverse_map_contains_only_observed_ids: schema sanity check
#
# Expected invariants:
#   MISS_ID == 0 → np.nan (MISS)
#   UNK_ID  == 1 → np.nan (UNK)
#   Observed codes → 2..N (invertible)
# =============================================================================

import numpy as np
import pytest

from utils.loader_utils import (
    MISS_ID, UNK_ID,
    build_cat_maps,
    encode_categorical_row,
    build_id_maps_inverse,
    decode_categorical_row,
    decode_categorical_batch,
)

@pytest.fixture
def feature_meta():
    # Two categorical features with explicit observed codes
    return {
        "features": [
            {
                "name": "landcover",
                "kind": "cat",
                "classes": [{"code": 11}, {"code": 21}, {"code": 22}, {"code": 23}],
            },
            {
                "name": "ownership",
                "kind": "cat",
                "classes": [{"code": 1}, {"code": 2}],
            },
            # include a continuous feature to ensure indices work with mixed vectors
            {"name": "elevation", "kind": "int", "stats": {"mean": 100, "std": 10}},
        ]
    }

@pytest.fixture
def names():
    # Order must match how the dataset builds feature vectors
    return ["landcover", "ownership", "elevation"]

@pytest.fixture
def cat_indices():
    # positions of categorical features within the full feature vector
    return [0, 1]

def test_forward_inverse_roundtrip_observed(feature_meta, names, cat_indices):
    cat_maps = build_cat_maps(feature_meta)
    inv_maps = build_id_maps_inverse(cat_maps)

    # raw_vec: include a continuous placeholder at index 2 (ignored by categorical funcs)
    raw_vec = np.array([21, 2, 123.0], dtype=np.float32)

    dense = encode_categorical_row(raw_vec, cat_indices, names, cat_maps)
    assert dense.shape == (2,)
    # Both observed → should map to ids >= 2
    assert dense[0] >= 2 and dense[1] >= 2

    # Decode back to raw codes (float output to allow NaN)
    decoded = decode_categorical_row(dense, cat_indices, names, inv_maps)
    # Decoded raw codes should match originals
    np.testing.assert_allclose(decoded, np.array([21.0, 2.0], dtype=np.float32))

def test_miss_and_unk_handling(feature_meta, names, cat_indices):
    cat_maps = build_cat_maps(feature_meta)
    inv_maps = build_id_maps_inverse(cat_maps)

    # MISS: NaN in raw → MISS_ID
    raw_vec_miss = np.array([np.nan, 1.0, 0.0], dtype=np.float32)
    dense_miss = encode_categorical_row(raw_vec_miss, cat_indices, names, cat_maps)
    assert dense_miss[0] == MISS_ID
    # UNK: raw code not in observed set → UNK_ID
    raw_vec_unk = np.array([999.0, 99.0, 0.0], dtype=np.float32)
    dense_unk = encode_categorical_row(raw_vec_unk, cat_indices, names, cat_maps)
    assert dense_unk[0] == UNK_ID or dense_unk[1] == UNK_ID

    # Decode with NaN placeholders (default)
    dec_miss = decode_categorical_row(dense_miss, cat_indices, names, inv_maps)
    dec_unk = decode_categorical_row(dense_unk, cat_indices, names, inv_maps)
    assert np.isnan(dec_miss[0])
    assert np.isnan(dec_unk[0]) or np.isnan(dec_unk[1])

    # Decode with explicit sentinel for UNK/MISS
    dec_miss_s = decode_categorical_row(
        dense_miss, cat_indices, names, inv_maps, miss_value=-1, unk_value=-1
    )
    dec_unk_s = decode_categorical_row(
        dense_unk, cat_indices, names, inv_maps, miss_value=-1, unk_value=-1
    )
    assert dec_miss_s[0] == -1
    assert -1 in dec_unk_s

def test_batch_decode_vectorized(feature_meta, names, cat_indices):
    cat_maps = build_cat_maps(feature_meta)
    inv_maps = build_id_maps_inverse(cat_maps)

    # Build a small batch of raw rows → encode → decode
    raw_rows = np.array([
        [11.0, 1.0, 0.0],     # all observed
        [23.0, np.nan, 0.0],  # MISS in second feature
        [999.0, 2.0, 0.0],    # UNK in first feature
    ], dtype=np.float32)

    dense_rows = np.vstack([
        encode_categorical_row(raw_rows[i], cat_indices, names, cat_maps)
        for i in range(raw_rows.shape[0])
    ])
    assert dense_rows.shape == (3, len(cat_indices))

    decoded_rows = decode_categorical_batch(
        dense_rows, cat_indices, names, inv_maps
    )
    # Row 0 should round-trip exactly
    np.testing.assert_allclose(decoded_rows[0], np.array([11.0, 1.0], dtype=np.float32))
    # Row 1: MISS becomes NaN in position 1
    assert np.isfinite(decoded_rows[1][0])
    assert np.isnan(decoded_rows[1][1])
    # Row 2: UNK becomes NaN in position 0
    assert np.isnan(decoded_rows[2][0])
    assert np.isfinite(decoded_rows[2][1])

def test_inverse_map_contains_only_observed_ids(feature_meta):
    cat_maps = build_cat_maps(feature_meta)
    inv_maps = build_id_maps_inverse(cat_maps)
    for name, inv in inv_maps.items():
        # No entries for MISS/UNK
        assert MISS_ID not in inv
        assert UNK_ID not in inv
        # All dense IDs should be >= 2
        assert all(did >= 2 for did in inv.keys())
