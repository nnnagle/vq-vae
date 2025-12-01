import torch
import pytest
from src.data.normalization import build_normalizer_from_meta


def test_zscore_normalizer_basic():
    # Fake stats for two continuous features
    feature_stats = {
        "lc_p_trees": {
            "kind": "continuous",
            "mean": 10.0,
            "std": 2.0,
        },
        "lcms_ysfc": {
            "kind": "continuous",
            "mean": 40.0,
            "std": 5.0,
        },
    }

    # YAML → parsed → something like this
    feature_norm_cfg = {
        "lc_p_trees": {"type": "z-score"},
        "lcms_ysfc": {"type": "z-score"},
    }

    feature_ids = ["lc_p_trees", "lcms_ysfc"]

    normalizer = build_normalizer_from_meta(
        feature_ids=feature_ids,
        stats_all=feature_stats,
        norm_cfg_all=feature_norm_cfg,
    )

    # Build a toy tensor: [T=1, C=2, H=2, W=2]
    x = torch.tensor(
        [
            [  # time=0
                [[10.0, 12.0],
                 [8.0, 10.0]],   # lc_p_trees
                [[40.0, 35.0],
                 [45.0, 40.0]],  # lcms_ysfc
            ]
        ]
    )  # shape [1, 2, 2, 2]

    # Use the normalizer as a callable
    x_norm = normalizer(x)

    assert x_norm.shape == x.shape

    # Flatten over T, H, W
    x0 = x_norm[:, 0].flatten()
    x1 = x_norm[:, 1].flatten()

    # Means should be ~0 given stats mean used for centering
    assert abs(x0.mean().item()) < 1e-5
    assert abs(x1.mean().item()) < 1e-5

    # Std should be ~1, but sample is small so allow some slack
    assert 0.8 < x0.std().item() < 1.2
    assert 0.8 < x1.std().item() < 1.2


def test_zscore_handles_zero_std():
    feature_stats = {
        "flat_band": {
            "kind": "continuous",
            "mean": 5.0,
            "std": 0.0,
        }
    }
    feature_norm_cfg = {"flat_band": {"type": "z-score"}}
    feature_ids = ["flat_band"]

    normalizer = build_normalizer_from_meta(
        feature_ids=feature_ids,
        stats_all=feature_stats,
        norm_cfg_all=feature_norm_cfg,
    )

    x = torch.full((1, 1, 2, 2), 5.0)
    x_norm = normalizer(x)

    # Everything identical → zero after normalization
    assert torch.allclose(x_norm, torch.zeros_like(x_norm))


def test_unknown_normalization_type_raises():
    feature_stats = {
        "weird": {
            "kind": "continuous",
            "mean": 0.0,
            "std": 1.0,
        }
    }
    feature_norm_cfg = {"weird": {"type": "alien-scaling"}}
    feature_ids = ["weird"]

    with pytest.raises(ValueError):
        build_normalizer_from_meta(
            feature_ids=feature_ids,
            stats_all=feature_stats,
            norm_cfg_all=feature_norm_cfg,
        )
