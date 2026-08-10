"""Regression test for the feature-normalization transform bug.

``FeatureBuilder._apply_normalization`` used to apply a channel's log/sqrt
transform into ``normalized_data[c_idx]`` and then z-score the **raw**
``data[c_idx]`` — discarding the transform and normalizing raw values against
transform-scale stats. For log-transformed channels (e.g.
``phase_ccdc.spectral_velocity``) that pushed every value past the clamp,
collapsing the channel to a constant ("dead channel"). This locks in the fix:
the transform must be honored so a log channel keeps its variance.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_apply_normalization_honors_log_transform():
    # Needs the data-loader stack (GDAL/zarr-free import of the builder module).
    fb_mod = pytest.importorskip("data.loaders.builders.feature_builder")
    FeatureBuilder = fb_mod.FeatureBuilder

    # Bypass __init__ (which would load a stats file / zarr) and inject minimal state.
    fb = object.__new__(FeatureBuilder)
    # Stats are on the LOG scale (computed on log-transformed data), like the real
    # spectral_velocity entry (mean -5.95, sd 0.78).
    fb.stats = {"feat": {"g.sv": {"mean": -5.95, "sd": 0.78}}}
    preset = SimpleNamespace(type="zscore",
                             clamp={"enabled": True, "min": -6.0, "max": 6.0})
    fb.config = SimpleNamespace(get_normalization_preset=lambda name: preset)
    feature_config = SimpleNamespace(channels={
        "g.sv": SimpleNamespace(transform={"name": "log", "epsilon": 0.001},
                                norm="zscore"),
    })

    raw = np.array([[[0.0, 0.001, 0.1, 1.0, 1.9]]], dtype=np.float64)  # [C=1, H=1, W=5]
    mask = np.ones(raw.shape[1:], dtype=bool)

    out = fb._apply_normalization(raw, "feat", feature_config, mask)[0].ravel()

    # Correct behavior: transform THEN normalize -> (log(x+eps) - mean)/sd, clamped.
    expected = np.clip((np.log(raw.ravel() + 0.001) + 5.95) / 0.78, -6.0, 6.0)
    assert np.allclose(out, expected, atol=1e-4)

    # It must NOT match the pre-fix raw-normalized collapse (all pinned at +6).
    buggy = np.clip((raw.ravel() + 5.95) / 0.78, -6.0, 6.0)
    assert out.std() > 0.5                       # real variance, not collapsed
    assert abs(out.min() - (-1.2279)) < 1e-2     # log floor, not the +6 clamp
    assert not np.allclose(out, buggy)           # distinct from the buggy path


def test_apply_normalization_untransformed_channel_unchanged():
    """Channels without a transform must be unaffected by the fix (raw z-score)."""
    fb_mod = pytest.importorskip("data.loaders.builders.feature_builder")
    FeatureBuilder = fb_mod.FeatureBuilder

    fb = object.__new__(FeatureBuilder)
    fb.stats = {"feat": {"g.plain": {"mean": 0.5, "sd": 2.0}}}
    preset = SimpleNamespace(type="zscore", clamp={"enabled": False})
    fb.config = SimpleNamespace(get_normalization_preset=lambda name: preset)
    feature_config = SimpleNamespace(channels={
        "g.plain": SimpleNamespace(transform=None, norm="zscore"),
    })

    raw = np.array([[[-1.0, 0.0, 0.5, 1.0, 3.0]]], dtype=np.float64)
    mask = np.ones(raw.shape[1:], dtype=bool)
    out = fb._apply_normalization(raw, "feat", feature_config, mask)[0].ravel()
    assert np.allclose(out, (raw.ravel() - 0.5) / 2.0, atol=1e-6)
