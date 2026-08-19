"""Wiring smoke test for the Diagnostic-A bilinear (type × phase) source.

Pure math is covered by ``test_interaction_readouts``. Here we exercise the A
streaming path — 3-pass fit (standardizer, whitened-PCA P, block normal
equations), the 2-D (λ_main, λ_bilinear) val sweep, and the within/total-R²
accumulator — on synthetic data with a genuinely type-conditional target (a
per-type GAIN on the recovery coordinate), stubbing the zarr extractor/target.
The additive z_phase probe could only fit one global gain; the bilinear one
recovers the type-varying gains, so within-R² should be high.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def _batch(seed, dt=3, zp=4, T=8, n=800, C=3):
    """Target is a GENUINE bilinear: a per-pixel gain LINEAR in z_type modulates the
    recovery coordinate. ``dt=3`` with ``rank=3`` (full rank) so the whitened-PCA
    projection is invertible and the interaction can represent the gain exactly —
    this isolates the *plumbing* (3-pass fit, block ridge, within-R²) from the
    rank-truncation modeling choice (which, on near-isotropic standardized z_type,
    picks variance-driven directions and would under-fit an arbitrary gain)."""
    g = np.random.default_rng(seed)
    # The type→gain and type→baseline maps are FIXED across batches (a single
    # consistent bilinear relationship); only the pixels/noise vary per batch.
    fixed = np.random.default_rng(12345)
    G = fixed.standard_normal((dt, C)).astype(np.float32)    # gain     = z_type @ G
    Bm = fixed.standard_normal((dt, C)).astype(np.float32)   # baseline = z_type @ Bm
    zt = g.standard_normal((n, dt)).astype(np.float32)
    recov = np.linspace(0, 1, T)[None, :] + 0.05 * g.standard_normal((n, T))   # within-pixel signal
    zp_arr = np.zeros((n, T, zp), np.float32)
    zp_arr[:, :, 0] = recov
    zp_arr[:, :, 1:] = 0.2 * g.standard_normal((n, T, zp - 1))
    gain = zt @ G                                             # [n, C], linear in z_type
    base = zt @ Bm                                           # [n, C]
    x = (base[:, None, :] + gain[:, None, :] * zp_arr[:, :, 0:1]
         + 0.03 * g.standard_normal((n, T, C))).astype(np.float32)
    return {
        "z_type": torch.from_numpy(zt),
        "z_phase": torch.from_numpy(zp_arr),
        "valid_tp": torch.ones(n, T, dtype=torch.bool),
        "_target": torch.from_numpy(x),
    }


def _install(monkeypatch):
    import training.phase_eval.reconstruction as rc
    monkeypatch.setattr(rc, "iter_batches", lambda loader, mb: iter(loader))
    monkeypatch.setattr(rc, "extract_pixel_series", lambda batch, *a, **k: batch)
    monkeypatch.setattr(rc, "_target_series", lambda data, provider, keep_idx: data["_target"])
    return rc


def test_bilinear_source_A_wiring(monkeypatch):
    rc = _install(monkeypatch)
    loader = [_batch(s) for s in range(3)]
    ctx = {"model": SimpleNamespace(z_type_dim=3, z_phase_dim=4)}
    res = rc._run_bilinear_source(
        ctx, loader, loader, loader, provider=None, keep_idx=None,
        kept_names=["c0", "c1", "c2"], kind="raw", halo=0, max_pixels=0,
        max_batches=0, rank=3,
    )
    assert res["rank"] == 3
    assert res["lambda_main"] in rc.BILINEAR_LAMBDA_MAIN_GRID
    assert res["lambda_bilinear"] in rc.BILINEAR_LAMBDA_INT_GRID
    rt = res["ridge_test"]
    assert rt["n_observations"] > 0
    assert np.isfinite(rt["r2_within_weighted"])
    # target is genuinely type-conditional (per-type gain) ⇒ the interaction
    # recovers the within-pixel signal well
    assert rt["r2_within_weighted"] > 0.5


def test_bilinear_in_default_sources(monkeypatch):
    import training.phase_eval.reconstruction as rc
    assert rc.BILINEAR_SOURCE in rc.FEATURE_SOURCES
