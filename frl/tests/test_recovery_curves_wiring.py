"""Wiring smoke test for the Diagnostic-B interaction readouts.

The pure math is covered by ``test_interaction_readouts``. Here we exercise the
*streaming plumbing* of ``_run_recovery_bilinear`` / ``_run_recovery_knn`` — valid
masking, normal-equation accumulation, block-ridge λ sweep, kNN reference build +
bandwidth sweep, and the shared reservoir/shape-agreement tail — on synthetic
data, stubbing the heavy zarr/GDAL extractor + reservoir so it runs without the
HPC stack. It asserts the pipeline runs and returns finite, well-formed metrics.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import torch


def _fake_phase_recovery_curves_module():
    m = types.ModuleType("training.phase_recovery_curves")

    class EvtReservoir:
        def __init__(self, max_per_evt=10_000, seed=0):
            self.d: dict[int, list] = {}

        def add_batch(self, evt, ysfc, pred, obs):
            for e, y, p, o in zip(evt, ysfc, pred, obs):
                self.d.setdefault(int(e), []).append((float(y), float(p), float(o)))

        def n_total(self):
            return sum(len(v) for v in self.d.values())

        def pixel_counts(self):
            return {c: len(v) for c, v in self.d.items()}

        def get(self, code):
            v = self.d.get(int(code))
            return np.array(v, dtype=np.float64) if v else None

    m.EvtReservoir = EvtReservoir
    m.plot_recovery_curves = lambda *a, **k: None
    m.save_csv = lambda *a, **k: None
    return m


def _synthetic_batch(seed, dt=6, zp=4, T=8, n_per=200):
    """A (data, nbr) pair with a genuinely type-conditional NBR: per-EVT baseline
    AND per-EVT slope on the recovery coordinate — so an interaction readout has
    something to find."""
    g = np.random.default_rng(seed)
    codes = [10, 20, 30]
    centers = {10: 0.0, 20: 4.0, 30: -4.0}
    baseline = {10: 0.1, 20: 0.6, 30: -0.5}
    slope = {10: 1.0, 20: -1.0, 30: 0.4}       # sign flips by type ⇒ needs interaction
    N = n_per * len(codes)
    evt = np.repeat(codes, n_per).astype(np.int32)

    zt = np.zeros((N, dt), np.float32)
    zt[:, 0] = np.array([centers[c] for c in evt]) + 0.2 * g.standard_normal(N)
    zt[:, 1:] = 0.3 * g.standard_normal((N, dt - 1))

    ysfc = np.tile(np.arange(T) * 3.0, (N, 1)).astype(np.float32)   # 0..21, in-range
    recov = (ysfc / (ysfc.max() + 1e-6)).astype(np.float32)         # [N, T] 0..1
    zp_arr = np.zeros((N, T, zp), np.float32)
    zp_arr[:, :, 0] = recov + 0.1 * g.standard_normal((N, T))
    zp_arr[:, :, 1:] = 0.2 * g.standard_normal((N, T, zp - 1))

    sl = np.array([slope[c] for c in evt])[:, None]
    bl = np.array([baseline[c] for c in evt])[:, None]
    nbr = (bl + sl * zp_arr[:, :, 0] + 0.05 * g.standard_normal((N, T))).astype(np.float32)

    data = {
        "z_type": torch.from_numpy(zt),
        "z_phase": torch.from_numpy(zp_arr),
        "ysfc": torch.from_numpy(ysfc),
        "evt": evt,
        "valid_tp": torch.ones(N, T, dtype=torch.bool),
    }
    return data, torch.from_numpy(nbr)


def _install(monkeypatch):
    import training.phase_eval.recovery_curves as rc
    sys.modules["training.phase_recovery_curves"] = _fake_phase_recovery_curves_module()
    monkeypatch.setattr(rc, "_extract_for_curves", lambda batch, *a, **k: batch)
    monkeypatch.setattr(rc, "iter_batches", lambda loader, mb: iter(loader))
    monkeypatch.setattr(rc, "_nbr_index", lambda fb: 0)
    return rc


def _ctx(dt=6, zp=4):
    return {"feature_builder": object(), "model": SimpleNamespace(z_type_dim=dt, z_phase_dim=zp)}


def _loader(n_batches=3, dt=6, zp=4):
    return [_synthetic_batch(s, dt=dt, zp=zp) for s in range(n_batches)]


def _shared_kwargs():
    return dict(
        evt_code_to_label={}, top_k_evt=3, halo=0, max_pixels_per_sample=0,
        max_batches=0, max_ysfc=30.0, max_samples_per_evt=10_000,
        min_bin_samples=3, output_dir=None, seed=0,
    )


def test_bilinear_readout_wiring(monkeypatch):
    rc = _install(monkeypatch)
    loader = _loader()
    res = rc._run_recovery_bilinear(
        _ctx(), loader, loader, loader, rank=3, **_shared_kwargs())
    assert res["design"] == "type-phase-bilinear"
    assert res["rank"] == 3
    assert res["lambda_main"] in rc.BILINEAR_LAMBDA_MAIN_GRID
    assert res["lambda_bilinear"] in rc.BILINEAR_LAMBDA_INT_GRID
    sa = res["shape_agreement"]
    assert sa["n_evt_scored"] >= 1
    assert np.isfinite(sa["mean"])


def test_knn_readout_wiring(monkeypatch):
    rc = _install(monkeypatch)
    loader = _loader()
    res = rc._run_recovery_knn(
        _ctx(), loader, loader, loader, ref_cap=500, **_shared_kwargs())
    assert res["design"] == "type-local-knn"
    assert res["sigma_type"] in rc.KNN_SIGMA_TYPE_GRID
    assert res["sigma_phase"] in rc.KNN_SIGMA_PHASE_GRID
    assert 0 < res["n_reference"] <= 500
    sa = res["shape_agreement"]
    assert sa["n_evt_scored"] >= 1
    assert np.isfinite(sa["mean"])


def test_runner_adds_interaction_keys_and_survives_failure(monkeypatch):
    """run_recovery_curves returns the additive designs + both interaction designs;
    a failure in an interaction readout is recorded, not fatal."""
    rc = _install(monkeypatch)
    loader = _loader()
    # force the bilinear readout to raise, to check the guard keeps the rest.
    monkeypatch.setattr(rc, "_run_recovery_bilinear",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = rc.run_recovery_curves(
        _ctx(), loader, loader, loader, evt_code_to_label={}, top_k_evt=3, halo=0,
        max_pixels_per_sample=0, max_batches=0, min_bin_samples=3, output_dir=None,
        knn_ref_cap=500,
    )
    assert set(out) >= {"phase-only", "type-phase", "type-phase-bilinear", "type-local-knn"}
    assert out["type-phase-bilinear"].get("error", "").startswith("RuntimeError")
    assert "shape_agreement" in out["type-local-knn"]
