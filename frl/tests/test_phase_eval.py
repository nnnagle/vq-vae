"""Tests for the Step-0 phase-eval harness math (training.phase_eval.*).

These exercise the decoupled, data-free helpers with tiny synthetic tensors — no
zarr, no checkpoint. They lock in the correctness of the within-pixel variance
decomposition, the total/within R² accumulator, the ejection jump+AUC logic, the
recovery-curve shape-agreement metric, and the deferred anomaly-target seam.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from training.phase_eval.common import AnomalyTargetProvider, variance_decompose
from training.phase_eval.ejection import transition_jumps
from training.phase_eval.reconstruction import _R2Accumulator, _Standardizer, _warn_lambda_edge
from training.phase_eval.recovery_curves import _bin_medians, _shape_agreement


# ── variance_decompose ─────────────────────────────────────────────────────

def test_variance_decompose_matches_hand_computation():
    # Pixel 0: [1,2,3] (pop var 2/3, mean 2); Pixel 1: [10,10,10] (var 0, mean 10)
    values = torch.tensor([[[1.0], [2.0], [3.0]], [[10.0], [10.0], [10.0]]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    within, between = variance_decompose(values, valid)
    assert within.item() == pytest.approx((2.0 / 3.0 + 0.0) / 2.0, rel=1e-6)
    # between = pop var of pixel means [2, 10] = 16
    assert between.item() == pytest.approx(16.0, rel=1e-6)


def test_variance_decompose_respects_validity_mask():
    values = torch.tensor([[[1.0], [2.0], [999.0]]])   # third timestep invalid
    valid = torch.tensor([[True, True, False]])
    within, _ = variance_decompose(values, valid)
    # only [1,2] counts → mean 1.5, pop var 0.25
    assert within.item() == pytest.approx(0.25, rel=1e-6)


# ── _R2Accumulator ─────────────────────────────────────────────────────────

def test_r2_accumulator_perfect_prediction():
    N, T, C = 4, 5, 2
    y = torch.randn(N, T, C)
    valid = torch.ones(N, T, dtype=torch.bool)
    acc = _R2Accumulator(C)
    acc.update(y.clone(), y.clone(), valid)
    res = acc.result([f"x{c}" for c in range(C)])
    assert res["r2_total_mean"] == pytest.approx(1.0, abs=1e-9)
    assert res["r2_within_mean"] == pytest.approx(1.0, abs=1e-9)


def test_r2_accumulator_mean_only_prediction_has_zero_within_r2():
    # Predicting each pixel's temporal mean → within-pixel anomaly all zero →
    # within-R² = 0 (explains none of the within-pixel variance).
    N, T, C = 6, 4, 1
    y = torch.randn(N, T, C)
    valid = torch.ones(N, T, dtype=torch.bool)
    pred = y.mean(dim=1, keepdim=True).expand(N, T, C).contiguous()
    acc = _R2Accumulator(C)
    acc.update(pred, y, valid)
    res = acc.result(["x0"])
    assert res["r2_within_mean"] == pytest.approx(0.0, abs=1e-6)


def test_r2_accumulator_weighted_ignores_dead_low_variance_channel():
    # The exact pathology the weighted aggregate guards against: channel x0 has
    # large within-pixel variance and is predicted perfectly (within-R²=1); channel
    # x1 has tiny within-pixel variance and is predicted anti-correlated
    # (within-R²=-3). The unweighted mean is dragged to -1, but the
    # variance-weighted aggregate stays ~1 because x1 has almost no variance to
    # get wrong. within_variance/within_mse expose x1 as numerically negligible.
    N, T = 4, 3
    anom0 = torch.tensor([-10.0, 0.0, 10.0])       # large within-pixel signal
    anom1 = torch.tensor([-0.01, 0.0, 0.01])       # near-dead within-pixel signal
    means = torch.arange(N).float().view(N, 1)     # distinct per-pixel levels
    y = torch.zeros(N, T, 2)
    y[:, :, 0] = means + anom0.view(1, T)
    y[:, :, 1] = means * 0.1 + anom1.view(1, T)
    pred = y.clone()
    pmean1 = y[:, :, 1].mean(dim=1, keepdim=True)
    pred[:, :, 1] = 2.0 * pmean1 - y[:, :, 1]      # flip x1 anomaly → R² = -3
    valid = torch.ones(N, T, dtype=torch.bool)
    acc = _R2Accumulator(2)
    acc.update(pred, y, valid)
    res = acc.result(["x0", "x1"])

    assert res["r2_within_per_channel"]["x0"] == pytest.approx(1.0, abs=1e-9)
    assert res["r2_within_per_channel"]["x1"] == pytest.approx(-3.0, abs=1e-6)
    assert res["r2_within_mean"] == pytest.approx(-1.0, abs=1e-6)   # unweighted: misleading
    assert res["r2_within_weighted"] > 0.99                         # weighted: honest
    # the per-channel variance column flags x1 as the negligible one
    wv = res["within_variance_per_channel"]
    assert wv["x0"] > 1e3 * wv["x1"]
    assert res["within_mse_per_channel"]["x0"] == pytest.approx(0.0, abs=1e-9)
    assert res["n_within_observations"] == N * T


# ── _warn_lambda_edge ──────────────────────────────────────────────────────

_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def test_lambda_edge_interior_pick_is_silent(caplog):
    with caplog.at_level("WARNING", logger="phase_eval.reconstruction"):
        _warn_lambda_edge(1e-2, _GRID, "A:test", scores=[0.1, 0.2, 0.3, 0.25, 0.2])
    assert not caplog.records


def test_lambda_edge_min_pick_warns(caplog):
    with caplog.at_level("WARNING", logger="phase_eval.reconstruction"):
        _warn_lambda_edge(1e-4, _GRID, "A:test", scores=[0.3, 0.25, 0.2, 0.1, 0.0])
    assert len(caplog.records) == 1
    assert "MINIMUM" in caplog.text


def test_lambda_edge_max_pick_warns(caplog):
    with caplog.at_level("WARNING", logger="phase_eval.reconstruction"):
        _warn_lambda_edge(1.0, _GRID, "A:test", scores=[0.0, 0.1, 0.2, 0.25, 0.3])
    assert len(caplog.records) == 1
    assert "MAXIMUM" in caplog.text


def test_lambda_edge_flat_sweep_is_silent(caplog):
    # z_type case: within-R² identically 0 across λ → boundary pick is arbitrary,
    # not a clamp; the guard must not false-alarm.
    with caplog.at_level("WARNING", logger="phase_eval.reconstruction"):
        _warn_lambda_edge(1e-4, _GRID, "A:z_type", scores=[0.0, 0.0, 0.0, 0.0, 0.0])
    assert not caplog.records


# ── _Standardizer ──────────────────────────────────────────────────────────

def test_standardizer_produces_unit_scale():
    torch.manual_seed(0)
    feat = torch.randn(1000, 3) * torch.tensor([2.0, 5.0, 0.1]) + 7.0
    std = _Standardizer(3)
    std.update(feat)
    std.finalize()
    out = std.apply(feat)
    assert torch.allclose(out.mean(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(out.std(dim=0, correction=0), torch.ones(3), atol=1e-2)


# ── ejection: transition_jumps + AUC ───────────────────────────────────────

def test_transition_jumps_separates_disturbance():
    # 1 pixel, T=4: big move into the ysfc=0 year, tiny moves otherwise.
    z = torch.tensor([[[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]])  # [1,4,2]
    ysfc = torch.tensor([[3.0, 4.0, 0.0, 1.0]])   # disturbance at t=2
    valid = torch.ones(1, 4, dtype=torch.bool)
    jumps, labels = transition_jumps(z, ysfc, valid)
    # transitions t=1,2,3 (t=0 has no predecessor) → 3 kept
    assert jumps.shape == (3,)
    assert labels.tolist() == [0.0, 1.0, 0.0]
    # the disturbance transition (into t=2) is the largest jump
    assert jumps[1] == jumps.max()
    assert roc_auc_score(labels, jumps) == pytest.approx(1.0)


def test_transition_jumps_skips_invalid_pairs():
    z = torch.zeros(1, 3, 2)
    ysfc = torch.tensor([[1.0, 0.0, 2.0]])
    valid = torch.tensor([[True, False, True]])   # t=1 invalid → both pairs dropped
    jumps, labels = transition_jumps(z, ysfc, valid)
    assert jumps.size == 0


# ── recovery curves: shape agreement ───────────────────────────────────────

def test_bin_medians_respects_min_n():
    ysfc = np.array([0.0, 0.0, 5.0])
    vals = np.array([1.0, 3.0, 9.0])
    out = _bin_medians(ysfc, vals, min_n=2)
    assert out[0] == pytest.approx(2.0)   # bin [0,1): two samples → median 2
    assert np.isnan(out[4])               # bin [5,8): one sample → NaN


def test_shape_agreement_high_when_curves_track():
    # Build a reservoir where predicted and observed both rise with ysfc.
    # EvtReservoir lives in the plotting/GDAL-bound module; skip if unavailable.
    EvtReservoir = pytest.importorskip("training.phase_recovery_curves").EvtReservoir
    res = EvtReservoir(max_per_evt=10_000, seed=0)
    rng = np.random.default_rng(0)
    ys, preds, obs = [], [], []
    for (lo, hi) in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8)]:
        for _ in range(50):
            y = rng.uniform(lo, hi - 1e-3)
            ys.append(y); preds.append(y + rng.normal(0, 0.01)); obs.append(y + rng.normal(0, 0.01))
    res.add_batch(
        np.full(len(ys), 100, dtype=np.int32),
        np.array(ys, np.float32), np.array(preds, np.float32), np.array(obs, np.float32),
    )
    out = _shape_agreement(res, [100], min_n=5)
    assert out["n_evt_scored"] == 1
    assert out["mean"] > 0.9


# ── deferred anomaly target seam ───────────────────────────────────────────

def test_anomaly_provider_raw_is_identity():
    x = torch.randn(4, 3, 5)
    z = torch.randn(4, 8)
    assert torch.equal(AnomalyTargetProvider("raw")(x, z), x)


def test_anomaly_provider_mature_baseline_is_deferred():
    with pytest.raises(NotImplementedError):
        AnomalyTargetProvider("mature_baseline")(torch.zeros(1, 1, 1), torch.zeros(1, 1))


def test_anomaly_provider_rejects_unknown_kind():
    with pytest.raises(ValueError):
        AnomalyTargetProvider("bogus")
