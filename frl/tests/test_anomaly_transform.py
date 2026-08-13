"""Tests for models.anomaly_transform — the Step-2 anomaly input transform."""

from __future__ import annotations

import pytest
import torch

from models.anomaly_transform import AnomalyTransform


def _mu_sigma(N, C, mu_val=0.0, sig_val=1.0):
    return torch.full((N, C), mu_val), torch.full((N, C), sig_val)


class TestBasics:
    def test_shapes_and_channel_layout(self):
        t = AnomalyTransform(n_channels=5)
        x = torch.randn(8, 5, 12)
        mu, sigma = _mu_sigma(8, 5)
        feats, valid = t(x, mu, sigma)
        assert feats.shape == (8, 10, 12)          # [a ; Δa] = 2C
        assert t.out_channels == 10
        assert valid.shape == (8, 12)
        assert valid.all()                          # all finite → all valid

    def test_grad_free(self):
        t = AnomalyTransform(n_channels=3)
        x = torch.randn(4, 3, 6, requires_grad=True)
        mu, sigma = _mu_sigma(4, 3)
        feats, _ = t(x, mu, sigma)
        assert not feats.requires_grad

    def test_delta_only_is_enforced(self):
        with pytest.raises(NotImplementedError):
            AnomalyTransform(n_channels=3, delta_only=False)


class TestAnomalyAndDelta:
    def test_mature_maps_near_zero(self):
        """x ≈ μ (mature) ⇒ a ≈ 0."""
        t = AnomalyTransform(n_channels=4)
        mu, sigma = _mu_sigma(6, 4, mu_val=2.0, sig_val=0.5)
        x = 2.0 + 0.01 * torch.randn(6, 4, 10)     # right at the baseline
        feats, _ = t(x, mu, sigma)
        a = feats[:, :4, :]
        assert a.abs().max() < 0.1

    def test_anomaly_matches_definition(self):
        t = AnomalyTransform(n_channels=3)
        x = torch.randn(5, 3, 8)
        mu, sigma = torch.randn(5, 3), torch.rand(5, 3) + 0.5
        feats, _ = t(x, mu, sigma)
        a = feats[:, :3, :]
        expected = (x - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        assert torch.allclose(a, expected, atol=1e-6)

    def test_delta_is_temporal_difference(self):
        """Δa[t] = a[t] − a[t−1]; Δa[0] = 0 (no predecessor)."""
        t = AnomalyTransform(n_channels=2)
        x = torch.randn(4, 2, 9)
        mu, sigma = _mu_sigma(4, 2)
        feats, _ = t(x, mu, sigma)
        a, da = feats[:, :2, :], feats[:, 2:, :]        # scale=1 (unlocked)
        assert torch.allclose(da[:, :, 0], torch.zeros_like(da[:, :, 0]))
        assert torch.allclose(da[:, :, 1:], a[:, :, 1:] - a[:, :, :-1], atol=1e-6)

    def test_fast_vs_slow_disturbance(self):
        """Fast ejecta → large |Δa|; slow ramp → small |Δa| but sustained a."""
        t = AnomalyTransform(n_channels=1)
        T = 10
        fast = torch.zeros(1, 1, T); fast[0, 0, 5:] = 4.0        # step at t=5
        slow = torch.zeros(1, 1, T)
        slow[0, 0, :] = torch.linspace(0, 4, T)                 # gradual ramp
        mu, sigma = _mu_sigma(1, 1)
        fa, _ = t(fast, mu, sigma)
        sa, _ = t(slow, mu, sigma)
        fast_dmax = fa[:, 1:, :].abs().max()
        slow_dmax = sa[:, 1:, :].abs().max()
        assert fast_dmax > 3 * slow_dmax                        # abruptness separates
        # both reach a sustained non-zero anomaly at the end
        assert fa[0, 0, -1] > 3 and sa[0, 0, -1] > 3


class TestMasking:
    def test_invalid_timesteps_zeroed(self):
        t = AnomalyTransform(n_channels=2)
        x = torch.randn(3, 2, 7)
        mu, sigma = _mu_sigma(3, 2)
        valid = torch.ones(3, 7, dtype=torch.bool)
        valid[0, 3] = False                                     # mask one timestep
        feats, valid_out = t(x, mu, sigma, valid=valid)
        a, da = feats[:, :2, :], feats[:, 2:, :]
        assert not valid_out[0, 3]
        assert a[0, :, 3].abs().max() == 0                      # a zeroed there
        # Δ spanning the invalid step (t=3 and t=4) is dropped
        assert da[0, :, 3].abs().max() == 0
        assert da[0, :, 4].abs().max() == 0

    def test_nonfinite_x_treated_invalid(self):
        t = AnomalyTransform(n_channels=2)
        x = torch.randn(2, 2, 6)
        x[1, 0, 2] = float("nan")
        mu, sigma = _mu_sigma(2, 2)
        feats, valid_out = t(x, mu, sigma)
        assert torch.isfinite(feats).all()                     # no NaN leaks out
        assert not valid_out[1, 2]


class TestDeltaScaleLifecycle:
    def test_identity_until_locked(self):
        t = AnomalyTransform(n_channels=2)
        assert float(t.delta_scale) == 1.0
        assert int(t.scale_locked) == 0

    def test_observe_then_lock_normalizes_delta(self):
        """After locking on warmup data, typical ‖Δa‖ ≈ 1."""
        torch.manual_seed(0)
        t = AnomalyTransform(n_channels=3)
        t.train()
        mu, sigma = _mu_sigma(64, 3)
        # Observe over several "warmup" batches (Δa has some intrinsic scale ~5).
        for _ in range(30):
            x = torch.cumsum(5.0 * torch.randn(64, 3, 15), dim=-1)
            t(x, mu, sigma)
        assert int(t.scale_locked) == 0                        # not locked yet
        locked = t.lock_delta_scale()
        assert locked > 0 and int(t.scale_locked) == 1
        # After lock, the median ‖Δa‖ on fresh like-distributed data is ~1.
        x = torch.cumsum(5.0 * torch.randn(64, 3, 15), dim=-1)
        feats, _ = t(x, mu, sigma)
        da = feats[:, 3:, :]
        med = torch.linalg.vector_norm(da[:, :, 1:], dim=1).median()
        assert 0.5 < med < 2.0

    def test_no_observation_when_frozen_or_eval(self):
        t = AnomalyTransform(n_channels=2)
        t.eval()
        mu, sigma = _mu_sigma(8, 2)
        t(torch.randn(8, 2, 10), mu, sigma)
        assert int(t._scale_obs) == 0                          # eval → no observe
        t.train(); t.lock_delta_scale()                        # lock (obs=0 → scale stays 1)
        before = float(t.delta_scale)
        t(torch.randn(8, 2, 10), mu, sigma)
        assert int(t._scale_obs) == 0 and float(t.delta_scale) == before
