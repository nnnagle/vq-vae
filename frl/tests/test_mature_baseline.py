"""Tests for models.mature_baseline — the Step-1 RFF mature-baseline readout."""

from __future__ import annotations

import math

import pytest
import torch

from models.mature_baseline import RFFMatureBaseline


# ── helpers ────────────────────────────────────────────────────────────────

def _make(in_dim=6, n_channels=4, n_features=1024, **kw):
    torch.manual_seed(0)
    return RFFMatureBaseline(
        in_dim=in_dim, n_channels=n_channels, n_features=n_features, seed=0, **kw
    )


def _reference_predict(m: RFFMatureBaseline, z_tr, x_tr, z_q):
    """Independent recompute of the full single-update pipeline: standardize →
    RFF → centered mean ridge → residual-variance ridge → predict. Shares only the
    fixed random basis (a hyperparameter), so it genuinely checks the module math.
    """
    M = z_tr.shape[0]
    zmean, zvar = z_tr.mean(0), z_tr.var(0, unbiased=False)

    def feats(z):
        zs = (z - zmean) / torch.sqrt(zvar + 1e-6)
        return math.sqrt(2.0 / m.n_features) * torch.cos(zs @ m.omega + m.rff_bias)

    phi = feats(z_tr)
    phi_mean, x_mean = phi.mean(0), x_tr.mean(0)
    Ac = (phi.t() @ phi) / M - torch.outer(phi_mean, phi_mean)
    cc = (phi.t() @ x_tr) / M - torch.outer(phi_mean, x_mean)
    K = Ac + m.ridge_lambda * torch.eye(m.n_features, dtype=Ac.dtype)
    Wm = torch.linalg.solve(K, cc)

    mu_tr = x_mean + (phi - phi_mean) @ Wm
    r2 = (x_tr - mu_tr) ** 2
    r2_mean = r2.mean(0)
    cc_r = (phi.t() @ r2) / M - torch.outer(phi_mean, r2_mean)
    Wr = torch.linalg.solve(K, cc_r)

    f = feats(z_q) - phi_mean
    mu = x_mean + f @ Wm
    var = torch.clamp(r2_mean + f @ Wr, min=m.sigma_floor ** 2)
    return mu, torch.sqrt(var)


# ── basics ─────────────────────────────────────────────────────────────────

class TestBasics:
    def test_shapes_and_sigma_floor(self):
        m = _make()
        m.update(torch.randn(50, 6), torch.randn(50, 4))
        mu, sigma = m.predict(torch.randn(10, 6))
        assert mu.shape == (10, 4) and sigma.shape == (10, 4)
        assert torch.all(sigma >= m.sigma_floor - 1e-6)

    def test_no_gradient_parameters(self):
        """The estimator is buffer-only: no params, and outputs are grad-free."""
        m = _make()
        assert list(m.parameters()) == []
        m.update(torch.randn(40, 6), torch.randn(40, 4))
        z = torch.randn(8, 6, requires_grad=True)
        mu, sigma = m.predict(z)
        assert not mu.requires_grad and not sigma.requires_grad
        assert not m.anomaly(torch.randn(8, 4), z).requires_grad

    def test_empty_update_is_noop(self):
        m = _make()
        before = m.num_updates.item()
        m.update(torch.zeros(0, 6), torch.zeros(0, 4))
        assert m.num_updates.item() == before

    def test_sigma_floor_must_be_positive(self):
        with pytest.raises(ValueError):
            _make(sigma_floor=0.0)


# ── the closed-form solve is correct ────────────────────────────────────────

class TestClosedForm:
    def test_single_update_matches_reference(self):
        """One update (decay→0) ⇒ μ and σ match a from-scratch centered
        mean-ridge + residual-variance-ridge solve, to numerical precision."""
        m = _make(n_features=96).double()
        torch.manual_seed(1)
        z_tr = torch.randn(200, 6, dtype=torch.float64)
        x_tr = torch.randn(200, 4, dtype=torch.float64)
        z_q = torch.randn(15, 6, dtype=torch.float64)

        m.update(z_tr, x_tr)
        mu, sigma = m.predict(z_q)
        mu_ref, sigma_ref = _reference_predict(m, z_tr, x_tr, z_q)

        assert torch.allclose(mu, mu_ref, atol=1e-8, rtol=1e-6)
        assert torch.allclose(sigma, sigma_ref, atol=1e-8, rtol=1e-6)

    def test_mu_recovers_a_smooth_target(self):
        """In-support μ tracks a smooth (bounded) function of z."""
        m = _make(n_features=1024, bandwidth=1.5, ridge_lambda=1e-3)
        torch.manual_seed(2)
        W = torch.randn(6, 4)
        z_tr = torch.randn(4000, 6)
        target = torch.tanh(z_tr @ W)
        m.update(z_tr, target + 0.05 * torch.randn(4000, 4))
        z_q = torch.randn(300, 6)
        mu, _ = m.predict(z_q)
        tgt_q = torch.tanh(z_q @ W)
        r2 = 1 - ((mu - tgt_q) ** 2).mean() / tgt_q.var()
        assert r2 > 0.85


# ── leverage / revert-to-prior (sharpen with D) ──────────────────────────────

class TestLeverageAndPrior:
    def test_leverage_in_unit_interval(self):
        m = _make()
        m.update(torch.randn(100, 6), torch.randn(100, 4))
        v = m.leverage(torch.randn(30, 6))
        assert torch.all(v >= 0.0) and torch.all(v <= 1.0)

    def test_sparse_support_has_higher_leverage(self):
        m = _make(n_features=1024, bandwidth=1.0)
        torch.manual_seed(3)
        z_tr = torch.randn(500, 6)            # data clustered near origin
        m.update(z_tr, z_tr @ torch.randn(6, 4))
        v_near = m.leverage(0.2 * torch.randn(50, 6)).median()
        v_far = m.leverage(6.0 + torch.zeros(50, 6)).median()
        assert v_far > v_near
        assert v_far > 0.4                    # leans substantially on the prior

    def test_far_support_reverts_toward_prior(self):
        """A query far from the mature data collapses *toward* the prior mean:
        its deviation from E[x] is smaller than the in-support signal spread (the
        readout does not wildly extrapolate). Revert sharpens with D; at finite D
        two distinct far points are not identical (each keeps an aliasing residual),
        so we assert the toward-prior collapse, not point equality."""
        m = _make(n_features=2048)
        torch.manual_seed(4)
        z_tr = torch.randn(600, 6)
        x_tr = z_tr @ torch.randn(6, 4) + 2.0    # prior mean offset by ~2
        m.update(z_tr, x_tr)
        mu_far = m.predict(torch.full((20, 6), 8.0))[0].mean(0)
        signal_spread = m.predict(torch.randn(500, 6))[0].std(0).mean()
        assert (mu_far - m.x_mean).abs().max() < signal_spread
        assert m.leverage(torch.full((20, 6), 8.0)).median() > 0.5


# ── heteroscedastic σ via residual-variance ridge ────────────────────────────

class TestSigma:
    def test_sigma_tracks_local_noise(self):
        m = _make(in_dim=2, n_channels=1, n_features=1024, bandwidth=1.0,
                  ridge_lambda=1e-3)
        torch.manual_seed(5)
        n = 3000
        zA = 3.0 + 0.3 * torch.randn(n, 2)        # low-noise region
        zB = -3.0 + 0.3 * torch.randn(n, 2)       # high-noise region
        xA = 0.1 * torch.randn(n, 1)
        xB = 1.0 * torch.randn(n, 1)
        m.update(torch.cat([zA, zB]), torch.cat([xA, xB]))
        _, sA = m.predict(3.0 + 0.3 * torch.randn(200, 2))
        _, sB = m.predict(-3.0 + 0.3 * torch.randn(200, 2))
        assert sB.median() > 2 * sA.median()
        assert sA.median() >= m.sigma_floor


# ── NLL usable as the selection criterion; anomaly is standardized ───────────

class TestNLLAndAnomaly:
    def test_heldout_nll_drops_after_fit(self):
        m = _make(n_features=1024, bandwidth=1.5, ridge_lambda=1e-3)
        torch.manual_seed(6)
        W = torch.randn(6, 4)
        z_tr = torch.randn(3000, 6)
        z_val = torch.randn(400, 6)
        x_tr = torch.tanh(z_tr @ W) + 0.1 * torch.randn(3000, 4)
        x_val = torch.tanh(z_val @ W) + 0.1 * torch.randn(400, 4)

        nll_before = m.nll(z_val, x_val).mean()
        m.update(z_tr, x_tr)
        assert m.nll(z_val, x_val).mean() < nll_before

    def test_anomaly_of_mature_is_standardized(self):
        """Anomaly of held-out in-distribution mature samples is ≈ zero-mean,
        ~unit-scale (per-channel)."""
        m = _make(in_dim=4, n_channels=3, n_features=1024, bandwidth=1.5,
                  ridge_lambda=1e-3)
        torch.manual_seed(7)
        W = torch.randn(4, 3)
        noise = torch.tensor([0.3, 0.6, 1.0])
        z = torch.randn(6000, 4)
        x = torch.tanh(z @ W) + noise * torch.randn(6000, 3)
        m.update(z, x)
        z_q = torch.randn(1000, 4)
        x_q = torch.tanh(z_q @ W) + noise * torch.randn(1000, 3)
        a = m.anomaly(x_q, z_q)
        assert a.mean().abs() < 0.25
        for c in range(3):                        # each channel ~ unit scale
            assert 0.6 < a[:, c].std() < 1.6
