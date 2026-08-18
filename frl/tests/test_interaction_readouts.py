"""Tests for phase_eval.interaction_readouts — the Diagnostic-B interaction readouts."""

from __future__ import annotations

import math

import torch

from training.phase_eval.interaction_readouts import (
    bilinear_features,
    bilinear_interaction,
    block_penalty,
    product_kernel_predict,
    solve_block_ridge,
    whitened_pca,
)


def _standardize(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def _augmented_normal_eqs(X, y):
    """(M-averaged) augmented normal equations [X, 1]ᵀ[X,1] and [X,1]ᵀy."""
    M = X.shape[0]
    Xa = torch.cat([X, torch.ones(M, 1, dtype=X.dtype)], dim=1)
    A = (Xa.t() @ Xa) / M
    B = (Xa.t() @ y) / M
    return A, B


# ── whitened PCA ─────────────────────────────────────────────────────────────

class TestWhitenedPCA:
    def test_shape_and_unit_variance_projection(self):
        torch.manual_seed(0)
        dt, r = 8, 3
        z = torch.randn(5000, dt) @ torch.randn(dt, dt)     # correlated
        zs = _standardize(z).double()
        cov = (zs.t() @ zs) / zs.shape[0]
        P = whitened_pca(cov, r)
        assert P.shape == (dt, r)
        proj = zs @ P                                        # [N, r]
        # whitening ⇒ projected coords ~ unit variance, decorrelated
        v = proj.var(0)
        assert torch.allclose(v, torch.ones(r, dtype=v.dtype), atol=0.05)

    def test_r_clipped_to_dim(self):
        cov = torch.eye(4)
        assert whitened_pca(cov, 10).shape == (4, 4)


# ── bilinear features ────────────────────────────────────────────────────────

class TestBilinearFeatures:
    def test_shapes(self):
        N, dt, zp, r = 10, 6, 4, 3
        zt, zp_s = torch.randn(N, dt), torch.randn(N, zp)
        P = torch.randn(dt, r)
        assert bilinear_interaction(zt, zp_s, P).shape == (N, r * zp)
        assert bilinear_features(zt, zp_s, P).shape == (N, dt + zp + r * zp)

    def test_interaction_is_product_structured(self):
        # block k of the interaction is (proj_k) * z_phase
        N, dt, zp, r = 7, 5, 4, 2
        zt, zp_s = torch.randn(N, dt), torch.randn(N, zp)
        P = torch.randn(dt, r)
        proj = zt @ P
        inter = bilinear_interaction(zt, zp_s, P).reshape(N, r, zp)
        for k in range(r):
            assert torch.allclose(inter[:, k, :], proj[:, k:k + 1] * zp_s, atol=1e-6)


# ── block ridge solve ────────────────────────────────────────────────────────

class TestBlockRidge:
    def test_matches_ols_at_zero_penalty(self):
        torch.manual_seed(1)
        X = _standardize(torch.randn(2000, 5)).double()
        w_true = torch.randn(5, 1, dtype=torch.float64)
        y = X @ w_true + 3.0 + 0.01 * torch.randn(2000, 1, dtype=torch.float64)
        A, B = _augmented_normal_eqs(X, y)
        pen = torch.zeros(6, dtype=torch.float64)
        W, b = solve_block_ridge(A, B, pen)
        assert torch.allclose(W, w_true, atol=1e-2)
        assert abs(float(b) - 3.0) < 1e-2

    def test_separate_penalties_shrink_their_own_block(self):
        """Heavy λ on the interaction block kills interaction coefs but leaves the
        main-effect coefs close to their lightly-penalized values."""
        torch.manual_seed(2)
        N, dt, zp, r = 4000, 4, 3, 2
        zt = _standardize(torch.randn(N, dt)).double()
        zp_s = _standardize(torch.randn(N, zp)).double()
        cov = (zt.t() @ zt) / N
        P = whitened_pca(cov, r).double()
        X = bilinear_features(zt, zp_s, P)
        X = _standardize(X)                                  # unit-diagonal ridge scale
        # target uses BOTH a main effect and an interaction
        w_true = torch.randn(X.shape[1], 1, dtype=torch.float64)
        y = X @ w_true + 0.05 * torch.randn(N, 1, dtype=torch.float64)
        A, B = _augmented_normal_eqs(X, y)
        n_main = dt + zp
        light = solve_block_ridge(A, B, block_penalty(dt, zp, r, 1e-4, 1e-4))[0]
        heavy_int = solve_block_ridge(A, B, block_penalty(dt, zp, r, 1e-4, 1e3))[0]
        main_mag = heavy_int[:n_main].abs().mean()
        int_mag = heavy_int[n_main:].abs().mean()
        # interaction block driven ~to zero; main block survives
        assert int_mag < 0.05 * main_mag
        # main coefs barely move relative to the lightly-penalized fit
        assert (heavy_int[:n_main] - light[:n_main]).abs().mean() < 0.2 * main_mag

    def test_bilinear_beats_additive_on_a_bilinear_target(self):
        """A genuinely type-conditional (bilinear) target: the additive [zt,zp]
        model cannot fit it; the bilinear one can. This is the whole motivation."""
        torch.manual_seed(3)
        N, dt, zp = 6000, 6, 4
        zt = _standardize(torch.randn(N, dt)).double()
        zp_s = _standardize(torch.randn(N, zp)).double()
        a = torch.randn(dt, 1, dtype=torch.float64)
        c = torch.randn(zp, 1, dtype=torch.float64)
        y = (zt @ a) * (zp_s @ c) + 0.05 * torch.randn(N, 1, dtype=torch.float64)  # rank-1 bilinear

        def _fit_r2(X):
            Xz = _standardize(X)
            A, B = _augmented_normal_eqs(Xz, y)
            pen = torch.zeros(Xz.shape[1] + 1, dtype=torch.float64)
            pen[:-1] = 1e-6
            W, b = solve_block_ridge(A, B, pen)
            pred = Xz @ W + b
            ss_res = ((y - pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            return float(1 - ss_res / ss_tot)

        r2_add = _fit_r2(torch.cat([zt, zp_s], dim=1))            # additive
        P = whitened_pca((zt.t() @ zt) / N, dt).double()         # full-rank interaction
        r2_bil = _fit_r2(bilinear_features(zt, zp_s, P))         # bilinear
        assert r2_add < 0.15                                      # additive ~helpless
        assert r2_bil > 0.9                                       # bilinear recovers it


# ── block penalty structure ──────────────────────────────────────────────────

def test_block_penalty_layout():
    pen = block_penalty(dt=4, zp=3, r=2, lam_main=0.1, lam_bilinear=0.5)
    assert pen.shape == (4 + 3 + 2 * 3 + 1,)
    assert torch.allclose(pen[:7], torch.full((7,), 0.1, dtype=pen.dtype))
    assert torch.allclose(pen[7:13], torch.full((6,), 0.5, dtype=pen.dtype))
    assert float(pen[-1]) == 0.0


# ── product-kernel kNN ───────────────────────────────────────────────────────

class TestProductKernel:
    def test_shape_and_convex_combination(self):
        torch.manual_seed(4)
        zt_r, zp_r = torch.randn(500, 5), torch.randn(500, 3)
        y_r = torch.randn(500)
        zt_q, zp_q = torch.randn(40, 5), torch.randn(40, 3)
        pred = product_kernel_predict(zt_q, zp_q, zt_r, zp_r, y_r, 1.0, 1.0)
        assert pred.shape == (40,)
        # predictions are weighted averages of y_ref → within its range
        assert float(pred.min()) >= float(y_r.min()) - 1e-6
        assert float(pred.max()) <= float(y_r.max()) + 1e-6

    def test_type_locality_needs_a_tight_type_bandwidth(self):
        """Target: two type clusters with OPPOSITE phase slopes. A tight σ_type
        (keeps clusters apart) predicts well; a huge σ_type (merges clusters,
        averaging the opposite slopes to ~0) does not — so the type bandwidth must
        be tunable separately from the phase bandwidth."""
        torch.manual_seed(5)
        n = 3000
        # cluster label from a well-separated type coordinate
        typ = torch.cat([torch.full((n,), -4.0), torch.full((n,), 4.0)])
        zt = (typ + 0.2 * torch.randn(2 * n)).unsqueeze(1)     # [2n, 1]
        zp = torch.randn(2 * n, 1)
        slope = torch.where(typ < 0, -1.0, 1.0)
        y = slope * zp.squeeze(1) + 0.05 * torch.randn(2 * n)

        idx = torch.randperm(2 * n)
        ref, qry = idx[: 2 * n - 400], idx[2 * n - 400:]

        def _r2(sig_type):
            pred = product_kernel_predict(
                zt[qry], zp[qry], zt[ref], zp[ref], y[ref],
                sigma_type=sig_type, sigma_phase=0.5,
            )
            yq = y[qry].double()
            ss_res = ((yq - pred) ** 2).sum()
            ss_tot = ((yq - yq.mean()) ** 2).sum()
            return float(1 - ss_res / ss_tot)

        r2_tight = _r2(0.5)      # resolves the two type clusters
        r2_merged = _r2(50.0)    # merges them → opposite slopes cancel
        assert r2_tight > 0.8
        assert r2_merged < 0.3
        assert r2_tight > r2_merged + 0.5
