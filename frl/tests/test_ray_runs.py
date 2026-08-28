"""Tests for losses.ray_runs — the ysfc-free ray/runs-kernel phase objective."""

from __future__ import annotations

import math

import torch

from losses.ray_runs import (
    pairwise_jump_gate,
    ray_contraction_anchor_loss,
    runs_kernel_matching_loss,
)


class TestJumpGate:
    def test_gate_closes_over_intervening_jump(self):
        # ‖Δa‖ small everywhere except a spike at t=3.
        d = torch.full((1, 8), 0.1)
        d[0, 3] = 20.0
        G = pairwise_jump_gate(d, tau=2.0)[0]        # [8, 8]
        # Δa at t=3 is the year-2→3 jump, so t=3 begins the post-disturbance run.
        # Any pair straddling it (interval contains s=3) is gated ~0: (2,4),(1,5),(0,7),(2,3).
        assert G[2, 4] < 0.01
        assert G[1, 5] < 0.01
        assert G[0, 7] < 0.01
        assert G[2, 3] < 0.01
        # Pairs entirely on one side of the jump are open ~1: pre-run (0,2) and
        # post-run (3,7),(4,7) — both endpoints on the same side.
        assert G[0, 2] > 0.99
        assert G[3, 7] > 0.99
        assert G[4, 7] > 0.99
        # Symmetric, unit diagonal.
        assert torch.allclose(G, G.T)
        assert torch.allclose(torch.diagonal(G), torch.ones(8))

    def test_graded_moderate_jump(self):
        d = torch.full((1, 5), 0.1)
        d[0, 2] = 2.0                                # moderate, not severe
        G = pairwise_jump_gate(d, tau=2.0)[0]
        assert 0.1 < G[1, 3] < 0.9                   # partial gate


class TestContractionAnchor:
    def _ray(self, N, T, d, rho, seed=0):
        g = torch.Generator().manual_seed(seed)
        e = torch.randn(N, 1, d, generator=g)
        t = torch.arange(T).float()
        zp = (rho ** t)[None, :, None] * e           # clean ρ-decay ray
        return zp

    def test_clean_ray_has_zero_contraction(self):
        rho = 0.86
        zp = self._ray(4, 12, 6, rho)
        a_norm = zp.norm(dim=-1)                      # a decays with the ray
        da = 0.1 * torch.ones(4, 12)                 # no jumps
        valid = torch.ones(4, 12, dtype=torch.bool)
        _, contraction, diag = ray_contraction_anchor_loss(
            zp, a_norm, da, valid, rho=rho, tau_jump=2.0, sigma_mature=0.5)
        assert contraction.item() < 1e-6
        assert diag["gate_mean"] > 0.99

    def test_wrong_rho_penalized(self):
        zp = self._ray(4, 12, 6, rho=0.86)
        a_norm = zp.norm(dim=-1)
        da = 0.1 * torch.ones(4, 12)
        valid = torch.ones(4, 12, dtype=torch.bool)
        _, c_right, _ = ray_contraction_anchor_loss(zp, a_norm, da, valid, 0.86, 2.0, 0.5)
        _, c_wrong, _ = ray_contraction_anchor_loss(zp, a_norm, da, valid, 0.60, 2.0, 0.5)
        assert c_right.item() < c_wrong.item()

    def test_anchor_pulls_mature_to_origin(self):
        # Mature timesteps (small ‖a‖) with nonzero z should incur anchor loss.
        N, T, d = 3, 6, 4
        zp = torch.ones(N, T, d)                      # z away from origin everywhere
        a_norm = torch.ones(N, T)
        a_norm[:, 4:] = 0.0                           # last two steps mature
        da = 0.1 * torch.ones(N, T)
        valid = torch.ones(N, T, dtype=torch.bool)
        anchor, _, diag = ray_contraction_anchor_loss(zp, a_norm, da, valid, 0.86, 2.0, 0.5)
        assert anchor.item() > 0                      # mature z≠0 penalized
        assert 0.0 < diag["mature_frac"] < 1.0

    def test_mature_quantile_self_calibrates(self):
        # ‖a‖ never approaches the absolute sigma_mature=0.5 → fixed threshold pins
        # nothing; the quantile threshold still pins the least-anomalous timesteps.
        N, T, d = 4, 8, 3
        zp = torch.ones(N, T, d)
        a_norm = 3.0 + torch.rand(N, T)               # ‖a‖ ∈ [3,4], all ≫ 0.5
        da = 0.1 * torch.ones(N, T)
        valid = torch.ones(N, T, dtype=torch.bool)
        _, _, d_abs = ray_contraction_anchor_loss(
            zp, a_norm, da, valid, 0.86, 2.0, 0.5, mature_quantile=0.0)
        _, _, d_q = ray_contraction_anchor_loss(
            zp, a_norm, da, valid, 0.86, 2.0, 0.5, mature_quantile=0.3)
        assert d_abs["mature_frac"] < 1e-3            # absolute threshold dead
        assert d_q["mature_frac"] > 0.05              # quantile threshold pins ~least ‖a‖
        # effective sigma tracks the batch ‖a‖ (near its 30th percentile), not 0.5.
        assert d_q["sigma_mature_eff"] > 1.0

    def test_contraction_gated_across_jump(self):
        # A disturbance mid-window: cross-jump lags must not be scored.
        N, T, d = 2, 8, 4
        zp = torch.randn(N, T, d)
        a_norm = zp.norm(dim=-1)
        da = 0.1 * torch.ones(N, T)
        da[:, 4] = 30.0                               # jump at t=4
        valid = torch.ones(N, T, dtype=torch.bool)
        _, _, diag = ray_contraction_anchor_loss(zp, a_norm, da, valid, 0.86, 2.0, 0.5)
        # gate_mean well below 1 because ~half the lag-pairs straddle the jump.
        assert diag["gate_mean"] < 0.8

    def test_gradients_flow(self):
        zp = self._ray(4, 10, 6, 0.86).clone().requires_grad_(True)
        a_norm = zp.detach().norm(dim=-1)
        da = 0.1 * torch.ones(4, 10)
        valid = torch.ones(4, 10, dtype=torch.bool)
        anchor, contraction, _ = ray_contraction_anchor_loss(zp, a_norm, da, valid, 0.86, 2.0, 0.5)
        (anchor + contraction).backward()
        assert zp.grad is not None and zp.grad.abs().sum() > 0


class TestRunsKernelMatching:
    def _two_fires(self, seed=0):
        """Pixel A: fire at t=2. Pixel B: fire at t=6. Same recovery shape, offset
        by 4 years. z_phase is set so the runs kernel *should* pull A's post-fire
        window onto B's."""
        g = torch.Generator().manual_seed(seed)
        N, T, F, d, dt = 4, 15, 2, 3, 5
        flow = 0.05 * torch.randn(N, T, F, generator=g)
        da = 0.1 * torch.ones(N, T)
        # A (pixel 0) fire at t=2, B (pixel 1) fire at t=6; C,D are stable controls.
        def recov(start):
            s = torch.zeros(T, F)
            for k in range(T - start):
                s[start + k, 0] = math.exp(-0.3 * k)      # a decays
            return s
        flow[0, :, :] += recov(2)
        flow[1, :, :] += recov(6)
        da[0, 2] = 20.0
        da[1, 6] = 20.0
        z_type = torch.zeros(N, dt)                        # all same type
        valid = torch.ones(N, T, dtype=torch.bool)
        # z_phase: make A and B's post-fire windows land together, controls apart.
        zp = 0.01 * torch.randn(N, T, d, generator=g)
        return zp, flow, da, z_type, valid

    def test_runs_kernel_sees_offset_matched_fires(self):
        zp, flow, da, z_type, valid = self._two_fires()
        loss, diag = runs_kernel_matching_loss(
            zp, flow, da, z_type, valid, tau_jump=2.0, half_window=5,
            window_sigma=3.0, sigma_flow=0.5, sigma_type=3.0, tau_metric=1.0,
            max_points=200, min_points=8)
        assert diag["active"] == 1.0
        assert math.isfinite(loss.item())
        # The reference runs-similarity must be higher among same-type pairs that
        # share a fire-recovery window than the floor — sanity that S is populated.
        assert diag["S_same"] > 0.0

    def test_gradients_and_finite(self):
        zp, flow, da, z_type, valid = self._two_fires(seed=1)
        zp = zp.requires_grad_(True)
        loss, _ = runs_kernel_matching_loss(
            zp, flow, da, z_type, valid, tau_jump=2.0, half_window=4,
            window_sigma=3.0, sigma_flow=0.5, sigma_type=3.0, tau_metric=1.0,
            max_points=200, min_points=8)
        loss.backward()
        assert torch.isfinite(loss) and zp.grad.abs().sum() > 0

    def test_matching_pulls_high_S_together(self):
        # If z_phase already matches S well, loss is lower than random z_phase.
        zp, flow, da, z_type, valid = self._two_fires(seed=2)
        kw = dict(tau_jump=2.0, half_window=5, window_sigma=3.0, sigma_flow=0.5,
                  sigma_type=3.0, tau_metric=1.0, max_points=200, min_points=8)
        loss_rand, _ = runs_kernel_matching_loss(zp, flow, da, z_type, valid, **kw)
        # Build a z_phase that mirrors the flow window structure (A~B close): use
        # the first flow channel trajectory as the embedding → high L where high S.
        zp2 = torch.zeros_like(zp)
        zp2[..., 0] = flow[..., 0]
        loss_aligned, _ = runs_kernel_matching_loss(zp2, flow, da, z_type, valid, **kw)
        assert loss_aligned.item() < loss_rand.item()

    def test_too_few_points_is_safe(self):
        zp = torch.randn(1, 3, 2)
        flow = torch.randn(1, 3, 2)
        da = 0.1 * torch.ones(1, 3)
        z_type = torch.zeros(1, 4)
        valid = torch.zeros(1, 3, dtype=torch.bool)   # nothing valid
        loss, diag = runs_kernel_matching_loss(
            zp, flow, da, z_type, valid, tau_jump=2.0, half_window=2,
            window_sigma=2.0, sigma_flow=0.5, sigma_type=3.0, tau_metric=1.0,
            max_points=100, min_points=8)
        assert loss.item() == 0.0 and diag["active"] == 0.0

    def _two_type_clusters(self, seed=0):
        """Two well-separated z_type clusters; identical flow so only the type gate
        differentiates pairs. Used to test the keep-threshold."""
        g = torch.Generator().manual_seed(seed)
        N, T, F, d, dt = 8, 12, 2, 3, 5
        flow = 0.1 * torch.randn(N, T, F, generator=g)
        da = 0.1 * torch.ones(N, T)
        z_type = torch.zeros(N, dt)
        z_type[N // 2:, 0] = 30.0          # cluster B far from cluster A along dim 0
        valid = torch.ones(N, T, dtype=torch.bool)
        zp = 0.05 * torch.randn(N, T, d, generator=g)
        return zp, flow, da, z_type, valid

    def test_threshold_drops_cross_type_pairs(self):
        zp, flow, da, z_type, valid = self._two_type_clusters()
        kw = dict(tau_jump=2.0, half_window=3, window_sigma=3.0, sigma_flow=0.5,
                  sigma_type=1.0, tau_metric=1.0, max_points=200, min_points=8)
        # No threshold: cross-type pairs are present (soft-weighted only).
        _, diag_all = runs_kernel_matching_loss(
            zp, flow, da, z_type, valid, type_keep_threshold=0.0, **kw)
        # High threshold: cross-type pairs (k_type≈0) are hard-dropped.
        _, diag_keep = runs_kernel_matching_loss(
            zp, flow, da, z_type, valid, type_keep_threshold=0.5, **kw)
        assert diag_keep["keep_frac"] < diag_all["keep_frac"]
        # Only same-type pairs survive ⇒ kept type gate is tight, distance small.
        assert diag_keep["k_type_kept"] > diag_all["k_type_kept"]
        assert diag_keep["dt_kept"] < diag_all["dt_kept"]
        # Bandwidth monitors are populated and sane.
        assert 0.0 <= diag_keep["keep_frac"] <= 1.0
        assert diag_keep["nbr_per_pt"] >= 0.0

    def test_type_grouped_pool_raises_neighbor_density(self):
        # Many small type clusters; random pooling starves same-type neighbors,
        # type-grouped pooling concentrates them.
        g = torch.Generator().manual_seed(7)
        n_clusters, per, T, F, d, dt = 20, 6, 12, 2, 3, 5
        N = n_clusters * per
        centers = 40.0 * torch.randn(n_clusters, dt, generator=g)
        z_type = centers.repeat_interleave(per, 0) + 0.1 * torch.randn(N, dt, generator=g)
        flow = 0.1 * torch.randn(N, T, F, generator=g)
        da = 0.1 * torch.ones(N, T)
        valid = torch.ones(N, T, dtype=torch.bool)
        zp = 0.05 * torch.randn(N, T, d, generator=g)
        kw = dict(tau_jump=2.0, half_window=3, window_sigma=3.0, sigma_flow=0.5,
                  sigma_type=1.0, tau_metric=1.0, max_points=400, min_points=8,
                  type_keep_threshold=0.5)
        gg = torch.Generator().manual_seed(0)
        _, d_rand = runs_kernel_matching_loss(
            zp, flow, da, z_type, valid, n_seeds=0, generator=gg, **kw)
        _, d_grp = runs_kernel_matching_loss(
            zp, flow, da, z_type, valid, n_seeds=6, group_size=per, generator=gg, **kw)
        # Type-grouped pooling gives each point many more same-type neighbors.
        assert d_grp["nbr_per_pt"] > 1.5 * d_rand["nbr_per_pt"]
        assert d_grp["keep_frac"] > d_rand["keep_frac"]
        assert d_grp["n_pixels"] > 0

    def test_threshold_default_keeps_all(self):
        zp, flow, da, z_type, valid = self._two_type_clusters(seed=3)
        kw = dict(tau_jump=2.0, half_window=3, window_sigma=3.0, sigma_flow=0.5,
                  sigma_type=1.0, tau_metric=1.0, max_points=200, min_points=8)
        _, diag = runs_kernel_matching_loss(
            zp, flow, da, z_type, valid, type_keep_threshold=0.0, **kw)
        # Inert mask: every off-diagonal pair is kept.
        assert abs(diag["keep_frac"] - 1.0) < 1e-6
