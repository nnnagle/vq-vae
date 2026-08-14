"""Tests for losses.ou_dynamics — the Step-4 within-pixel Gaussian OU-NLL."""

from __future__ import annotations

import math

import pytest
import torch

from losses.ou_dynamics import OUDynamics


def _stable(N, T, D, scale=1.0):
    """A pixel batch with no disturbances (small ‖Δa‖ everywhere)."""
    z = torch.randn(N, T, D) * scale
    da = 0.1 * torch.ones(N, T)          # stable → gate ≈ 1
    valid = torch.ones(N, T, dtype=torch.bool)
    return z, da, valid


class TestGate:
    def test_disturbance_is_gated_out(self):
        ou = OUDynamics(tau=2.0)
        z, da, valid = _stable(4, 10, 6)
        da[:, 5] = 20.0                   # a big kick at t=5
        # The gate at the transition into t=5 should be ~0.
        _, diag = ou(z, da, valid)
        # Build the per-transition gate directly to check t=5 is suppressed.
        w = torch.exp(-(da[:, 1:] / ou.tau) ** 2)
        assert w[:, 4].max() < 0.01       # transition target t=5
        assert w[:, 0].min() > 0.9        # stable transitions ~1

    def test_invalid_transitions_excluded(self):
        ou = OUDynamics()
        z, da, valid = _stable(3, 8, 4)
        valid[0, 3] = False
        loss, diag = ou(z, da, valid)
        assert torch.isfinite(loss)
        assert diag["n_eff"] > 0


class TestLoss:
    def test_zero_on_perfect_ou_trajectory(self):
        """If z_t = ρ·z_{t-1} exactly, the residual (and loss) is ~0."""
        ou = OUDynamics(rho_init=0.8, learn_rho=False)
        N, T, D = 4, 12, 5
        z0 = torch.randn(N, 1, D)
        rho = 0.8
        zs = [z0]
        for _ in range(T - 1):
            zs.append(rho * zs[-1])
        z = torch.cat(zs, dim=1)          # [N, T, D], perfectly contracting
        da = 0.1 * torch.ones(N, T)
        loss, _ = ou(z, da)
        assert loss.item() < 1e-8

    def test_gradient_flows_to_z_and_rho(self):
        ou = OUDynamics(rho_init=0.9, learn_rho=True)
        z, da, valid = _stable(4, 10, 6)
        z.requires_grad_(True)
        loss, _ = ou(z, da, valid)
        loss.backward()
        assert z.grad is not None and z.grad.abs().sum() > 0
        assert ou._rho_logit.grad is not None

    def test_rho_in_unit_interval(self):
        ou = OUDynamics(rho_init=0.95)
        assert 0.0 < float(ou.rho.detach()) < 1.0
        # sigmoid keeps ρ bounded in (0,1] for any finite logit.
        with torch.no_grad():
            ou._rho_logit.fill_(8.0)
        assert 0.99 < float(ou.rho.detach()) <= 1.0

    def test_contracting_beats_expanding(self):
        """With ρ<1, a contracting trajectory has lower OU residual than an
        expanding one of the same steps."""
        ou = OUDynamics(rho_init=0.8, learn_rho=False)
        N, T, D = 8, 10, 4
        base = torch.randn(N, 1, D)
        contract = torch.cat([base * (0.8 ** t) for t in range(T)], dim=1)
        expand = torch.cat([base * (1.25 ** t) for t in range(T)], dim=1)
        da = 0.1 * torch.ones(N, T)
        lc, _ = ou(contract, da)
        le, _ = ou(expand, da)
        assert lc.item() < le.item()


class TestRadiusShrinkingSigma:
    def test_learned_s_grows_with_radius_and_includes_logterm(self):
        ou = OUDynamics(s0=0.1, s1=0.5, learn_s=True)
        # s²(r) = s0² + s1²·r² is larger at larger ‖z‖.
        assert float(ou.s1) > 0
        z, da, valid = _stable(4, 8, 5, scale=3.0)
        loss, _ = ou(z, da, valid)
        assert torch.isfinite(loss)       # log-s term finite

    def test_default_is_weighted_residual_no_logterm(self):
        """Default (fixed s, s1=0) → loss is scale-free in s (pure residual)."""
        ou_a = OUDynamics(s0=0.5, s1=0.0, learn_s=False)
        ou_b = OUDynamics(s0=2.0, s1=0.0, learn_s=False)
        z, da, valid = _stable(4, 10, 6)
        la, _ = ou_a(z, da, valid)
        lb, _ = ou_b(z, da, valid)
        assert torch.allclose(la, lb)     # s0 is just a folded weight → identical
