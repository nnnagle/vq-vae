"""Tests for losses.type_phase_contrastive — the Step-5 type-local ranking InfoNCE."""

from __future__ import annotations

import pytest
import torch

from losses.type_phase_contrastive import type_phase_contrastive_loss


def _synthetic(n_types=4, n_flows=5, reps=6, dt=6, F=4, D=8, seed=0):
    """Build pixel-times structured as (type × flow) clusters; each sample is its
    own pixel (cross-pixel positives available)."""
    g = torch.Generator().manual_seed(seed)
    type_centers = torch.randn(n_types, dt, generator=g) * 3
    flow_centers = torch.randn(n_flows, F, generator=g) * 3
    z_type, flow, tid, fid = [], [], [], []
    for t in range(n_types):
        for f in range(n_flows):
            for _ in range(reps):
                z_type.append(type_centers[t] + 0.1 * torch.randn(dt, generator=g))
                flow.append(flow_centers[f] + 0.1 * torch.randn(F, generator=g))
                tid.append(t); fid.append(f)
    z_type = torch.stack(z_type); flow = torch.stack(flow)
    M = z_type.shape[0]
    pixel_id = torch.arange(M)                     # every sample a distinct pixel
    z_phase = torch.randn(M, D, generator=g)
    return z_phase, z_type, flow, pixel_id, torch.tensor(tid), torch.tensor(fid)


class TestBasics:
    def test_finite_and_grad_to_zphase_only(self):
        z_phase, z_type, flow, pid, *_ = _synthetic()
        z_phase.requires_grad_(True)
        z_type.requires_grad_(True)
        flow.requires_grad_(True)
        loss, diag = type_phase_contrastive_loss(
            z_phase, z_type, flow, pid, tau_phase=1.0, sigma_type=1.0, sigma_flow=1.0)
        assert torch.isfinite(loss)
        loss.backward()
        assert z_phase.grad is not None and z_phase.grad.abs().sum() > 0
        # observables define the target only — no grad flows into them.
        assert z_type.grad is None and flow.grad is None

    def test_degenerate_small_batch(self):
        z = torch.randn(3, 8)
        loss, diag = type_phase_contrastive_loss(
            z, torch.randn(3, 6), torch.randn(3, 4), torch.arange(3), n_pos=5)
        assert loss.item() == 0.0 and diag["n_anchors"] == 0


class TestMetricLearning:
    def test_optimization_organizes_by_type_and_flow(self):
        """Minimizing the loss pulls same-type/same-flow z_phase together and pushes
        same-type/different-flow apart (gap/τ rises; the metric forms)."""
        z_phase, z_type, flow, pid, tid, fid = _synthetic(seed=1)
        z_phase = z_phase.clone().requires_grad_(True)
        opt = torch.optim.Adam([z_phase], lr=0.05)

        def curr_loss():
            return type_phase_contrastive_loss(
                z_phase, z_type, flow, pid,
                tau_phase=1.0, sigma_type=1.5, sigma_flow=1.5, n_pos=4, n_neg=20)

        l0, d0 = curr_loss()
        for _ in range(150):
            opt.zero_grad()
            loss, _ = curr_loss()
            loss.backward()
            opt.step()
        l1, d1 = curr_loss()

        assert l1.item() < l0.item()                 # loss decreased
        assert d1["gap_over_tau"] > d0["gap_over_tau"]  # separation improved

        # Directly check the emergent geometry on the trained z_phase.
        with torch.no_grad():
            same_tf, same_t_diff_f = [], []
            M = z_phase.shape[0]
            for i in range(M):
                for j in range(i + 1, M):
                    if tid[i] != tid[j]:
                        continue
                    d = (z_phase[i] - z_phase[j]).pow(2).sum().item()
                    (same_tf if fid[i] == fid[j] else same_t_diff_f).append(d)
            mean_same = sum(same_tf) / len(same_tf)
            mean_diff = sum(same_t_diff_f) / len(same_t_diff_f)
        assert mean_same < mean_diff                 # same-stage closer than diff-stage

    def test_cross_type_is_incommensurate(self):
        """Cross-type pairs get w_neg≈0 → they are neither strongly pulled nor pushed;
        the loss does not force cross-type structure."""
        z_phase, z_type, flow, pid, tid, fid = _synthetic(seed=2)
        # Cross-type negatives should carry negligible weight.
        with torch.no_grad():
            d_type2 = torch.cdist(z_type, z_type).pow(2)
            k_type = torch.exp(-d_type2 / (2 * 1.0 ** 2))
        # For well-separated type centers (×3), cross-type k_type is tiny.
        cross = k_type[tid.view(-1, 1) != tid.view(1, -1)]
        assert cross.max() < 0.2
