"""Tests for losses.triplet_phase — pair builder and loss function."""

from __future__ import annotations

import pytest
import torch

from losses.triplet_phase import (
    MarginClass,
    build_triplet_constraints_batch,
    classify_triplet,
    phase_triplet_loss,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_ysfc(T: int, disturbance_years: list[int]) -> torch.Tensor:
    """Create a ysfc vector of length T.

    Non-disturbance years get value > 0 (years since last change);
    disturbance years get 0.
    """
    ysfc = torch.ones(T, dtype=torch.float32)
    for y in disturbance_years:
        ysfc[y] = 0
    return ysfc


# ── classify_triplet ──────────────────────────────────────────────────────

class TestClassifyTriplet:
    """Unit tests for the per-pixel triplet classification."""

    def test_d01_only(self):
        """Disturbance between t0 and t1 only → 3 constraints (2L + 1S)."""
        # T=15 (years 0..14), disturbance at year 5
        ysfc = _make_ysfc(15, disturbance_years=[5])
        # t0=2, t1=8, t2=12 → disturbance at 5 is in (2,8), none in (8,12)
        constraints = classify_triplet(ysfc, t0_idx=2, t1_idx=8, t2_idx=12)
        assert constraints is not None
        assert len(constraints) == 3
        large = [c for c in constraints if c.margin_class == MarginClass.LARGE]
        small = [c for c in constraints if c.margin_class == MarginClass.SMALL]
        assert len(large) == 2
        assert len(small) == 1

    def test_d12_only(self):
        """Disturbance between t1 and t2 only → 3 constraints (2L + 1S)."""
        ysfc = _make_ysfc(15, disturbance_years=[10])
        # t0=2, t1=6, t2=13 → disturbance at 10 is in (6,13), none in (2,6)
        constraints = classify_triplet(ysfc, t0_idx=2, t1_idx=6, t2_idx=13)
        assert constraints is not None
        assert len(constraints) == 3
        large = [c for c in constraints if c.margin_class == MarginClass.LARGE]
        small = [c for c in constraints if c.margin_class == MarginClass.SMALL]
        assert len(large) == 2
        assert len(small) == 1

    def test_no_disturbance(self):
        """No disturbance anywhere → 2 small-margin constraints."""
        ysfc = _make_ysfc(15, disturbance_years=[])
        constraints = classify_triplet(ysfc, t0_idx=2, t1_idx=6, t2_idx=13)
        assert constraints is not None
        assert len(constraints) == 2
        assert all(c.margin_class == MarginClass.SMALL for c in constraints)

    def test_disturbance_both_intervals_skipped(self):
        """Disturbance in both intervals → skip."""
        ysfc = _make_ysfc(15, disturbance_years=[4, 10])
        # t0=2, t1=6, t2=13 → dist at 4 in (2,6), dist at 10 in (6,13)
        constraints = classify_triplet(ysfc, t0_idx=2, t1_idx=6, t2_idx=13)
        assert constraints is None

    def test_disturbance_at_sampled_year_skipped(self):
        """Disturbance at one of the sampled years → skip."""
        ysfc = _make_ysfc(15, disturbance_years=[6])
        constraints = classify_triplet(ysfc, t0_idx=2, t1_idx=6, t2_idx=13)
        assert constraints is None

    def test_disturbance_at_t0_skipped(self):
        ysfc = _make_ysfc(15, disturbance_years=[2])
        constraints = classify_triplet(ysfc, t0_idx=2, t1_idx=6, t2_idx=13)
        assert constraints is None

    def test_disturbance_at_t2_skipped(self):
        ysfc = _make_ysfc(15, disturbance_years=[13])
        constraints = classify_triplet(ysfc, t0_idx=2, t1_idx=6, t2_idx=13)
        assert constraints is None

    def test_adjacent_years_no_interval(self):
        """Consecutive years → no interval to check, treated as no disturbance."""
        ysfc = _make_ysfc(15, disturbance_years=[])
        # t0=5, t1=6, t2=7 → intervals are empty
        constraints = classify_triplet(ysfc, t0_idx=5, t1_idx=6, t2_idx=7)
        assert constraints is not None
        assert len(constraints) == 2

    def test_disturbance_outside_intervals(self):
        """Disturbance exists but outside both intervals → no disturbance case."""
        ysfc = _make_ysfc(15, disturbance_years=[0, 14])
        constraints = classify_triplet(ysfc, t0_idx=2, t1_idx=6, t2_idx=12)
        assert constraints is not None
        assert len(constraints) == 2


# ── build_triplet_constraints_batch ───────────────────────────────────────

class TestBuildBatch:
    """Tests for the vectorised batch constraint builder."""

    def test_mixed_cases(self):
        """Batch with one pixel per case: d01, d12, none, skip."""
        T = 15
        ysfc = torch.ones(4, T)
        # Pixel 0: disturbance at year 5 → d01 (t0=2,t1=8,t2=12)
        ysfc[0, 5] = 0
        # Pixel 1: disturbance at year 10 → d12 (t0=2,t1=6,t2=13)
        ysfc[1, 10] = 0
        # Pixel 2: no disturbance → none
        # (already all ones)
        # Pixel 3: disturbance at both 4 and 10 → skip
        ysfc[3, 4] = 0
        ysfc[3, 10] = 0

        closer, farther, is_large, pix = build_triplet_constraints_batch(
            ysfc, t0_idx=2, t1_idx=8, t2_idx=12
        )

        # Pixel 0: d01 → 3 constraints
        # Pixel 1: disturbance at 10 is in (8,12) → d12 → 3 constraints
        # Pixel 2: no disturbance → 2 constraints
        # Pixel 3: both intervals → skip
        assert closer.shape[0] == 3 + 3 + 2  # 8 total
        assert (pix == 0).sum().item() == 3
        assert (pix == 1).sum().item() == 3
        assert (pix == 2).sum().item() == 2
        assert (pix == 3).sum().item() == 0

    def test_all_skipped(self):
        """All pixels skipped → empty tensors."""
        T = 15
        ysfc = torch.ones(3, T)
        # Disturbance at a sampled year for all pixels.
        ysfc[:, 6] = 0

        closer, farther, is_large, pix = build_triplet_constraints_batch(
            ysfc, t0_idx=2, t1_idx=6, t2_idx=12
        )
        assert closer.shape[0] == 0

    def test_slot_indices_are_valid(self):
        """All slot indices should be in {0, 1, 2}."""
        T = 15
        ysfc = torch.ones(10, T)
        closer, farther, _, _ = build_triplet_constraints_batch(
            ysfc, t0_idx=2, t1_idx=8, t2_idx=12
        )
        if closer.numel() > 0:
            assert closer.min().item() >= 0
            assert closer.max().item() <= 2
            assert farther.min().item() >= 0
            assert farther.max().item() <= 2


# ── phase_triplet_loss ────────────────────────────────────────────────────

class TestPhaseTripletLoss:
    """Tests for the end-to-end loss function."""

    def test_zero_loss_when_constraints_satisfied(self):
        """If d_far >> d_close, loss should be near zero."""
        N, D, T = 10, 16, 15
        ysfc = torch.ones(N, T)  # no disturbance → case none

        # Embeddings: t0 ≈ t1 (close), t2 far away.
        # For no-disturbance: d(0,2)>d(0,1) and d(0,2)>d(1,2).
        torch.manual_seed(42)
        base = torch.randn(N, D)
        emb_t0 = base
        emb_t1 = base + 0.01 * torch.randn(N, D)  # very close to t0
        emb_t2 = base + 5.0 * torch.randn(N, D)    # far from both

        loss, stats = phase_triplet_loss(
            emb_t0, emb_t1, emb_t2, ysfc,
            t0_idx=2, t1_idx=6, t2_idx=12,
            large_margin=1.0, small_margin=0.3,
        )
        # Loss should be small (constraints well satisfied).
        assert loss.item() < 0.5
        assert stats["n_constraints"] == N * 2  # 2 per pixel, no disturbance
        assert stats["frac_satisfied"] > 0.5

    def test_positive_loss_when_violated(self):
        """If d_close > d_far, loss should be large."""
        N, D, T = 10, 16, 15
        ysfc = torch.ones(N, T)  # no disturbance

        # Embeddings: t0 and t2 are close, t1 is far.
        # For no-disturbance we want d(0,2) > d(0,1), but we set d(0,2) ≈ 0.
        torch.manual_seed(42)
        base = torch.randn(N, D)
        emb_t0 = base
        emb_t1 = base + 5.0 * torch.ones(N, D)  # far from t0
        emb_t2 = base + 0.01 * torch.randn(N, D)  # close to t0

        loss, stats = phase_triplet_loss(
            emb_t0, emb_t1, emb_t2, ysfc,
            t0_idx=2, t1_idx=6, t2_idx=12,
        )
        assert loss.item() > 1.0
        assert stats["frac_satisfied"] < 0.5

    def test_all_skipped_returns_zero(self):
        """When all pixels are skipped, loss = 0."""
        N, D, T = 5, 8, 15
        ysfc = torch.ones(N, T)
        ysfc[:, 6] = 0  # disturbance at sampled year → skip all

        emb = torch.randn(N, D)
        loss, stats = phase_triplet_loss(
            emb, emb, emb, ysfc,
            t0_idx=2, t1_idx=6, t2_idx=12,
        )
        assert loss.item() == 0.0
        assert stats["n_constraints"] == 0
        assert stats["n_pixels_skipped"] == N

    def test_gradient_flows(self):
        """Verify gradients propagate to embeddings."""
        N, D, T = 8, 16, 15
        ysfc = torch.ones(N, T)

        emb_t0 = torch.randn(N, D, requires_grad=True)
        emb_t1 = torch.randn(N, D, requires_grad=True)
        emb_t2 = torch.randn(N, D, requires_grad=True)

        loss, _ = phase_triplet_loss(
            emb_t0, emb_t1, emb_t2, ysfc,
            t0_idx=2, t1_idx=6, t2_idx=12,
        )
        loss.backward()
        assert emb_t0.grad is not None
        assert emb_t1.grad is not None
        assert emb_t2.grad is not None
        # At least some gradients should be non-zero.
        assert emb_t0.grad.abs().sum() > 0

    def test_d01_case_disturbance_respected(self):
        """With disturbance between t0-t1, embeddings reflecting that
        should have lower loss than embeddings contradicting it."""
        N, D, T = 20, 16, 15
        ysfc = torch.ones(N, T)
        ysfc[:, 5] = 0  # disturbance at year 5, between t0=2 and t1=8

        torch.manual_seed(0)
        base = torch.randn(N, D)

        # "Correct" layout: t0 far from t1 (disturbance), t1 close to t2 (recovery)
        emb_t0_good = base
        emb_t1_good = base + 5.0 * torch.randn(N, D)
        emb_t2_good = emb_t1_good + 0.1 * torch.randn(N, D)

        # "Wrong" layout: t0 close to t1, t1 far from t2
        emb_t0_bad = base
        emb_t1_bad = base + 0.1 * torch.randn(N, D)
        emb_t2_bad = base + 5.0 * torch.randn(N, D)

        loss_good, _ = phase_triplet_loss(
            emb_t0_good, emb_t1_good, emb_t2_good, ysfc,
            t0_idx=2, t1_idx=8, t2_idx=12,
        )
        loss_bad, _ = phase_triplet_loss(
            emb_t0_bad, emb_t1_bad, emb_t2_bad, ysfc,
            t0_idx=2, t1_idx=8, t2_idx=12,
        )
        assert loss_good.item() < loss_bad.item()

    def test_margin_values_affect_loss(self):
        """Larger margins should increase loss for the same embeddings."""
        N, D, T = 10, 16, 15
        ysfc = torch.ones(N, T)
        torch.manual_seed(42)
        emb_t0 = torch.randn(N, D)
        emb_t1 = torch.randn(N, D)
        emb_t2 = torch.randn(N, D)

        loss_small, _ = phase_triplet_loss(
            emb_t0, emb_t1, emb_t2, ysfc,
            t0_idx=2, t1_idx=6, t2_idx=12,
            large_margin=0.5, small_margin=0.1,
        )
        loss_large, _ = phase_triplet_loss(
            emb_t0, emb_t1, emb_t2, ysfc,
            t0_idx=2, t1_idx=6, t2_idx=12,
            large_margin=2.0, small_margin=1.0,
        )
        assert loss_large.item() > loss_small.item()
