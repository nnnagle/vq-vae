"""Tests for losses.soft_neighborhood and losses.phase_neighborhood."""

from __future__ import annotations

import pytest
import torch

from losses.soft_neighborhood import soft_neighborhood_matching_loss
from losses.phase_neighborhood import (
    build_ysfc_overlap_mask,
    align_distance_matrices,
    build_phase_neighborhood_batch,
    phase_neighborhood_loss,
)


# ══════════════════════════════════════════════════════════════════════════
# Generic soft neighborhood matching loss
# ══════════════════════════════════════════════════════════════════════════


class TestSoftNeighborhoodLoss:
    """Tests for the generic soft_neighborhood_matching_loss."""

    def test_identical_distributions_give_zero_loss(self):
        """When reference and learned distances are identical, KL = 0."""
        B, M = 4, 6
        d = torch.rand(B, M, M).abs()
        d = (d + d.transpose(1, 2)) / 2  # symmetric

        mask = torch.ones(B, M, M, dtype=torch.bool)
        mask[:, range(M), range(M)] = False  # exclude diagonal

        loss, stats = soft_neighborhood_matching_loss(
            d_reference=d,
            d_learned=d,
            mask=mask,
            tau_ref=0.1,
            tau_learned=0.1,
        )
        assert loss.item() < 1e-5
        assert stats["n_rows_valid"] == B * M

    def test_different_distributions_give_positive_loss(self):
        """Different distance matrices should produce positive KL."""
        B, M = 4, 6
        torch.manual_seed(42)
        d_ref = torch.rand(B, M, M).abs()
        d_learned = torch.rand(B, M, M).abs()

        mask = torch.ones(B, M, M, dtype=torch.bool)
        mask[:, range(M), range(M)] = False

        loss, stats = soft_neighborhood_matching_loss(
            d_reference=d_ref,
            d_learned=d_learned,
            mask=mask,
            tau_ref=0.1,
            tau_learned=0.1,
        )
        assert loss.item() > 0.0

    def test_fully_masked_returns_zero(self):
        """All entries masked → zero loss."""
        B, M = 2, 4
        d = torch.rand(B, M, M)
        mask = torch.zeros(B, M, M, dtype=torch.bool)

        loss, stats = soft_neighborhood_matching_loss(d, d, mask)
        assert loss.item() == 0.0
        assert stats["n_rows_valid"] == 0

    def test_partial_mask_excludes_padding(self):
        """Rows with fewer than min_valid_per_row entries are skipped."""
        B, M = 1, 5
        d = torch.rand(B, M, M).abs()
        mask = torch.zeros(B, M, M, dtype=torch.bool)
        # Only rows 0 and 1 have enough valid entries.
        mask[0, 0, 1:4] = True  # 3 valid
        mask[0, 1, [0, 2, 3]] = True  # 3 valid
        mask[0, 2, 3] = True  # only 1 valid → skip
        # Rows 3, 4: all masked → skip

        loss, stats = soft_neighborhood_matching_loss(
            d, d, mask, min_valid_per_row=2,
        )
        assert stats["n_rows_valid"] == 2

    def test_pair_weights(self):
        """Higher weight on a pair with larger KL should increase total loss."""
        B, M = 2, 5
        torch.manual_seed(0)

        mask = torch.ones(B, M, M, dtype=torch.bool)
        mask[:, range(M), range(M)] = False

        d_ref = torch.rand(B, M, M).abs()
        d_learned = torch.rand(B, M, M).abs()

        # Make pair 0 have very different distances (high KL).
        d_learned[0] = d_ref[0] * 5.0

        # Weight pair 0 high.
        w_high_on_0 = torch.tensor([10.0, 1.0])
        loss_high, _ = soft_neighborhood_matching_loss(
            d_ref, d_learned, mask, pair_weights=w_high_on_0,
        )

        # Weight pair 0 low.
        w_low_on_0 = torch.tensor([0.1, 1.0])
        loss_low, _ = soft_neighborhood_matching_loss(
            d_ref, d_learned, mask, pair_weights=w_low_on_0,
        )

        assert loss_high.item() > loss_low.item()

    def test_gradient_flows(self):
        """Gradients should propagate through d_learned."""
        B, M = 3, 6
        d_ref = torch.rand(B, M, M).abs()
        d_learned = torch.rand(B, M, M, requires_grad=True)

        mask = torch.ones(B, M, M, dtype=torch.bool)
        mask[:, range(M), range(M)] = False

        loss, _ = soft_neighborhood_matching_loss(
            d_ref, d_learned, mask,
        )
        loss.backward()
        assert d_learned.grad is not None
        assert d_learned.grad.abs().sum() > 0

    def test_temperature_affects_sharpness(self):
        """Lower temperature should produce sharper distributions and
        potentially different loss values."""
        B, M = 2, 6
        torch.manual_seed(42)
        d_ref = torch.rand(B, M, M).abs()
        d_learned = torch.rand(B, M, M).abs()

        mask = torch.ones(B, M, M, dtype=torch.bool)
        mask[:, range(M), range(M)] = False

        loss_sharp, _ = soft_neighborhood_matching_loss(
            d_ref, d_learned, mask, tau_ref=0.01, tau_learned=0.01,
        )
        loss_soft, _ = soft_neighborhood_matching_loss(
            d_ref, d_learned, mask, tau_ref=1.0, tau_learned=1.0,
        )
        # These should differ (not asserting direction — just that
        # temperature has an effect).
        assert abs(loss_sharp.item() - loss_soft.item()) > 1e-4

    def test_min_valid_per_row_validation(self):
        """min_valid_per_row < 2 should raise."""
        with pytest.raises(ValueError, match="min_valid_per_row"):
            soft_neighborhood_matching_loss(
                torch.zeros(1, 3, 3),
                torch.zeros(1, 3, 3),
                torch.ones(1, 3, 3, dtype=torch.bool),
                min_valid_per_row=1,
            )


# ══════════════════════════════════════════════════════════════════════════
# ysfc overlap utilities
# ══════════════════════════════════════════════════════════════════════════


class TestYsfcOverlap:
    """Tests for build_ysfc_overlap_mask."""

    def test_full_overlap_self_pair(self):
        """A pixel paired with itself has full overlap."""
        ysfc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
        idx_i, idx_j, vals = build_ysfc_overlap_mask(ysfc, ysfc)

        # Every time step maps to itself.
        assert idx_i.shape[0] == 5
        assert idx_j.shape[0] == 5
        # Indices should be identical for self-pair.
        assert torch.equal(idx_i, idx_j)

    def test_partial_overlap(self):
        """Two pixels with partial ysfc overlap."""
        ysfc_i = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
        ysfc_j = torch.tensor([3, 4, 5, 6, 7], dtype=torch.float32)
        idx_i, idx_j, vals = build_ysfc_overlap_mask(ysfc_i, ysfc_j)

        # Overlap values: {3, 4}
        assert set(vals.tolist()) == {3.0, 4.0}
        assert idx_i.shape[0] == 2  # one time step per value, no duplicates

        # Check correct time indices.
        # ysfc=3 at i: t=3, at j: t=0
        # ysfc=4 at i: t=4, at j: t=1
        val_3_mask = vals == 3.0
        assert idx_i[val_3_mask].item() == 3
        assert idx_j[val_3_mask].item() == 0

    def test_no_overlap(self):
        """Disjoint ysfc ranges produce empty overlap."""
        ysfc_i = torch.tensor([0, 1, 2], dtype=torch.float32)
        ysfc_j = torch.tensor([5, 6, 7], dtype=torch.float32)
        idx_i, idx_j, vals = build_ysfc_overlap_mask(ysfc_i, ysfc_j)

        assert idx_i.shape[0] == 0
        assert idx_j.shape[0] == 0

    def test_stuttering_ysfc(self):
        """Repeated ysfc values (stuttering) produce cross-product entries."""
        ysfc_i = torch.tensor([0, 0, 1, 2, 3], dtype=torch.float32)
        ysfc_j = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
        idx_i, idx_j, vals = build_ysfc_overlap_mask(ysfc_i, ysfc_j)

        # ysfc=0: i has 2 occurrences (t=0,1), j has 1 (t=0) → 2 entries
        # ysfc=1: i has 1 (t=2), j has 1 (t=1) → 1 entry
        # ysfc=2: 1 × 1 = 1
        # ysfc=3: 1 × 1 = 1
        # Total: 5
        assert idx_i.shape[0] == 5
        n_val_0 = (vals == 0.0).sum().item()
        assert n_val_0 == 2

    def test_pre_and_post_disturbance_overlap(self):
        """Pixel with pre- and post-disturbance ysfc values overlaps
        with a post-disturbance-only pixel."""
        # Pixel i: disturbed at t=2 → ysfc = [20, 21, 0, 1, 2]
        ysfc_i = torch.tensor([20, 21, 0, 1, 2], dtype=torch.float32)
        # Pixel j: disturbed at t=0 → ysfc = [0, 1, 2, 3, 4]
        ysfc_j = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)

        idx_i, idx_j, vals = build_ysfc_overlap_mask(ysfc_i, ysfc_j)

        # Overlap: {0, 1, 2}
        assert set(vals.tolist()) == {0.0, 1.0, 2.0}
        assert idx_i.shape[0] == 3


# ══════════════════════════════════════════════════════════════════════════
# Alignment
# ══════════════════════════════════════════════════════════════════════════


class TestAlignDistanceMatrices:

    def test_basic_alignment(self):
        """Aligned matrix has correct values from the full matrix."""
        T = 5
        # Symmetric distance matrix.
        d_full = torch.arange(T * T, dtype=torch.float32).reshape(T, T)
        d_full = (d_full + d_full.T) / 2

        # Suppose overlap maps to time indices [1, 3] at this pixel.
        indices = torch.tensor([1, 3])
        M = 3  # pad to 3

        d_ref, d_learn, mask = align_distance_matrices(
            d_full, d_full, indices, indices, M,
        )

        # Top-left 2×2 should have values from d_full at [1,3]×[1,3].
        assert d_ref[0, 0].item() == d_full[1, 1].item()
        assert d_ref[0, 1].item() == d_full[1, 3].item()
        assert d_ref[1, 0].item() == d_full[3, 1].item()
        assert d_ref[1, 1].item() == d_full[3, 3].item()

        # Padding row/col 2 should be zero.
        assert d_ref[2, :].sum().item() == 0.0
        assert d_ref[:, 2].sum().item() == 0.0

        # Mask: diagonal false, padding false.
        assert mask[0, 1].item() is True
        assert mask[1, 0].item() is True
        assert mask[0, 0].item() is False  # diagonal
        assert mask[2, 0].item() is False  # padding

    def test_empty_overlap(self):
        """Empty indices produce all-zero matrices and all-false mask."""
        d_full = torch.rand(5, 5)
        indices = torch.empty(0, dtype=torch.long)
        d_ref, d_learn, mask = align_distance_matrices(
            d_full, d_full, indices, indices, M=5,
        )
        assert d_ref.sum().item() == 0.0
        assert mask.any().item() is False


# ══════════════════════════════════════════════════════════════════════════
# Batch construction
# ══════════════════════════════════════════════════════════════════════════


class TestBuildBatch:

    def test_self_pairs_full_overlap(self):
        """Self-pairs should have full T overlap."""
        N, T, C, D = 3, 5, 4, 8
        torch.manual_seed(0)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)
        ysfc = torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [3, 4, 5, 6, 7],
        ], dtype=torch.float32)

        # Self-pairs: (0,0), (1,1), (2,2)
        pairs = torch.tensor([[0, 0], [1, 1], [2, 2]])

        d_ref, d_learn, mask, valid, M = build_phase_neighborhood_batch(
            spec, emb, ysfc, pairs, min_overlap=3,
        )

        assert valid.all()
        assert M == 5  # full overlap for self-pairs
        # All self-pairs should have 5×5 masks (minus diagonal).
        assert mask.sum(dim=(1, 2)).tolist() == [20, 20, 20]

    def test_insufficient_overlap_filtered(self):
        """Pairs with too little overlap are excluded."""
        N, T, C, D = 2, 5, 4, 8
        torch.manual_seed(0)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)

        # Only 2 values overlap — below min_overlap=3.
        ysfc = torch.tensor([
            [0, 1, 2, 3, 4],
            [3, 4, 5, 6, 7],
        ], dtype=torch.float32)

        pairs = torch.tensor([[0, 1]])
        d_ref, d_learn, mask, valid, M = build_phase_neighborhood_batch(
            spec, emb, ysfc, pairs, min_overlap=3,
        )

        assert not valid[0].item()
        assert d_ref.shape[0] == 0

    def test_mixed_valid_and_invalid_pairs(self):
        """Batch with both valid and invalid pairs."""
        N, T, C, D = 3, 5, 4, 8
        torch.manual_seed(0)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)
        ysfc = torch.tensor([
            [0, 1, 2, 3, 4],  # pixel 0
            [2, 3, 4, 5, 6],  # pixel 1: overlap with 0 = {2,3,4} = 3
            [8, 9, 10, 11, 12],  # pixel 2: no overlap with 0
        ], dtype=torch.float32)

        pairs = torch.tensor([[0, 0], [0, 1], [0, 2]])
        d_ref, d_learn, mask, valid, M = build_phase_neighborhood_batch(
            spec, emb, ysfc, pairs, min_overlap=3,
        )

        assert valid.tolist() == [True, True, False]
        assert d_ref.shape[0] == 2  # 2 valid pairs


# ══════════════════════════════════════════════════════════════════════════
# End-to-end phase neighborhood loss
# ══════════════════════════════════════════════════════════════════════════


class TestPhaseNeighborhoodLoss:

    def test_self_pair_identical_embeddings_low_loss(self):
        """Self-pair where embeddings reproduce spectral structure → low loss."""
        N, T, C, D = 1, 8, 4, 4
        torch.manual_seed(42)
        spec = torch.randn(N, T, C)
        # Make embeddings proportional to spectral features.
        emb = spec.clone()

        ysfc = torch.arange(T, dtype=torch.float32).unsqueeze(0)
        pairs = torch.tensor([[0, 0]])

        loss, stats = phase_neighborhood_loss(
            spectral_features=spec,
            phase_embeddings=emb,
            ysfc=ysfc,
            pair_indices=pairs,
            tau_ref=0.1,
            tau_learned=0.1,
            min_overlap=3,
        )
        assert loss.item() < 0.1
        assert stats["n_pairs_sufficient_overlap"] == 1

    def test_random_embeddings_higher_loss(self):
        """Random embeddings should produce higher loss than matched ones."""
        N, T, C, D = 2, 8, 4, 8
        torch.manual_seed(42)
        spec = torch.randn(N, T, C)

        ysfc = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(N, T)
        pairs = torch.tensor([[0, 0], [1, 1]])

        # Matched embeddings.
        emb_good = spec[:, :, :D] if D <= C else torch.cat(
            [spec, torch.randn(N, T, D - C)], dim=2
        )
        # Use spectral features directly as embeddings (same distance structure).
        emb_good = spec.clone()

        loss_good, _ = phase_neighborhood_loss(
            spec, emb_good, ysfc, pairs,
            tau_ref=0.1, tau_learned=0.1,
        )

        # Random embeddings.
        torch.manual_seed(999)
        emb_bad = torch.randn(N, T, C)

        loss_bad, _ = phase_neighborhood_loss(
            spec, emb_bad, ysfc, pairs,
            tau_ref=0.1, tau_learned=0.1,
        )

        assert loss_good.item() < loss_bad.item()

    def test_gradient_flows_to_embeddings(self):
        """Gradients should propagate to phase_embeddings."""
        N, T, C, D = 3, 6, 4, 8
        torch.manual_seed(0)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D, requires_grad=True)
        ysfc = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(N, T)

        pairs = torch.tensor([[0, 0], [0, 1], [1, 2]])

        loss, _ = phase_neighborhood_loss(
            spec, emb, ysfc, pairs,
            tau_ref=0.1, tau_learned=0.1,
        )
        loss.backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum() > 0

    def test_no_valid_pairs_returns_zero(self):
        """When all pairs have insufficient overlap, loss is zero."""
        N, T, C, D = 2, 5, 4, 8
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)
        # Disjoint ysfc.
        ysfc = torch.tensor([
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
        ], dtype=torch.float32)

        pairs = torch.tensor([[0, 1]])

        loss, stats = phase_neighborhood_loss(
            spec, emb, ysfc, pairs, min_overlap=3,
        )
        assert loss.item() == 0.0
        assert stats["n_pairs_sufficient_overlap"] == 0

    def test_pair_weights_affect_loss(self):
        """Type similarity weights should modulate the loss."""
        N, T, C, D = 3, 8, 4, 8
        torch.manual_seed(42)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)
        ysfc = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(N, T)

        pairs = torch.tensor([[0, 0], [1, 1], [2, 2]])

        # Make pixel 0's embeddings very wrong.
        emb_mod = emb.clone()
        emb_mod[0] = emb_mod[0] * 10.0

        w_high_on_0 = torch.tensor([10.0, 1.0, 1.0])
        loss_high, _ = phase_neighborhood_loss(
            spec, emb_mod, ysfc, pairs,
            pair_weights=w_high_on_0,
        )

        w_low_on_0 = torch.tensor([0.1, 1.0, 1.0])
        loss_low, _ = phase_neighborhood_loss(
            spec, emb_mod, ysfc, pairs,
            pair_weights=w_low_on_0,
        )

        assert loss_high.item() > loss_low.item()

    def test_cross_pixel_disturbance_alignment(self):
        """Two pixels disturbed at different calendar years but with
        overlapping ysfc should produce a valid loss."""
        T, C, D = 10, 4, 8
        torch.manual_seed(0)

        # Pixel 0: disturbed at t=2 → ysfc = [20, 21, 0, 1, 2, 3, 4, 5, 6, 7]
        # Pixel 1: disturbed at t=5 → ysfc = [20, 21, 22, 23, 24, 0, 1, 2, 3, 4]
        ysfc = torch.tensor([
            [20, 21, 0, 1, 2, 3, 4, 5, 6, 7],
            [20, 21, 22, 23, 24, 0, 1, 2, 3, 4],
        ], dtype=torch.float32)

        spec = torch.randn(2, T, C)
        emb = torch.randn(2, T, D, requires_grad=True)
        pairs = torch.tensor([[0, 1]])

        loss, stats = phase_neighborhood_loss(
            spec, emb, ysfc, pairs,
            min_overlap=3,
        )

        # Should have overlap on {0, 1, 2, 3, 4, 20, 21} = 7 values.
        assert stats["n_pairs_sufficient_overlap"] == 1
        assert loss.item() > 0.0

        loss.backward()
        assert emb.grad is not None
