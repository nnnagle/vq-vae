"""Tests for losses.soft_neighborhood and losses.phase_neighborhood."""

from __future__ import annotations

import pytest
import torch

from losses.soft_neighborhood import soft_neighborhood_matching_loss
from losses.phase_neighborhood import (
    average_features_by_ysfc,
    build_phase_neighborhood_batch,
    build_ysfc_overlap,
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
    """Tests for build_ysfc_overlap."""

    def test_full_overlap_self_pair(self):
        """A pixel paired with itself has full overlap."""
        ysfc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
        shared, groups_i, groups_j = build_ysfc_overlap(ysfc, ysfc)

        assert shared.shape[0] == 5
        assert len(groups_i) == 5
        assert len(groups_j) == 5
        # For self-pair, groups should be identical.
        for gi, gj in zip(groups_i, groups_j):
            assert torch.equal(gi, gj)

    def test_partial_overlap(self):
        """Two pixels with partial ysfc overlap."""
        ysfc_i = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
        ysfc_j = torch.tensor([3, 4, 5, 6, 7], dtype=torch.float32)
        shared, groups_i, groups_j = build_ysfc_overlap(ysfc_i, ysfc_j)

        assert set(shared.tolist()) == {3.0, 4.0}
        assert len(groups_i) == 2

        # ysfc=3 at i: t=3, at j: t=0
        idx_3 = (shared == 3.0).nonzero(as_tuple=False).item()
        assert groups_i[idx_3].tolist() == [3]
        assert groups_j[idx_3].tolist() == [0]

    def test_no_overlap(self):
        """Disjoint ysfc ranges produce empty overlap."""
        ysfc_i = torch.tensor([0, 1, 2], dtype=torch.float32)
        ysfc_j = torch.tensor([5, 6, 7], dtype=torch.float32)
        shared, groups_i, groups_j = build_ysfc_overlap(ysfc_i, ysfc_j)

        assert shared.shape[0] == 0
        assert len(groups_i) == 0

    def test_stuttering_ysfc_groups(self):
        """Repeated ysfc values produce groups with multiple time indices."""
        ysfc_i = torch.tensor([0, 0, 1, 2, 3], dtype=torch.float32)
        ysfc_j = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)
        shared, groups_i, groups_j = build_ysfc_overlap(ysfc_i, ysfc_j)

        # ysfc=0: i has 2 time steps, j has 1.
        idx_0 = (shared == 0.0).nonzero(as_tuple=False).item()
        assert groups_i[idx_0].tolist() == [0, 1]  # two occurrences
        assert groups_j[idx_0].tolist() == [0]  # one occurrence

    def test_pre_and_post_disturbance_overlap(self):
        """Pixel with pre- and post-disturbance ysfc values overlaps
        with a post-disturbance-only pixel."""
        ysfc_i = torch.tensor([20, 21, 0, 1, 2], dtype=torch.float32)
        ysfc_j = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32)

        shared, groups_i, groups_j = build_ysfc_overlap(ysfc_i, ysfc_j)
        assert set(shared.tolist()) == {0.0, 1.0, 2.0}

    def test_sorted_output(self):
        """Shared values should be sorted."""
        ysfc_i = torch.tensor([4, 2, 0, 1, 3], dtype=torch.float32)
        ysfc_j = torch.tensor([3, 1, 4, 0, 2], dtype=torch.float32)
        shared, _, _ = build_ysfc_overlap(ysfc_i, ysfc_j)
        assert shared.tolist() == sorted(shared.tolist())


# ══════════════════════════════════════════════════════════════════════════
# Feature averaging
# ══════════════════════════════════════════════════════════════════════════


class TestAverageFeatures:

    def test_single_entry_groups(self):
        """When each group has one entry, averaging is identity."""
        T, C = 5, 3
        features = torch.randn(T, C)
        groups = [torch.tensor([t]) for t in range(T)]
        result = average_features_by_ysfc(features, groups, K=T)
        assert torch.allclose(result, features)

    def test_averaging_reduces_duplicates(self):
        """Groups with multiple entries produce the mean."""
        features = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        # Group 0: t=0,1 → mean [2, 3]
        # Group 1: t=2,3 → mean [6, 7]
        groups = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        result = average_features_by_ysfc(features, groups, K=2)

        expected = torch.tensor([[2.0, 3.0], [6.0, 7.0]])
        assert torch.allclose(result, expected)

    def test_gradient_flows_through_averaging(self):
        """Gradients should propagate through the averaging."""
        features = torch.randn(5, 3, requires_grad=True)
        groups = [torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([3, 4])]
        result = average_features_by_ysfc(features, groups, K=3)
        result.sum().backward()

        assert features.grad is not None
        # t=0 and t=1 are in a group of 2, so gradient = 1/2.
        assert torch.allclose(features.grad[0], torch.tensor([0.5, 0.5, 0.5]))
        # t=2 is alone, gradient = 1.
        assert torch.allclose(features.grad[2], torch.tensor([1.0, 1.0, 1.0]))


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

        pairs = torch.tensor([[0, 0], [1, 1], [2, 2]])
        batch = build_phase_neighborhood_batch(spec, emb, ysfc, pairs, min_overlap=3)

        assert batch["valid_pair_mask"].all()
        assert batch["M"] == 5
        # Self-similarity mask: 5×5 minus diagonal = 20 per pair.
        assert batch["mask_self"].sum(dim=(1, 2)).tolist() == [20, 20, 20]
        # Cross-pixel mask: 5×5 = 25 per pair (diagonal included).
        assert batch["mask_cross"].sum(dim=(1, 2)).tolist() == [25, 25, 25]

    def test_insufficient_overlap_filtered(self):
        """Pairs with too little overlap are excluded."""
        N, T, C, D = 2, 5, 4, 8
        torch.manual_seed(0)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)

        ysfc = torch.tensor([
            [0, 1, 2, 3, 4],
            [3, 4, 5, 6, 7],
        ], dtype=torch.float32)

        pairs = torch.tensor([[0, 1]])
        batch = build_phase_neighborhood_batch(spec, emb, ysfc, pairs, min_overlap=3)

        assert not batch["valid_pair_mask"][0].item()
        assert batch["d_ref_self"].shape[0] == 0

    def test_stuttering_ysfc_averaging(self):
        """Pixels with repeated ysfc values should produce averaged features."""
        N, T, C, D = 1, 5, 2, 2
        # ysfc has ysfc=0 at t=0 and t=1.
        ysfc = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.float32)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)

        pairs = torch.tensor([[0, 0]])
        batch = build_phase_neighborhood_batch(spec, emb, ysfc, pairs, min_overlap=3)

        # 4 unique ysfc values: {0, 1, 2, 3}
        assert batch["valid_pair_mask"][0].item()
        assert batch["M"] == 4

    def test_both_distance_types_computed(self):
        """Both self-similarity and cross-pixel matrices should be non-zero."""
        N, T, C, D = 3, 8, 4, 8
        torch.manual_seed(0)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)
        ysfc = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(N, T)

        pairs = torch.tensor([[0, 1], [1, 2]])
        batch = build_phase_neighborhood_batch(spec, emb, ysfc, pairs, min_overlap=3)

        assert batch["d_ref_self"].abs().sum() > 0
        assert batch["d_ref_cross"].abs().sum() > 0
        assert batch["d_learned_self"].abs().sum() > 0
        assert batch["d_learned_cross"].abs().sum() > 0


# ══════════════════════════════════════════════════════════════════════════
# End-to-end phase neighborhood loss
# ══════════════════════════════════════════════════════════════════════════


class TestPhaseNeighborhoodLoss:

    def test_self_pair_identical_embeddings_low_loss(self):
        """Self-pair where embeddings reproduce spectral structure → low loss."""
        N, T, C = 1, 8, 4
        torch.manual_seed(42)
        spec = torch.randn(N, T, C)
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
        assert stats["loss_self"] < 0.1
        assert stats["n_pairs_sufficient_overlap"] == 1

    def test_random_embeddings_higher_loss(self):
        """Random embeddings should produce higher loss than matched ones."""
        N, T, C = 2, 8, 4
        torch.manual_seed(42)
        spec = torch.randn(N, T, C)

        ysfc = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(N, T)
        pairs = torch.tensor([[0, 0], [1, 1]])

        emb_good = spec.clone()
        loss_good, _ = phase_neighborhood_loss(
            spec, emb_good, ysfc, pairs, tau_ref=0.1, tau_learned=0.1,
        )

        torch.manual_seed(999)
        emb_bad = torch.randn(N, T, C)
        loss_bad, _ = phase_neighborhood_loss(
            spec, emb_bad, ysfc, pairs, tau_ref=0.1, tau_learned=0.1,
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
            spec, emb, ysfc, pairs, tau_ref=0.1, tau_learned=0.1,
        )
        loss.backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum() > 0

    def test_no_valid_pairs_returns_zero(self):
        """When all pairs have insufficient overlap, loss is zero."""
        N, T, C, D = 2, 5, 4, 8
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)
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

        emb_mod = emb.clone()
        emb_mod[0] = emb_mod[0] * 10.0

        w_high_on_0 = torch.tensor([10.0, 1.0, 1.0])
        loss_high, _ = phase_neighborhood_loss(
            spec, emb_mod, ysfc, pairs, pair_weights=w_high_on_0,
        )

        w_low_on_0 = torch.tensor([0.1, 1.0, 1.0])
        loss_low, _ = phase_neighborhood_loss(
            spec, emb_mod, ysfc, pairs, pair_weights=w_low_on_0,
        )

        assert loss_high.item() > loss_low.item()

    def test_cross_pixel_disturbance_alignment(self):
        """Two pixels disturbed at different calendar years should produce
        a valid loss via ysfc alignment."""
        T, C, D = 10, 4, 8
        torch.manual_seed(0)

        ysfc = torch.tensor([
            [20, 21, 0, 1, 2, 3, 4, 5, 6, 7],
            [20, 21, 22, 23, 24, 0, 1, 2, 3, 4],
        ], dtype=torch.float32)

        spec = torch.randn(2, T, C)
        emb = torch.randn(2, T, D, requires_grad=True)
        pairs = torch.tensor([[0, 1]])

        loss, stats = phase_neighborhood_loss(
            spec, emb, ysfc, pairs, min_overlap=3,
        )

        assert stats["n_pairs_sufficient_overlap"] == 1
        assert loss.item() > 0.0
        loss.backward()
        assert emb.grad is not None

    def test_self_and_cross_loss_reported_separately(self):
        """Stats should report self-similarity and cross-pixel losses."""
        N, T, C, D = 3, 6, 4, 8
        torch.manual_seed(0)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)
        ysfc = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(N, T)

        pairs = torch.tensor([[0, 0], [0, 1]])

        loss, stats = phase_neighborhood_loss(
            spec, emb, ysfc, pairs, tau_ref=0.1, tau_learned=0.1,
        )

        assert "loss_self" in stats
        assert "loss_cross" in stats
        assert stats["loss_self"] >= 0.0
        assert stats["loss_cross"] >= 0.0

    def test_loss_weights(self):
        """self_similarity_weight and cross_pixel_weight should control
        the contribution of each term."""
        N, T, C, D = 2, 8, 4, 8
        torch.manual_seed(42)
        spec = torch.randn(N, T, C)
        emb = torch.randn(N, T, D)
        ysfc = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(N, T)

        pairs = torch.tensor([[0, 1]])

        loss_self_only, stats_self_only = phase_neighborhood_loss(
            spec, emb, ysfc, pairs,
            self_similarity_weight=1.0, cross_pixel_weight=0.0,
        )
        loss_cross_only, stats_cross_only = phase_neighborhood_loss(
            spec, emb, ysfc, pairs,
            self_similarity_weight=0.0, cross_pixel_weight=1.0,
        )

        assert abs(loss_self_only.item() - stats_self_only["loss_self"]) < 1e-5
        assert abs(loss_cross_only.item() - stats_cross_only["loss_cross"]) < 1e-5
