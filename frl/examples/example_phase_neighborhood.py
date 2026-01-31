"""
Example: Phase neighborhood matching loss with ysfc alignment.

Demonstrates the full flow:

1. Simulated landscape with a mix of disturbed and undisturbed pixels.
2. ysfc alignment identifies shared recovery stages across pixels.
3. Reference distributions from Mahalanobis-whitened LS8 spectra.
4. Learned distributions from phase embeddings (synthetic here).
5. KL divergence trains embeddings to match spectral structure.
6. Type-similarity weighting focuses the loss on comparable pixels.

In real training, replace synthetic data with:
    - ``spectral_features`` from ``FeatureBuilder.build_feature('phase_ls8', sample)``
    - ``phase_embeddings`` from the TCN encoder
    - ``ysfc`` from ``FeatureBuilder.build_feature('ysfc', sample)``
    - ``pair_weights`` from type embedding similarity
"""

import torch

from losses.phase_neighborhood import (
    build_ysfc_overlap_mask,
    phase_neighborhood_loss,
)


def make_synthetic_landscape(
    N: int = 20,
    T: int = 15,
    C: int = 7,
    D: int = 64,
    n_disturbed: int = 8,
    seed: int = 42,
):
    """Create a synthetic landscape for demonstration.

    Returns spectral features, phase embeddings, ysfc arrays,
    pair indices, and pair weights.
    """
    torch.manual_seed(seed)

    # --- ysfc values -------------------------------------------------------
    # Undisturbed pixels: high ysfc counting up (long since any disturbance).
    # Disturbed pixels: have a disturbance event during the window.
    ysfc = torch.zeros(N, T)

    for i in range(N):
        if i < n_disturbed:
            # Disturbed pixel: disturbance at a random year.
            dist_year = torch.randint(2, T - 2, (1,)).item()
            # Pre-disturbance: counting up from some previous event.
            pre_start = torch.randint(10, 30, (1,)).item()
            for t in range(T):
                if t < dist_year:
                    ysfc[i, t] = pre_start + t
                else:
                    ysfc[i, t] = t - dist_year
        else:
            # Undisturbed: high ysfc, counting up.
            start = torch.randint(20, 40, (1,)).item()
            ysfc[i] = torch.arange(T).float() + start

    # --- Spectral features -------------------------------------------------
    # Simulate Mahalanobis-whitened LS8 spectra.
    # Undisturbed pixels: slow random walk.
    # Disturbed pixels: sharp spectral shift at disturbance, then recovery.
    spectral = torch.randn(N, T, C) * 0.1  # baseline noise

    for i in range(n_disturbed):
        dist_year = (ysfc[i] == 0).nonzero(as_tuple=False)[0].item()
        # Pre-disturbance: stable with noise.
        pre_level = torch.randn(C)
        spectral[i, :dist_year] = pre_level + torch.randn(dist_year, C) * 0.05

        # Disturbance: large spectral shift.
        shift = torch.randn(C) * 2.0
        # Post-disturbance: exponential recovery toward pre_level.
        for t in range(dist_year, T):
            years_since = t - dist_year
            recovery = 1 - torch.exp(torch.tensor(-years_since / 5.0))
            spectral[i, t] = pre_level + shift * (1 - recovery) + torch.randn(C) * 0.05

    # --- Phase embeddings (synthetic) --------------------------------------
    # In real training these come from the TCN encoder.
    phase_embeddings = torch.randn(N, T, D, requires_grad=True)

    # --- Pair construction -------------------------------------------------
    # Include self-pairs for all pixels, and cross-pixel pairs among
    # disturbed pixels (where ysfc overlap is informative).
    pairs_list = []

    # Self-pairs.
    for i in range(N):
        pairs_list.append([i, i])

    # Cross-pixel pairs among disturbed pixels.
    for i in range(n_disturbed):
        for j in range(i + 1, n_disturbed):
            pairs_list.append([i, j])
            pairs_list.append([j, i])  # asymmetric loss: both directions

    pair_indices = torch.tensor(pairs_list)

    # --- Pair weights from type similarity ---------------------------------
    # Simulate type embeddings and compute similarity weights.
    type_embeddings = torch.randn(N, 32)
    # Add cluster structure: first 10 pixels are type A, rest type B.
    type_embeddings[:10] += torch.tensor([1.0] * 16 + [0.0] * 16)
    type_embeddings[10:] += torch.tensor([0.0] * 16 + [1.0] * 16)

    pair_weights = torch.zeros(len(pairs_list))
    for b, (i, j) in enumerate(pairs_list):
        if i == j:
            pair_weights[b] = 1.0  # self-pairs always weight 1
        else:
            # Weight by type similarity (Gaussian kernel on L2 distance).
            type_dist = (type_embeddings[i] - type_embeddings[j]).pow(2).sum().sqrt()
            pair_weights[b] = torch.exp(-type_dist / 5.0).item()

    return spectral, phase_embeddings, ysfc, pair_indices, pair_weights


def example_overlap_inspection():
    """Show the ysfc overlap structure between example pixel pairs."""
    print("=" * 60)
    print("Example 1: ysfc overlap inspection")
    print("=" * 60)

    # Two pixels disturbed at different times.
    # Pixel 0: disturbed at t=3 → ysfc = [20, 21, 22, 0, 1, 2, 3, 4]
    # Pixel 1: disturbed at t=6 → ysfc = [15, 16, 17, 18, 19, 20, 0, 1]
    ysfc_0 = torch.tensor([20, 21, 22, 0, 1, 2, 3, 4], dtype=torch.float32)
    ysfc_1 = torch.tensor([15, 16, 17, 18, 19, 20, 0, 1], dtype=torch.float32)

    idx_0, idx_1, vals = build_ysfc_overlap_mask(ysfc_0, ysfc_1)

    print(f"\nPixel 0 ysfc: {ysfc_0.tolist()}")
    print(f"Pixel 1 ysfc: {ysfc_1.tolist()}")
    print(f"\nOverlap ysfc values: {sorted(set(vals.tolist()))}")
    print(f"Number of overlap entries: {idx_0.shape[0]}")
    print(f"\nAlignment mapping:")
    for k in range(idx_0.shape[0]):
        t_i = idx_0[k].item()
        t_j = idx_1[k].item()
        y = vals[k].item()
        print(
            f"  ysfc={y:>3.0f} → pixel 0 at t={t_i} "
            f"(calendar {2010+t_i}), pixel 1 at t={t_j} "
            f"(calendar {2010+t_j})"
        )


def example_loss_computation():
    """Compute the phase neighborhood loss on a synthetic landscape."""
    print("\n" + "=" * 60)
    print("Example 2: Phase neighborhood loss computation")
    print("=" * 60)

    spectral, embeddings, ysfc, pairs, weights = make_synthetic_landscape()

    N, T = ysfc.shape
    print(f"\nLandscape: {N} pixels, {T} time steps")
    print(f"Pairs: {pairs.shape[0]} ({N} self + {pairs.shape[0] - N} cross-pixel)")
    print(f"Disturbed pixels: 8")
    print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Compute loss.
    loss, stats = phase_neighborhood_loss(
        spectral_features=spectral,
        phase_embeddings=embeddings,
        ysfc=ysfc,
        pair_indices=pairs,
        pair_weights=weights,
        tau_ref=0.1,
        tau_learned=0.1,
        min_overlap=3,
    )

    print(f"\nLoss: {loss.item():.4f}")
    print(f"\nStatistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Verify gradient flow.
    loss.backward()
    grad_norm = embeddings.grad.norm().item()
    print(f"\nGradient norm on phase embeddings: {grad_norm:.4f}")


def example_self_vs_cross():
    """Compare loss contributions from self-pairs vs cross-pixel pairs."""
    print("\n" + "=" * 60)
    print("Example 3: Self-pair vs cross-pixel loss")
    print("=" * 60)

    spectral, embeddings, ysfc, _, _ = make_synthetic_landscape()
    N = spectral.shape[0]

    # Self-pairs only.
    self_pairs = torch.tensor([[i, i] for i in range(N)])
    emb_detached = embeddings.detach().requires_grad_(True)

    loss_self, stats_self = phase_neighborhood_loss(
        spectral, emb_detached, ysfc, self_pairs,
        tau_ref=0.1, tau_learned=0.1,
    )
    print(f"\nSelf-pairs only:")
    print(f"  Loss: {loss_self.item():.4f}")
    print(f"  Active pairs: {stats_self['n_pairs_active']}")
    print(f"  Valid rows: {stats_self['n_rows_valid']}")

    # Cross-pixel pairs among first 8 (disturbed) pixels.
    cross_pairs = []
    for i in range(8):
        for j in range(i + 1, 8):
            cross_pairs.append([i, j])
    cross_pairs = torch.tensor(cross_pairs)

    emb_detached2 = embeddings.detach().requires_grad_(True)
    loss_cross, stats_cross = phase_neighborhood_loss(
        spectral, emb_detached2, ysfc, cross_pairs,
        tau_ref=0.1, tau_learned=0.1,
    )
    print(f"\nCross-pixel pairs (disturbed only):")
    print(f"  Loss: {loss_cross.item():.4f}")
    print(f"  Active pairs: {stats_cross['n_pairs_active']}")
    print(f"  Valid rows: {stats_cross['n_rows_valid']}")
    print(f"  Mean overlap: {stats_cross['mean_overlap']:.1f}")


if __name__ == "__main__":
    example_overlap_inspection()
    example_loss_computation()
    example_self_vs_cross()
