"""
Example: Phase triplet loss with ysfc from the data pipeline.

Shows how to:
1. Build the ``ysfc`` feature from a dataset sample
2. Sample three years from the time window
3. Extract embeddings at spatial anchor points
4. Compute the phase triplet loss

This is a sketch of how the loss integrates into training.
It uses synthetic embeddings in place of the real TCN encoder.
"""

import numpy as np
import torch

from losses.triplet_phase import phase_triplet_loss


def example_phase_triplet_loss():
    # ------------------------------------------------------------------
    # 1. Simulate what the data pipeline produces for ysfc
    # ------------------------------------------------------------------
    # Time window 2010–2024 → T = 15 annual steps
    T = 15
    H, W = 256, 256

    # ysfc feature comes from FeatureBuilder as [C=1, T, H, W] float32.
    # ysfc(t) == 0 means disturbance in that year.
    # Here: simulate a landscape where ~5% of pixels have a disturbance
    # at year index 7 (i.e., calendar year 2017).
    rng = np.random.default_rng(42)
    ysfc_raw = np.ones((1, T, H, W), dtype=np.float32)
    disturbed_mask = rng.random((H, W)) < 0.05
    ysfc_raw[0, 7, disturbed_mask] = 0  # fast change in 2017

    # In the real pipeline this comes from:
    #   ysfc_result = feature_builder.build_feature('ysfc', sample)
    #   ysfc_raw = ysfc_result.data  # [1, T, H, W]

    # ------------------------------------------------------------------
    # 2. Sample three years
    # ------------------------------------------------------------------
    # Final year = last index; two others that are non-consecutive.
    t2_idx = T - 1          # 2024 (index 14)
    t0_idx = 2              # 2012 (index 2)
    t1_idx = 8              # 2018 (index 8)
    # Sorted: t0 < t1 < t2

    # ------------------------------------------------------------------
    # 3. Extract ysfc at spatial anchor points
    # ------------------------------------------------------------------
    # In training, anchors are selected on a grid + supplement.
    # Here: sample N random spatial locations.
    N = 200
    rows = rng.integers(0, H, size=N)
    cols = rng.integers(0, W, size=N)

    # ysfc for each anchor pixel: [N, T]
    # Squeeze out the C=1 dimension first → [T, H, W]
    ysfc_thw = ysfc_raw[0]  # [T, H, W]
    ysfc_pixels = torch.from_numpy(
        ysfc_thw[:, rows, cols].T  # [N, T]
    )

    # ------------------------------------------------------------------
    # 4. Synthetic embeddings (replace with TCN output in real training)
    # ------------------------------------------------------------------
    D = 64  # embedding dimension
    torch.manual_seed(0)

    # In reality:
    #   phase_ls8 = feature_builder.build_feature('phase_ls8', sample)
    #   tcn_out = tcn_encoder(phase_ls8.data)  # [D, T, H, W]
    #   emb_t0 = tcn_out[:, t0_idx, rows, cols].T  # [N, D]
    #   emb_t1 = tcn_out[:, t1_idx, rows, cols].T
    #   emb_t2 = tcn_out[:, t2_idx, rows, cols].T

    emb_t0 = torch.randn(N, D, requires_grad=True)
    emb_t1 = torch.randn(N, D, requires_grad=True)
    emb_t2 = torch.randn(N, D, requires_grad=True)

    # ------------------------------------------------------------------
    # 5. Compute loss
    # ------------------------------------------------------------------
    loss, stats = phase_triplet_loss(
        emb_t0, emb_t1, emb_t2,
        ysfc_pixels,
        t0_idx=t0_idx,
        t1_idx=t1_idx,
        t2_idx=t2_idx,
        large_margin=1.0,
        small_margin=0.3,
    )

    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
    print(f"  Total constraints: {stats['n_constraints']}")
    print(f"  Large margin:      {stats['n_large']}")
    print(f"  Small margin:      {stats['n_small']}")
    print(f"  Pixels valid:      {stats['n_pixels_valid']}")
    print(f"  Pixels skipped:    {stats['n_pixels_skipped']}")
    print(f"  Frac satisfied:    {stats['frac_satisfied']:.2%}")

    # Verify gradients flow
    loss.backward()
    print(f"\n  grad norm t0: {emb_t0.grad.norm().item():.4f}")
    print(f"  grad norm t1: {emb_t1.grad.norm().item():.4f}")
    print(f"  grad norm t2: {emb_t2.grad.norm().item():.4f}")


if __name__ == "__main__":
    example_phase_triplet_loss()
