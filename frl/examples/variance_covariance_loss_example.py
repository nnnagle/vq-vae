"""
Examples demonstrating the variance-covariance loss for preventing embedding collapse.

This loss is based on VICReg (Variance-Invariance-Covariance Regularization)
and is useful for self-supervised learning to ensure the embedding space
is fully utilized.
"""

import torch

from losses import covariance_loss, variance_covariance_loss, variance_loss


def example_1_basic_usage():
    """Basic usage with random embeddings."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create random embeddings [N, D]
    # N = 1000 samples, D = 64 embedding dimensions
    embeddings = torch.randn(1000, 64)

    # Compute combined loss
    total, var_loss, cov_loss = variance_covariance_loss(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Variance loss: {var_loss.item():.6f}")
    print(f"Covariance loss: {cov_loss.item():.6f}")
    print(f"Total loss: {total.item():.6f}")
    print()


def example_2_collapsed_embeddings():
    """Show high loss for collapsed embeddings."""
    print("=" * 60)
    print("Example 2: Collapsed Embeddings (High Loss)")
    print("=" * 60)

    # Simulate collapsed embeddings where all dimensions are similar
    # This is what we want to prevent!
    base = torch.randn(1000, 1)
    collapsed = base.expand(1000, 64) + torch.randn(1000, 64) * 0.01

    total, var_loss, cov_loss = variance_covariance_loss(collapsed)

    print(f"Collapsed embeddings shape: {collapsed.shape}")
    print(f"Variance loss: {var_loss.item():.6f} (high = dimensions have low variance)")
    print(f"Covariance loss: {cov_loss.item():.6f} (high = dimensions are correlated)")
    print(f"Total loss: {total.item():.6f}")
    print()


def example_3_healthy_embeddings():
    """Show low loss for healthy uncorrelated embeddings."""
    print("=" * 60)
    print("Example 3: Healthy Embeddings (Low Loss)")
    print("=" * 60)

    # Well-behaved embeddings: each dimension has good variance, dimensions uncorrelated
    embeddings = torch.randn(1000, 64)

    total, var_loss, cov_loss = variance_covariance_loss(embeddings)

    print(f"Healthy embeddings shape: {embeddings.shape}")
    print(f"Variance loss: {var_loss.item():.6f} (low = good variance per dimension)")
    print(f"Covariance loss: {cov_loss.item():.6f} (low = dimensions are decorrelated)")
    print(f"Total loss: {total.item():.6f}")
    print()


def example_4_remote_sensing_spatial():
    """Example for remote sensing with spatial embeddings [B, D, H, W]."""
    print("=" * 60)
    print("Example 4: Remote Sensing Spatial Embeddings")
    print("=" * 60)

    # Simulate latents from a VQ-VAE encoder
    # B=8 patches, D=64 embedding dim, H=W=32 spatial resolution
    B, D, H, W = 8, 64, 32, 32
    latents = torch.randn(B, D, H, W)

    print(f"Original latent shape: {latents.shape}")

    # Reshape [B, D, H, W] -> [N, D] where N = B * H * W
    # Each spatial location becomes a sample
    embeddings = latents.permute(0, 2, 3, 1).reshape(-1, D)

    print(f"Reshaped for loss: {embeddings.shape}")

    total, var_loss, cov_loss = variance_covariance_loss(embeddings)

    print(f"Variance loss: {var_loss.item():.6f}")
    print(f"Covariance loss: {cov_loss.item():.6f}")
    print(f"Total loss: {total.item():.6f}")
    print()


def example_5_custom_weights():
    """Using custom weights for variance and covariance terms."""
    print("=" * 60)
    print("Example 5: Custom Weights (VICReg-style)")
    print("=" * 60)

    embeddings = torch.randn(1000, 64)

    # VICReg paper uses weights of 25 for both terms
    total, var_loss, cov_loss = variance_covariance_loss(
        embeddings,
        variance_weight=25.0,
        covariance_weight=25.0,
    )

    print(f"Using VICReg weights (25, 25)")
    print(f"Variance loss (unweighted): {var_loss.item():.6f}")
    print(f"Covariance loss (unweighted): {cov_loss.item():.6f}")
    print(f"Total loss (weighted): {total.item():.6f}")
    print()


def example_6_variance_target():
    """Adjusting the variance target."""
    print("=" * 60)
    print("Example 6: Custom Variance Target")
    print("=" * 60)

    # Create embeddings with std ~0.5
    embeddings = torch.randn(1000, 64) * 0.5

    # With default target=1.0, there will be variance loss
    total_default, var_default, _ = variance_covariance_loss(
        embeddings, variance_target=1.0
    )

    # With target=0.5, no variance loss
    total_custom, var_custom, _ = variance_covariance_loss(
        embeddings, variance_target=0.5
    )

    print(f"Embeddings std: ~0.5")
    print(f"With variance_target=1.0: var_loss = {var_default.item():.6f}")
    print(f"With variance_target=0.5: var_loss = {var_custom.item():.6f}")
    print()


def example_7_individual_losses():
    """Using individual variance_loss and covariance_loss functions."""
    print("=" * 60)
    print("Example 7: Individual Loss Functions")
    print("=" * 60)

    embeddings = torch.randn(1000, 64)

    # Compute losses individually
    var_l = variance_loss(embeddings, target=1.0)
    cov_l = covariance_loss(embeddings)

    print(f"Using individual functions:")
    print(f"  variance_loss(): {var_l.item():.6f}")
    print(f"  covariance_loss(): {cov_l.item():.6f}")
    print()


def example_8_gradient_flow():
    """Verify gradient flows through the loss."""
    print("=" * 60)
    print("Example 8: Gradient Flow Verification")
    print("=" * 60)

    # Create embeddings with gradients
    embeddings = torch.randn(1000, 64, requires_grad=True)

    total, var_loss, cov_loss = variance_covariance_loss(embeddings)

    # Backpropagate
    total.backward()

    print(f"Total loss: {total.item():.6f}")
    print(f"Gradient shape: {embeddings.grad.shape}")
    print(f"Gradient norm: {embeddings.grad.norm().item():.6f}")
    print(f"Gradient flows correctly!")
    print()


def example_9_training_loop():
    """Simulated training loop with variance-covariance regularization."""
    print("=" * 60)
    print("Example 9: Training Loop Simulation")
    print("=" * 60)

    # Simulate an encoder that maps [B, C, H, W] -> [B, D, H, W]
    class SimpleEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    encoder = SimpleEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # Training loop
    for step in range(5):
        # Fake batch of 256x256 patches
        x = torch.randn(4, 3, 64, 64)

        # Forward pass
        latents = encoder(x)  # [B, D, H, W]

        # Reshape for loss: [B, D, H, W] -> [B*H*W, D]
        B, D, H, W = latents.shape
        embeddings = latents.permute(0, 2, 3, 1).reshape(-1, D)

        # Compute loss
        total, var_loss, cov_loss = variance_covariance_loss(
            embeddings,
            variance_weight=1.0,
            covariance_weight=0.04,  # Often smaller than variance weight
        )

        # Backward pass
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if step % 1 == 0:
            print(f"Step {step}: total={total.item():.4f}, var={var_loss.item():.4f}, cov={cov_loss.item():.4f}")

    print()


def example_10_combined_with_other_losses():
    """Combining variance-covariance loss with reconstruction loss."""
    print("=" * 60)
    print("Example 10: Combined with Other Losses")
    print("=" * 60)

    from losses import reconstruction_loss

    # Simulated encoder-decoder
    embeddings = torch.randn(1000, 64, requires_grad=True)
    reconstructed = torch.randn(1000, 3)
    target = torch.randn(1000, 3)

    # Reconstruction loss
    recon_loss = reconstruction_loss(reconstructed, target, loss_type="l2")

    # Regularization loss
    total_reg, var_loss, cov_loss = variance_covariance_loss(
        embeddings,
        variance_weight=1.0,
        covariance_weight=0.04,
    )

    # Combined loss
    combined = recon_loss + 0.1 * total_reg

    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"Variance loss: {var_loss.item():.6f}")
    print(f"Covariance loss: {cov_loss.item():.6f}")
    print(f"Regularization total: {total_reg.item():.6f}")
    print(f"Combined loss: {combined.item():.4f}")
    print()


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_collapsed_embeddings()
    example_3_healthy_embeddings()
    example_4_remote_sensing_spatial()
    example_5_custom_weights()
    example_6_variance_target()
    example_7_individual_losses()
    example_8_gradient_flow()
    example_9_training_loop()
    example_10_combined_with_other_losses()
