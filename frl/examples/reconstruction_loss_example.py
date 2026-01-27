"""
Example usage of reconstruction loss functions.

This script demonstrates how to use:
1. reconstruction_loss - for continuous targets (L1, L2, Huber)
2. categorical_loss - for discrete/categorical targets (cross-entropy)
3. count_loss - for count data (Poisson, Negative Binomial)
"""

import torch
from losses import reconstruction_loss, categorical_loss, count_loss


# =============================================================================
# CONTINUOUS RECONSTRUCTION LOSS
# =============================================================================


def example_continuous_basic():
    """
    Example: Basic continuous reconstruction loss.
    """
    print("\n" + "=" * 70)
    print("Continuous Reconstruction Loss - Basic Usage")
    print("=" * 70)

    # Simulated autoencoder output
    input = torch.randn(4, 3, 32, 32)  # Reconstructed
    target = torch.randn(4, 3, 32, 32)  # Original

    print(f"Input shape: {input.shape}")
    print(f"Target shape: {target.shape}")

    # Different loss types
    for loss_type in ["l1", "l2", "huber", "smooth_l1"]:
        loss = reconstruction_loss(input, target, loss_type=loss_type)
        print(f"  {loss_type:10s}: {loss.item():.4f}")


def example_continuous_with_mask():
    """
    Example: Continuous loss with masking (e.g., ignore clouds).
    """
    print("\n" + "=" * 70)
    print("Continuous Reconstruction Loss - With Mask")
    print("=" * 70)

    input = torch.randn(4, 3, 32, 32)
    target = torch.randn(4, 3, 32, 32)

    # Mask: True = valid, False = ignore
    mask = torch.ones(4, 32, 32, dtype=torch.bool)
    mask[:, :10, :10] = False  # Mask out top-left 10x10 region (e.g., clouds)

    valid_ratio = mask.float().mean()
    print(f"Valid pixels: {valid_ratio:.1%}")

    loss_no_mask = reconstruction_loss(input, target)
    loss_with_mask = reconstruction_loss(input, target, mask)

    print(f"Loss without mask: {loss_no_mask.item():.4f}")
    print(f"Loss with mask:    {loss_with_mask.item():.4f}")


def example_continuous_huber_delta():
    """
    Example: Huber loss with different delta values.
    """
    print("\n" + "=" * 70)
    print("Continuous Reconstruction Loss - Huber Delta")
    print("=" * 70)

    # Create data with some outliers
    input = torch.randn(4, 3, 32, 32)
    target = torch.randn(4, 3, 32, 32)

    # Add outliers
    target[0, 0, :5, :5] += 10.0

    print("Huber loss with different delta values:")
    print("(Lower delta = more robust to outliers)")
    for delta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        loss = reconstruction_loss(input, target, loss_type="huber", delta=delta)
        print(f"  delta={delta:.1f}: {loss.item():.4f}")

    # Compare with L1 and L2
    l1 = reconstruction_loss(input, target, loss_type="l1")
    l2 = reconstruction_loss(input, target, loss_type="l2")
    print(f"\nL1 (most robust):  {l1.item():.4f}")
    print(f"L2 (least robust): {l2.item():.4f}")


def example_continuous_reduction():
    """
    Example: Different reduction modes.
    """
    print("\n" + "=" * 70)
    print("Continuous Reconstruction Loss - Reduction Modes")
    print("=" * 70)

    input = torch.randn(4, 3, 32, 32)
    target = torch.randn(4, 3, 32, 32)

    loss_mean = reconstruction_loss(input, target, reduction="mean")
    loss_sum = reconstruction_loss(input, target, reduction="sum")
    loss_none = reconstruction_loss(input, target, reduction="none")

    print(f"reduction='mean': {loss_mean.item():.4f} (scalar)")
    print(f"reduction='sum':  {loss_sum.item():.4f} (scalar)")
    print(f"reduction='none': shape {loss_none.shape} (per-element)")
    print(f"  mean of 'none': {loss_none.mean().item():.4f}")


# =============================================================================
# CATEGORICAL LOSS
# =============================================================================


def example_categorical_basic():
    """
    Example: Basic categorical (cross-entropy) loss.
    """
    print("\n" + "=" * 70)
    print("Categorical Loss - Basic Usage")
    print("=" * 70)

    num_classes = 10

    # Model outputs logits
    logits = torch.randn(4, num_classes, 32, 32)
    target = torch.randint(0, num_classes, (4, 32, 32))

    print(f"Logits shape: {logits.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Num classes: {num_classes}")

    loss = categorical_loss(logits, target)
    print(f"\nCross-entropy loss: {loss.item():.4f}")

    # Theoretical minimum (perfect prediction)
    print(f"Theoretical min (perfect): 0.0000")
    print(f"Random baseline (10 classes): {torch.log(torch.tensor(10.0)).item():.4f}")


def example_categorical_with_mask():
    """
    Example: Categorical loss with masking (e.g., ignore background).
    """
    print("\n" + "=" * 70)
    print("Categorical Loss - With Mask")
    print("=" * 70)

    num_classes = 5
    logits = torch.randn(4, num_classes, 32, 32)
    target = torch.randint(0, num_classes, (4, 32, 32))

    # Mask out class 0 (e.g., background)
    mask = target != 0
    valid_ratio = mask.float().mean()

    print(f"Ignoring class 0 (background)")
    print(f"Valid pixels: {valid_ratio:.1%}")

    loss_no_mask = categorical_loss(logits, target)
    loss_with_mask = categorical_loss(logits, target, mask)

    print(f"Loss without mask: {loss_no_mask.item():.4f}")
    print(f"Loss with mask:    {loss_with_mask.item():.4f}")


def example_categorical_class_weights():
    """
    Example: Class weights for imbalanced data.
    """
    print("\n" + "=" * 70)
    print("Categorical Loss - Class Weights")
    print("=" * 70)

    num_classes = 5
    logits = torch.randn(4, num_classes, 32, 32)

    # Create imbalanced targets (class 0 is dominant)
    target = torch.zeros(4, 32, 32, dtype=torch.long)
    target[:, :5, :] = 1  # Small region is class 1
    target[:, :2, :] = 2  # Even smaller region is class 2

    # Count class frequencies
    for c in range(num_classes):
        count = (target == c).sum()
        print(f"Class {c}: {count} pixels ({count / target.numel():.1%})")

    # Inverse frequency weights
    counts = torch.tensor([(target == c).sum().float() for c in range(num_classes)])
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * num_classes  # Normalize

    print(f"\nClass weights: {weights.tolist()}")

    loss_uniform = categorical_loss(logits, target)
    loss_weighted = categorical_loss(logits, target, class_weights=weights)

    print(f"\nLoss (uniform weights):   {loss_uniform.item():.4f}")
    print(f"Loss (inverse freq weights): {loss_weighted.item():.4f}")


def example_categorical_label_smoothing():
    """
    Example: Label smoothing for regularization.
    """
    print("\n" + "=" * 70)
    print("Categorical Loss - Label Smoothing")
    print("=" * 70)

    num_classes = 10
    logits = torch.randn(4, num_classes, 32, 32)
    target = torch.randint(0, num_classes, (4, 32, 32))

    print("Label smoothing effect:")
    for smoothing in [0.0, 0.05, 0.1, 0.2]:
        loss = categorical_loss(logits, target, label_smoothing=smoothing)
        print(f"  smoothing={smoothing:.2f}: {loss.item():.4f}")

    print("\nHigher smoothing -> softer targets -> typically higher loss")
    print("But helps prevent overconfidence and improves generalization")


# =============================================================================
# COUNT LOSS
# =============================================================================


def example_count_poisson():
    """
    Example: Poisson loss for count data.
    """
    print("\n" + "=" * 70)
    print("Count Loss - Poisson")
    print("=" * 70)

    # Model predicts log-rate, convert to rate (must be positive)
    log_rate = torch.randn(4, 1, 32, 32)
    rate = torch.exp(log_rate)  # Ensure positive

    # Target counts (non-negative integers)
    target = torch.poisson(rate)  # Generate realistic targets

    print(f"Rate shape: {rate.shape}")
    print(f"Rate range: [{rate.min().item():.2f}, {rate.max().item():.2f}]")
    print(f"Target range: [{target.min().item():.0f}, {target.max().item():.0f}]")

    loss = count_loss(rate, target, loss_type="poisson")
    print(f"\nPoisson loss: {loss.item():.4f}")


def example_count_with_mask():
    """
    Example: Count loss with masking.
    """
    print("\n" + "=" * 70)
    print("Count Loss - With Mask")
    print("=" * 70)

    rate = torch.exp(torch.randn(4, 1, 32, 32))
    target = torch.poisson(rate)

    # Mask: some locations have missing count data
    mask = torch.ones(4, 32, 32, dtype=torch.bool)
    mask[:, :8, :8] = False  # Missing data in corner

    valid_ratio = mask.float().mean()
    print(f"Valid pixels: {valid_ratio:.1%}")

    loss_no_mask = count_loss(rate, target, loss_type="poisson")
    loss_with_mask = count_loss(rate, target, mask=mask, loss_type="poisson")

    print(f"Loss without mask: {loss_no_mask.item():.4f}")
    print(f"Loss with mask:    {loss_with_mask.item():.4f}")


def example_count_negative_binomial():
    """
    Example: Negative Binomial for overdispersed counts.
    """
    print("\n" + "=" * 70)
    print("Count Loss - Negative Binomial (Overdispersion)")
    print("=" * 70)

    rate = torch.exp(torch.randn(4, 1, 32, 32) + 2)  # Higher rates
    target = torch.poisson(rate)

    print("Negative Binomial with different dispersion values:")
    print("(Higher dispersion -> closer to Poisson)")

    for dispersion in [0.1, 1.0, 10.0, 100.0]:
        loss = count_loss(rate, target, loss_type="negative_binomial", dispersion=dispersion)
        print(f"  dispersion={dispersion:5.1f}: {loss.item():.4f}")

    # Compare with Poisson
    poisson_loss = count_loss(rate, target, loss_type="poisson")
    print(f"\nPoisson loss:        {poisson_loss.item():.4f}")
    print("(NegBin with high dispersion ≈ Poisson)")


def example_count_learned_dispersion():
    """
    Example: Spatially-varying dispersion (learned).
    """
    print("\n" + "=" * 70)
    print("Count Loss - Learned Dispersion")
    print("=" * 70)

    # Model outputs both rate and dispersion
    rate = torch.exp(torch.randn(4, 1, 32, 32))
    dispersion = torch.exp(torch.randn(4, 1, 32, 32))  # Per-pixel dispersion

    target = torch.poisson(rate)

    print(f"Rate shape: {rate.shape}")
    print(f"Dispersion shape: {dispersion.shape}")

    loss = count_loss(
        rate, target,
        loss_type="negative_binomial",
        dispersion=dispersion
    )
    print(f"\nNegBin loss with learned dispersion: {loss.item():.4f}")


# =============================================================================
# COMBINED EXAMPLE
# =============================================================================


def example_multi_output_model():
    """
    Example: Multi-output model with different loss types.
    """
    print("\n" + "=" * 70)
    print("Multi-Output Model Example")
    print("=" * 70)

    batch_size = 4
    H, W = 32, 32

    # Simulated model outputs
    continuous_pred = torch.randn(batch_size, 3, H, W)  # RGB reconstruction
    categorical_pred = torch.randn(batch_size, 10, H, W)  # 10-class segmentation
    count_pred = torch.exp(torch.randn(batch_size, 1, H, W))  # Count prediction

    # Targets
    continuous_target = torch.randn(batch_size, 3, H, W)
    categorical_target = torch.randint(0, 10, (batch_size, H, W))
    count_target = torch.poisson(count_pred)

    # Shared mask (e.g., valid pixels)
    mask = torch.rand(batch_size, H, W) > 0.1  # 90% valid

    # Compute losses
    loss_continuous = reconstruction_loss(
        continuous_pred, continuous_target, mask, loss_type="l2"
    )
    loss_categorical = categorical_loss(
        categorical_pred, categorical_target, mask
    )
    loss_count = count_loss(
        count_pred, count_target, mask, loss_type="poisson"
    )

    # Weighted combination
    lambda_continuous = 1.0
    lambda_categorical = 0.5
    lambda_count = 0.1

    total_loss = (
        lambda_continuous * loss_continuous
        + lambda_categorical * loss_categorical
        + lambda_count * loss_count
    )

    print("Individual losses:")
    print(f"  Continuous (L2):    {loss_continuous.item():.4f} × {lambda_continuous}")
    print(f"  Categorical (CE):   {loss_categorical.item():.4f} × {lambda_categorical}")
    print(f"  Count (Poisson):    {loss_count.item():.4f} × {lambda_count}")
    print(f"\nTotal weighted loss: {total_loss.item():.4f}")


# =============================================================================
# GRADIENT FLOW
# =============================================================================


def example_gradient_flow():
    """
    Example: Verify gradients flow correctly through all losses.
    """
    print("\n" + "=" * 70)
    print("Gradient Flow Verification")
    print("=" * 70)

    # Continuous
    input_cont = torch.randn(4, 3, 8, 8, requires_grad=True)
    target_cont = torch.randn(4, 3, 8, 8)
    loss_cont = reconstruction_loss(input_cont, target_cont)
    loss_cont.backward()
    print(f"Continuous - grad norm: {input_cont.grad.norm().item():.4f}")

    # Categorical
    logits = torch.randn(4, 10, 8, 8, requires_grad=True)
    target_cat = torch.randint(0, 10, (4, 8, 8))
    loss_cat = categorical_loss(logits, target_cat)
    loss_cat.backward()
    print(f"Categorical - grad norm: {logits.grad.norm().item():.4f}")

    # Count
    log_rate = torch.randn(4, 1, 8, 8, requires_grad=True)
    rate = torch.exp(log_rate)
    target_count = torch.poisson(rate.detach())
    loss_count = count_loss(rate, target_count, loss_type="poisson")
    loss_count.backward()
    print(f"Count - grad norm: {log_rate.grad.norm().item():.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RECONSTRUCTION LOSS EXAMPLES")
    print("=" * 70)

    # Continuous
    example_continuous_basic()
    example_continuous_with_mask()
    example_continuous_huber_delta()
    example_continuous_reduction()

    # Categorical
    example_categorical_basic()
    example_categorical_with_mask()
    example_categorical_class_weights()
    example_categorical_label_smoothing()

    # Count
    example_count_poisson()
    example_count_with_mask()
    example_count_negative_binomial()
    example_count_learned_dispersion()

    # Combined
    example_multi_output_model()
    example_gradient_flow()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
