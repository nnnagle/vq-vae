"""
Example usage of contrastive loss function.

This script demonstrates how to:
1. Set up embeddings, positive pairs, and negative pairs
2. Use different similarity functions (l2, cosine, dot)
3. Apply pair weights for importance sampling
4. Understand the grouping-by-anchor behavior
"""

import torch
from losses import contrastive_loss


def example_basic_usage():
    """
    Example: Basic contrastive loss with default settings.

    Scenario: 10 samples with 128-dim embeddings. We define which pairs
    are similar (positive) and which are dissimilar (negative).
    """
    print("\n" + "=" * 70)
    print("Basic Contrastive Loss Usage")
    print("=" * 70)

    # 10 samples, 128-dimensional embeddings
    embeddings = torch.randn(10, 128)

    # Positive pairs: (anchor_idx, positive_idx)
    # These pairs should be pulled together
    pos_pairs = torch.tensor([
        [0, 1],  # sample 0 is similar to sample 1
        [0, 2],  # sample 0 is similar to sample 2
        [3, 4],  # sample 3 is similar to sample 4
    ])

    # Negative pairs: (anchor_idx, negative_idx)
    # These pairs should be pushed apart
    neg_pairs = torch.tensor([
        [0, 5],  # sample 0 is dissimilar to sample 5
        [0, 6],  # sample 0 is dissimilar to sample 6
        [3, 7],  # sample 3 is dissimilar to sample 7
        [3, 8],  # sample 3 is dissimilar to sample 8
    ])

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Positive pairs: {pos_pairs.shape[0]} pairs")
    print(f"Negative pairs: {neg_pairs.shape[0]} pairs")
    print(f"Unique anchors: {torch.unique(pos_pairs[:, 0]).tolist()}")

    # Compute loss
    loss = contrastive_loss(embeddings, pos_pairs, neg_pairs)

    print(f"\nLoss: {loss.item():.4f}")
    print("(Loss is averaged over unique anchors: 0 and 3)")


def example_anchor_grouping():
    """
    Example: Understanding how pairs are grouped by anchor.

    The loss is computed per-anchor, then averaged. Each anchor's loss
    considers ALL its positive and negative pairs together in one softmax.
    """
    print("\n" + "=" * 70)
    print("Anchor Grouping Behavior")
    print("=" * 70)

    # Create embeddings where we control the distances
    # Anchor 0 at origin, positives nearby, negatives far
    embeddings = torch.zeros(10, 4)
    embeddings[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])  # anchor 0
    embeddings[1] = torch.tensor([0.1, 0.0, 0.0, 0.0])  # positive (close)
    embeddings[2] = torch.tensor([0.0, 0.1, 0.0, 0.0])  # positive (close)
    embeddings[5] = torch.tensor([5.0, 0.0, 0.0, 0.0])  # negative (far)
    embeddings[6] = torch.tensor([0.0, 5.0, 0.0, 0.0])  # negative (far)

    # Anchor 3 setup (positives close, negatives far)
    embeddings[3] = torch.tensor([0.0, 0.0, 0.0, 0.0])  # anchor 3
    embeddings[4] = torch.tensor([0.2, 0.0, 0.0, 0.0])  # positive (slightly farther)
    embeddings[7] = torch.tensor([3.0, 0.0, 0.0, 0.0])  # negative (closer than anchor 0's)
    embeddings[8] = torch.tensor([0.0, 3.0, 0.0, 0.0])  # negative

    pos_pairs = torch.tensor([[0, 1], [0, 2], [3, 4]])
    neg_pairs = torch.tensor([[0, 5], [0, 6], [3, 7], [3, 8]])

    loss = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=1.0)

    print("Setup:")
    print("  Anchor 0: 2 positives at dist ~0.1, 2 negatives at dist 5.0")
    print("  Anchor 3: 1 positive at dist 0.2, 2 negatives at dist 3.0")
    print(f"\nLoss: {loss.item():.4f}")
    print("(Anchor 0 should have lower loss due to better separation)")

    # Show per-anchor behavior by computing separately
    loss_anchor0 = contrastive_loss(
        embeddings,
        pos_pairs[pos_pairs[:, 0] == 0],
        neg_pairs[neg_pairs[:, 0] == 0],
        temperature=1.0,
    )
    loss_anchor3 = contrastive_loss(
        embeddings,
        pos_pairs[pos_pairs[:, 0] == 3],
        neg_pairs[neg_pairs[:, 0] == 3],
        temperature=1.0,
    )
    print(f"\nPer-anchor losses:")
    print(f"  Anchor 0: {loss_anchor0.item():.4f}")
    print(f"  Anchor 3: {loss_anchor3.item():.4f}")
    print(f"  Average:  {(loss_anchor0 + loss_anchor3).item() / 2:.4f}")


def example_similarity_functions():
    """
    Example: Comparing different similarity functions.

    - l2: Euclidean distance in embedding space (default)
    - cosine: Angular similarity (normalized)
    - dot: Raw dot product
    """
    print("\n" + "=" * 70)
    print("Similarity Functions Comparison")
    print("=" * 70)

    # Random embeddings (unnormalized)
    torch.manual_seed(42)
    embeddings = torch.randn(10, 64)

    pos_pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    neg_pairs = torch.tensor([[0, 6], [2, 7], [4, 8]])

    print("With UNNORMALIZED embeddings:")
    for sim in ["l2", "cosine", "dot"]:
        loss = contrastive_loss(
            embeddings, pos_pairs, neg_pairs, similarity=sim, temperature=1.0
        )
        print(f"  {sim:8s}: {loss.item():.4f}")

    # Now with normalized embeddings
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    print("\nWith NORMALIZED embeddings:")
    for sim in ["l2", "cosine", "dot"]:
        loss = contrastive_loss(
            embeddings_norm, pos_pairs, neg_pairs, similarity=sim, temperature=1.0
        )
        print(f"  {sim:8s}: {loss.item():.4f}")

    print("\nNote: For normalized embeddings, l2 with temp=t equals dot with temp=t/2")
    print("      (cosine and dot are identical for normalized embeddings)")


def example_weighted_pairs():
    """
    Example: Using weights to emphasize certain pairs.

    Weights can represent:
    - Confidence in the label
    - Importance sampling corrections
    - Hard negative mining emphasis
    """
    print("\n" + "=" * 70)
    print("Weighted Pairs")
    print("=" * 70)

    embeddings = torch.randn(10, 64)

    pos_pairs = torch.tensor([[0, 1], [0, 2], [0, 3]])
    neg_pairs = torch.tensor([[0, 5], [0, 6], [0, 7]])

    # Uniform weights
    loss_uniform = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=1.0)
    print(f"Uniform weights: {loss_uniform.item():.4f}")

    # Emphasize first positive pair (e.g., high-confidence label)
    pos_weights = torch.tensor([5.0, 1.0, 1.0])
    neg_weights = torch.tensor([1.0, 1.0, 1.0])
    loss_pos_weighted = contrastive_loss(
        embeddings, pos_pairs, neg_pairs, pos_weights, neg_weights, temperature=1.0
    )
    print(f"Emphasize pos[0] (w=5): {loss_pos_weighted.item():.4f}")

    # Emphasize hard negatives (e.g., semi-hard negative mining)
    pos_weights = torch.tensor([1.0, 1.0, 1.0])
    neg_weights = torch.tensor([5.0, 1.0, 1.0])
    loss_neg_weighted = contrastive_loss(
        embeddings, pos_pairs, neg_pairs, pos_weights, neg_weights, temperature=1.0
    )
    print(f"Emphasize neg[0] (w=5): {loss_neg_weighted.item():.4f}")


def example_temperature_effect():
    """
    Example: Effect of temperature on the loss.

    Lower temperature = sharper distribution = harder contrastive learning
    Higher temperature = softer distribution = smoother gradients
    """
    print("\n" + "=" * 70)
    print("Temperature Effect")
    print("=" * 70)

    torch.manual_seed(42)
    embeddings = torch.randn(10, 64)

    pos_pairs = torch.tensor([[0, 1], [2, 3]])
    neg_pairs = torch.tensor([[0, 5], [0, 6], [2, 7], [2, 8]])

    print("Temperature scaling effect:")
    for temp in [0.01, 0.1, 0.5, 1.0, 2.0]:
        loss = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=temp)
        print(f"  temp={temp:4.2f}: loss={loss.item():.4f}")

    print("\nLower temp -> loss depends more on the single closest negative")
    print("Higher temp -> loss considers all negatives more equally")


def example_gradient_flow():
    """
    Example: Verifying gradient flow for training.

    The loss should provide gradients that:
    - Pull positive pairs together
    - Push negative pairs apart
    """
    print("\n" + "=" * 70)
    print("Gradient Flow Verification")
    print("=" * 70)

    # Embeddings with gradients
    embeddings = torch.randn(10, 64, requires_grad=True)

    pos_pairs = torch.tensor([[0, 1], [2, 3]])
    neg_pairs = torch.tensor([[0, 5], [2, 6]])

    loss = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.5)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient shape: {embeddings.grad.shape}")
    print(f"Gradient norm: {embeddings.grad.norm().item():.4f}")

    # Check which embeddings have non-zero gradients
    grad_norms = embeddings.grad.norm(dim=1)
    active_indices = (grad_norms > 1e-6).nonzero().squeeze().tolist()
    print(f"Embeddings with gradients: {active_indices}")
    print("(Only embeddings involved in pairs receive gradients)")


def example_training_loop_snippet():
    """
    Example: How to use in a training loop.

    Shows integration with a simple encoder and optimizer.
    """
    print("\n" + "=" * 70)
    print("Training Loop Integration")
    print("=" * 70)

    # Simple encoder (in practice, this would be your model)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 128),
    )

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # Simulate a few training steps
    print("Simulated training steps:")
    for step in range(5):
        # Batch of raw features
        batch_features = torch.randn(16, 32)

        # Get embeddings from encoder
        embeddings = encoder(batch_features)

        # Define pairs (in practice, from your data loader)
        # Here: consecutive samples are positive pairs, random are negative
        pos_pairs = torch.tensor([[i, i + 1] for i in range(0, 14, 2)])
        neg_pairs = torch.tensor([[i, (i + 8) % 16] for i in range(0, 16, 2)])

        # Compute loss
        loss = contrastive_loss(
            embeddings,
            pos_pairs,
            neg_pairs,
            temperature=0.1,
            similarity="l2",
        )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    print("\nTraining loop complete!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CONTRASTIVE LOSS EXAMPLES")
    print("=" * 70)

    example_basic_usage()
    example_anchor_grouping()
    example_similarity_functions()
    example_weighted_pairs()
    example_temperature_effect()
    example_gradient_flow()
    example_training_loop_snippet()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
