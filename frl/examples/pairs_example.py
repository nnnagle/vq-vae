"""
Example usage of pair generation functions for contrastive learning.

This script demonstrates how to:
1. Generate pairs from distance matrices using different strategies
2. Use KNN, mutual KNN, quantile, and radius-based pair selection
3. Handle rectangular distance matrices with anchor_cols
4. Apply validity masks to exclude certain indices
5. Integrate with contrastive_loss for end-to-end training
"""

import torch
from losses import (
    contrastive_loss,
    pairs_knn,
    pairs_mutual_knn,
    pairs_quantile,
    pairs_radius,
)


def example_basic_knn():
    """
    Example: Basic KNN pair generation from a distance matrix.

    Selects the k nearest neighbors for each anchor as pairs.
    """
    print("\n" + "=" * 70)
    print("Basic KNN Pairs")
    print("=" * 70)

    # Create embeddings and compute pairwise distances
    torch.manual_seed(42)
    embeddings = torch.randn(20, 64)
    distances = torch.cdist(embeddings, embeddings)  # [20, 20]

    print(f"Embeddings: {embeddings.shape}")
    print(f"Distance matrix: {distances.shape}")

    # Get 5 nearest neighbors for each point
    pairs = pairs_knn(distances, k=5)

    print(f"\nKNN pairs (k=5): {pairs.shape}")
    print(f"Expected: ~{20 * 5} pairs (20 anchors × 5 neighbors)")
    print(f"Sample pairs:\n{pairs[:10]}")

    # Verify: each anchor should have 5 pairs
    anchors, counts = torch.unique(pairs[:, 0], return_counts=True)
    print(f"\nPairs per anchor: min={counts.min()}, max={counts.max()}")


def example_symmetric_knn():
    """
    Example: Symmetric KNN pairs.

    With symmetric=True, if (i, j) is a pair, (j, i) is also added.
    This is useful when you want bidirectional relationships.
    """
    print("\n" + "=" * 70)
    print("Symmetric KNN Pairs")
    print("=" * 70)

    torch.manual_seed(42)
    embeddings = torch.randn(20, 64)
    distances = torch.cdist(embeddings, embeddings)

    # Without symmetric
    pairs_asym = pairs_knn(distances, k=3, symmetric=False)
    print(f"Asymmetric pairs (k=3): {pairs_asym.shape[0]}")

    # With symmetric
    pairs_sym = pairs_knn(distances, k=3, symmetric=True)
    print(f"Symmetric pairs (k=3): {pairs_sym.shape[0]}")

    # Check for bidirectional pairs
    pair_set_sym = set((a.item(), b.item()) for a, b in pairs_sym)
    bidirectional = sum(1 for a, b in pair_set_sym if (b, a) in pair_set_sym)
    print(f"Bidirectional pairs in symmetric: {bidirectional // 2}")


def example_mutual_knn():
    """
    Example: Mutual KNN pairs.

    Only includes pairs where BOTH points are in each other's k-NN.
    This creates high-confidence similarity pairs.
    """
    print("\n" + "=" * 70)
    print("Mutual KNN Pairs")
    print("=" * 70)

    torch.manual_seed(42)
    embeddings = torch.randn(50, 64)
    distances = torch.cdist(embeddings, embeddings)

    # Compare regular KNN vs mutual KNN
    knn_pairs = pairs_knn(distances, k=10, symmetric=True)
    mutual_pairs = pairs_mutual_knn(distances, k=10)

    print(f"Regular KNN pairs (k=10, symmetric): {knn_pairs.shape[0]}")
    print(f"Mutual KNN pairs (k=10): {mutual_pairs.shape[0]}")
    print("Mutual KNN is more selective (both must be neighbors)")

    # Verify mutual property
    pair_set = set((a.item(), b.item()) for a, b in mutual_pairs)
    all_mutual = all((b, a) in pair_set for a, b in pair_set)
    print(f"All pairs are mutual: {all_mutual}")


def example_quantile_pairs():
    """
    Example: Quantile-based pair selection.

    Select pairs based on their position in the distance distribution.
    Useful for selecting positives (low quantile) and negatives (high quantile).
    """
    print("\n" + "=" * 70)
    print("Quantile-Based Pairs")
    print("=" * 70)

    torch.manual_seed(42)
    embeddings = torch.randn(100, 64)
    distances = torch.cdist(embeddings, embeddings)

    # Closest 5% as positive pairs
    pos_pairs = pairs_quantile(distances, low=0.0, high=0.05)
    print(f"Positive pairs (0-5% quantile): {pos_pairs.shape[0]}")

    # 50-75% as negative pairs (medium-far distances)
    neg_pairs = pairs_quantile(distances, low=0.5, high=0.75)
    print(f"Negative pairs (50-75% quantile): {neg_pairs.shape[0]}")

    # Show distance statistics for each set
    pos_dists = distances[pos_pairs[:, 0], pos_pairs[:, 1]]
    neg_dists = distances[neg_pairs[:, 0], neg_pairs[:, 1]]
    print(f"\nPositive pair distances: mean={pos_dists.mean():.3f}, std={pos_dists.std():.3f}")
    print(f"Negative pair distances: mean={neg_dists.mean():.3f}, std={neg_dists.std():.3f}")


def example_radius_pairs():
    """
    Example: Radius-based pair selection.

    Select pairs within absolute distance thresholds.
    Useful when you have meaningful distance scales.
    """
    print("\n" + "=" * 70)
    print("Radius-Based Pairs")
    print("=" * 70)

    torch.manual_seed(42)
    embeddings = torch.randn(100, 64)
    distances = torch.cdist(embeddings, embeddings)

    # Check distance distribution
    valid_dists = distances[distances > 0]  # exclude diagonal
    print(f"Distance range: [{valid_dists.min():.2f}, {valid_dists.max():.2f}]")
    print(f"Distance mean: {valid_dists.mean():.2f}, std: {valid_dists.std():.2f}")

    # Close pairs (within 1 std below mean)
    threshold = valid_dists.mean() - valid_dists.std()
    close_pairs = pairs_radius(distances, min_dist=0.0, max_dist=threshold.item())
    print(f"\nClose pairs (dist < {threshold:.2f}): {close_pairs.shape[0]}")

    # Far pairs (above mean + 1 std)
    far_threshold = valid_dists.mean() + valid_dists.std()
    far_pairs = pairs_radius(distances, min_dist=far_threshold.item(), max_dist=float("inf"))
    print(f"Far pairs (dist > {far_threshold:.2f}): {far_pairs.shape[0]}")


def example_rectangular_matrix():
    """
    Example: Rectangular distance matrix (anchors vs candidates).

    Common scenario: N query embeddings compared against M reference embeddings,
    where M > N and the queries are a subset of references.
    """
    print("\n" + "=" * 70)
    print("Rectangular Distance Matrix")
    print("=" * 70)

    torch.manual_seed(42)

    # Scenario: 10 anchors, 100 candidates
    # Anchors correspond to candidates 0-9
    n_anchors = 10
    n_candidates = 100

    # Create embeddings
    anchor_embeddings = torch.randn(n_anchors, 64)
    candidate_embeddings = torch.randn(n_candidates, 64)

    # Place anchors at the beginning of candidates (common pattern)
    candidate_embeddings[:n_anchors] = anchor_embeddings

    # Compute rectangular distance matrix
    distances = torch.cdist(anchor_embeddings, candidate_embeddings)  # [10, 100]
    print(f"Distance matrix shape: {distances.shape}")

    # anchor_cols maps each row to its corresponding column (for self-exclusion)
    anchor_cols = torch.arange(n_anchors)  # [0, 1, 2, ..., 9]
    print(f"anchor_cols: {anchor_cols}")

    # Get KNN pairs
    pairs = pairs_knn(distances, k=5, anchor_cols=anchor_cols)
    print(f"\nKNN pairs: {pairs.shape}")
    print(f"Sample pairs:\n{pairs[:10]}")

    # Verify: anchor IDs come from anchor_cols, targets are column indices
    print(f"\nAnchor IDs in pairs: {torch.unique(pairs[:, 0]).tolist()}")
    print("(These match anchor_cols values)")


def example_validity_mask():
    """
    Example: Using valid_mask to exclude certain indices.

    Useful when some embeddings are missing, invalid, or should be excluded
    from pair generation.
    """
    print("\n" + "=" * 70)
    print("Validity Mask")
    print("=" * 70)

    torch.manual_seed(42)
    embeddings = torch.randn(50, 64)
    distances = torch.cdist(embeddings, embeddings)

    # Mark some indices as invalid
    valid_mask = torch.ones(50)
    invalid_indices = [5, 10, 15, 20, 25]
    valid_mask[invalid_indices] = 0
    print(f"Invalid indices: {invalid_indices}")

    # Without mask
    pairs_no_mask = pairs_knn(distances, k=5)
    print(f"\nPairs without mask: {pairs_no_mask.shape[0]}")

    # With mask
    pairs_masked = pairs_knn(distances, k=5, valid_mask=valid_mask)
    print(f"Pairs with mask: {pairs_masked.shape[0]}")

    # Verify no invalid indices in masked pairs
    all_indices = torch.cat([pairs_masked[:, 0], pairs_masked[:, 1]])
    has_invalid = any(idx in invalid_indices for idx in all_indices.tolist())
    print(f"Contains invalid indices: {has_invalid}")


def example_max_pairs_sampling():
    """
    Example: Limiting the number of pairs with random sampling.

    When pair count is very large, use max_pairs to cap the output.
    """
    print("\n" + "=" * 70)
    print("Max Pairs Sampling")
    print("=" * 70)

    torch.manual_seed(42)
    embeddings = torch.randn(200, 64)
    distances = torch.cdist(embeddings, embeddings)

    # Quantile selection can produce many pairs
    pairs_full = pairs_quantile(distances, low=0.0, high=0.2)
    print(f"Full pairs (0-20% quantile): {pairs_full.shape[0]}")

    # Limit to max_pairs
    pairs_sampled = pairs_quantile(distances, low=0.0, high=0.2, max_pairs=1000)
    print(f"Sampled pairs (max=1000): {pairs_sampled.shape[0]}")

    # Sampling is random
    pairs_sampled2 = pairs_quantile(distances, low=0.0, high=0.2, max_pairs=1000)
    overlap = len(set(map(tuple, pairs_sampled.tolist())) &
                  set(map(tuple, pairs_sampled2.tolist())))
    print(f"Overlap between two samplings: {overlap} pairs")


def example_handling_inf_nan():
    """
    Example: Automatic handling of inf/nan in distance matrices.

    inf and nan values are treated as invalid and excluded from pairs.
    """
    print("\n" + "=" * 70)
    print("Handling inf/nan Values")
    print("=" * 70)

    torch.manual_seed(42)
    distances = torch.randn(20, 20).abs() + 0.1  # positive distances

    # Introduce some inf/nan values
    distances[0, 5] = float("inf")
    distances[0, 6] = float("nan")
    distances[3, :] = float("inf")  # entire row invalid
    print("Introduced: inf at [0,5], nan at [0,6], all inf at row 3")

    pairs = pairs_knn(distances, k=5)

    # Check that problematic pairs are excluded
    has_anchor_3 = (pairs[:, 0] == 3).any().item()
    has_target_5_from_0 = ((pairs[:, 0] == 0) & (pairs[:, 1] == 5)).any().item()
    has_target_6_from_0 = ((pairs[:, 0] == 0) & (pairs[:, 1] == 6)).any().item()

    print(f"\nAnchor 3 in pairs: {has_anchor_3} (should be False)")
    print(f"Pair (0, 5) in pairs: {has_target_5_from_0} (should be False)")
    print(f"Pair (0, 6) in pairs: {has_target_6_from_0} (should be False)")


def example_end_to_end_contrastive():
    """
    Example: End-to-end usage with contrastive_loss.

    Shows the complete workflow from embeddings to loss computation.
    """
    print("\n" + "=" * 70)
    print("End-to-End Contrastive Learning")
    print("=" * 70)

    torch.manual_seed(42)

    # Create embeddings
    embeddings = torch.randn(100, 128, requires_grad=True)
    distances = torch.cdist(embeddings.detach(), embeddings.detach())

    print(f"Embeddings: {embeddings.shape}")

    # Generate positive pairs: closest 5% (excluding self)
    pos_pairs = pairs_quantile(distances, low=0.0, high=0.05)
    print(f"Positive pairs: {pos_pairs.shape[0]}")

    # Generate negative pairs: 50-75% quantile range
    neg_pairs = pairs_quantile(distances, low=0.5, high=0.75)
    print(f"Negative pairs: {neg_pairs.shape[0]}")

    # Compute contrastive loss
    loss = contrastive_loss(
        embeddings,
        pos_pairs,
        neg_pairs,
        temperature=0.1,
        similarity="l2",
    )

    print(f"\nContrastive loss: {loss.item():.4f}")

    # Verify gradients flow
    loss.backward()
    print(f"Gradient norm: {embeddings.grad.norm().item():.4f}")


def example_training_loop():
    """
    Example: Integration in a training loop.

    Shows how pairs can be regenerated periodically as embeddings change.
    """
    print("\n" + "=" * 70)
    print("Training Loop Integration")
    print("=" * 70)

    # Simple encoder
    encoder = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 128),
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # Fixed input batch (in practice, this would vary)
    torch.manual_seed(42)
    batch = torch.randn(50, 32)

    print("Training with pair regeneration every 2 steps:")
    for step in range(6):
        # Get current embeddings
        embeddings = encoder(batch)

        # Regenerate pairs periodically (as embeddings change)
        if step % 2 == 0:
            with torch.no_grad():
                distances = torch.cdist(embeddings, embeddings)
                pos_pairs = pairs_knn(distances, k=5)
                neg_pairs = pairs_quantile(distances, low=0.5, high=0.8, max_pairs=500)
            print(f"  Step {step}: regenerated pairs "
                  f"(pos={pos_pairs.shape[0]}, neg={neg_pairs.shape[0]})")

        # Compute loss and update
        loss = contrastive_loss(embeddings, pos_pairs, neg_pairs, temperature=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step}: loss = {loss.item():.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PAIR GENERATION EXAMPLES")
    print("=" * 70)

    example_basic_knn()
    example_symmetric_knn()
    example_mutual_knn()
    example_quantile_pairs()
    example_radius_pairs()
    example_rectangular_matrix()
    example_validity_mask()
    example_max_pairs_sampling()
    example_handling_inf_nan()
    example_end_to_end_contrastive()
    example_training_loop()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
