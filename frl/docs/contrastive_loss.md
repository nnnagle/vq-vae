# Contrastive Loss

The `contrastive_loss` function implements an InfoNCE-style contrastive loss with support for weighted positive/negative pairs and efficient grouping by anchor indices.

## Overview

```
Embeddings [N, D]
     │
     ▼
┌─────────────────────────────────────┐
│         contrastive_loss            │
│                                     │
│  1. Compute pairwise similarities   │
│  2. Group pairs by anchor           │
│  3. Apply temperature scaling       │
│  4. Compute softmax per anchor      │
│  5. Average loss across anchors     │
│                                     │
└─────────────────────────────────────┘
     │
     ▼
  Scalar Loss
```

## Quick Start

```python
from losses import contrastive_loss
import torch

# Embeddings: 100 samples, 128 dimensions
embeddings = torch.randn(100, 128)

# Positive pairs: (anchor_idx, positive_idx)
pos_pairs = torch.tensor([
    [0, 1],   # sample 0 similar to sample 1
    [0, 2],   # sample 0 similar to sample 2
    [3, 4],   # sample 3 similar to sample 4
])

# Negative pairs: (anchor_idx, negative_idx)
neg_pairs = torch.tensor([
    [0, 5],   # sample 0 dissimilar to sample 5
    [0, 6],   # sample 0 dissimilar to sample 6
    [3, 7],   # sample 3 dissimilar to sample 7
])

# Compute loss
loss = contrastive_loss(embeddings, pos_pairs, neg_pairs)
loss.backward()
```

## API Reference

### contrastive_loss

```python
def contrastive_loss(
    embeddings: torch.Tensor,
    pos_pairs: torch.Tensor,
    neg_pairs: torch.Tensor,
    pos_weights: torch.Tensor | None = None,
    neg_weights: torch.Tensor | None = None,
    temperature: float = 0.07,
    similarity: Literal["l2", "cosine", "dot"] = "l2",
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embeddings` | `[N, D]` tensor | N samples with D-dimensional embeddings |
| `pos_pairs` | `[P, 2]` tensor | Positive pairs as `(anchor_idx, positive_idx)` |
| `neg_pairs` | `[M, 2]` tensor | Negative pairs as `(anchor_idx, negative_idx)` |
| `pos_weights` | `[P]` tensor or None | Optional weights for positive pairs |
| `neg_weights` | `[M]` tensor or None | Optional weights for negative pairs |
| `temperature` | float | Softmax temperature (default: 0.07) |
| `similarity` | str | One of `"l2"`, `"cosine"`, `"dot"` (default: `"l2"`) |

**Returns:** Scalar loss tensor (averaged over unique anchors)

## Loss Formula

For each anchor `a`, the loss is:

```
L_a = -log( Σ_p w_p·exp(sim(a,p)/t) / (Σ_p w_p·exp(sim(a,p)/t) + Σ_n w_n·exp(sim(a,n)/t)) )
```

Where:
- `p` iterates over positive pairs for anchor `a`
- `n` iterates over negative pairs for anchor `a`
- `w_p`, `w_n` are the pair weights
- `t` is the temperature
- `sim(·,·)` is the similarity function

The total loss is averaged over all unique anchors.

## Similarity Functions

### L2 (Recommended)

```
sim(a, b) = -||a - b||² / D
```

- Measures spatial distance in embedding space
- Normalized by dimension D for consistent temperature scaling
- Works well with default temperature (0.07)
- **Recommended for general use**

### Cosine

```
sim(a, b) = (a · b) / (||a|| ||b||)
```

- Measures angular similarity (ignores magnitude)
- Bounded in [-1, 1]
- Works well with default temperature

### Dot Product

```
sim(a, b) = a · b
```

- Raw dot product (unbounded)
- **Use only with normalized embeddings** or higher temperature
- For unnormalized embeddings, values scale with dimension and can cause numerical issues

### Equivalence for Normalized Embeddings

When embeddings are L2-normalized:
- `cosine` = `dot` (identical)
- `l2` is a linear transformation: `-||a-b||²/D = (-2 + 2·dot)/D`

## Temperature

Temperature controls the "hardness" of contrastive learning:

| Temperature | Effect |
|-------------|--------|
| Low (0.01-0.1) | Sharp softmax, focuses on closest pair. Loss → 0 if positive is closest. |
| Medium (0.1-0.5) | Balanced, considers multiple pairs |
| High (0.5-2.0) | Soft softmax, all pairs contribute more equally |

```python
# Hard contrastive learning
loss = contrastive_loss(emb, pos, neg, temperature=0.05)

# Soft contrastive learning
loss = contrastive_loss(emb, pos, neg, temperature=1.0)
```

## Pair Weights

Weights allow importance sampling or hard negative mining:

```python
# Emphasize confident positive pairs
pos_weights = torch.tensor([1.0, 0.5, 1.0])  # Less weight on uncertain pair

# Hard negative mining - emphasize difficult negatives
neg_weights = torch.tensor([5.0, 1.0, 1.0])  # More weight on hard negative

loss = contrastive_loss(
    embeddings, pos_pairs, neg_pairs,
    pos_weights=pos_weights,
    neg_weights=neg_weights
)
```

## Anchor Grouping

Pairs are grouped by their anchor (first) index. Each anchor's loss is computed over ALL its positive and negative pairs together:

```python
# Anchor 0 has 2 positives and 3 negatives
pos_pairs = torch.tensor([[0, 1], [0, 2], [1, 3]])
neg_pairs = torch.tensor([[0, 5], [0, 6], [0, 7], [1, 8]])

# Anchor 0: softmax over pairs (0,1), (0,2), (0,5), (0,6), (0,7)
# Anchor 1: softmax over pairs (1,3), (1,8)
# Final loss = average of anchor 0 and anchor 1 losses
```

## Edge Cases

| Case | Behavior |
|------|----------|
| Anchor has positives but no negatives | Loss = 0 for that anchor (nothing to contrast) |
| Anchor has negatives but no positives | Anchor is ignored (filtered out) |
| Empty `pos_pairs` | Returns 0.0 |
| Very low temperature | Loss → 0 if positive is closest (perfect separation) |

## Numerical Stability

The implementation uses the log-sum-exp trick to avoid overflow/underflow:

```
logsumexp(x) = max(x) + log(Σ exp(x - max(x)))
```

This allows stable computation even with:
- Very low temperatures (0.01)
- High-dimensional embeddings
- Large similarity differences

## Example: Training Loop

```python
import torch
from losses import contrastive_loss

# Simple encoder
encoder = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 128),
)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

for epoch in range(100):
    # Get batch of features
    features = torch.randn(16, 32)

    # Encode to embeddings
    embeddings = encoder(features)

    # Define pairs (from your data/sampling strategy)
    pos_pairs = torch.tensor([[i, i+1] for i in range(0, 14, 2)])
    neg_pairs = torch.tensor([[i, (i+8) % 16] for i in range(0, 14, 2)])

    # Compute loss and update
    loss = contrastive_loss(
        embeddings,
        pos_pairs,
        neg_pairs,
        temperature=0.1,
        similarity="l2"
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Example: With Pre-normalized Embeddings

```python
import torch.nn.functional as F
from losses import contrastive_loss

# Normalize embeddings before loss
embeddings = encoder(features)
embeddings = F.normalize(embeddings, p=2, dim=1)

# Now cosine, dot, and l2 are all related
# Use any similarity function
loss = contrastive_loss(
    embeddings, pos_pairs, neg_pairs,
    similarity="cosine",  # or "dot" (equivalent for normalized)
    temperature=0.07
)
```

## Comparison with Other Losses

| Loss | Pairs | Grouping | Weights |
|------|-------|----------|---------|
| `contrastive_loss` | Explicit pos/neg pairs | By anchor | Per-pair |
| Triplet Loss | (anchor, pos, neg) triplets | Per triplet | No |
| NT-Xent (SimCLR) | All-pairs in batch | Per sample | No |
| SupCon | Label-based pairs | Per sample | No |

`contrastive_loss` is most flexible when you have explicit pair relationships (e.g., from spatial/temporal proximity, metadata, etc.) rather than class labels.
