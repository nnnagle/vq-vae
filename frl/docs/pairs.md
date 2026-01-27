# Pair Generation for Contrastive Learning

The pair generation functions create positive and negative pair indices from distance matrices, for use with `contrastive_loss`.

## Overview

```
Distance Matrix [N, M]
         │
         ▼
┌─────────────────────────────────┐
│     Pair Generation Function    │
│                                 │
│  • pairs_knn (k-nearest)        │
│  • pairs_mutual_knn             │
│  • pairs_quantile               │
│  • pairs_radius                 │
│                                 │
│  Options:                       │
│  • symmetric (square only)      │
│  • anchor_cols (rectangular)    │
│  • valid_mask (exclusions)      │
│  • max_pairs (sampling)         │
│                                 │
└─────────────────────────────────┘
         │
         ▼
   Pairs [P, 2]
   (anchor_id, target_id)
```

## Quick Start

```python
import torch
from losses import pairs_knn, pairs_quantile, contrastive_loss

# Create embeddings and distance matrix
embeddings = torch.randn(100, 128)
distances = torch.cdist(embeddings, embeddings)  # [100, 100]

# Generate positive pairs: 5 nearest neighbors per anchor
pos_pairs = pairs_knn(distances, k=5)

# Generate negative pairs: 50-75% quantile range
neg_pairs = pairs_quantile(distances, low=0.5, high=0.75)

# Use with contrastive loss
loss = contrastive_loss(embeddings, pos_pairs, neg_pairs)
```

## API Reference

### pairs_knn

Select k-nearest neighbors for each anchor.

```python
def pairs_knn(
    distances: torch.Tensor,
    k: int,
    symmetric: bool = False,
    anchor_cols: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `distances` | `[N, M]` tensor | Distance matrix (N anchors, M candidates) |
| `k` | int | Number of nearest neighbors per anchor |
| `symmetric` | bool | Add reverse pairs (j,i) for each (i,j). Square matrices only. |
| `anchor_cols` | `[N]` tensor or None | Column index for each anchor (self-exclusion + ID mapping). Required for rectangular. |
| `valid_mask` | `[M]` tensor or None | 1=valid, 0=invalid. Excludes invalid indices from pairs. |
| `max_pairs` | int or None | Maximum pairs to return (random sampling if exceeded) |

**Returns:** `[P, 2]` tensor of `(anchor_id, target_id)` pairs

### pairs_mutual_knn

Select pairs where both points are in each other's k-NN. Inherently symmetric.

```python
def pairs_mutual_knn(
    distances: torch.Tensor,
    k: int,
    valid_mask: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `distances` | `[N, N]` tensor | Square distance matrix |
| `k` | int | Number of nearest neighbors to consider |
| `valid_mask` | `[N]` tensor or None | 1=valid, 0=invalid |
| `max_pairs` | int or None | Maximum pairs to return |

**Returns:** `[P, 2]` tensor of `(anchor_id, target_id)` pairs

### pairs_quantile

Select pairs whose distances fall within a quantile range.

```python
def pairs_quantile(
    distances: torch.Tensor,
    low: float = 0.0,
    high: float = 0.1,
    symmetric: bool = False,
    anchor_cols: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `distances` | `[N, M]` tensor | Distance matrix |
| `low` | float | Lower quantile bound (inclusive), in [0, 1] |
| `high` | float | Upper quantile bound (exclusive), in [0, 1] |
| `symmetric` | bool | Add reverse pairs. Square matrices only. |
| `anchor_cols` | `[N]` tensor or None | Required for rectangular matrices |
| `valid_mask` | `[M]` tensor or None | 1=valid, 0=invalid |
| `max_pairs` | int or None | Maximum pairs to return |

**Returns:** `[P, 2]` tensor of `(anchor_id, target_id)` pairs

### pairs_radius

Select pairs whose distances fall within an absolute range.

```python
def pairs_radius(
    distances: torch.Tensor,
    min_dist: float = 0.0,
    max_dist: float = float("inf"),
    symmetric: bool = False,
    anchor_cols: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    max_pairs: int | None = None,
) -> torch.Tensor
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `distances` | `[N, M]` tensor | Distance matrix |
| `min_dist` | float | Minimum distance (inclusive) |
| `max_dist` | float | Maximum distance (exclusive) |
| `symmetric` | bool | Add reverse pairs. Square matrices only. |
| `anchor_cols` | `[N]` tensor or None | Required for rectangular matrices |
| `valid_mask` | `[M]` tensor or None | 1=valid, 0=invalid |
| `max_pairs` | int or None | Maximum pairs to return |

**Returns:** `[P, 2]` tensor of `(anchor_id, target_id)` pairs

## Choosing a Strategy

| Strategy | Use Case | Pair Count |
|----------|----------|------------|
| `pairs_knn` | Fixed number of positives per anchor | N × k |
| `pairs_mutual_knn` | High-confidence similarity (both agree) | Variable, usually < N × k |
| `pairs_quantile` | Relative distance thresholds | ~(high - low) × N × M |
| `pairs_radius` | Absolute distance thresholds | Variable |

### Typical Combinations

```python
# Positives: nearest neighbors
# Negatives: medium-far distances
pos_pairs = pairs_knn(distances, k=5)
neg_pairs = pairs_quantile(distances, low=0.5, high=0.75)

# Positives: mutual nearest neighbors (high confidence)
# Negatives: far distances
pos_pairs = pairs_mutual_knn(distances, k=10)
neg_pairs = pairs_quantile(distances, low=0.75, high=1.0)

# Positives: within radius
# Negatives: outside radius
pos_pairs = pairs_radius(distances, max_dist=0.5)
neg_pairs = pairs_radius(distances, min_dist=2.0)
```

## Square vs Rectangular Matrices

### Square Matrix (N × N)

Self-comparisons where rows and columns share the same index space.

```python
embeddings = torch.randn(100, 128)
distances = torch.cdist(embeddings, embeddings)  # [100, 100]

# anchor_cols defaults to [0, 1, ..., N-1]
# Self-pairs (diagonal) automatically excluded
pairs = pairs_knn(distances, k=5)
```

### Rectangular Matrix (N × M)

N anchors compared against M candidates (M ≥ N). Anchors are typically a subset of candidates.

```python
n_anchors = 256
n_candidates = 256 * 256  # 65536

# Anchors are first 256 candidates
anchor_embeddings = all_embeddings[:n_anchors]
distances = torch.cdist(anchor_embeddings, all_embeddings)  # [256, 65536]

# anchor_cols maps row i to column i (anchors are columns 0-255)
anchor_cols = torch.arange(n_anchors)

pairs = pairs_knn(distances, k=10, anchor_cols=anchor_cols)
```

### anchor_cols Explained

`anchor_cols` serves two purposes:

1. **Self-exclusion**: For row `i`, column `anchor_cols[i]` is excluded
2. **Anchor ID mapping**: Returned pairs use `anchor_cols[i]` as the anchor ID

```python
# Example: 3 anchors, 100 candidates
# Anchors are at candidate indices 50, 75, 82
anchor_cols = torch.tensor([50, 75, 82])

# Row 0 → excludes column 50, returns pairs like (50, target)
# Row 1 → excludes column 75, returns pairs like (75, target)
# Row 2 → excludes column 82, returns pairs like (82, target)
```

## Symmetric Pairs

With `symmetric=True`, for each pair (i, j), the reverse (j, i) is also added.

```python
# Without symmetric: if j is in i's KNN, add (i, j)
pairs = pairs_knn(distances, k=5, symmetric=False)

# With symmetric: also add (j, i) if j is a valid anchor
pairs = pairs_knn(distances, k=5, symmetric=True)
```

**Constraints:**
- `symmetric=True` requires a square distance matrix
- `symmetric=True` cannot be combined with `anchor_cols`

**Note:** Symmetric pairs are NOT deduplicated. If both (i,j) and (j,i) would be selected naturally, both appear twice. This is intentional for contrastive loss, where each anchor needs its own pairs.

## Validity Mask

Exclude certain indices from pair generation:

```python
valid_mask = torch.ones(100)
valid_mask[[5, 10, 15]] = 0  # Mark indices 5, 10, 15 as invalid

pairs = pairs_knn(distances, k=5, valid_mask=valid_mask)
# No pairs will include indices 5, 10, or 15 (as anchor or target)
```

**Automatic invalid handling:**
- `inf` values in distances → treated as invalid
- `nan` values in distances → treated as invalid
- Invalid anchors → entire row excluded
- Invalid targets → specific pairs excluded

## Max Pairs and Sampling

For large distance matrices, pair count can be very large:

```python
# 256 × 65536 matrix, 10% quantile → ~1.6M pairs
pairs = pairs_quantile(distances, low=0.0, high=0.1)  # Could be huge!

# Limit with random sampling
pairs = pairs_quantile(distances, low=0.0, high=0.1, max_pairs=10000)
```

Sampling is random (using `torch.randperm`), so different calls may return different subsets.

## Edge Cases

| Case | Behavior |
|------|----------|
| k > valid candidates | Return fewer pairs (graceful degradation) |
| Empty result | Return `torch.empty((0, 2), dtype=torch.long)` |
| All distances invalid | Return empty tensor |
| Quantile low == high | Error (require low < high) |
| Rectangular + symmetric | Error |
| anchor_cols + symmetric | Error |

## Performance Considerations

**Quantile computation:**
- Requires sorting: O(N·M · log(N·M))
- For 256 × 65536: ~20-50ms on GPU

**KNN computation:**
- Uses `torch.topk`: O(N·M · log(k))
- Generally faster than quantile for small k

**Recommendation:** If computing both positive and negative pairs from the same distance matrix using quantile, the quantile thresholds are computed independently. For very large matrices in tight loops, consider caching the sorted distances.

## Example: Complete Training Pipeline

```python
import torch
from losses import pairs_knn, pairs_quantile, contrastive_loss

# Model and optimizer
encoder = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 128),
)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

for epoch in range(100):
    for batch in dataloader:
        # Get embeddings
        embeddings = encoder(batch)

        # Compute distances (detach to avoid graph through distance computation)
        with torch.no_grad():
            distances = torch.cdist(embeddings, embeddings)

            # Generate pairs
            pos_pairs = pairs_knn(distances, k=5)
            neg_pairs = pairs_quantile(distances, low=0.5, high=0.8, max_pairs=1000)

        # Compute loss (gradients flow through embeddings, not distances)
        loss = contrastive_loss(
            embeddings, pos_pairs, neg_pairs,
            temperature=0.1, similarity="l2"
        )

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Example: Rectangular with Validity Mask

```python
# Scenario: 100 query images, 10000 reference images
# Queries are references 0-99
# Some references are marked as invalid

n_queries = 100
n_references = 10000

query_emb = torch.randn(n_queries, 128)
ref_emb = torch.randn(n_references, 128)
ref_emb[:n_queries] = query_emb  # Queries are first 100 references

distances = torch.cdist(query_emb, ref_emb)  # [100, 10000]

# Queries map to columns 0-99
anchor_cols = torch.arange(n_queries)

# Mark some references as invalid
valid_mask = torch.ones(n_references)
invalid_refs = [50, 200, 500, 1000]  # Bad images
valid_mask[invalid_refs] = 0

# Generate pairs
pos_pairs = pairs_knn(distances, k=10, anchor_cols=anchor_cols, valid_mask=valid_mask)
neg_pairs = pairs_quantile(
    distances, low=0.5, high=0.8,
    anchor_cols=anchor_cols, valid_mask=valid_mask,
    max_pairs=5000
)
```
