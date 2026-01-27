# Variance-Covariance Loss

VICReg-style loss for preventing embedding collapse in self-supervised learning.

## Overview

This loss encourages:
1. **Variance**: Each embedding dimension has sufficient variance (prevents collapse to constant)
2. **Covariance**: Embedding dimensions are decorrelated (encourages full use of embedding space)

## API

### `variance_covariance_loss`

```python
def variance_covariance_loss(
    embeddings: torch.Tensor,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**Parameters:**
- `embeddings`: `[N, D]` tensor where N is samples, D is embedding dimension
- `variance_weight`: Weight for variance term (default: 1.0)
- `covariance_weight`: Weight for covariance term (default: 1.0)
- `variance_target`: Target standard deviation per dimension (default: 1.0)
- `eps`: Numerical stability constant (default: 1e-4)

**Returns:**
- `(total_loss, variance_loss, covariance_loss)` tuple

### `variance_loss`

```python
def variance_loss(
    embeddings: torch.Tensor,
    target: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
```

Standalone variance component.

### `covariance_loss`

```python
def covariance_loss(
    embeddings: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
```

Standalone covariance component.

## Mathematical Details

### Variance Loss

For each dimension d:
```
var_loss_d = max(0, target - std(embeddings[:, d]))
```

Final loss is the mean over all dimensions. This is a hinge loss: no penalty when std >= target.

### Covariance Loss

1. Center embeddings: `z = embeddings - mean(embeddings)`
2. Compute covariance: `C = (z^T @ z) / (N - 1)`
3. Zero diagonal
4. Sum squared off-diagonal: `loss = sum(C_offdiag^2) / D`

## Usage

### Basic Usage

```python
from losses import variance_covariance_loss

embeddings = torch.randn(1000, 64)  # [N, D]
total, var_loss, cov_loss = variance_covariance_loss(embeddings)
```

### Remote Sensing with Spatial Latents

For latents of shape `[B, D, H, W]`, reshape to `[N, D]` first:

```python
# Latents from encoder: [B, D, H, W]
latents = encoder(images)

# Reshape: [B, D, H, W] -> [B*H*W, D]
B, D, H, W = latents.shape
embeddings = latents.permute(0, 2, 3, 1).reshape(-1, D)

# Compute loss
total, var_loss, cov_loss = variance_covariance_loss(embeddings)
```

### With Custom Weights

VICReg paper uses weights of 25 for both terms:

```python
total, var_loss, cov_loss = variance_covariance_loss(
    embeddings,
    variance_weight=25.0,
    covariance_weight=25.0,
)
```

### Combined with Other Losses

```python
from losses import variance_covariance_loss, reconstruction_loss

# Reconstruction loss
recon_loss = reconstruction_loss(reconstructed, target, loss_type="l2")

# Regularization
total_reg, _, _ = variance_covariance_loss(embeddings)

# Combined
loss = recon_loss + 0.1 * total_reg
```

## When to Use

- **Self-supervised learning**: Prevents representation collapse
- **VQ-VAE**: Encourages use of full codebook embedding space
- **Contrastive learning**: Use alongside contrastive loss for regularization
- **Remote sensing**: Ensures embedding space captures variation across entire region

## Typical Weight Values

| Use Case | variance_weight | covariance_weight |
|----------|----------------|-------------------|
| VICReg paper | 25.0 | 25.0 |
| Light regularization | 1.0 | 0.04 |
| Strong regularization | 10.0 | 10.0 |

## Detecting Collapse

High variance loss indicates dimensions are collapsing (low variance).
High covariance loss indicates dimensions are correlated (redundant).

```python
total, var_loss, cov_loss = variance_covariance_loss(embeddings)

if var_loss > 0.5:
    print("Warning: Some dimensions may be collapsing")
if cov_loss > 1.0:
    print("Warning: High correlation between dimensions")
```
