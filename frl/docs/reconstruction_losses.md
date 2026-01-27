# Reconstruction Losses

This module provides loss functions for reconstructing different types of data:

- **Continuous** (`reconstruction_loss`): L1, L2, Huber for real-valued targets
- **Categorical** (`categorical_loss`): Cross-entropy for discrete class labels
- **Count** (`count_loss`): Poisson, Negative Binomial for count data

All losses support masking to ignore invalid regions (clouds, missing data, etc.).

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Reconstruction Losses                           │
├─────────────────────┬─────────────────────┬─────────────────────────┤
│      Continuous     │     Categorical     │         Count           │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│  reconstruction_loss│  categorical_loss   │      count_loss         │
│                     │                     │                         │
│  • L1 (MAE)         │  • Cross-Entropy    │  • Poisson              │
│  • L2 (MSE)         │  • Label Smoothing  │  • Negative Binomial    │
│  • Huber            │  • Class Weights    │  • Learned Dispersion   │
│  • Smooth L1        │                     │                         │
└─────────────────────┴─────────────────────┴─────────────────────────┘
```

## Quick Start

```python
from losses import reconstruction_loss, categorical_loss, count_loss
import torch

# Continuous (e.g., image reconstruction)
pred = model(x)
loss = reconstruction_loss(pred, target, mask, loss_type="l2")

# Categorical (e.g., segmentation)
logits = model(x)  # [B, C, H, W]
loss = categorical_loss(logits, class_labels, mask)

# Count (e.g., density estimation)
rate = torch.exp(model(x))  # Must be positive
loss = count_loss(rate, counts, mask, loss_type="poisson")
```

---

## Continuous: `reconstruction_loss`

For real-valued targets (e.g., spectral values, normalized features).

### API

```python
def reconstruction_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    loss_type: Literal["l1", "l2", "mse", "huber", "smooth_l1"] = "l2",
    reduction: Literal["mean", "sum", "none"] = "mean",
    delta: float = 1.0,
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | Tensor | Predicted values, any shape |
| `target` | Tensor | Target values, broadcastable with input |
| `mask` | Tensor or None | Boolean mask, True = valid |
| `loss_type` | str | `"l1"`, `"l2"`, `"mse"`, `"huber"`, `"smooth_l1"` |
| `reduction` | str | `"mean"`, `"sum"`, `"none"` |
| `delta` | float | Huber loss threshold (default: 1.0) |

### Loss Types

| Type | Formula | Use Case |
|------|---------|----------|
| `l1` | `\|x - y\|` | Robust to outliers |
| `l2` / `mse` | `(x - y)²` | Standard, differentiable |
| `huber` | Quadratic if `\|x-y\| < δ`, else linear | Best of both |
| `smooth_l1` | PyTorch's smooth L1 | Object detection |

### Example

```python
# Basic usage
loss = reconstruction_loss(pred, target)

# With mask (ignore clouds)
mask = ~cloud_mask  # True = valid
loss = reconstruction_loss(pred, target, mask, loss_type="l1")

# Huber loss (robust to outliers)
loss = reconstruction_loss(pred, target, loss_type="huber", delta=0.5)
```

### When to Use Each Loss Type

- **L2/MSE**: Default choice, penalizes large errors quadratically
- **L1/MAE**: When data has outliers, more robust
- **Huber**: When you want L2 for small errors, L1 for large (adjust `delta`)
- **Smooth L1**: Common in detection tasks, similar to Huber with `delta=1`

---

## Categorical: `categorical_loss`

For discrete class labels (e.g., land cover type, segmentation masks).

### API

```python
def categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `logits` | `[B, C, ...]` | Raw model outputs (before softmax) |
| `target` | `[B, ...]` | Class indices in `[0, C-1]` |
| `mask` | Tensor or None | Boolean mask, True = valid |
| `class_weights` | `[C]` or None | Per-class weights for imbalance |
| `reduction` | str | `"mean"`, `"sum"`, `"none"` |
| `label_smoothing` | float | Smoothing factor `[0, 1]` |
| `ignore_index` | int | Target value to ignore (default: -100) |

### Example

```python
# Basic segmentation
logits = model(image)  # [B, 10, H, W] for 10 classes
target = labels  # [B, H, W] with values 0-9
loss = categorical_loss(logits, target)

# Ignore background (class 0)
mask = target != 0
loss = categorical_loss(logits, target, mask)

# Handle class imbalance
weights = torch.tensor([0.1, 1.0, 2.0, 1.5, ...])  # Per-class
loss = categorical_loss(logits, target, class_weights=weights)

# Label smoothing for regularization
loss = categorical_loss(logits, target, label_smoothing=0.1)
```

### Class Weights for Imbalanced Data

```python
# Compute inverse frequency weights
class_counts = torch.bincount(target.flatten(), minlength=num_classes)
weights = 1.0 / (class_counts.float() + 1)
weights = weights / weights.sum() * num_classes  # Normalize

loss = categorical_loss(logits, target, class_weights=weights)
```

### Label Smoothing

Converts hard targets to soft targets:
- Without smoothing: `[0, 0, 1, 0, 0]` (one-hot)
- With smoothing=0.1: `[0.02, 0.02, 0.92, 0.02, 0.02]`

Benefits:
- Prevents overconfident predictions
- Improves generalization
- Acts as regularization

---

## Count: `count_loss`

For count/rate data (e.g., tree counts, event frequencies).

### API

```python
def count_loss(
    rate: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    loss_type: Literal["poisson", "negative_binomial"] = "poisson",
    reduction: Literal["mean", "sum", "none"] = "mean",
    dispersion: torch.Tensor | float = 1.0,
    full: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `rate` | Tensor | Predicted rate λ (must be positive) |
| `target` | Tensor | Target counts (non-negative) |
| `mask` | Tensor or None | Boolean mask, True = valid |
| `loss_type` | str | `"poisson"` or `"negative_binomial"` |
| `reduction` | str | `"mean"`, `"sum"`, `"none"` |
| `dispersion` | float or Tensor | NegBin dispersion parameter |
| `full` | bool | Include log(k!) term for Poisson (default: False) |
| `eps` | float | Numerical stability (default: 1e-8) |

### Loss Types

| Type | Variance | Use Case |
|------|----------|----------|
| `poisson` | variance = mean | Standard count data |
| `negative_binomial` | variance = mean + mean²/r | Overdispersed counts |

### Example

```python
# Poisson for count prediction
log_rate = model(features)
rate = torch.exp(log_rate)  # Ensure positive via exp
loss = count_loss(rate, target_counts, loss_type="poisson")

# Full Poisson NLL (non-negative, comparable to NegBin)
loss = count_loss(rate, target_counts, loss_type="poisson", full=True)

# Alternative: use softplus for positivity
rate = F.softplus(model(features))

# Negative Binomial for overdispersed data
loss = count_loss(rate, target, loss_type="negative_binomial", dispersion=10.0)

# Learned per-pixel dispersion
rate, log_disp = model(features)  # Model outputs both
dispersion = torch.exp(log_disp)
loss = count_loss(rate, target, loss_type="negative_binomial", dispersion=dispersion)
```

### Poisson vs Negative Binomial

**Poisson**: Assumes variance = mean
- Good for "well-behaved" count data
- Simpler (one parameter: λ)

**Negative Binomial**: Allows variance > mean (overdispersion)
- Better for real-world data where variance often exceeds mean
- Extra parameter: dispersion `r`
- As `r → ∞`, approaches Poisson

```python
# Check for overdispersion
mean = counts.mean()
var = counts.var()
print(f"Mean: {mean:.2f}, Var: {var:.2f}, Ratio: {var/mean:.2f}")
# If ratio >> 1, use negative_binomial
```

---

## Masking

All losses support boolean masks where `True = valid`, `False = ignore`.

### Mask Broadcasting

Masks are automatically broadcast to match the loss shape:

```python
# Spatial mask [B, H, W] works with [B, C, H, W] tensor
mask = torch.ones(4, 32, 32, dtype=torch.bool)
loss = reconstruction_loss(pred, target, mask)  # pred is [4, 3, 32, 32]
```

### Common Masking Patterns

```python
# Ignore clouds
mask = ~cloud_mask

# Ignore NaN values
mask = ~torch.isnan(target)
target = torch.nan_to_num(target, 0)

# Ignore specific class (e.g., background)
mask = target != 0

# Combine multiple masks
mask = valid_mask & ~cloud_mask & (target >= 0)

# All masked -> returns 0.0
mask = torch.zeros_like(target, dtype=torch.bool)
loss = reconstruction_loss(pred, target, mask)  # Returns 0.0
```

---

## Multi-Output Models

Combine different losses for multi-task learning:

```python
# Model outputs
rgb_pred = model.rgb_head(features)      # Continuous
class_logits = model.class_head(features)  # Categorical
count_rate = model.count_head(features)   # Count

# Individual losses
loss_rgb = reconstruction_loss(rgb_pred, rgb_target, mask, loss_type="l2")
loss_class = categorical_loss(class_logits, class_target, mask)
loss_count = count_loss(count_rate, count_target, mask, loss_type="poisson")

# Weighted combination
total_loss = (
    1.0 * loss_rgb
    + 0.5 * loss_class
    + 0.1 * loss_count
)
```

---

## Edge Cases

| Case | Behavior |
|------|----------|
| All masked (`mask.all() == False`) | Returns 0.0 |
| Empty tensor | Returns 0.0 |
| NaN in masked regions | Ignored (not propagated) |
| Negative rate in `count_loss` | Clamped to `eps` (use exp/softplus) |

---

## Summary Table

| Function | Data Type | Input Shape | Target Shape | Key Parameters |
|----------|-----------|-------------|--------------|----------------|
| `reconstruction_loss` | Continuous | `[B, ...]` | `[B, ...]` | `loss_type`, `delta` |
| `categorical_loss` | Discrete | `[B, C, ...]` | `[B, ...]` | `class_weights`, `label_smoothing` |
| `count_loss` | Counts | `[B, ...]` | `[B, ...]` | `loss_type`, `dispersion` |
