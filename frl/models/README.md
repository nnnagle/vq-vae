# Encoder Modules for Forest State Representation VQ-VAE

This directory contains neural network encoder modules for processing temporal and static geospatial data.

## Module Overview

### 1. TCN Encoder (`tcn.py`)

Temporal Convolutional Network for processing `[B, C, T, H, W]` tensors.

**Features:**
- Dilated 1D convolutions along time dimension
- Gated residual connections with learnable projections
- Pre-convolution dropout
- Pre-activation GroupNorm
- Optional temporal pooling (statistics or none)
- **Mask-aware temporal pooling** for handling clouds, missing data, etc.
- Post-pooling LayerNorm

**Example:**
```python
from frl.models import build_tcn_from_config

config = {
    'in_channels': 7,
    'channels': [128, 128, 128],
    'kernel_size': 3,
    'dilations': [1, 2, 4],
    'residual': {'projection_channels': 128},
    'dropout': {'rate': 0.10},
    'norm': {'num_groups': 16},
    'pooling': {'kind': 'stats'},  # or 'none'
    'post_pool_norm': {'kind': 'layernorm'}
}

encoder = build_tcn_from_config(config)

# Without mask
x = torch.randn(2, 7, 10, 32, 32)  # [B, C, T, H, W]
out = encoder(x)  # [B, 256, 32, 32] if pooling='stats' (mean+std)

# With mask (exclude clouds/invalid timesteps)
mask = torch.ones(2, 10, 32, 32, dtype=torch.bool)  # [B, T, H, W]
mask[:, 2:5, :, :] = False  # Mask out timesteps 2-4
out = encoder(x, mask=mask)  # Statistics computed only from valid timesteps
```

**Mask API:**
- Mask shape: `[B, T, H, W]`
- Values: `True`/`1.0` = valid, `False`/`0.0` = masked (excluded)
- When provided, masked timesteps are excluded from statistics computation
- Handles edge cases: all-masked locations default to count=1 to avoid NaN

### 2. Conv2D Encoder (`conv2d_encoder.py`)

Convolutional encoder for static `[B, C, H, W]` tensors.

**Features:**
- Multiple 2D convolutional layers
- Post-convolution dropout and GroupNorm
- Configurable kernel sizes (typically 1x1 or 3x3)
- ReLU activation

**Example:**
```python
from frl.models import build_conv2d_from_config

config = {
    'in_channels': 47,
    'channels': [128, 64],
    'kernel_size': 1,
    'padding': 0,
    'dropout': {'rate': 0.10},
    'norm': {'num_groups': [16, 8]},
    'activation': 'relu'
}

encoder = build_conv2d_from_config(config)
# Input: [B, 47, H, W]
# Output: [B, 64, H, W]
```

### 3. Gated Residual Conv2D (`spatial.py`)

Adaptive spatial smoothing with learned gating.

**Features:**
- Multiple 2D conv layers with gating mechanism
- Preserves sharp edges while smoothing homogeneous regions
- Channel-wise gates control blend of smoothed vs. original features

**Example:**
```python
from frl.models import build_gated_residual_conv2d_from_config

config = {
    'channels': 128,
    'conv': {'layers': 2, 'kernel_size': 3, 'padding': 1},
    'gate': {'hidden': 64, 'kernel_size': 1}
}

module = build_gated_residual_conv2d_from_config(config)
# Input: [B, 128, H, W]
# Output: [B, 128, H, W]
```

### 4. FiLM Conditioning (`conditioning.py`)

Feature-wise Linear Modulation for conditioning one pathway on another.

**Features:**
- Generates scale (gamma) and shift (beta) parameters
- Used to condition phase pathway on type latents
- Includes temporal broadcasting utilities

**Example:**
```python
from frl.models import FiLMLayer, broadcast_to_time

# Create FiLM layer
film = FiLMLayer(cond_dim=64, target_dim=128, hidden_dim=64)

# Generate modulation parameters
z_type = ...  # [B, 64, H, W]
gamma, beta = film(z_type)  # Each: [B, 128, H, W]

# Apply to features
h_phase = ...  # [B, 128, H, W]
modulated = film.modulate(h_phase, gamma, beta)

# Broadcast to temporal dimension for temporal features
gamma_t = broadcast_to_time(gamma, T=10)  # [B, 128, 10, H, W]
```

### 5. Prediction Heads (`heads.py`)

Various head architectures for different prediction tasks.

**MLP Head:**
```python
from frl.models import build_mlp_from_config

config = {
    'in_dim': 128,
    'layers': [256, 64],
    'output_dim': 64,
    'dropout': 0.0
}

head = build_mlp_from_config(config)
# Input: [B, 128, H, W]
# Output: [B, 64, H, W]
```

**Linear Head:**
```python
from frl.models import LinearHead

head = LinearHead(in_dim=64, out_dim=50)
# Simple 1x1 conv projection
```

**Conv2D Head:**
```python
from frl.models import Conv2DHead

head = Conv2DHead(
    in_channels=64,
    channels=[128, 64, 32],
    out_channels=20,
    kernel_size=1,
    activation='none'
)
# Multi-layer convolutional head
```

## Configuration Format

All modules support configuration via dictionaries that match the YAML structure from your model specification. The `build_*_from_config` functions parse these dictionaries and instantiate the appropriate modules.

### Example: Full Type Pathway

```python
from frl.models import (
    build_tcn_from_config,
    build_conv2d_from_config,
    build_gated_residual_conv2d_from_config,
    build_mlp_from_config
)
import torch

# 1. TCN for temporal features (ls8_delta)
tcn = build_tcn_from_config({
    'in_channels': 7,
    'channels': [128, 128, 128],
    'kernel_size': 3,
    'dilations': [1, 2, 4],
    'residual': {'projection_channels': 128},
    'dropout': {'rate': 0.10},
    'norm': {'num_groups': 16},
    'pooling': {'kind': 'stats'}
})

# 2. Conv2D for static features (ccdc_history)
conv_ccdc = build_conv2d_from_config({
    'in_channels': 47,
    'channels': [128, 64],
    'kernel_size': 1,
    'dropout': {'rate': 0.10},
    'norm': {'num_groups': [16, 8]},
    'activation': 'relu'
})

# 3. Conv2D for topography
conv_topo = build_conv2d_from_config({
    'in_channels': 8,
    'channels': [8, 6],
    'kernel_size': 1,
    'dropout': {'rate': 0.40},
    'norm': {'num_groups': 2},
    'activation': 'relu'
})

# 4. Trunk for fusion
trunk = build_conv2d_from_config({
    'in_channels': 326,  # 256 + 64 + 6
    'channels': [256, 128],
    'kernel_size': 1,
    'dropout': {'rate': 0.10},
    'norm': {'num_groups': [32, 16]},
    'activation': 'relu'
})

# 5. Spatial smoothing
spatial = build_gated_residual_conv2d_from_config({
    'channels': 128,
    'conv': {'layers': 2, 'kernel_size': 3, 'padding': 1},
    'gate': {'hidden': 64, 'kernel_size': 1}
})

# 6. Head to latent space
head = build_mlp_from_config({
    'in_dim': 128,
    'layers': [256, 64],
    'output_dim': 64,
    'dropout': 0.0
})

# Forward pass
ls8_delta = torch.randn(2, 7, 10, 32, 32)  # [B, C, T, H, W]
ccdc_hist = torch.randn(2, 47, 32, 32)     # [B, C, H, W]
topo = torch.randn(2, 8, 32, 32)           # [B, C, H, W]

feat_ls8 = tcn(ls8_delta)           # [2, 256, 32, 32]
feat_ccdc = conv_ccdc(ccdc_hist)    # [2, 64, 32, 32]
feat_topo = conv_topo(topo)         # [2, 6, 32, 32]

h_type = torch.cat([feat_ls8, feat_ccdc, feat_topo], dim=1)  # [2, 326, 32, 32]
h_type = trunk(h_type)              # [2, 128, 32, 32]
h_type_smooth = spatial(h_type)     # [2, 128, 32, 32]
z_type_cont = head(h_type_smooth)   # [2, 64, 32, 32]
```

## Tensor Conventions

As specified in your model YAML:

- **Temporal tensors**: `[B, C, T, H, W]` (batch, channels, time, height, width)
- **Static tensors**: `[B, C, H, W]` (batch, channels, height, width)

The TCN encoder handles the temporal dimension internally by reshaping to `[B*H*W, C, T]` for 1D convolutions, then reshaping back.

## Architecture Patterns

### Gated Residuals

The gated residual pattern used in TCN:
```
output = gate * conv_output + (1 - gate) * residual
```

This allows the network to learn when to use new features vs. preserve the input.

### FiLM Conditioning

Feature-wise Linear Modulation:
```
modulated = gamma * features + beta
```

Used to condition the phase pathway on type latents (with stop-gradient).

### Adaptive Spatial Smoothing

The GatedResidualConv2D learns to smooth homogeneous regions while preserving edges:
```
smoothed = conv_layers(input)
gate = gate_network(input)
output = gate * smoothed + (1 - gate) * input
```

## Testing

Run the example script to verify all modules:

```bash
python -m frl.examples.encoder_examples
```

This will instantiate all encoders with sample data and verify output shapes.

## Notes

- All modules use GroupNorm for stability with small batch sizes
- Dropout placement is configurable (preconv for TCN, postconv for Conv2D)
- Post-pooling LayerNorm helps stabilize training
- FiLM gamma is initialized near 1.0 (identity) for stable training start
