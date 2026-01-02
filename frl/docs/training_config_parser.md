# Training Configuration Parser - Usage Guide

## Overview

The `training_config_parser.py` provides a complete system for loading, parsing, validating, and querying training configurations for the forest representation learning model.

## Quick Start

```python
from training_config_parser import TrainingConfigParser, load_training_config

# Method 1: Full control
parser = TrainingConfigParser('frl_training_v0.yaml')
config = parser.parse()
parser.validate()
print(parser.summary())

# Method 2: Convenience function
config = load_training_config('frl_training_v0.yaml')

# Access settings
print(f"Batch size: {config.training.batch_size}")
print(f"Learning rate: {config.optimizer.lr}")
print(f"Debug mode: {config.is_debug_mode}")
```

## Key Features

âœ… **Structured access** - Dotted notation for all settings  
âœ… **Type safety** - Dataclasses with proper types  
âœ… **Validation** - Comprehensive consistency checks  
âœ… **Loss schedules** - Query weights at any epoch  
âœ… **Summaries** - Human-readable configuration overview  
âœ… **Error checking** - Clear error messages  

---

## Core Classes

### TrainingConfigParser

Main parser class for loading and validating configurations.

```python
parser = TrainingConfigParser('config.yaml')
config = parser.parse()          # Parse YAML
parser.validate()                 # Validate consistency
summary = parser.summary()        # Get summary
schedule = parser.get_loss_schedule()  # Get loss schedule
```

### TrainingConfiguration

Complete configuration with structured access to all settings.

```python
config = parser.parse()

# Access nested settings with dot notation
batch_size = config.training.batch_size
lr = config.optimizer.lr
patch_size = config.sampling.patch_size
debug_mode = config.spatial_domain.debug_mode

# Properties
is_debug = config.is_debug_mode
total_steps = config.total_training_steps
active_window = config.spatial_domain.active_window
```

---

## Configuration Sections

### 1. Run Configuration

```python
config.run.experiment_name       # 'frl_v0_exp001'
config.run.run_root              # 'runs'
config.run.ckpt_dir              # 'checkpoints'

# Checkpoint settings
config.run.checkpoint.save_every_n_epochs  # 5
config.run.checkpoint.save_top_k           # 3
config.run.checkpoint.monitor              # 'val/loss_total'
config.run.checkpoint.mode                 # 'min'
```

**Example:**
```python
# Check checkpoint configuration
ckpt = config.run.checkpoint
print(f"Saving checkpoints every {ckpt.save_every_n_epochs} epochs")
print(f"Keeping top {ckpt.save_top_k} by {ckpt.monitor}")
```

### 2. Hardware Configuration

```python
config.hardware.device                # 'cuda'
config.hardware.gpu_ids               # [0]
config.hardware.num_workers           # 4
config.hardware.pin_memory            # True

# Mixed precision
config.hardware.mixed_precision.enabled  # True
config.hardware.mixed_precision.dtype    # 'bfloat16'
```

**Example:**
```python
# Set up device
import torch
device = torch.device(config.hardware.device)
num_gpus = len(config.hardware.gpu_ids)
print(f"Using {num_gpus} GPU(s): {config.hardware.gpu_ids}")
```

### 3. Training Configuration

```python
config.training.num_epochs        # 100
config.training.batch_size        # 4

# Gradient clipping
config.training.gradient_clip.enabled    # True
config.training.gradient_clip.max_norm   # 1.0

# Early stopping
config.training.early_stopping.enabled   # True
config.training.early_stopping.patience  # 15

# Validation
config.training.validation.enabled          # True
config.training.validation.val_every_n_epochs  # 1
config.training.validation.val_fraction     # 0.15
```

**Example:**
```python
# Training loop setup
for epoch in range(config.training.num_epochs):
    # Training
    train_one_epoch(...)
    
    # Clip gradients
    if config.training.gradient_clip.enabled:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.gradient_clip.max_norm
        )
    
    # Validation
    if epoch % config.training.validation.val_every_n_epochs == 0:
        validate(...)
```

### 4. Optimizer Configuration

```python
config.optimizer.name            # 'adamw'
config.optimizer.lr              # 1e-4
config.optimizer.weight_decay    # 0.01
```

**Example:**
```python
# Create optimizer
import torch.optim as optim

if config.optimizer.name == 'adamw':
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay
    )
elif config.optimizer.name == 'adam':
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.optimizer.lr
    )
```

### 5. Scheduler Configuration

```python
config.scheduler.name             # 'cosine_warmup'

# Warmup
config.scheduler.warmup.enabled   # True
config.scheduler.warmup.epochs    # 5

# Cosine settings
config.scheduler.T_max            # 95
config.scheduler.eta_min          # 1e-6
```

**Example:**
```python
# Create scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

if config.scheduler.name == 'cosine_warmup':
    # Custom warmup + cosine implementation
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=config.scheduler.warmup.epochs,
        T_max=config.scheduler.T_max,
        eta_min=config.scheduler.eta_min
    )
```

### 6. Spatial Domain Configuration

```python
config.spatial_domain.debug_mode           # True

# Debug window
config.spatial_domain.debug_window.origin  # [8192, 8192]
config.spatial_domain.debug_window.size    # [1024, 1024]

# Full domain
config.spatial_domain.full_domain.origin   # [0, 0]
config.spatial_domain.full_domain.size     # [13056, 23552]

# Active window (automatically selects based on debug_mode)
config.spatial_domain.active_window.origin
config.spatial_domain.active_window.size
config.spatial_domain.active_window.height
config.spatial_domain.active_window.width
```

**Example:**
```python
# Get active spatial extent
window = config.spatial_domain.active_window
print(f"Training on region: {window.origin} + {window.size}")
print(f"  Row: {window.row_start} to {window.row_start + window.height}")
print(f"  Col: {window.col_start} to {window.col_start + window.width}")

# Calculate number of patches
num_patches_h = window.height // config.sampling.patch_size
num_patches_w = window.width // config.sampling.patch_size
total_patches = num_patches_h * num_patches_w
print(f"Total patches: {total_patches:,}")
```

### 7. Temporal Domain Configuration

```python
config.temporal_domain.end_years       # [2020, 2022, 2024]
config.temporal_domain.window_length   # 10

# Bundle configuration
config.temporal_domain.bundle.enabled  # True
config.temporal_domain.bundle.size     # 3
config.temporal_domain.bundle.offsets  # [0, -2, -4]

# Sampling
config.temporal_domain.sampling.mode      # 'weighted'
config.temporal_domain.sampling.weights   # {2024: 0.4, 2022: 0.3, 2020: 0.3}
```

**Example:**
```python
# Sample a temporal window
import random
import numpy as np

if config.temporal_domain.sampling.mode == 'weighted':
    weights_dict = config.temporal_domain.sampling.weights
    years = list(weights_dict.keys())
    weights = list(weights_dict.values())
    
    anchor_year = random.choices(years, weights=weights)[0]
else:
    # Uniform sampling
    anchor_year = random.choice(config.temporal_domain.end_years)

# Get bundle offsets
offsets = config.temporal_domain.bundle.offsets
bundle_years = [anchor_year + offset for offset in offsets]
print(f"Sampled bundle: {bundle_years}")
```

### 8. Sampling Configuration

```python
config.sampling.patch_size                 # 256

# Grid subsample
config.sampling.grid_subsample.enabled     # False
config.sampling.grid_subsample.grid_size   # [16, 16]

# Forest samples
config.sampling.forest_samples.enabled     # False
config.sampling.forest_samples.per_patch   # 64

# Augmentation
config.sampling.augmentation.enabled              # False
config.sampling.augmentation.random_flip          # {prob: 0.5}
config.sampling.augmentation.random_rotation      # {angles: [0, 90, 180, 270]}
```

**Example:**
```python
# Apply augmentation
import torch
import torch.nn.functional as F

def augment_patch(patch):
    if not config.sampling.augmentation.enabled:
        return patch
    
    # Random flip
    flip_cfg = config.sampling.augmentation.random_flip
    if flip_cfg and random.random() < flip_cfg['prob']:
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=[-1])  # Horizontal
        else:
            patch = torch.flip(patch, dims=[-2])  # Vertical
    
    # Random rotation
    rot_cfg = config.sampling.augmentation.random_rotation
    if rot_cfg and random.random() < rot_cfg.get('prob', 0.5):
        angle = random.choice(rot_cfg['angles'])
        k = angle // 90  # Number of 90° rotations
        patch = torch.rot90(patch, k=k, dims=[-2, -1])
    
    return patch
```

### 9. Losses Configuration

```python
# VQ loss
config.losses.vq_loss.enabled          # False (in your config)
config.losses.vq_loss.weight           # 1.0
config.losses.vq_loss.commitment_cost  # 0.25

# Triplet loss
config.losses.z_type_triplet.enabled   # True
config.losses.z_type_triplet.triplet_mining  # {strategy: 'hardest_in_batch', margin: 0.2}

# Phase monotonicity
config.losses.phase_monotonicity.enabled      # True
config.losses.phase_monotonicity.constraints  # List[MonotonicityConstraint]
```

**Example - Get loss weights at specific epoch:**
```python
# Get all loss weights for epoch 25
weights = config.losses.get_total_weight_at_epoch(25)
print(f"Epoch 25 weights: {weights}")
# Output: {'z_type_triplet': 1.0, 'vq_loss': 0.0, 'phase_monotonicity': 0.5}

# Check individual losses
epoch = 15
triplet_weight = config.losses.z_type_triplet.get_weight_at_epoch(epoch)
print(f"Triplet weight at epoch {epoch}: {triplet_weight}")
```

**Example - Compute total loss:**
```python
def compute_total_loss(losses_dict, epoch):
    """Compute weighted total loss"""
    weights = config.losses.get_total_weight_at_epoch(epoch)
    
    total = 0.0
    for loss_name, loss_value in losses_dict.items():
        if loss_name in weights:
            weighted = loss_value * weights[loss_name]
            total += weighted
            print(f"  {loss_name}: {loss_value:.4f} × {weights[loss_name]:.2f} = {weighted:.4f}")
    
    return total

# Usage
losses = {
    'z_type_triplet': 0.523,
    'vq_loss': 1.234,
    'phase_monotonicity': 0.089
}
total_loss = compute_total_loss(losses, epoch=25)
```

### 10. Masking Configuration

```python
config.masking.global_mask       # ['shared.masks.aoi', 'shared.masks.forest']
config.masking.per_loss_masks    # {'z_type_triplet': [...]}
config.masking.global_weight     # 'shared.quality.forest_weight'
```

**Example:**
```python
# Get masks for a specific loss
loss_name = 'z_type_triplet'
masks = config.masking.per_loss_masks.get(loss_name, [])
print(f"Masks for {loss_name}: {masks}")

# Global weight reference
weight_ref = config.masking.global_weight
print(f"Using global weight: {weight_ref}")
```

### 11. Metrics Configuration

```python
config.metrics.train              # ['loss_total', 'loss_vq', ...]
config.metrics.validation         # ['loss_total', 'loss_vq']

# Latent analysis
config.metrics.latent_analysis    # {'enabled': True, 'compute_every_n_epochs': 10}
```

**Example:**
```python
# Set up metric tracking
from collections import defaultdict

metrics = {
    'train': defaultdict(list),
    'val': defaultdict(list)
}

# During training
for metric_name in config.metrics.train:
    value = compute_metric(metric_name)
    metrics['train'][metric_name].append(value)

# Latent analysis
latent_cfg = config.metrics.latent_analysis
if latent_cfg and latent_cfg.get('enabled'):
    if epoch % latent_cfg.get('compute_every_n_epochs', 10) == 0:
        analyze_latent_space(model)
```

### 12. Visualization Configuration

```python
config.visualization.enabled      # False (in your config)
config.visualization.tensorboard  # {'log_images_every_n_epochs': 5}
```

**Example:**
```python
# Tensorboard logging
if config.visualization.enabled:
    tb_cfg = config.visualization.tensorboard
    if epoch % tb_cfg.get('log_images_every_n_epochs', 5) == 0:
        log_images_to_tensorboard(writer, images, epoch)
```

### 13. Reproducibility Configuration

```python
config.reproducibility.seed       # 42
config.reproducibility.benchmark  # True
```

**Example:**
```python
# Set random seeds
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if not config.reproducibility.benchmark:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

set_seed(config.reproducibility.seed)
```

---

## Advanced Usage

### Get Loss Schedule

```python
# Print loss schedule summary
print(parser.get_loss_schedule())

# Output:
# Loss Weight Schedule
# ==================================================
# Epoch      VQ         Triplet    Monoton.  
# --------------------------------------------------
# 0          0.00       0.00       0.00      
# 10         0.00       0.00       0.50      
# 20         0.00       1.00       0.50      
# 30         0.00       1.00       1.00      
# 50         0.00       1.00       1.00      
# 99         0.00       1.00       1.00
```

### Plot Loss Schedule

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(0, config.training.num_epochs)
weights = {
    'VQ': [],
    'Triplet': [],
    'Monotonicity': []
}

for epoch in epochs:
    w = config.losses.get_total_weight_at_epoch(epoch)
    weights['VQ'].append(w['vq_loss'])
    weights['Triplet'].append(w['z_type_triplet'])
    weights['Monotonicity'].append(w['phase_monotonicity'])

plt.figure(figsize=(10, 6))
for name, values in weights.items():
    plt.plot(epochs, values, label=name, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss Weight')
plt.title('Loss Weight Schedule')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_schedule.png')
```

### Validate Against Bindings

```python
from data.loaders.bindings.parser import BindingsParser

# Load both configs
training_parser = TrainingConfigParser('frl_training_v0.yaml')
training_config = training_parser.parse()

bindings_parser = BindingsParser(training_config.config_paths['bindings_path'])
bindings_config = bindings_parser.parse()

# Check consistency
assert training_config.temporal_domain.window_length == \
       bindings_config['training']['windowing']['length_years'], \
       "Window length mismatch!"

assert set(training_config.temporal_domain.end_years) == \
       set(bindings_config['training']['windowing']['anchor_sampling']['end_years']), \
       "End years mismatch!"

print("✓ Training config consistent with bindings")
```

### Create Config Programmatically

```python
from training_config_parser import (
    TrainingConfiguration, RunConfig, HardwareConfig,
    TrainingConfig, OptimizerConfig, SchedulerConfig
)

# Create config from scratch
config = TrainingConfiguration(
    version="1.0",
    name="custom_experiment",
    config_paths={
        'bindings_path': 'config/frl_bindings_v0.yaml',
        'model_path': 'config/frl_model_v0.yaml'
    },
    run=RunConfig(experiment_name="exp_001"),
    hardware=HardwareConfig(device='cuda', gpu_ids=[0, 1]),
    training=TrainingConfig(num_epochs=50, batch_size=8),
    # ... other configs
)
```

### Export Modified Config

```python
import yaml

# Modify config
config.training.batch_size = 8
config.optimizer.lr = 5e-4

# Convert back to dict (you'll need to implement to_dict methods)
# Or just modify the raw YAML and re-save

with open('modified_config.yaml', 'w') as f:
    yaml.dump(parser.raw_config, f, default_flow_style=False)
```

---

## Common Patterns

### Pattern 1: Training Loop Setup

```python
from training_config_parser import load_training_config

# Load config
config = load_training_config('frl_training_v0.yaml')

# Initialize training
device = torch.device(config.hardware.device)
model = build_model(config).to(device)

optimizer = create_optimizer(model, config.optimizer)
scheduler = create_scheduler(optimizer, config.scheduler)

# Training loop
for epoch in range(config.training.num_epochs):
    # Get loss weights for this epoch
    loss_weights = config.losses.get_total_weight_at_epoch(epoch)
    
    # Train
    train_loss = train_epoch(model, dataloader, optimizer, loss_weights)
    
    # Clip gradients
    if config.training.gradient_clip.enabled:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.gradient_clip.max_norm
        )
    
    # Update learning rate
    scheduler.step()
    
    # Validate
    if epoch % config.training.validation.val_every_n_epochs == 0:
        val_loss = validate(model, val_dataloader)
    
    # Checkpoint
    if epoch % config.run.checkpoint.save_every_n_epochs == 0:
        save_checkpoint(model, optimizer, epoch)
```

### Pattern 2: Dataset Creation

```python
from training_config_parser import load_training_config
from data.loaders.data_reader import DataReader

config = load_training_config('frl_training_v0.yaml')

# Get active spatial window
window = config.spatial_domain.active_window

# Create dataset with config settings
dataset = ForestDataset(
    spatial_origin=window.origin,
    spatial_size=window.size,
    patch_size=config.sampling.patch_size,
    temporal_end_years=config.temporal_domain.end_years,
    window_length=config.temporal_domain.window_length,
    bundle_offsets=config.temporal_domain.bundle.offsets
)

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    num_workers=config.hardware.num_workers,
    pin_memory=config.hardware.pin_memory
)
```

### Pattern 3: Multi-GPU Setup

```python
config = load_training_config('frl_training_v0.yaml')

if len(config.hardware.gpu_ids) > 1:
    # Multi-GPU
    model = torch.nn.DataParallel(
        model,
        device_ids=config.hardware.gpu_ids
    )
    print(f"Using {len(config.hardware.gpu_ids)} GPUs")
else:
    # Single GPU
    device = torch.device(f"cuda:{config.hardware.gpu_ids[0]}")
    model = model.to(device)
```

---

## Error Handling

### Common Errors

**1. Missing config files**
```python
try:
    config = load_training_config('missing.yaml')
except FileNotFoundError as e:
    print(f"Config file not found: {e}")
```

**2. Validation errors**
```python
try:
    parser = TrainingConfigParser('config.yaml')
    config = parser.parse()
    parser.validate()
except ValueError as e:
    print(f"Validation failed:\n{e}")
```

**3. Invalid settings**
```python
# Parser will catch these during validation:
# - Batch size < 1
# - Learning rate <= 0
# - Missing GPU IDs when device='cuda'
# - Bundle size mismatch
# - Missing config paths
```

---

## Testing

```python
# Test parsing
def test_parse():
    parser = TrainingConfigParser('frl_training_v0.yaml')
    config = parser.parse()
    
    assert config.version == "1.0"
    assert config.training.batch_size == 4
    assert config.optimizer.lr == 1e-4

# Test validation
def test_validate():
    parser = TrainingConfigParser('frl_training_v0.yaml')
    config = parser.parse()
    assert parser.validate() == True

# Test loss schedule
def test_loss_schedule():
    parser = TrainingConfigParser('frl_training_v0.yaml')
    config = parser.parse()
    
    # Epoch 0: triplet should be 0
    weights = config.losses.get_total_weight_at_epoch(0)
    assert weights['z_type_triplet'] == 0.0
    
    # Epoch 20: triplet should be 1.0
    weights = config.losses.get_total_weight_at_epoch(20)
    assert weights['z_type_triplet'] == 1.0
```

---

## Summary

The training config parser provides:

✅ **Clean API** - Dotted notation for all settings  
✅ **Type safety** - Dataclasses with proper types  
✅ **Validation** - Catches configuration errors early  
✅ **Loss scheduling** - Query weights at any epoch  
✅ **Summaries** - Quick overview of configuration  
✅ **Integration-ready** - Works with bindings and model configs  

Use it as the single source of truth for all training parameters!