# Training

This document describes the representation learning training pipeline, the
`RepresentationModel` architecture, checkpoint format, and the linear probe
evaluation script.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  train_representation.py                     │
│                                                             │
│  1. Build datasets (train / val)                            │
│  2. Construct RepresentationModel                           │
│  3. Train with spectral + spatial InfoNCE losses            │
│  4. Save checkpoints and experiment artifacts               │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │  checkpoint (.pt)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  fit_linear_probe.py                         │
│                                                             │
│  1. Load frozen RepresentationModel from checkpoint         │
│  2. Fit closed-form ridge regression probe                  │
│  3. Evaluate MSE / R² per target metric                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## RepresentationModel

The model is defined in `models/representation.py` and provides the full
encoder pipeline as a single `nn.Module`:

```
Input [B, 16, H, W]  (ccdc_history)
        │
        ▼
  Conv2DEncoder
    conv1×1 → GroupNorm → ReLU → Dropout  (16 → 128)
    conv1×1 → GroupNorm → ReLU → Dropout  (128 → 64)
        │
        ▼
  GatedResidualConv2D
    spatial conv layers (3×3, 2 layers)
    gating network (per-pixel, per-channel)
        │
        ▼
Output [B, 64, H, W]  (embedding z)
```

### Version tracking

`RepresentationModel.VERSION` is a string (currently `"1"`) that is saved in
every checkpoint. When loading, `from_checkpoint()` checks the version and
raises a clear error on mismatch rather than producing a cryptic shape error.

Bump `VERSION` whenever the forward pipeline changes in a
checkpoint-incompatible way.

### Usage

```python
from models import RepresentationModel

# Training
model = RepresentationModel().to(device)
z = model(x)                       # [B, 64, H, W]
z, gate = model(x, return_gate=True)  # also get gate values

# Inference from checkpoint
model = RepresentationModel.from_checkpoint("path/to/ckpt.pt", device="cuda")
```

### Source file archival

`RepresentationModel.source_file()` returns the path to `representation.py`,
which the training script copies into the experiment directory so you can
inspect exactly which architecture produced a given set of checkpoints.

## Training script

```
python training/train_representation.py \
    --bindings config/frl_binding_v1.yaml \
    --training config/frl_training_v1.yaml
```

### Command-line arguments

| Argument           | Default                          | Description                                  |
|--------------------|----------------------------------|----------------------------------------------|
| `--bindings`       | `config/frl_binding_v1.yaml`     | Path to dataset bindings config              |
| `--training`       | `config/frl_training_v1.yaml`    | Path to training config                      |
| `--epochs`         | from config                      | Number of training epochs (override)         |
| `--batch-size`     | from config                      | Batch size (override)                        |
| `--lr`             | from config                      | Learning rate (override)                     |
| `--device`         | from config                      | Device, e.g. `cuda:0` or `cpu` (override)    |
| `--checkpoint-dir` | `{experiment_dir}/{ckpt_dir}`    | Checkpoint save directory (override)         |
| `--overwrite`      | `False`                          | Remove existing experiment dir before start  |

### Experiment directory structure

```
{run_root}/{experiment_name}/
├── {ckpt_dir}/
│   ├── encoder_epoch_001.pt
│   ├── encoder_epoch_002.pt
│   └── ...
├── {log_dir}/
│   └── training.log
├── frl_binding_v1.yaml        # copied config
├── frl_training_v1.yaml       # copied config
└── representation.py          # archived model source
```

The `--overwrite` flag is required to reuse an existing experiment directory.
Without it, the script exits with an error to prevent accidental data loss.

### Checkpoint format

Each `.pt` file contains:

| Key                    | Type              | Description                          |
|------------------------|-------------------|--------------------------------------|
| `model_version`        | `str`             | `RepresentationModel.VERSION`        |
| `model_state_dict`     | `dict`            | `model.state_dict()`                 |
| `optimizer_state_dict` | `dict`            | Optimizer state                      |
| `scheduler_state_dict` | `dict`            | LR scheduler state                   |
| `epoch`                | `int`             | 1-indexed epoch number               |
| `train_loss`           | `float`           | Mean training loss for the epoch     |
| `train_spectral_loss`  | `float`           | Spectral component of training loss  |
| `train_spatial_loss`   | `float`           | Spatial component of training loss   |
| `val_loss`             | `float`           | Mean validation loss for the epoch   |
| `val_spectral_loss`    | `float`           | Spectral component of validation loss|
| `val_spatial_loss`     | `float`           | Spatial component of validation loss |

### Loss function

Training uses two InfoNCE contrastive losses combined additively:

1. **Spectral loss** — pairs selected by spectral (Mahalanobis) distance with
   a spatial separation constraint. Encourages spectrally similar pixels to
   have similar embeddings regardless of location.

2. **Spatial loss** — pairs selected by spatial proximity (KNN for positives,
   random sampling at distance for negatives), weighted by spectral similarity.
   Encourages nearby pixels to have similar embeddings, with the weighting
   reducing the influence of pairs that cross land-cover boundaries.

Both losses share the same embedding space and temperature parameter.

## Linear probe

```
python training/fit_linear_probe.py \
    --checkpoint path/to/encoder_epoch_050.pt \
    --bindings config/frl_binding_v1.yaml \
    --training config/frl_training_v1.yaml
```

### Command-line arguments

| Argument              | Default                          | Description                               |
|-----------------------|----------------------------------|-------------------------------------------|
| `--checkpoint`        | (required)                       | Path to encoder checkpoint                |
| `--bindings`          | `config/frl_binding_v1.yaml`     | Bindings YAML                             |
| `--training`          | `config/frl_training_v1.yaml`    | Training YAML                             |
| `--batch-size`        | from config                      | Batch size override                       |
| `--device`            | from config                      | Device override                           |
| `--ridge-lambda`      | `1e-3`                           | Ridge regression penalty (weights only)   |
| `--max-batches-train` | `0` (all)                        | Cap batches for fitting                   |
| `--max-batches-eval`  | `0` (all)                        | Cap batches for evaluation                |
| `--output`            | `{ckpt_dir}/linear_probe_closed_form.pt` | Output path for probe weights  |

### How it works

The probe evaluates representation quality by fitting a linear map from the
frozen 64-dim embeddings to target vegetation metrics:

1. Load `RepresentationModel` from checkpoint (frozen, eval mode).
2. Stream training data through the model, accumulating the normal equations
   `A = X'X` and `B = X'Y` for ridge regression in float64.
3. Solve `(A + λI)W = B` in closed form — no SGD, no learning rate tuning.
4. Evaluate per-metric MSE and R² on both train and validation splits.

Target metrics: `mean_ndvi`, `mean_ndmi`, `mean_nbr`, `mean_seasonal_amp_nir`,
`variance_ndvi`.

### Adding a new model version

When you change the architecture:

1. Update `RepresentationModel.__init__` and `forward` in
   `models/representation.py`.
2. Bump `RepresentationModel.VERSION` (e.g. `"1"` → `"2"`).
3. Old checkpoints will raise a version mismatch error on load, preventing
   silent misuse.
4. The archived `representation.py` in each experiment directory records
   exactly which version produced those checkpoints.
