# FoR-EST: Forest Estimation with Embedded State Trajectories

## Project Overview

**Note on naming:** This repo is called `vq-vae` but that name is vestigial from an early design. The actual model is **not** a VQ-VAE. It is a dual-pathway contrastive representation learner using InfoNCE and VICReg losses.

**Model status:** As of experiment `frl_v0_exp017`, the model is considered **complete and fit for purpose**. The addition of `phase_recovery_discrimination_loss` (`frl/losses/triplet_phase.py`) was the final missing piece: prior losses (`soft_neighborhood_phase`, phase VICReg) enforced only relative distance ordering in z_phase, which allowed the model to satisfy all training objectives while compressing all recovery stages into an arbitrarily small region of embedding space. The recovery discrimination loss adds an absolute margin constraint — within each pixel, disturbed-state embeddings (ysfc ≤ 1) must be at least `margin` apart from recovered-state embeddings (ysfc ≥ 5) — forcing the temporal dynamics to be metrically meaningful, not just rank-ordered.

### Scientific Goal

Learn a metric embedding space for 30×30m forest pixels where distance reflects similarity of forest status. The embedding is designed for use with USFS Forest Inventory and Analysis (FIA) data in:
- Post-stratification
- Small area analysis
- Identifying poorly sampled forest conditions

### Embedding Structure

The model produces two separate embeddings per pixel:

| Embedding | Shape | Semantics |
|-----------|-------|-----------|
| `z_type` | `[B, 64, H, W]` | **What kind of forest** — atemporal: structure, species, density. Trained with spectral + spatial contrastive losses. |
| `z_phase` | `[B, 12, T, H, W]` | **Temporal dynamics for that type** — per-timestep, conditioned on z_type via FiLM. Trained with temporal neighborhood/triplet losses. |

---

## Repository Structure

```
frl/                  # Representation learning package (main focus)
  models/             # Neural network architecture
  data/               # Data loading, normalization, sampling
  losses/             # Loss functions
  training/           # Training loop and evaluation
  config/             # YAML configuration files
  utils/              # Spatial utilities
  examples/           # Usage examples
  tests/              # Unit tests

scripts/              # Upstream preprocessing — extracts satellite data, builds Zarr
zarr_builder/         # Zarr archive construction
utils/                # General geospatial utilities
```

The `frl/` package is self-contained. Everything outside it is upstream preprocessing that you typically don't need to modify.

---

## Model Architecture

### Dual-Pathway Encoder (`frl/models/representation.py`)

```
TYPE PATHWAY (atemporal):
  Input: [B, C_type, H, W]
  → Conv2DEncoder               frl/models/conv2d_encoder.py
  → EdgeAwareSmoothingConv2D    frl/models/spatial.py
  → z_type: [B, 64, H, W]      (unconstrained magnitude; no L2 norm on output)

PHASE PATHWAY (temporal):
  Input: [B, C_phase, T, H, W]
  → TCNEncoder             frl/models/tcn.py  (dilated convolutions, multi-scale temporal RF)
  → 1×1 bottleneck Conv    → h  (pre-FiLM; type-agnostic trajectory prototype)
  → L2 normalize
  → FiLMLayer              frl/models/conditioning.py  (conditioned on z_type, STOP-GRADIENT)
  → z_phase: [B, 12, T, H, W]
```

### Key Design Decisions

- **SimCLR projection head on z_type** (`frl/models/heads.py` — `MLPProjectionHead`) — A small MLP (Linear → BN → ReLU → Linear, optionally L2-normalized) sits between `z_type` and the spectral/spatial InfoNCE losses during training. Following the SimCLR convention, this head absorbs distortions introduced by contrastive training so the backbone embedding stays clean for downstream tasks. It is discarded at inference time — `z_type` is used directly. VICReg operates on raw `z_type` (not projected) to guard against backbone collapse in the original embedding space. Enabled/disabled and sized via `type_projection` in `frl_repr_model_v1.yaml`; default output_dim=8 (small relative to z_type_dim=48, reflecting ~3–4 significant spectral PCs). `embed_locations.py` writes the projected embedding as `g_type_*` columns alongside `z_type_*`.
- **Stop-gradient on `z_type` before FiLM** — `z_type` is `.detach()`'d before being passed to `FiLMLayer`. This is intentional to prevent circular conditioning. Do not remove this.
- **EdgeAwareSmoothingConv2D** (`frl/models/spatial.py`) — replaces `GatedResidualConv2D` starting from exp018. Uses a fixed directional filter bank (K=8: 4 orientations × 2 scales — fine 3×3 and coarse dilated-3×3) with **rank-R factored per-channel mixing weights** predicted from per-channel Sobel gradients. Mixing weights are factored as W[k,c] = Σ_r A[k,r]·B[c,r]: A holds R shared direction-basis patterns (K-way softmax each), B holds per-channel mixture coefficients over those R patterns (R-way softmax per channel). This enforces cross-channel correlation while preserving per-channel flexibility — e.g. topographic channels can load on coarse-scale bases, spectral channels on fine-scale bases. A learned residual gate (`output = smoothed + gate·(x−smoothed)`) preserves features **across** edges and at corners. `GatedResidualConv2D` is retained in `spatial.py` for historical reference.
- **z_type is unconstrained in magnitude** — `F.normalize()` was removed from `forward_type` (festive-shannon sync). VICReg enforces std ≥ 1 per dimension and decorrelates dimensions, but does not constrain the mean. Downstream users should center z_type (e.g. via `StandardScaler`) before linear probes or clustering. Spatial InfoNCE operates directly on raw z_type (no projection head), so dot-product similarities are in the 10–40 range rather than [-1, 1]; the temperature of 0.07 is calibrated to this scale via the effective loss value, not the raw logit range.
- **L2 normalization before FiLM** — the pre-FiLM bottleneck h is L2-normalized before being passed to FiLM; FiLM gamma then owns the scaling. Per-channel batch demeaning of h was removed — it was ineffective because all phase losses operate on relative distances (pairwise or within-pixel), so any additive offset cancels.
- **Frobenius type-leakage penalty** — `||cov(h, z_type)||_F` is added to the training loss (weight 0.01, active only when phase curriculum is active). Stop-gradient on z_type so only the TCN receives gradient. This discourages the pre-FiLM bottleneck h from encoding forest-type information, keeping type separated from temporal dynamics. Configured via `phase_type_leakage_weight` in `frl_binding_v1.yaml` under `soft_neighborhood_phase`.
- **Sparse forward pass** — `forward_phase_at_locations()` runs the phase encoder only at sampled anchor pixel locations (not the full spatial grid) for training efficiency.
- **FiLM initialized near identity** — gamma≈1, beta≈0 at initialization for stable early training.
- **FiLM gamma amplification observed** — after training, FiLM gamma converges to ~3.5 (from init=1.0). The TCN produces ~78% temporal variance in pre-FiLM z; post-FiLM z_phase retains ~32%. EVT-stratified diagnostics (`phase_evt_diagnostics.py`) confirm gamma is type-conditional: plantation/pine types (e.g. EVT 9322, 7368) receive above-average gamma especially in channel 4 (NBR-sensitive); stable oak types receive below-average gamma. Channels 8–10 have near-zero temporal variance fraction (largely redundant with z_type); channel 11 is most temporally active. Pre-exp017 recovery curve analysis (`phase_recovery_curves.py`) showed the phase embedding mostly encoded pixel identity rather than recovery stage — post-disturbance NBR did not rise clearly with ysfc across most EVT types. This was the root failure that `phase_recovery_discrimination_loss` was designed to fix.

### Model Entry Points

```python
# Instantiate from config
model = RepresentationModel.from_config(cfg, type_in_channels, phase_in_channels)

# Load from checkpoint (v4 format)
model = RepresentationModel.from_checkpoint(path, device='cuda', freeze=False)
```

---

## Data Pipeline

```
Zarr archive (built by scripts/ and zarr_builder/)
  ↓
ForestDatasetV2          frl/data/loaders/dataset/forest_dataset_v2.py
  Checkerboard train/val/test split, loads raw patches from Zarr
  ↓
FeatureBuilder           frl/data/loaders/builders/feature_builder.py
  Loads precomputed stats (mean, std, quantiles, covariance)
  Applies normalization: zscore, robust_iqr, clamp, fixed range, identity
  Handles masks: global mask, channel-level masks, NaN
  ↓
DataBundle / TrainingBundle
  Standard [B, C, T, H, W] tensors + binary anchor_mask
```

### Anchor Sampling (`frl/data/sampling/anchor_sampling.py`)

Training runs on sampled anchor pixels (not full grids):
- `sample_anchors_grid()` — regular grid with jitter
- `sample_anchors_grid_plus_supplement()` — grid + random supplement (typical default)

### Dataset Bindings (`frl/data/loaders/config/dataset_bindings_parser.py`)

The bindings YAML defines dataset groups:
- `static_mask` — time-invariant masks
- `annual_mask` — per-year masks
- `annual` — annual time series features
- `irregular` — irregularly-sampled time series

---

## Loss Functions (`frl/losses/`)

| Loss | File | Purpose |
|------|------|---------|
| InfoNCE contrastive | `contrastive.py` | Metric learning via positive/negative pairs |
| Pair generation | `pairs.py` | kNN, mutual-kNN, quantile, radius, spatial-constrained |
| VICReg | `variance_covariance.py` | Collapse prevention: enforces variance + decorrelation |
| Phase neighborhood | `phase_neighborhood.py` | Temporal consistency of z_phase |
| Phase triplet | `phase_triplet.py` | Temporal ordering constraints (defined but not wired into training loop) |
| Soft neighborhood | `soft_neighborhood.py` | Soft KL matching of relative z_phase distance structure at shared ysfc |
| Phase recovery discrimination | `triplet_phase.py` | **Absolute** margin between disturbed (ysfc≤1) and recovered (ysfc≥5) embeddings within each pixel — the loss that makes recovery stage metrically separable |
| Frobenius leakage penalty | (inline in `train_representation.py`) | `\|\|cov(h, z_type)\|\|_F` — discourages type information in the pre-FiLM bottleneck h |
| Reconstruction | `reconstruction.py` | Optional L1/L2/Huber reconstruction |

**Pair construction:**
- Spectral positive pairs: cross-batch mutual kNN in whitened feature space (not within-patch)
- Spectral negative pairs: cross-batch random sampling, scaled to `spectral_neg_per_anchor × N_total` pixels (default: 20 per anchor)
- Spatial positive pairs: within-patch spatial kNN
- Spatial negative pairs: beyond distance threshold, weighted by spectral dissimilarity

---

## Training (`frl/training/train_representation.py`)

The training loop applies the following loss components:

1. **Spectral InfoNCE** — contrastive loss on `z_type` using cross-batch mutual kNN positives and cross-batch random negatives (scaled by `spectral_neg_per_anchor`, default 20)
2. **Spatial InfoNCE** — contrastive loss on `z_type` using spatial kNN pairs
3. **VICReg** — variance + covariance regularization on `z_type`
4. **Soft neighborhood phase** — KL-divergence matching of relative z_phase distance structure at shared ysfc values between pixel pairs
5. **Phase spread ranking** — pixels with higher inter-annual spectral variance must have more spread-out z_phase trajectories
6. **Phase VICReg** — collapse prevention on z_phase dimensions (note: operates on the wrong population for recovery; see Known Limitations)
7. **Phase recovery discrimination** — absolute margin loss: within each pixel, embeddings at ysfc ≤ 1 must be at least `margin` apart from embeddings at ysfc ≥ 5. This is the loss that closes the gap between relative ordering and metrically meaningful recovery stage representation.
8. **Frobenius type-leakage penalty** — `||cov(h, z_type)||_F` penalises type information in the pre-FiLM bottleneck h. Stop-gradient on z_type; active only when phase curriculum weight > 0.

### Important: Phase Loss Curriculum

The phase loss uses **curriculum learning** — it is **zero for the first N epochs** (warmup), then ramps up. This is intentional. If you see phase loss = 0 early in training, that is expected behavior, not a bug. The warmup epoch count is set in the training config.

### Optimizer Setup

- AdamW (lr=1e-4, weight_decay=0.01)
- Cosine annealing with warmup (10 epochs)
- Mixed precision: bfloat16
- Gradient clipping

---

## Configuration (`frl/config/`)

Three YAML files control everything:

| File | Controls | Change when... |
|------|----------|----------------|
| `frl_repr_model_v1.yaml` | Architecture: encoder channels `[128→64]`, TCN dilations `[1,2,4]`, z_type_dim=64, z_phase_dim=12, dropout schedule | Changing model capacity or structure |
| `frl_binding_v1.yaml` | Zarr path, time window (2010-2024), dataset groups, channel definitions, formulas, thresholding, normalization presets | Adding/removing input features or data sources |
| `frl_training_v1.yaml` | Optimizer, scheduler, loss weights, batch size=12, epochs=200, checkpointing, validation | Tuning training hyperparameters |

The training config references the bindings config internally — you typically only need to pass `--training` on the CLI.

---

## Workflow

### Prerequisites

Statistics must be computed before training (for normalization):
```bash
python frl/examples/data/example_compute_stats.py
```

### Train the Representation Model

```bash
python frl/training/train_representation.py \
    --training frl/config/frl_training_v1.yaml
```

If the experiment directory already exists, training **auto-resumes** from `encoder_last.pt` (restores model, optimizer, and scheduler; appends to the existing log). To start fresh, use `--overwrite` (deletes the directory). To prevent auto-resume without overwriting, use `--no-resume`.

### Evaluate with Linear Probe

```bash
python frl/training/fit_linear_probe.py \
    --checkpoint runs/checkpoints/model.pt
```

### Important: Encoder Feature Name

All inference and evaluation scripts must read the encoder feature name from the
training config rather than hardcoding it:

```python
enc_feature_name = training_config.model_input.type_encoder_feature
# e.g. "type_encoder_input"  (34 channels)
```

The old name `"ccdc_history"` (22 channels) is stale and will cause a channel mismatch
error. All scripts in `frl/training/` follow this pattern.

### Upstream Preprocessing (rarely needed)

```bash
# Build Zarr archive from satellite data (see scripts/ and zarr_builder/)
python -m scripts.build_zarr --config scripts/config.yaml
```

---

## Key File Index

```
frl/models/representation.py              Main model: RepresentationModel
frl/models/conv2d_encoder.py              Type pathway: 2D conv encoder
frl/models/tcn.py                         Phase pathway: TCN encoder
frl/models/spatial.py                     GatedResidualConv2D
frl/models/conditioning.py                FiLMLayer
frl/models/heads.py                       Prediction heads (MLP, Linear, Conv2D)

frl/data/loaders/dataset/forest_dataset_v2.py       PyTorch Dataset (Zarr → samples)
frl/data/loaders/builders/feature_builder.py        Normalization + masking
frl/data/loaders/config/dataset_bindings_parser.py  YAML bindings parser
frl/data/loaders/config/training_config_parser.py   Training YAML parser
frl/data/sampling/anchor_sampling.py                Anchor pixel sampling
frl/utils/spatial.py                                Spatial distance + kNN utilities
frl/utils/sampling.py                               ReservoirSampler (Algorithm R streaming sampler)

frl/losses/contrastive.py                InfoNCE loss
frl/losses/pairs.py                      Pair generation strategies
frl/losses/variance_covariance.py        VICReg loss
frl/losses/phase_neighborhood.py         Phase temporal loss
frl/losses/phase_triplet.py              Phase triplet loss
frl/losses/reconstruction.py             Reconstruction loss

frl/training/train_representation.py          Main training script
frl/training/fit_linear_probe.py              Downstream type embedding linear probe (z_type → FIA targets)
frl/training/fit_phase_linear_probe.py        Phase embedding linear probe (temporal R²)
frl/training/fit_gmm_clusters.py              Fit GMM on z_type embeddings
frl/training/compare_gmm_evt.py               Compare GMM clusters vs EVT forest types
frl/training/visualize_test_patches.py        Visualize model output on test patches
frl/training/visualize_forest_diagnostics.py  Forest-wide embedding diagnostics
frl/training/phase_evt_diagnostics.py         EVT-stratified FiLM gamma + z_phase temporal variance
frl/training/phase_recovery_curves.py         Per-EVT NBR recovery curves vs. ysfc (requires probe)

frl/config/frl_repr_model_v1.yaml        Architecture config
frl/config/frl_binding_v1.yaml           Dataset bindings config
frl/config/frl_training_v1.yaml          Training hyperparameter config
```

---

## Extending the Model

The codebase is flexible with no rigid extension conventions.

**Add a new loss function:**
1. Create in `frl/losses/`
2. Import and call in `frl/training/train_representation.py`
3. Add loss weight to `frl_training_v1.yaml`

**Add a new encoder component:**
- Type pathway tensors: `[B, C, H, W]`
- Phase pathway tensors: `[B, C, T, H, W]`
- Match these shapes when adding new modules

**Add a new input data source:**
1. Define channels in `frl_binding_v1.yaml` under the appropriate group (`static`, `annual`, `irregular`)
2. Compute normalization statistics
3. Add normalization preset

**New downstream task:**
```python
model = RepresentationModel.from_checkpoint(path, freeze=True)
head = MLPHead(in_dim=64, out_dim=n_classes)  # frl/models/heads.py
```

---

## Known Issues / Gotchas

**Invalid values in boundary/masked patches.**
```
frl/data/loaders/builders/feature_builder.py:559: RuntimeWarning: invalid value encountered in matmul
  transformed_flat = whitening_matrix @ centered_flat
```
Patches that touch the domain boundary or contain heavily masked pixels can have NaN or Inf values that survive into the whitening transform. These are zeroed by `nan_to_num` before the matmul and then clamped to ±5 — so they are harmless to training. If you see this warning it means more boundary patches than usual are reaching that code path (e.g. after increasing batch size). The downstream NaN loss check in `process_batch` will skip any sample whose loss goes non-finite.

---

## Known Limitations / Future Work

~~**TODO: Weight cross-patch negatives by spectral distance.** Currently cross-patch negatives are unweighted (uniform), which accepts false negatives — spectrally similar forests from different patches that get incorrectly pushed apart. A principled fix: compute spectral distances between cross-patch pairs and apply `neg_weights = 1 - exp(-d_spec / tau)`, consistent with how spatial InfoNCE negatives are already weighted (`frl/training/train_representation.py`, spatial weighting block). This requires computing spectral distances for sampled cross-patch pairs only (not the full O(N²B²) matrix).~~ *(implemented)*

**TODO: Improve encoding of temporal variance and variance-like measures (variance_ndvi, spectral_velocity).** These targets have weak linear probe R² (~0.25–0.63). The cause is not clearly an architecture issue — it may be a loss issue: after whitening, these features contribute only ~1/22 of the InfoNCE pair-selection signal, so gradient pressure is weak. Alternatively, low R² may be appropriate for stable forest types and only problematic for dynamic types. Options to investigate:
- Upweight variance-like features in the spectral distance computation
- Add an auxiliary reconstruction loss targeting these specific channels
- ~~Stratify probe diagnostics by EVT forest type before concluding the signal is missing~~ *(done — EVT stratification shows weak phase signal is broadly true across types, not just a stable-forest artifact)*

~~**TODO: Compute EVT-forest-type-stratified diagnostics for phase signal strength.**~~ *(implemented — see `phase_evt_diagnostics.py` and `phase_recovery_curves.py`; key findings recorded in the FiLM gamma bullet above)*

~~**TODO: Make z_phase encode recovery stage, not just pixel identity.** The `soft_neighborhood_phase` loss enforced only relative distance ordering (KL-softmax), which is equivariant to uniform scaling of the embedding space. A model could satisfy it perfectly while compressing all recovery stages into an arbitrarily small region. Phase VICReg did not fix this because it operated on the wrong population (N_phase×T flattened timesteps, dominated by within-pixel temporal variation rather than across-pixel recovery-stage variation). The fix — `phase_recovery_discrimination_loss` in `frl/losses/triplet_phase.py` — adds an absolute margin constraint directly between disturbed and recovered timesteps within each pixel.~~ *(implemented in exp017; model considered complete)*
