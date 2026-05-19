# FoR-EST: Methods Outline for Remote Sensing of Environment

*Prepared from codebase analysis — exp017 checkpoint configuration.*
*File references are relative to the repository root.*

---

## 1. Scientific Problem and Motivation

Forest monitoring at national scale requires characterizing not only what *type* of forest occupies a given pixel, but what *stage* of disturbance or recovery it currently represents. The USFS Forest Inventory and Analysis (FIA) program samples the forested land base using a probabilistic design; plot-level measurements are extended to area estimates via post-stratification, where satellite-derived strata define groups of similar forest pixels. The precision of those estimates depends directly on how well the strata capture within-stratum homogeneity in the quantities of interest.

Existing approaches to satellite-based forest stratification either capture atemporal spectral composition (forest type) without encoding recovery trajectory, or encode trajectory through simple spectral indices that conflate recovery stage with forest type. Pixels at different recovery stages can be spectrally similar in raw reflectance space, and pixels at the same recovery stage can differ substantially by type — these two signals are entangled in the input feature space.

FoR-EST is designed to disentangle them by learning two separate metric embedding spaces: `z_type`, encoding what kind of forest a pixel is (atemporal), and `z_phase`, encoding where in its disturbance-recovery trajectory that forest currently sits (temporal, conditioned on type). The motivating applications are FIA post-stratification and small-area estimation, where both dimensions of similarity matter for variance reduction.

A specific root failure in earlier models was that phase losses based solely on relative distance ordering (soft KL matching) could be satisfied by a model that compresses all recovery stages into an arbitrarily small region of embedding space — satisfying the ordering while making the space metrically meaningless. The final loss (`phase_recovery_discrimination_loss`, `frl/losses/triplet_phase.py`) was designed to close this gap with an absolute margin constraint.

---

## 2. Input Data

### 2.1 Spatial Resolution and Geographic Scope

- **Spatial resolution**: 30×30m (Landsat-native; implied by CCDC and LCMS sources; not explicitly stated in YAML configs — requires author confirmation).
- **Geographic scope**: Virginia (VA), USA. Full training domain: 13,056 × 23,552 pixels (`frl/config/frl_training_v1.yaml`, `spatial_domain.full_domain`). Zarr archive: `/data/VA/zarr/va_vae_dataset.zarr` (`frl/config/frl_binding_v1.yaml`).
- **Time window**: 2010–2024, 15 annual timesteps (`frl_binding_v1.yaml`, `time_window: {start: 2010, end: 2024}`).

### 2.2 Type Encoder Inputs (`type_encoder_input`, 34 channels, static `[C, H, W]`)

Defined in `frl/config/frl_binding_v1.yaml` under `features.type_encoder_input`.

**Topographic (8 channels)** — from `static.topo`:
- `elevation`, `slope`, `northness`, `eastness` — DEM-derived; z-score normalized, masked by `dem_mask`
- `hand` (Height Above Nearest Drainage), `twi` (Topographic Wetness Index) — masked by respective quality masks
- `tpi` (Topographic Position Index, 500m radius) — masked by `tpi_mask`
- `ksat` (log₁₀ saturated hydraulic conductivity) — masked by `ksat_mask`

**CCDC History (26 channels)** — from `static.ccdc_metrics_history`, representing time-collapsed summaries of Landsat CCDC trajectories over the full historical record:
- Spectral means: `mean_green`, `mean_red`, `mean_nir`, `mean_swir1`, `mean_swir2`, `mean_ndvi`, `mean_nbr`, `mean_ndmi`
- Seasonal amplitude means: `mean_seasonal_amp_red`, `mean_seasonal_amp_nir`, `mean_seasonal_amp_swir1`, `mean_seasonal_amp_swir2`
- Variability: `variance_seasonal_amp_red`, `variance_seasonal_amp_nir`, `variance_ndvi`, `max_seasonal_amp_nir`, `max_seasonal_amp_swir1`, `variance_seasonal_amp_swir1`, `variance_seasonal_amp_swir2`
- Change metrics: `spectral_distance_per_decade`, `mean_decline_rate`, `net_change_ndvi`, `num_segments`, `mean_segment_duration`, `mean_break_magnitude`, `rapid_loss_mean_magnitude`

All 26 CCDC channels: robust IQR normalization (median/IQR, clamped ±8). Variance-like and change-rate channels additionally receive a log transform before normalization (`frl_binding_v1.yaml`, `infonce_type_spectral` feature definition with `transform: {name: log}`).

### 2.3 Phase Encoder Inputs (`phase_ccdc`, 13 channels, temporal `[C, T, H, W]`)

Defined in `frl/config/frl_binding_v1.yaml` under `features.phase_ccdc`. All annual, 2010–2024.

- `temporal_position`: fractional year index $t / (T-1)$ in [0, 1]; no normalization (`identity`)
- CCDC annual model evaluated at each year: `red`, `nir`, `swir1`, `swir2`, `nbr`, `ndmi`, `ndvi`, `spectral_velocity` (log + z-score), `seas_amp_red`, `seas_amp_nir`, `seas_amp_swir1`, `seas_amp_swir2` — all z-score normalized, clamped ±6σ

### 2.4 Auxiliary Label for Phase Loss Construction

- `ysfc` (years since fast change, LCMS `lcms_ysfc_value_1985_2024`): per-pixel annual indicator of time elapsed since the most recent disturbance. `ysfc=0` indicates a disturbance year; higher values indicate recovery stage. Loaded as-is (`identity` normalization). Used only for pair selection and loss construction — not as encoder input.

### 2.5 Normalization and Masking

Normalization policies (`frl_binding_v1.yaml`, `normalization.presets`):

| Preset | Method | Clamp |
|---|---|---|
| `zscore` | (x − μ) / σ, from precomputed Zarr stats | ±6 |
| `robust_iqr` | (x − median) / IQR | ±8 |
| `prob01` | clamp only | [0, 1] |
| `identity` | none | — |
| `minmax_0_40` | linear rescale [0, 40] → [0, 1] | [0, 1] |

Statistics are computed over 500 randomly sampled 256×256 patches (500,000 pixel reservoir per channel) from the masked forested area.

Global pixel mask: `static_mask.aoi` AND `static_mask.forest`. The forest mask is derived from LCMS land-use probability for 2024 (`lcms_lu_p_forest ≥ 0.25`). Channel-level quality masks are applied per feature group (e.g., `dem_mask`, `twi_mask`). Channels with fill values (−9999) receive NaN treatment in the feature builder.

---

## 3. Model Architecture

### 3.1 Dual-Pathway Design

The `RepresentationModel` (`frl/models/representation.py`) produces two embeddings:

| Embedding | Shape | Semantics |
|---|---|---|
| `z_type` | `[B, 64, H, W]` | Atemporal forest type — structure, species, density. Trained with spectral + spatial contrastive losses. |
| `z_phase` | `[N_phase, T, 12]` (training) or `[B, 12, T, H, W]` (inference) | Per-timestep dynamics conditioned on forest type. |

### 3.2 Type Pathway

**`Conv2DEncoder`** (`frl/models/conv2d_encoder.py`):

Input: `[B, 34, H, W]`

Architecture: `[input_dropout] → (Conv2d → GroupNorm → ReLU → Dropout2d) × 2`

| Layer | Op | Output channels | Kernel | GroupNorm groups | Dropout |
|---|---|---|---|---|---|
| Input dropout | `Dropout2d` | 34 | — | — | 0→10% over 20 epochs (linear schedule) |
| 1 | Conv2d | 128 | 1×1, padding=0 | 8 | 0.1 |
| 2 | Conv2d | 64 | 1×1, padding=0 | 8 | 0.1 |

Output: `[B, 64, H, W]`. Note: kernel_size=1 means the type encoder is a pixelwise MLP — no spatial aggregation at this stage.

**`GatedResidualConv2D`** (`frl/models/spatial.py`):

Input: `[B, 64, H, W]` from `Conv2DEncoder`.

Architecture (current "NEW" formulation, lines 125–128 of `spatial.py`):
```
smoothed = conv_layers(x)          # 2× Conv2d(64,64,k=3,pad=1) with GroupNorm + ReLU between layers
residual  = x - smoothed           # high-frequency content (edges, sharp boundaries)
gate      = gate_network(residual) # gate_network: Conv2d(64,64,1) → ReLU → Conv2d(64,64,1) → Sigmoid
output    = x + gate * residual    # add gated high-frequency content back to input
```

This is adaptive sharpening: `output = x + gate × (x − smoothed)`. When `gate = 0`, output = x (pass-through). When `gate = 1`, output = `2x − smoothed` (full high-frequency amplification). The gate is driven by the residual itself, so regions with significant high-frequency content (edges, boundaries) learn large gate values. The module preserves spatial gradients while the 1×1 type encoder has no spatial receptive field.

Output: `z_type [B, 64, H, W]`.

> **Note**: The module docstring and comments describe the OLD formulation (`gate * smoothed + (1-gate) * x`), which is the inverse. Only the NEW code at lines 125–128 is executed. The paper should describe the actual behavior.

### 3.3 Phase Pathway

**`TCNEncoder`** (`frl/models/tcn.py`):

During training, the TCN operates on anchor pixel time-series only (`forward_phase_at_locations`): input `[N_phase, 13, T]`.

Three `GatedResidualBlock` layers:

| Block | in_ch | out_ch | dilation | Temporal RF |
|---|---|---|---|---|
| 1 | 13 | 64 | 1 | 3 |
| 2 | 64 | 64 | 2 | 3 + 4 = 7 (cumulative) |
| 3 | 64 | 64 | 4 | 7 + 8 = 15 (cumulative) |

Kernel size=3, non-causal padding = `(kernel_size − 1) × dilation // 2`. Total temporal receptive field with these dilations is at most 15 timesteps (the full time window when T=15).

Per-block: `Dropout1d(0.1, preconv) → Conv1d → GroupNorm(8) → gate = sigmoid(Conv1d_1×1) → ReLU → gate × out + (1−gate) × residual`. Channel-changing layers include a 1×1 residual projection.

Pooling: `none`. Output: `[N_phase, 64, T]`.

**Bottleneck** (`phase_head`, a `Conv2d(64, 12, 1)`):
- Reshape: `[N_phase, 64, T] → [N_phase×T, 64, 1, 1] → Conv2d → [N_phase×T, 12, 1, 1] → [N_phase, 12, T]`
- Projects TCN features to `z_phase_dim=12`.

**L2 normalization**:
```python
h = F.normalize(h.flatten(1, 2), dim=1).unflatten(1, (12, T))
```
Flattens the `[N, 12, T]` tensor to `[N, 12×T]`, normalizes each pixel's full spatiotemporal embedding to unit L2 norm, then restores the `[N, 12, T]` shape. The normalization is joint over channels and time — one scalar norm per pixel. This means the TCN controls direction in `(channel × time)` space while FiLM gamma controls scale.

**`FiLMLayer`** (`frl/models/conditioning.py`):

Generates per-pixel affine parameters from `z_type`:
```
gamma_network: Conv2d(64, 32, 1) → ReLU → Conv2d(32, 12, 1)
beta_network:  Conv2d(64, 32, 1) → ReLU → Conv2d(32, 12, 1)
```
Initialization: `gamma[-1].weight ~ N(0, 0.01)`, `gamma[-1].bias = 1`; `beta[-1].weight ~ N(0, 0.01)`, `beta[-1].bias = 0`. This gives near-identity behavior at initialization (gamma ≈ 1, beta ≈ 0).

FiLM application:
```python
z_phase = gamma.unsqueeze(2) * h + beta.unsqueeze(2)  # [N, 12, T]
z_phase = z_phase.permute(0, 2, 1)                    # [N, T, 12]
```

The conditioning input is `z_type.detach()` — z_type is stop-gradiented before being passed to FiLM. See Section 7.

### 3.4 Embedding Dimensions Summary

| Symbol | Shape (training) | Description |
|---|---|---|
| `z_type` | `[B, 64, H, W]` | Type embedding (dense spatial grid) |
| `z_anchors` | `[N_anchors, 64]` | z_type extracted at anchor pixel locations |
| `z_phase_at_anchors` | `[N_phase, T, 12]` | Phase embedding at sampled anchors |

---

## 4. Loss Functions

All losses are computed in `process_batch()` in `frl/training/train_representation.py`.

### 4.1 Spectral InfoNCE (cross-batch)

**Purpose**: Learn location-invariant forest type embedding — spectrally similar forests from different patches should be close in z_type.

**Pair distance space**: `infonce_type_spectral` feature — 22 CCDC-history channels with robust_iqr normalization and log transforms on variance/change-rate channels. A precomputed per-patch covariance is used to compute Mahalanobis (whitened L2) distances.

**Positive pairs** (chunked mutual kNN, `pairs_mutual_knn_chunked`, `frl/losses/pairs.py`):
- All anchor embeddings from all patches in a batch are pooled into `[N_total, C]`.
- For each query point (chunk_size=128 at a time), compute L2 distances to all N_total targets in the whitened spectral space.
- Within-patch pairs with spatial distance < 4 pixels are masked to infinity (excluded from kNN).
- Cross-patch pairs carry no spatial constraint (always spatially distant).
- Mutual kNN with k=16: pair (i, j) is a positive only if j ∈ kNN(i) AND i ∈ kNN(j).
- Both (i, j) and (j, i) are returned.

**Negative pairs** (random cross-patch sampling):
- For each ordered patch pair (p_i, p_j) with p_i ≠ p_j: sample `n_per = max(1, n_neg // n_patch_pairs)` random index pairs.
- Total target negatives: `N_total × spectral_neg_per_anchor` (default: 20 per anchor).
- **False-negative suppression**: `neg_weight = (1 − exp(−d_spec / τ)).clamp(min=0.05)`, τ=1.0. Pairs that are spectrally similar get low weight, preventing the model from being penalized for pushing together truly similar cross-patch forests.

**Loss**:
```
L_spectral = contrastive_loss(z_all, pos_pairs, neg_pairs, neg_weights=neg_weights,
                               temperature=0.07, similarity='l2')
```
where similarity='l2' computes `−||a − b||² / D`.

### 4.2 Spatial InfoNCE (within-patch)

**Purpose**: Learn spatial smoothness — nearby pixels of the same forest type should be close in z_type.

**Positive pairs**: spatial kNN (k=4, max_radius=8 pixels).
- Weights: `pos_weight = exp(−d_spec / τ_s).clamp(min=0.05)`, τ_s=1.0 (upweights spatially close but also spectrally similar pairs; downweights inadvertent edge pairs).

**Negative pairs**: spatial range (min_dist=16, max_dist=∞, n_per_anchor=4).
- Weights: `neg_weight = (1 − exp(−d_spec / τ_s)).clamp(min=0.05)` (downweights false negatives that are spectrally similar but spatially far).

**Loss**: `contrastive_loss(..., temperature=0.07, similarity='l2')`.

### 4.3 VICReg on z_type

**Purpose**: Prevent dimensional collapse in z_type across anchor pixels.

Applied to `z_anchors [N_anchors, 64]`:
- **Variance** (hinge): `relu(1.0 − std_d).mean()` over all 64 dimensions. Penalizes dimensions with std < 1.
- **Covariance**: `sum(off-diagonal cov²) / 64`. Penalizes pairwise correlations.
- Combined: `var_weight=1.0 × var_loss + cov_weight=1.0 × cov_loss`, scaled by `vcr_weight=0.1`.

### 4.4 Soft Neighborhood Phase Loss

**File**: `frl/losses/phase_neighborhood.py`, `frl/losses/soft_neighborhood.py`  
**Curriculum**: zero for epochs 0–29; linear ramp epochs 30–39; full weight from epoch 40.

**Purpose**: Teach z_phase to reproduce the temporal distance structure of the spectral input — for each pixel pair, the relative distances between recovery stages in the phase embedding should match the relative distances in the spectral feature space at the same recovery stages.

**What it enforces**: Relative ordering of recovery stages — pixels at similar ysfc values should be closer in z_phase than pixels at dissimilar ysfc values.

**What failure mode it addresses**: Without absolute constraints, a model can satisfy this loss by mapping all recovery stages to the same point (degenerate solution). The distances are then all zero, and the softmax distribution is uniform — which can match a near-uniform reference distribution without actually encoding recovery.

**Pair construction** (`build_phase_pairs`, `frl/losses/phase_pairs.py`):
1. Stage 1 — kNN in whitened spectral space (k=16): find k nearest spectral neighbors per anchor.
2. Stage 2 — ysfc overlap filter: keep only pairs sharing ≥ 3 unique ysfc values across their time series.
3. Drop anchors with fewer than 5 surviving cross-pixel pairs.
4. Self-pairs (i, i) are always added for surviving anchors.
5. Pair weight: `w_ij = exp(−||spec_i − spec_j|| / 5.0)` for cross pairs; 1.0 for self-pairs.

**ysfc alignment** (`build_phase_neighborhood_batch`):
- For each shared ysfc value, select one representative timestep: the timestep from the longest recovery sequence containing that value (tie-break: most recent). This handles pixels with multiple disturbances by preferring the timestep embedded in the longest monotone recovery.
- Reference features: the phase encoder inputs (`phase_ccdc`, same 13-channel temporal features) centered over shared ysfc values. Per-pair demeaning over shared ysfc positions before computing distances removes the static baseline spectral signature, so distances reflect temporal trajectory shape only.

**Loss formulation** (two terms, both via `soft_neighborhood_matching_loss`):

For each valid pair (i, j) with K shared ysfc values, let M = K (padded dimension):

Define masked softmax distributions over shared ysfc positions:
$$p_{tt'} = \frac{\mathbf{1}[\text{mask}(t,t')] \cdot \exp(-d_{\text{ref}}(t,t') / \tau_{\text{ref}})}{\sum_{t''} \mathbf{1}[\text{mask}(t,t'')] \cdot \exp(-d_{\text{ref}}(t,t'') / \tau_{\text{ref}})}$$

- **Self-similarity term**: `d_ref` = spectral self-distance at pixel j; `d_learned` = embedding self-distance at pixel i. Mask excludes diagonal. Teaches trajectory shape: pixel i's embedding distances across time should match pixel j's spectral distances across the same ysfc values.
- **Cross-pixel term**: `d_ref` = spectral distance between pixels i and j; `d_learned` = embedding distance between i and j. Diagonal included. Anchors both pixels in a shared metric space.

Loss per pair: $\mathcal{L}_b = \sum_t \mathbf{1}[\text{row valid}] \cdot D_{\text{KL}}(p_{t,\cdot} \| q_{t,\cdot})$

Final loss: weighted mean across pairs (`pair_weights` from spectral similarity, min_valid_per_row=2). τ_ref=0.1, τ_learned=0.1.

Combined: `self_similarity_weight=1.0 × loss_self + cross_pixel_weight=1.0 × loss_cross`, scaled by `weight=1.0 × curriculum_w`.

### 4.5 Phase Spread Ranking Loss

**File**: `compute_phase_spread_ranking`, `frl/losses/phase_neighborhood.py`  
**Curriculum**: start_epoch=30, ramp_epochs=10 (same schedule as soft neighborhood phase).  
**Weight**: 0.5.

**Purpose**: Pixels with higher inter-annual spectral dynamism must have more spread-out phase trajectories across shared ysfc stages. Addresses underdetermination in soft_neighborhood_phase: a pixel with a static spectrum could have near-zero phase spread and still minimize the KL loss (uniform distribution matching uniform reference).

**Dynamism reference** (`phase_dynamism_supervision` feature): mean of 7 whitened variance/change-rate channels (variance_ndvi, variance_nir, variance_seasonal_amp_*, spectral_distance_per_decade), after robust_iqr + log transform and Mahalanobis whitening via precomputed covariance. Returns a scalar per anchor.

**Constraint**: For pair (i, j) where `|dynamism_ref[i] − dynamism_ref[j]| > delta (0.5)`:
- Phase spread = mean off-diagonal distance in the ysfc-aligned self-distance matrix (`d_learned_self`).
- Soft-margin: `softplus(spread_less_dynamic − spread_more_dynamic + margin)`, margin=0.1.
- Pairs below the delta threshold are silently skipped.

### 4.6 Phase VICReg

**Curriculum**: same curriculum weight as soft neighborhood phase.  
**Weight**: 0.1.

**Purpose**: Prevent dimensional collapse in z_phase.

Applied to `z_phase_at_anchors` reshaped from `[N_phase, T, 12]` to `[N_phase × T, 12]`. Same variance hinge + covariance penalty as z_type VICReg. Note: this operates on the combined distribution of all pixel-timestep pairs, which is dominated by within-pixel temporal variation. This is not the primary recovery-stage population.

### 4.7 Phase Recovery Discrimination Loss

**File**: `phase_recovery_discrimination_loss`, `frl/losses/triplet_phase.py`  
**Curriculum**: start_epoch=30, ramp_epochs=10.  
**Weight**: 1.0.

**Purpose**: Enforce absolute metric separation between disturbed and recovered states within each pixel.

**Why it is necessary**: The soft_neighborhood_phase loss is a relative ordering loss — it enforces that z_phase distances at shared ysfc values reproduce the spectral distance ordering. This is equivariant to a uniform isotropic scaling of the embedding space: a model can map all recovery stages to a small cluster and satisfy the KL loss with near-uniform distributions on both sides. The recovery discrimination loss adds an absolute constraint that does not reduce under isotropic compression.

**What it enforces**: For each anchor pixel that has both a "disturbed" timestep (ysfc ≤ 1.0) and a "recovered" timestep (ysfc ≥ 5.0):
$$\mathcal{L}_{\text{disc}} = \text{mean}_{(t_{\text{low}}, t_{\text{high}})} \left[ \text{softplus}\!\left(\text{margin} - \|z_{\text{phase}}(t_{\text{low}}) - z_{\text{phase}}(t_{\text{high}})\|_2\right) \right]$$
where margin=0.5, `low_ysfc_max=1.0` ("disturbed": ysfc=0 or 1), `high_ysfc_min=5.0` ("recovered").

All (disturbed, recovered) timestep pairs within each qualifying pixel contribute. Pixels without both classes are skipped.

**The distinction from relative ordering losses**: Soft_neighborhood_phase and phase_spread_ranking enforce that *if* two ysfc stages differ, their embeddings should *relatively* differ. Phase_recovery_discrimination enforces that specific pairs of ysfc stages differ by *at least* margin in absolute L2 distance. The two losses are complementary and both necessary.

### 4.8 EVT Soft Neighborhood Loss (Disabled)

**File**: `frl/losses/evt_soft_neighborhood.py`  
**Weight**: 0.0 (disabled in exp017).

Encourages z_type to mirror the distributional structure of the LANDFIRE EVT confusion graph via diffusion distances. The EVT-stratified anchor sampler (`grid-plus-supplement-evt`, supplement_n=768, inverse-frequency weighting by EVT code) is still built for cross-batch kNN sampling, but the loss itself contributes zero weight.

---

## 5. Training Procedure

### 5.1 Curriculum Learning Schedule for Phase Losses

Phase losses (soft_neighborhood_phase, phase_spread_ranking, phase_vcr, phase_recovery_discrimination) follow a four-phase curriculum (`frl/config/frl_training_v1.yaml`, `scheduler`):

| Phase | Epochs | LR | Phase loss weight |
|---|---|---|---|
| Type warmup | 0–10 | Linear 0 → peak (1e-4) | 0 |
| Type training | 10–30 | Cosine decay | 0 |
| Phase re-warmup | 30–35 | Drops to 5%×peak, ramps back to peak | 0 → 1 (ramp) |
| Joint training | 35–400 | Cosine decay 1e-4 → 1e-6 | 1 |

The LR drop at epoch 30 (`phase_warmup.start_factor=0.05`) is motivated by AdamW's momentum state: phase-pathway parameters have zero accumulated second-moment estimates at curriculum entry, so without a LR drop, the first gradient update would be a unit-norm step at the full current LR — potentially destabilizing. The re-warmup allows the optimizer to accumulate reliable estimates before restoring full LR.

Phase loss curriculum weight within the ramp: `curriculum_w = (epoch − start_epoch) / ramp_epochs`, clamped to [0, 1].

### 5.2 Optimizer and Scheduler

- **Optimizer**: AdamW, lr=1e-4, weight_decay=0.01.
- **Scheduler**: cosine warmup, T_max derived from curriculum structure, eta_min=1e-6.
- **Mixed precision**: bfloat16 (`hardware.mixed_precision.dtype`).
- **Gradient clipping**: `clip_grad_norm_, max_norm=1.0`.
- **Early stopping**: patience=15 epochs, monitoring `val/loss_total`.

### 5.3 Batch Structure and Epoch Sampling

- **Patch size**: 256×256 pixels.
- **Batch size**: 12 patches.
- **Epoch mode**: `number` — 500 patches sampled per epoch (random without replacement from the training split). This is ~10% of the training domain per epoch, not a full pass.
- **Total epochs**: 400.
- **Validation**: every epoch, 15% of patches.

### 5.4 Anchor Sampling Strategy

Two samplers operate per batch:

**Type sampler** (for z_type contrastive losses):
- `sample_anchors_grid_plus_supplement`: stride=16, border=16, jitter_radius=4 (training), supplement_n=104 weighted by `aoi × forest` mask.
- For a 256×256 patch with 16-pixel border and 16-pixel stride: grid yields approximately `((256-32)/16)² = 196` grid points, plus 104 supplement = ~300 anchors per patch.
- Jitter (±4 pixels from grid origin) provides augmentation in training; disabled for validation.

**Phase sampler** (for phase losses):
- Same `grid-plus-supplement` strategy, additionally gated by ysfc validity mask (requires valid ysfc at all timesteps).
- EVT-stratified variant (`grid-plus-supplement-evt`, supplement_n=768) is configured for the EVT-stratified kNN sampler (used even when EVT loss weight=0.0, to oversample rare EVT codes in cross-batch kNN pair construction).

### 5.5 Train/Val/Test Split

Spatial block split: 7×7 block grid over the full 13,056×23,552 domain. Patches within each block are assigned to a split. This separates training and validation/test areas spatially, preventing spatial autocorrelation from inflating validation metrics. Minimum 30% valid AOI fraction required per patch. Implemented in `ForestDatasetV2._filter_by_split`, `frl/data/loaders/dataset/forest_dataset_v2.py`.

---

## 6. Downstream Use

### 6.1 FIA Post-Stratification

The intended use is post-stratification and small-area estimation with FIA data. The `z_type` embedding provides a 64-dimensional representation of forest structure, species composition, and density signals that is learned without FIA label supervision. FIA plots are overlaid on the embedding space; strata are defined by clustering in z_type (GMM or k-means); within-stratum variance in FIA measurements drives the precision gain.

### 6.2 Linear Probe for z_type

`frl/training/fit_linear_probe.py`: fits a linear regression from frozen z_type embeddings to FIA-derived spectral targets. The encoder feature name is read from `training_config.model_input.type_encoder_feature` (currently `type_encoder_input`). This tests linear separability of forest attributes in the z_type space without end-to-end supervision.

### 6.3 Linear Probe for z_phase

`frl/training/fit_phase_linear_probe.py`: evaluates how well z_phase encodes temporal dynamics via R² against time-varying targets. Probed targets include annual spectral indices. Designed to measure whether recovery stage variation is metrically embedded across the temporal dimension.

### 6.4 GMM Clustering

`frl/training/fit_gmm_clusters.py`: fits a Gaussian Mixture Model on z_type embeddings. Clusters are compared against LANDFIRE EVT forest types (`compare_gmm_evt.py`) to assess alignment between learned and ecologically defined forest categories.

### 6.5 Diagnostic Tools

- `phase_evt_diagnostics.py`: EVT-stratified analysis of FiLM gamma magnitude and z_phase temporal variance fraction per channel. Revealed that gamma ≈ 3.5 post-training and is type-conditional.
- `phase_recovery_curves.py`: per-EVT NBR recovery curves as a function of ysfc; used to diagnose whether z_phase encodes recovery stage or pixel identity.

---

## 7. Key Design Decisions and Rationale

### 7.1 Stop-Gradient on z_type Before FiLM

`z_full.detach()` is called before extracting `z_type_at_anchors` for the phase encoder (`train_representation.py`, line ~480). The phase loss gradient cannot flow back through z_type via the FiLM conditioning path. Without this stop-gradient, phase-specific loss terms would update z_type through FiLM conditioning, allowing z_type to specialize for phase-loss minimization at the expense of its own spectral/spatial contrastive objectives. The type and phase pathways are trained by separate loss signals.

### 7.2 L2 Normalization Before FiLM

`F.normalize(h.flatten(1,2), dim=1)` normalizes the full (channel × time) vector of each pixel to unit L2 norm. This means the TCN determines the *direction* of the trajectory in the (channel × time) subspace, while FiLM gamma fully controls the *scale* per channel. Without this normalization, the scale of TCN outputs is unconstrained and interacts with FiLM initialization unpredictably.

### 7.3 Sparse Forward Pass

`forward_phase_at_locations` runs the TCN on sampled anchor pixels only — typically ~300 pixels per 256×256 patch rather than the full 65,536. This 200× reduction in the spatial dimension is the primary reason the phase pathway can be trained jointly with the type pathway in a single batch without prohibitive GPU memory. The method produces identical results to extracting from the dense forward pass at the same locations (documented in the method's docstring).

### 7.4 Cross-Batch vs. Within-Patch Negative Sampling

Spectral InfoNCE negatives are sampled cross-patch, not within-patch. This has two consequences: (1) cross-patch pairs are by construction spatially distant, so the spatial distance constraint for negatives is trivially satisfied without computation; (2) within-patch negatives risk sampling geographically adjacent pixels of the same forest type (false negatives), which is especially problematic for spatially homogeneous forests. Cross-batch negatives are weighted by spectral distance to further suppress false negatives.

### 7.5 Chunked Mutual kNN for Spectral Positives

`pairs_mutual_knn_chunked` processes `chunk_size=128` query points at a time against all N_total points, rather than materializing the full N_total × N_total distance matrix. Peak memory is O(chunk_size × N_total × C) instead of O(N_total² × C). The within-patch spatial constraint (pairs closer than 4 pixels masked to ∞) is applied only to the diagonal blocks (within-patch pairs); cross-patch blocks need no masking.

### 7.6 Phase VICReg Population Mismatch (Known Limitation)

Phase VICReg is applied to `z_phase_at_anchors.reshape(-1, 12)` — i.e., `[N_phase × T, 12]`. This flattening mixes within-pixel temporal variation (all T timesteps of each anchor pixel) with across-pixel variation. The dominant source of variance in this population is within-pixel temporal variation, not across-pixel recovery-stage variation, so VICReg's variance hinge may not effectively prevent the collapse of recovery-stage information. The phase_recovery_discrimination_loss directly addresses the across-pixel recovery collapse that VICReg cannot.

### 7.7 ysfc Alignment with Multiple Disturbances

When a ysfc value appears at multiple timesteps (e.g., a pixel that was disturbed twice), the reference timestep for that ysfc value is selected as the timestep from the *longest recovery sequence* containing it, with ties broken by most-recent (`select_features_by_ysfc`, `frl/losses/phase_neighborhood.py`). This prevents the phase loss from averaging over incomparable recovery sequences.

### 7.8 Input Channel Dropout Schedule

`type_encoder_input_dropout`: linearly ramped from 0% to 10% over the first 20 epochs using `Dropout2d` (zeros entire channel maps). The rationale in the model YAML: "Forces the encoder to learn cross-band structure rather than per-band features." The ramping (rather than constant dropout from epoch 0) allows the model to first establish a rough structural understanding before being forced to rely on cross-band correlations.

---

## 8. Open Questions for Paper Writing

The following items are ambiguous in the code, underdocumented, or require author clarification before the methods section can be finalized.

**8.1 Spatial resolution and geographic bounds**  
The config files confirm the data is in Virginia (`/data/VA/zarr/`) at 30m resolution (implied by CCDC/LCMS sources) with a full domain of 13,056 × 23,552 pixels, but neither the pixel size nor the CRS (UTM zone, EPSG code) are declared in any YAML. Exact geographic bounds and datum should be confirmed.

**8.2 GatedResidualConv2D forward pass**  
The current code (lines 125–128, `spatial.py`) computes `output = x + gate × (x − smoothed)`, which is an adaptive high-frequency amplification, not the blending described in the class docstring (OLD formulation). The paper will need to describe the actual behavior: it is adaptive sharpening where the gate is driven by the high-frequency residual, not a smooth/edge blend. Author should confirm which formulation is intended.

**8.3 L2 normalization scope in the phase pathway**  
`F.normalize(h.flatten(1,2), dim=1)` normalizes over the full `12 × T` dimensional vector per pixel. This is a joint normalization across channels and timesteps, not per-timestep. The paper should clarify whether this is the intended design, and what invariance it imparts: the unit sphere in (channel × time) space.

**8.4 Actual number of timesteps T**  
The time window is 2010–2024 inclusive, implying T=15. However, ysfc data extends back to 1985 (annual dataset `lcms_ysfc_value_1985_2024`). The phase encoder only processes 2010–2024 (`phase_ccdc`), but ysfc history used for pair construction (`ysfc` feature, annual.ysfc) is also in the 2010–2024 window. The effective T=15 should be confirmed, and whether any temporal padding or masking handles missing years.

**8.5 EVT soft neighborhood loss**  
The loss is defined, implemented, and configured with a full set of hyperparameters, but has `weight: 0.0`. The EVT-stratified sampler is active. The paper needs to clarify: (a) was this loss used in any training run that informs the final model? (b) is the EVT-stratified sampler providing real benefit for cross-batch kNN pair diversity even without the loss?

**8.6 Phase VICReg interaction with recovery discrimination**  
Phase VICReg operates on the wrong population (within-pixel-temporal flattening) for its stated purpose of preventing recovery-stage collapse. The recovery discrimination loss targets the correct population (within-pixel, between recovery stages). It is unclear whether Phase VICReg provides any benefit over what recovery discrimination already enforces, or whether the two have additive or conflicting gradient signals.

**8.7 Loss weight configuration pathway**  
The `losses:` section of `frl_binding_v1.yaml` defines weights for each loss component. The training script reads these from the `config` dict passed to `process_batch`. The relationship between the YAML `losses` section and the `config` dict needs documentation — specifically, where the spectral loss weight, spatial loss weight, vcr_weight, etc. are resolved into `config` for `process_batch`.

**8.8 Train/val/test block assignment rule**  
The 7×7 block grid is specified, but the actual assignment logic (how block indices map to train/val/test) is in `ForestDatasetV2._filter_by_split` which was not fully read. The paper needs to state the split fractions and whether the split is purely spatial (geographically separated blocks) or random within blocks.

**8.9 Number of qualifying anchor pixels per batch for phase losses**  
The minimum qualifying conditions (ysfc valid at all timesteps, ≥ 10 phase anchors, pairs surviving both kNN and ysfc overlap filter with ≥ 5 pairs per anchor) mean phase losses are computed on a variable fraction of patches and anchors per batch. Whether typical batch utilization rates are sufficient for stable gradients should be characterized.

**8.10 Relationship between `infonce_type_spectral` feature and Mahalanobis whitening**  
The `infonce_type_spectral` feature has `covariance: {calculate: true, stat_domain: patch}`. The pair selection and negative weighting use L2 distances in this feature space, described in comments as "Mahalanobis (L2 in whitened space)." The actual whitening transform (whether the precomputed covariance is used for PCA whitening, Cholesky whitening, or simple diagonal scaling) should be confirmed in `FeatureBuilder`.

**8.11 Temporal position channel as phase encoder input**  
The `phase_ccdc` feature includes `temporal_position` (fractional year index, 0→1) as its first channel with `identity` normalization. This provides the model with an explicit time index. The paper should address whether this encodes calendar time (useful for phenology confounds) or purely ordinal position, and its expected role in the phase trajectory.

**8.12 Spectral distance for spatial pair weighting uses static features, not type embedding**  
Spatial InfoNCE pair weights are computed from `spec_dist_data` (the `infonce_type_spectral` static feature), not from z_type itself. This means the weighting is data-driven from the raw features, not adaptive to the current state of the learned embedding. Whether this is intended (avoids circular dependency) or a limitation should be noted.
