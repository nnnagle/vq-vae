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

- **SimCLR projection head on z_type** (`frl/models/heads.py` — `MLPProjectionHead`) — A small MLP (Linear → BN → ReLU → Linear, optionally L2-normalized) sits between `z_type` and the spectral/spatial InfoNCE losses during training. Following the SimCLR convention, this head absorbs distortions introduced by contrastive training so the backbone embedding stays clean for downstream tasks. It is discarded at inference time — `z_type` is used directly. VICReg operates on raw `z_type` (not projected) to guard against backbone collapse in the original embedding space. Enabled/disabled and sized via `type_projection` in `frl_repr_model_v1.yaml`; default output_dim=8 (small relative to z_type_dim=48, reflecting ~3–4 significant spectral PCs). `embed_locations.py` writes the projected embedding as `g_type_*` columns alongside `z_type_*`. **Currently disabled (`type_projection.enabled: false`) — head-free ablation from exp032 onward:** with the head off, `model.project_type()` is an identity map and the spectral InfoNCE trains directly on raw `z_type` (like the spatial loss), so its temperature was raised 0.07 → 0.5. The **"Spectral sims"** log line (pos/neg similarity, `gap/T`) is the calibration check — target `gap/T ~2-3` like the spatial loss. Re-enable by setting `enabled: true` and reverting the spectral temperature.
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
| OU dynamics | `ou_dynamics.py` | Within-pixel Gaussian OU transition NLL (plug-in ‖z_t−ρz_{t−1}‖²); scalar global ρ. The **joint/complete-data MAP** — biases ρ toward 0 by the reliability ratio (attenuation). Superseded by the Kalman filter below. |
| Differentiable Kalman filter | `kalman_filter.py` | **Marginal**-likelihood within-pixel AR(1)+noise NLL (state integrated out ⇒ **de-attenuated ρ**). Reduced-rank linear-Gaussian SSM on the anomaly; type-conditional ρ(z_type)/Q; filtered state → z_phase. Outward-jump gating via ysfc reset. See "Phase pathway: differentiable Kalman filter" below. **Built + unit-tested; not yet wired into training (Phase 2).** |
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

### Phase pathway: differentiable Kalman filter (planned Step 6 — built, not yet wired)

**Motivation (from the exp035↔exp039 downstream analysis).** For FIA kNN, exp039
regressed on *removals* — a phase-reset / harvest signal where z_phase is the
**primary** predictor and the neighborhood is type-conditional. exp039's z_phase
ejects disturbances hard (C jump ratio 41.5) but the disturbance representation
is not **within-type coherent** (C ROC-AUC fell to 0.815; removals reliable-k
collapsed 20–40 → 10), because the OU+contrastive replaced the old
soft-neighborhood's *smooth within-type recovery-state metric* with a within-pixel
smoothness prior (OU) + ordinal ranking (contrastive). The fix is a
**type-conditional recovery rate** ρ(z_type) that restores a smooth within-type
recovery-stage ordering — everything within type (oak and pine harvests stay
incommensurate).

**The model.** A reduced-rank linear-Gaussian state-space model, per pixel, on
the anomaly (`losses/kalman_filter.py`):
`x_t = diag(ρ(z_type)) x_{t−1} + w_t` (Q(z_type)), `a_t = C x_t + v_t` (R), with
`z_phase = filtered state x_{t|t}`. This **moves the OU from a loss to an
architecture**: the Kalman gain is the optimal data-driven drive, so there is no
free `f` to trade off against ρ (the ρ↔f identifiability degeneracy is removed by
construction), and because the filter integrates the state out the loss is the
**marginal** likelihood → **ρ is de-attenuated** (the plug-in `ou_dynamics`
penalty is the joint MAP and biases ρ toward 0 by the reliability ratio
γ_x/(γ_x+R); unit-tested: true ρ=0.85 → filter 0.85, naive lag-1 0.39).

**Self-pairs only** (native): the filter is purely within-pixel; cross-pixel
structure stays in the contrastive. Filter NLL ↔ the old soft-neighborhood *self*
term; contrastive ↔ its *cross* term.

**Outward-jump gating** (the "don't apply over large outward jumps" rule):
recovery is the *inward* relaxation that identifies ρ; a disturbance is a large
*outward* jump that is not AR(1)-modelable. Disturbance years (`reset`, from
ysfc==0 or a ysfc decrease) restart the segment prior and are **assimilated but
not scored**, so the jump never enters the likelihood — ρ is estimated on the
recovery regime only. `NIS` (normalized innovation squared, ≈ C_obs when
calibrated) is logged as the free filter-consistency / identifiability check.

**Decisions in force** (revisit notes for later):
- **Phase VICReg disabled** — the NLL + emission prevent state collapse.
- **Filtered** state as z_phase — *revisit: RTS smoother* (better estimate + the
  lag-1 cross-cov EM sufficient statistic).
- **ysfc reset gate** — *revisit: add an unlabelled/innovation-threshold gate* to
  catch disturbances LCMS misses.
- **Shared emission C** — *revisit: type-conditional C(z_type)*.
- **d = z_phase_dim = 8** is enough because the dynamics (ρ, Q, prior) vary with
  z_type — a low-dim state with type-adapted dynamics is expressive.
- If the Phase-0 fit shows AR(1) is inadequate widely, add **AR(2) / complex-
  diagonal modes** (damped-oscillatory, for non-monotone recovery).

**Rollout.** Phase 0 = classical per-EVT AR(1)+noise fit for a data prior ρ̂(EVT)
+ AR-order verdict (`analysis/ar1_recovery.py` core, unit-tested;
`analysis/run_ar1_recovery.py` ISAAC CLI). Phase 1 = the differentiable filter
module + tests (`losses/kalman_filter.py`, `tests/test_kalman_filter.py` —
**done**). Phase 2 (next) = type-conditional ρ/Q/C/prior heads on
`RepresentationModel` + wire the NLL into `process_batch` behind a `phase_kalman`
config block, filtered state → z_phase, contrastive kept, VCR off. Phase 3 =
diagnostics (NIS, ρ spread) + ISAAC validation (within-type removals reliable-k,
C-AUC, recovery curves).

### Code structure & data flow (`frl/training/`)

`train_representation.py` is a thin CLI entry point (~620 lines): argument parsing, config/dataset/model/optimizer wiring, the epoch loop, and per-epoch checkpoint orchestration. All training logic lives in the `frl/training/representation/` subpackage:

| Module | Responsibility |
|--------|----------------|
| `step.py` | `process_batch()` — the full per-batch step (data flow below) |
| `loops.py` | `train_epoch` / `validate_epoch` — iterate the dataloader, call `process_batch`, accumulate + log |
| `config_builders.py` | Parsed YAML → `loss_config` and the phase / spread / recovery-disc / EVT setup dicts (+ EVT metric & sampler) |
| `scheduler.py` | `build_scheduler` — warmup / two-phase phase-warmup cosine / plain cosine, + auto-resume state restore |
| `checkpointing.py` | `CheckpointManager` (last/periodic/top-k save + prune/rename) and `resume_from_checkpoint` |
| `epoch_logging.py` | `log_epoch` — the per-epoch diagnostic log block |
| `curriculum.py` | Pure epoch→scalar schedules: input dropout, the shared phase-loss `ramp_weight`, spatial-smoothing gate |
| `profiling.py` | The `--profile` flag (`set_profile` / `is_profiling`), shared across modules |

Import graph is acyclic: `train_representation.main → loops → step → {curriculum, profiling}`; `main` additionally imports `config_builders`, `scheduler`, `checkpointing`, `epoch_logging`. Two modules pin `logging.getLogger("training.train_representation")` so log records keep their original name after the move.

**Data flow through `process_batch()` (`step.py`)** — one batch of B patches:

1. **Curriculum weights** — `ramp_weight(epoch, …)` sets the phase / spread / recovery-disc loss weights (0 during warmup, ramping to 1).
2. **PASS 1 — CPU prep, per sample** (fills `prep_list`): sample anchors (grid+supplement, or EVT-stratified when the EVT loss is on), extract the spectral-distance feature at anchors, build spatial-InfoNCE pairs + spectral weights, and build phase pairs (`build_phase_pairs`). The encoder forward is **deferred**. Worker-precomputed spatial pairs are reused when present (see Feature Precomputation).
3. **BATCHED GPU FORWARD** — a single chunked `[B,C,H,W]` encoder forward over all valid samples → `z_type`, `gate` (chunked by `enc_chunk_size` to bound peak memory).
4. **PASS 2 — per sample**: extract `z_type` at anchors; compute **spatial InfoNCE**, **VICReg**, **EVT** soft-neighborhood; run the **phase TCN at phase-anchor pixels** (`forward_phase_at_locations`, conditioned on stop-gradient `z_type` via FiLM) → `z_phase` + **phase VICReg**; accumulate cross-batch collectors (`cross_patch_*` for spectral, `cross_phase_*` for phase). Per-patch loss = spatial + vcr + phase_vcr + evt; non-finite samples are skipped.
5. **CROSS-BATCH SPECTRAL** — pool all anchors across the batch: chunked mutual-kNN positives + random cross-patch negatives (weighted by spectral distance) → **global spectral InfoNCE**, plus the **Spectral sims** kernel-sizing diagnostic (`gap/T`, for the head-free calibration).
6. **CROSS-BATCH PHASE** — pool all phase anchors: randomized-PCA + kNN in `z_type` space builds a type-local spectral baseline (for demeaning), then the **phase neighborhood**, **spread ranking**, **recovery discrimination**, and **Frobenius type-leakage** losses.
7. **Backward + optimizer step** (training only), then assemble the stats/timing dict returned to the epoch loop and consumed by `log_epoch`.

Steps 1–4 are per-sample; **steps 5–6 are computed once over the whole batch** — which is why spectral positives can be spectrally-similar pixels from *different* patches (location-invariant forest type). `loops.py` sums these per-batch stat dicts into per-epoch means and forwards the last batch's diagnostic sub-dicts (gate, sims, phase, FiLM, type-leakage) to `log_epoch`.

### Important: Phase Loss Curriculum

The phase loss uses **curriculum learning** — it is **zero for the first N epochs** (warmup), then ramps up. This is intentional. If you see phase loss = 0 early in training, that is expected behavior, not a bug. The warmup epoch count is set in the training config.

### Important: Feature Precomputation in DataLoader Workers

`feature_builder.build_feature()` (Mahalanobis whitening + normalization) runs in the DataLoader worker processes, not in the main training loop. This keeps the GPU from sitting idle during CPU-bound preprocessing.

`ForestDatasetV2` is given a `feature_builder` and a `precompute_features` list at construction time (built in `main()` in `train_representation.py`). Each worker calls `build_feature()` in `__getitem__` and stores the results in the batch dict under `__feat_{name}_data` / `__feat_{name}_mask`. The `_get_feature()` helper in `process_batch()` (`frl/training/representation/step.py`) reads these pre-built arrays from the batch; it falls back to calling `feature_builder.build_feature()` silently if a name is missing — meaning the fallback is correct but slow.

**When adding a new `_get_feature()` call in `process_batch()` (`step.py`), also add the feature name to the `precompute_features` list in `main()`.** Omitting it won't break training, but the feature will be built in the main process and the speedup won't apply.

**Only add spatial (2D) features to `precompute_features` — not temporal ones.** Temporal features like `ysfc` and `phase_encoder_feature` are `[C, T, H, W]` arrays; stacking them across a full batch (e.g. 32 samples × 22 channels × 15 years × 256×256 pixels) causes OOM, so they are not precomputed in workers. They are consumed only at ~100–300 anchor pixel locations in `process_batch()`, so instead of building the full grid and extracting, build them **only at the anchor coords** via `FeatureBuilder.build_feature_at_locations(name, sample, coords)`. Because normalization and Mahalanobis whitening use fixed precomputed stats and are pointwise per pixel, this is bit-identical to the full-grid build (verified, `max|diff|=0`) at ~H·W/N (~230×) less cost. Building these features full-grid per sample in the main process used to be the dominant training-step cost — see Performance.

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

### Step-0 Phase-Pathway Eval Harness (`frl/training/phase_eval/`)

The falsifiable eval for the phase-pathway rethink (`docs/phase_rethink_design.md`
Step 0). Runs diagnostics **A–C**, **fit on train / λ-tuned on val / reported on
test**, and writes one `metrics.json` per checkpoint; `compare_eval.py` diffs a new
run against the exp034/exp035 baseline. See
`docs/phase_eval_step0_findings.md` for the exp035 results and what Steps 1–5
should target.

```bash
# one checkpoint (diagnostics A,B,C)
PYTHONPATH=frl python -m training.phase_eval.run_eval \
    --checkpoint runs/frl_v0_exp035/checkpoints/encoder_best_1_epoch_380.pt \
    --training config/frl_training_v1.yaml --evt-map ../data/LF2024_EVT.csv \
    --output-dir runs/frl_v0_exp035/phase_eval/
# new vs baseline
PYTHONPATH=frl python -m training.phase_eval.compare_eval \
    --new runs/frl_v0_exp035/phase_eval/metrics.json \
    --baseline runs/frl_v0_exp034/phase_eval/metrics.json
```

- **A — reconstruction** (`reconstruction.py`): ridge + MLP-ceiling probes
  `features → raw phase-input x`, for four feature sources: post-FiLM `z_phase`,
  **pre-FiLM `h`** (the TCN bottleneck, the FiLM-free contrast), `z_type`
  (atemporal control), and **`type-phase-bilinear`** — the type×phase interaction
  source (rank-`r` whitened-PCA `[z_type, z_phase, (Pᵀz_type)⊗z_phase]`, separate
  ridge on the main-effect vs interaction blocks; no MLP, the per-source MLPs are
  the nonlinear ceiling). Because `z_type` is atemporal, an *additive* `[z_type,
  z_phase]` adds **zero** within-pixel signal — the interaction is the *only* route
  by which type can raise the within-pixel reconstruction, so this source's
  within-R² lift over `z_phase` alone measures how much type-conditional reading
  recovers. **Caveat:** the rank-`r` truncation uses whitened PCA of `z_type`,
  which on near-isotropic (VICReg'd) standardized `z_type` picks variance-driven
  rather than supervised directions — so it's a *conservative* (lower-bound)
  estimate of the interaction benefit; raise `BILINEAR_RANK` or switch to a
  supervised reduced-rank fit if it looks capped. Reports **total** and
  **within-pixel** R² (the within-pixel
  R² is *the phase signal*), as a **variance-weighted aggregate** (pool residual/
  total SS across channels before the ratio — so low-within-variance channels
  can't dominate an unweighted mean) plus per-channel R²/variance/MSE. `z_type`
  within-R² is **0 by construction** (atemporal → broadcast over T → zero
  within-pixel prediction variance): that is the control, not a bug. The
  `__summary__` block reports the **h → z_phase within-R² gap** = how much
  temporal signal FiLM loses. `temporal_position` (the calendar-index target) is
  excluded via `EXCLUDE_TARGET_CHANNELS` — it is trivially predictable and was
  inflating the aggregate.
- **B — recovery curves** (`recovery_curves.py`): `z_phase → NBR` probe, per-EVT
  actual-vs-predicted recovery curves and a **shape-agreement** metric. Runs four
  readouts (`interaction_readouts.py` holds the pure math). Two **additive**
  (linear): **phase-only** and **type-phase** (`[z_type, z_phase]` concat) — the
  latter supplies the type-specific baseline z_phase alone lacks. Two
  **interaction** readouts (type MODULATES the read of phase — z_phase is a
  type-collapsed shadow, only meaningful conditional on type; an additive concat
  gives z_phase one global gain and under-reads it, which is a prime suspect for
  the exp037 amplitude under-shoot): **type-phase-bilinear** — a rank-`r` (default
  3) whitened-PCA bilinear `(Pᵀz_type)⊗z_phase` with **separate** ridge on the
  main-effect vs interaction blocks (the interaction is higher-variance, wants more
  shrinkage; both λ tuned jointly on val); and **type-local-knn** — product-kernel
  Nadaraya–Watson with type/phase bandwidths tuned **independently** (mirrors the
  downstream kNN post-stratification use). **Backward-compat:** the two interaction
  designs are NEW keys under `B_recovery_curves`, absent from exp035-and-earlier
  `metrics.json`; `compare_eval.py` reports them as new-only (baseline None, no
  crash). `run_interaction=False` reproduces the additive-only B.
- **C — ejection** (`ejection.py`): jump magnitude `‖z_phase[t] − z_phase[t−1]‖`
  at disturbance years (`ysfc==0`) vs stable, and the ROC-AUC of disturbance-from-
  jump.
- **E — type-conditioning leakage (DEFERRED, not yet implemented).** A probe
  measuring whether the anomaly `a` still encodes `z_type`: R² of `z_type` from the
  per-pixel mean anomaly `mean_t a` (or `‖cov(mean_t a, z_type)‖_F`). The anomaly is
  meant to be type-conditional (`a = (x − μ(z_type))/σ(z_type)`, mature ⇒ a≈0), so
  residual predictability of `z_type` from `a` means `μ` under-removed the type
  baseline (readout under-fit / bandwidth too smooth) and the contrastive's
  "conditional on same type" comparison is leaking. This is the **direct**
  measurement of the `h ≫ σ_type` design constraint (RFF readout bandwidth much
  smoother than the contrastive type kernel — see the `type_phase_contrastive.sigma_type`
  comment in `frl_binding_v1.yaml`), better than proxying it by the `h/σ_type` ratio.
  Cheap to add — reuses the `z_type` + anomaly already extracted in `common.py`.
  (D = LCMS change-agent clustering, also deferred; see `lcms_agents.py`.)

**Ridge normal equations are averaged by the observation count M** (`A/M`, `B/M`)
before adding `λI`, so the λ grid is on a dataset-size-independent scale — with
standardized features `A/M` has a unit diagonal. Without the `1/M`, `A`'s diagonal
is ~M (10⁷–10⁸) and every `λ≤1` is a ~10⁻⁷ perturbation: the sweep goes flat and
the fit is silently unregularized. A `_warn_lambda_edge` guard logs a warning when
the selected λ lands on the grid boundary (optimum outside the grid), skipping
degenerate flat sweeps (e.g. z_type). **Both A and B fit their own ridge — this
averaging must hold in both.**

ISAAC launchers: `eval_isaac_v2.sh` (GPU) / `eval_isaac_bigmem.sh` (CPU bigmem,
sidesteps the GPU-idle watchdog on the light Step-0 eval).

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
frl/losses/ou_dynamics.py                Within-pixel OU transition NLL (plug-in; superseded by kalman_filter)
frl/losses/kalman_filter.py              Differentiable within-pixel Kalman filter + marginal NLL (de-attenuated ρ)
frl/losses/reconstruction.py             Reconstruction loss

frl/analysis/ar1_recovery.py             Phase-0 classical AR(1)+noise recovery estimator (data prior ρ̂; unit-tested core)
frl/analysis/run_ar1_recovery.py         Phase-0 ISAAC CLI: per-EVT AR(1)+noise fit on the anomaly
frl/analysis/fia_knn_models.Rmd          Downstream FIA kNN eval (reliable-k, post-stratification RE/ESS)

frl/training/train_representation.py                 Training CLI entry point (arg parsing + wiring + epoch loop)
frl/training/representation/step.py                  process_batch() — the per-batch training/val step
frl/training/representation/loops.py                 train_epoch / validate_epoch
frl/training/representation/config_builders.py       YAML → loss_config / phase / spread / recovery-disc / EVT
frl/training/representation/scheduler.py             build_scheduler (warmup / two-phase cosine / resume)
frl/training/representation/checkpointing.py         CheckpointManager (top-k save) + resume_from_checkpoint
frl/training/representation/epoch_logging.py         log_epoch — per-epoch diagnostic logging
frl/training/representation/curriculum.py            Epoch→scalar schedules (dropout, ramp_weight, smoothing gate)
frl/training/representation/profiling.py             --profile flag (set_profile / is_profiling)
frl/training/fit_linear_probe.py              Downstream type embedding linear probe (z_type → FIA targets)
frl/training/fit_phase_linear_probe.py        Phase embedding linear probe (temporal R²)
frl/training/fit_gmm_clusters.py              Fit GMM on z_type embeddings
frl/training/compare_gmm_evt.py               Compare GMM clusters vs EVT forest types
frl/training/visualize_test_patches.py        Visualize model output on test patches
frl/training/visualize_forest_diagnostics.py  Forest-wide embedding diagnostics
frl/training/phase_evt_diagnostics.py         EVT-stratified FiLM gamma + z_phase temporal variance
frl/training/phase_recovery_curves.py         Per-EVT NBR recovery curves vs. ysfc (requires probe)

frl/training/phase_eval/common.py             Step-0 harness plumbing (loaders, extract_pixel_series, variance_decompose)
frl/training/phase_eval/reconstruction.py     Diagnostic A — z_phase/h/z_type → x reconstruction (within-pixel R²)
frl/training/phase_eval/recovery_curves.py    Diagnostic B — z_phase→NBR recovery curves (phase-only + type-phase)
frl/training/phase_eval/ejection.py           Diagnostic C — disturbance-year jump magnitude + ROC-AUC
frl/training/phase_eval/run_eval.py           Step-0 CLI runner (train-fit / val-tune / test-report → metrics.json)
frl/training/phase_eval/compare_eval.py       Diff two metrics.json (new vs exp034/exp035 baseline)
frl/training/phase_eval/lcms_agents.py        LCMS change-agent class codes (Diagnostic D, deferred)

frl/config/frl_repr_model_v1.yaml        Architecture config
frl/config/frl_binding_v1.yaml           Dataset bindings config
frl/config/frl_training_v1.yaml          Training hyperparameter config
```

---

## Extending the Model

The codebase is flexible with no rigid extension conventions.

**Add a new loss function:**
1. Create in `frl/losses/`
2. Import and call it inside `process_batch()` in `frl/training/representation/step.py` (per-sample in Pass 2, or once-per-batch in the cross-batch section); thread any new diagnostic through `loops.py` and `epoch_logging.py` if you want it logged
3. If it needs config, add a builder/keys in `frl/training/representation/config_builders.py`; loss weights are defined in the bindings YAML (`frl_binding_v1.yaml`), not the training YAML

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

## Training Infrastructure (ISAAC HPC)

### Data

The Zarr archive lives on Lustre at `/lustre/isaac24/scratch/nnagle/zarr/` (284 GB, 3.9M files). A pre-built tar archive is at `/lustre/isaac24/scratch/nnagle/zarr.tar` (269 GB) for fast job-start extraction — use this instead of `cp -r`, which takes 90+ minutes due to Lustre metadata overhead on 3.9M files.

**Tar extraction layout.** Build the tar so its members are single-nested under
`zarr/` — extract to `/dev/shm/` (or `/tmp/`) and you get
`/dev/shm/zarr/va_vae_dataset.zarr`, so `ZARR_ROOT=/dev/shm/zarr`. Reproduce the
layout without copying the data via GNU tar's `--transform`:
`cd /data/VA/zarr_v2 && tar -cf zarr_v2.tar --transform='s,^,zarr/,' va_vae_dataset.zarr`.
The **v2 dataset** (`zarr_v2.tar`, includes `lcms_chg_class`) and
`train_isaac_ram_v2.sh` use this corrected layout.

*Legacy note:* the original `zarr.tar` was accidentally built double-nested
(`zarr/zarr/va_vae_dataset.zarr/`), so the original `train_isaac_ram.sh` /
`train_isaac_dev*.sh` set `ZARR_ROOT=/dev/shm/zarr/zarr`. Those remain paired with
that old tar; new work should use the v2 tar + `train_isaac_ram_v2.sh`.

**Sidecar files:** Stats files (`*.json`, `*.csv`) are not inside the tar — they are copied separately from Lustre after extraction.

### Training Scripts

| Script | Partition | Data location | Purpose |
|--------|-----------|---------------|---------|
| `train_isaac_ram_v2.sh` | campus-gpu-large | `/dev/shm` (RAM) | **v2 dataset** (with `lcms_chg_class`); corrected single-nest tar |
| `train_isaac.sh` | campus-gpu-large | Lustre (slow) | Original production script (v1) |
| `train_isaac_ram.sh` | campus-gpu-large | `/dev/shm` (RAM) | v1 production, auto-resumes; legacy double-nest tar |
| `train_isaac_dev.sh` | campus-gpu-bigmem | `/tmp` (NVMe) | v1 dev, `--overwrite`; legacy double-nest tar |
| `train_isaac_dev_ram.sh` | campus-gpu-large | `/dev/shm` (RAM) | v1 dev, `--overwrite`; legacy double-nest tar |

### Performance

The per-batch training step was reduced ~6× (~6.85s → ~1.1s/batch, phase-active dev config), taking epoch time from >10 min (earlier sessions) to ~1 min. The bottleneck was **not** the dataloader or the Mahalanobis whitening, as previously assumed — moving data to RAM never helped because the pipeline was never I/O-bound. The real costs were in the main process and were found with `--profile` (per-epoch dataloader wait/step split + per-component step breakdown; off by default so it adds no `cuda.synchronize()` overhead):

1. **A per-batch gate-value diagnostic** — `compute_stats(all_gate_values)` ran `cat` + `randperm` + `quantile` over ~67M values *every batch* (~3.5s), feeding a log line printed only once per epoch (and frozen at 1.0 during the smoothing curriculum). Fixed by subsampling to `_GATE_STATS_SAMPLES` (4096) values per patch.
2. **Per-sample full-grid temporal feature builds** — `phase_ccdc`/`phase_dynamism_supervision` were normalized over the entire `[C,T,256,256]` grid per sample, then read at only ~280 anchor pixels (~2.4s/batch when the phase curriculum is active). Fixed with `build_feature_at_locations()` (anchor-only; see Feature Precomputation note).

Supporting changes: `OPENBLAS/MKL/NUMEXPR/OMP_NUM_THREADS=1` in the launch scripts (uncapped BLAS thread pools oversubscribe cores across many workers), a per-worker `torch.set_num_threads(1)` `worker_init_fn`, and `train_isaac_ram.sh` now uses `--cpus-per-task=48` (→46 workers, matching `num_workers` in the config). At ~1.1s/batch, the old 4 workers would re-starve the loop. Re-measure after any change with `--profile` (already enabled in the `*_dev*.sh` scripts); production scripts run without it.

### SLURM Notes

- **Authorized QOS:** `campus`, `campus-bigmem`, `campus-gpu`, `long`, `long-bigmem`, `long-gpu`
- **campus-gpu-large is GPU-heterogeneous and the GRES tag lies** (all report `gpu:v100s:2`). Actual (from `gpuprobe`): **uniform 32 GB** = `clrv1107`, `clrv1205`; **mixed 32 GB + 16 GB V100** = `clrv1103`, `clrv1105`, `clrv1201`; **`clrv1101`** = 16 GB / degraded. On a mixed node `--gpus=1` can hand you the 16 GB card (→ OOM), and you can't select by type because the 16 GB V100 is mislabeled `v100s`.
- **Do NOT use `--exclusive`** on these 2-GPU nodes — it idles the second GPU and the scheduler cancels the job for holding an unused GPU. Likewise avoid reserving cores/GPUs you don't use.
- **GPU launch recipe (`train_isaac_ram_v2.sh`, `eval_isaac_v2.sh`):** `--gpus=1` + `--exclude=clrv1101,clrv1103,clrv1105,clrv1201` (restrict to the uniform-32 GB nodes so any single GPU is 32 GB) + the runtime largest-memory GPU pin (`CUDA_VISIBLE_DEVICES`, defense). `/dev/shm` is node-wide, so the scripts pre-clean `/dev/shm/zarr` and fail-fast if < ~300 GB free (a co-tenant contending) — just resubmit.
- **campus-gpu-bigmem** (`ilpa1209`, A40 + NVMe `/tmp`) — frequently in maintenance; use campus-gpu-large as fallback
- **campus-gpu-large** nodes have 770 GB RAM — enough to hold the zarr in `/dev/shm` with `--mem=500G` (no `--exclusive` needed)
- **CPU eval** (`eval_isaac_bigmem.sh`) runs on `campus-bigmem` with `--device cpu` — sidesteps the campus-gpu GPU-utilization watchdog, which cancels the GPU-light Step-0 eval "for not using GPU".
- **ai-tenn** partition (H100s) requires a separate allocation not currently held

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

**Likely root cause found — feature-normalization bug (fixed).** The weak variance/velocity signal was, at least in large part, a data-pipeline bug, not a loss/architecture issue: `FeatureBuilder._apply_normalization` applied each channel's `log`/`sqrt` transform but then z-scored the **raw** channel against the **transform-scale** stats, pushing every value past the clamp and collapsing all log-transformed channels to a constant ("dead channel"). Exactly the variance-like measures below use `transform: log` — `variance_ndvi`, `variance_*`, `spectral_velocity`, `spectral_distance_per_decade`, etc. — so they were dead in **both** `build_feature` (eval) and `build_feature_at_locations` (training), across model inputs (`phase_ccdc.spectral_velocity`), loss targets (`soft_neighborhood_phase_target`, `phase_dynamism_supervision`), and the InfoNCE pair-selection space (`infonce_type_spectral`). Fixed in `feature_builder._apply_normalization` (normalize the transformed array; regression test in `frl/tests/test_feature_normalization.py`). Re-evaluate these probes after **retraining with the fix** before pursuing the loss/architecture options below.

**TODO: Improve encoding of temporal variance and variance-like measures (variance_ndvi, spectral_velocity).** These targets have weak linear probe R² (~0.25–0.63) *in pre-fix runs (exp034 and earlier)*. The cause is not clearly an architecture issue — it may be a loss issue: after whitening, these features contribute only ~1/22 of the InfoNCE pair-selection signal, so gradient pressure is weak. Alternatively, low R² may be appropriate for stable forest types and only problematic for dynamic types. Options to investigate:
- Upweight variance-like features in the spectral distance computation
- Add an auxiliary reconstruction loss targeting these specific channels
- ~~Stratify probe diagnostics by EVT forest type before concluding the signal is missing~~ *(done — EVT stratification shows weak phase signal is broadly true across types, not just a stable-forest artifact)*

~~**TODO: Compute EVT-forest-type-stratified diagnostics for phase signal strength.**~~ *(implemented — see `phase_evt_diagnostics.py` and `phase_recovery_curves.py`; key findings recorded in the FiLM gamma bullet above)*

~~**TODO: Make z_phase encode recovery stage, not just pixel identity.** The `soft_neighborhood_phase` loss enforced only relative distance ordering (KL-softmax), which is equivariant to uniform scaling of the embedding space. A model could satisfy it perfectly while compressing all recovery stages into an arbitrarily small region. Phase VICReg did not fix this because it operated on the wrong population (N_phase×T flattened timesteps, dominated by within-pixel temporal variation rather than across-pixel recovery-stage variation). The fix — `phase_recovery_discrimination_loss` in `frl/losses/triplet_phase.py` — adds an absolute margin constraint directly between disturbed and recovered timesteps within each pixel.~~ *(implemented in exp017; model considered complete)*

**FUTURE (deferred): attention-pool trajectory descriptor head on the phase pathway.** Alongside the per-timestep `z_phase`, a small temporal attention-pool head could emit a per-pixel trajectory descriptor — a handful of learned query tokens summarizing e.g. {pre-disturbance baseline, trough depth, recovery slope, time-since-trough}. It would serve as an auxiliary conditioning signal for the per-timestep head and/or as a second retrieval key for trajectory-shape similarity. Not needed for the phase-pathway rethink (the primary output stays per-timestep); recorded here as a possible later enhancement and set aside for now.
