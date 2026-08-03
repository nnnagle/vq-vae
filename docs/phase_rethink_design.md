# Phase-Pathway Rethink — Design & Evaluation Spec

Status: **discussion / pre-implementation.** Baseline for all comparisons: **exp034**.

This document is the working spec for the phase-pathway refactor. It is not yet
implemented. The checklist at the bottom is the order of work; step 0 (the eval)
is fleshed out first, on purpose, so the later steps are falsifiable.

---

## Motivation

The current phase pathway is fed globally z-scored spectral (`phase_ccdc`), so it
keys on absolute level (= forest-type / pixel identity) rather than
departure-and-return. The ysfc-driven losses only enforce *relative* distance rank
(scale-equivariant), so recovery stages collapse into a small region — patched, not
fixed, by `phase_recovery_discrimination_loss`. ysfc as a *target* also conflates
disturbance agents (fire vs harvest vs insect), which we don't want.

## Locked design decisions

- **Inputs → type-conditional anomalies** `(x_it − μ_i)/σ_i`, where μ_i/σ_i are the
  mean/scale of **mature** forest (selected by `ysfc > threshold`) **local in
  z_type space**. Multi-channel (keep per-band shape so agents separate on their
  own). Add Δ / Δ² temporal-difference channels for abruptness.
- **Geometry** (here input-space ≈ embedding target): anomaly origin ≈ 0 = mature;
  mature forests wiggle/bounce around 0; disturbances are **ejecta** from the
  origin that relax back over time. Must handle **fast** ejecta (harvest, fire)
  *and* **slow** ejecta (insect).
- **Output**: **per-timestep** `z_phase[t]` — a point that drifts toward the mature
  basin and is kicked out by disturbance. **No clustering loss** on disturbance
  regions (density structure is a hypothesis we will *measure*, not impose).
- **Type modulation**: keep `FiLM(stop-grad z_type)` at the output. Its role is now
  clean — place each type's phase manifold in its own region. **Post-FiLM z_phase =
  joint (type × phase) coordinate** = the deliverable.
- **ysfc**: demoted from loss *target* → mature-set *selector* (and, optionally, a
  weak within-recovery monotonicity prior only).
- **Mature-set threshold** `ysfc > mature_ysfc_threshold` is an **input parameter**,
  not hard-coded. It is region-dependent — forests mature faster in the East than
  the West. Suggested starting values: **~10–12 in the East, ~20–25 in the West**.
  Left as a tunable so we can sweep it and (later) set it per-region.
- **Attention-pool trajectory descriptor head**: deferred (noted in CLAUDE.md).

## Key risk

μ()/σ() must be **smooth** functions of z_type (kNN/kernel regression that is not
too wiggly), and the mature neighborhood depends on the *learned* embedding →
circularity. Mitigation: stage it (frozen z_type first; EMA/target-network only if
warranted), regularize smoothness.

---

## Step 0 — Evaluation spec (must exist before refactoring)

All probes are fit on **train**, tuned on **val**, reported on **test**, using the
checkerboard split already inside `ForestDatasetV2`. Every metric is reported for
the **new model vs the exp034 baseline** so "better" is defined. Unless stated,
probe the **post-FiLM z_phase[t]** (the joint type×phase coord); where noted, also
probe **pre-FiLM h** and **z_type** for contrast.

### A. Reconstruction probes — does z_phase retain the anomaly trajectory?

Fit light probes (ridge, and a small MLP as ceiling) `z_phase[t] → x_it` and
`z_phase[t] → anomaly_it`.

- Report **total variance explained** *and* — the important one — **within-pixel
  variance explained**: decompose target variance into between-pixel and
  within-pixel components and report R² on the within-pixel part. Within-pixel
  variance *is* the phase signal; a model that scores high on total but low on
  within-pixel has just re-encoded type/level.
- Predicting the **anomaly** is more diagnostic than raw `x` (raw `x` is dominated
  by type/level). Report both.

### B. Type-conditional recovery curves (upgrade of `phase_recovery_curves.py`)

For selected EVT classes: (i) plot the **actual** mean anomaly vs ysfc, and (ii)
fit a light probe `z_phase[t] → NBR` (or NBR anomaly) and plot **predicted mean vs
ysfc**, stratified by EVT. Different EVT types have different trajectories vs ysfc;
success = the embedding reproduces the *type-specific* shape, not a single pooled
curve. ysfc is imperfect but adequate here as an x-axis.

### C. Disturbance produces ejection

At `ysfc = 0`, how often is there a **big jump** `‖z_phase[t] − z_phase[t−1]‖`?
Report the jump-magnitude distribution at ysfc=0 vs ysfc>0 and a separation metric
(e.g. ROC-AUC of "is this a disturbance year" from jump magnitude alone). Validates
the slow-feature + ejecta geometry.

### D. Do ejecta and recovery pathways organize by change agent?

When `ysfc = 0` **and** there is a jump: do the **jump/ejection locations** in
z_phase cluster by change agent? Do the **return trajectories** to maturity cluster
by agent (fast harvest/fire V-shapes vs slow insect L-shapes)? This is a
*diagnostic*, not a loss — we are checking whether the "few origins + pathways"
structure emerges on its own.
- **Data dependency (deferred):** change-agent labels come from **LCMS** (Landscape
  Change Monitoring System). Not wired in yet — add later as a bindings source and
  join to the anchor pixels; diagnostic D runs once it's available.

### E. FIA downstream validation (existing machinery in `frl/analysis/`)

Extract embeddings at FIA plot locations; use kNN in embedding space
(`z_phase`, and `[z_type, z_phase]`) as the retrieval metric.
- Do **Basal Area, Volume**, etc. agree with kNN neighbors (report kNN-regression
  R² per attribute)? Reuses `fia_knn_models.Rmd`, `fia_embedding_evaluation.Rmd`.
- For plots **with removals**, are their kNN neighbors *also* removals? Reuses
  `fia_removals_stratification.Rmd`. This is the closest proxy to the actual
  post-stratification / small-area goal.
- **Data dependency:** FIA plot → attribute join and plot-location embedding
  extraction (`embed_locations.py`, `fit_linear_probe.py` conventions).

---

## Step 1 — Mature-baseline estimator (spec)

Goal: two functions **μ(z_type)** and **σ(z_type)** that, for any pixel's type
embedding, return the per-channel mean and scale of **mature** forest of that type.
These define the anomaly input `(x_it − μ_i)/σ_i` used by the rethought phase
pathway. The estimator is **live during training**: μ/σ are computed from a large
**reservoir** of mature pixels whose z_type is refreshed as the model trains (see
"Reservoir mechanism" below). There is **no separate frozen/co-trained split** — it
is one path. An optional one-off build against a frozen exp034 z_type is kept only
as a cheap offline sanity prototype, not the production mechanism.

- **`x_it` is the phase input exactly as defined in the bindings YAML** (currently
  `phase_ccdc`; whatever it evolves to). The estimator is agnostic to the specific
  channel list — it normalizes the bindings-defined phase feature. Any pass-through
  channels (e.g. `temporal_position`) are handled by the bindings spec, not here.
- **Reservoir mechanism.** A large fixed-capacity **reservoir**
  (`frl/utils/sampling.py::ReservoirSampler`, Algorithm R) is fed
  `(z_type_detached, mature x_it, ysfc, evt)` from the anchor pixels **already being
  embedded each batch** — no separate re-embedding pass. Reference samples are
  **per mature timestep** (`ysfc > mature_ysfc_threshold`), not per-pixel means:
  pooling individual mature timesteps preserves the ergodic year-to-year variation
  of a mature forest, which σ_i is meant to capture. This is a persisted,
  cross-batch enlargement of the per-batch "randomized-PCA + kNN in z_type space"
  demeaning pool that already exists in `process_batch` step 6. Data is Virginia
  (all-East) for now → a single reservoir, threshold ≈ 10–12; region-tagged
  reservoirs come with the East/West generalization.
- **Freshness / staleness (design decision).** Vanilla Algorithm R never updates a
  stored vector after insertion and samples uniformly over *all* history, so a
  never-reset reservoir accumulates **stale z_type embeddings** (including early bad
  ones) and μ/σ would reflect a blur of past geometries. Because z_type drifts, the
  reservoir must track the **current** embedding space: either **periodic reset**
  (per-epoch, or every few epochs) or a **forgetting / sliding** variant. Reset
  cadence vs. decay is an open decision; whichever we pick, it must keep the
  reservoir fresh relative to the rate z_type is still moving.
- **Estimator form (open decision — two candidates for the same target).** Both
  estimate the *conditional moments of mature spectral given type*:
  `x_it_mature ~ N(μ(z_type), σ(z_type)²)`. z_type is unconstrained in magnitude, so
  **standardize z_type first** either way. μ/σ are a **stop-grad, slowly-moving
  baseline** fed **detached z_type** — the phase loss never pushes z_type through
  μ/σ (mirrors FiLM stop-grad; keeps type/phase separated).
  - **(a) Non-parametric kNN / kernel regression** over the mature reservoir.
    Robust, interpretable bandwidth (`h > σ_ij`), degrades gracefully to "nearest
    observed data" in under-sampled z_type regions. Costs: needs the large reservoir
    + a refresh policy (freshness bullet above); not deployable without shipping the
    reservoir; O(N_query·N_res·d) per batch.
  - **(b) Parametric readout** — a small **RBF network** (preferred: smooth by
    construction, explicit kernel width = the interpretable bandwidth) or a
    **Lipschitz-controlled MLP** (spectral norm / gradient penalty; the Lipschitz
    constant `L ≈ 1/h` is the bandwidth analog, so `h > σ_ij` becomes `L < 1/σ_ij`;
    smooth activations, weight decay). Fit online by **heteroscedastic Gaussian NLL**
    on mature samples, σ via softplus/exp. Cheap constant-time query, differentiable,
    deployable. Its staleness is **parameter lag** (weights trailing a
    slowly-drifting z_type), which tracks better than frozen reservoir snapshots — so
    the large reservoir can shrink to a **small replay buffer** for variance
    reduction. Cost: can extrapolate confidently-but-wrongly in low-density z_type
    regions (mitigate with low capacity / coarse-prior shrinkage / an uncertainty
    head), and can silently fail (σ→0, μ ignoring type).
  - **Recommendation:** build **(a) kNN first** as the robust reference/oracle and to
    de-risk the anomaly-input idea; move production to **(b)** once trusted,
    **cross-checking (b)'s μ/σ against (a)** on held-out mature pixels to catch
    silent failure. The parametric form also dissolves most of the reservoir-refresh
    question (→ replay-buffer size).
- **Settling via warmup.** Since μ/σ move with z_type early in training, the
  existing **phase-loss curriculum warmup** now does double duty: it lets the
  reservoir-based μ/σ settle before the anomaly-input phase pathway starts learning.
  It likely needs to be **longer** than the current FiLM-stability warmup — μ/σ
  settling is a stricter condition than z_type merely being non-degenerate.
- **σ_i = pooled.** The scale pools both the mature **temporal wiggle** (across
  mature timesteps) and the **between-pixel** spread of mature type-i forests —
  i.e. the std of the pooled mature-timestep reference in the z_type neighborhood.
  Rationale: a disturbance should stand out against a mature forest's *normal*
  wiggle, and per-timestep pooling makes that wiggle part of the reference
  distribution directly.
- **Smoothness guard.** μ()/σ() must stay **slowly-varying functions of z_type**
  (they feed every downstream input; a wiggly μ makes same-type pixels'
  anomalies incomparable, confuses estimator jitter with real anomalies, and
  destabilizes the step-7 co-trained target). Mechanisms:
  - *Primary knob* — neighborhood size (k / kernel bandwidth); bigger = smoother.
  - *Selection criterion* — leave-one-out prediction of a held-out mature
    timestep's `x_it` from its neighbors, swept over bandwidth (small → high
    variance/overfit, large → high bias/types blur). Pick the smoothest setting
    that still tracks known type differences; **report effective degrees of
    freedom**.
  - *Graceful degradation* — minimum-neighbor floor + **shrinkage toward a coarser
    prior** (EVT-class mean, else global mean) where local support is thin. Ties to
    the per-EVT coverage diagnostic: rare / chronically-disturbed types fall back to
    a stable prior instead of a noisy local fit.
  - For the **parametric readout (b)** the smoothness knob is the RBF kernel width
    or the MLP's Lipschitz bound / weight decay (see "Estimator form"); pick it by
    the same LOO criterion and cross-check against kNN.
- **Scale hierarchy vs. the pairwise comparison scale σ_ij.** The mature-baseline
  bandwidth `h` should be **coarser (larger) than the type-similarity comparison
  scale σ_ij** used to weight loss pairs. Why: (1) σ_i is a second moment, so its
  neighborhood must hold many mature samples; (2) consistency — the loss treats
  pixels within σ_ij as same-type/comparable, so the baseline must be ≈ constant
  across a σ_ij neighborhood, else "same-type" pixels get different baselines and
  their anomalies stop being comparable; (3) it gives the smoothness guard headroom.
  Two caveats:
  - **Different metric spaces today.** The current pair weight is
    `w_ij = exp(−‖spec_i − spec_j‖₂ / sigma)`, `sigma = 5.0`
    (`frl/losses/phase_pairs.py`), defined in **Mahalanobis-whitened spectral**
    units — *not* z_type units. `h` is in **standardized z_type** units. So
    `h > σ_ij` is a principle about relative coarseness **on a shared ruler**, not a
    literal comparison to 5.0. To operationalize it, define the loss's type-
    similarity scale in the **same standardized-z_type metric** as `h` (which we
    want for consistency anyway).
  - **Not too large.** `h ≫ σ_ij` nearly-globalizes the baseline and leaves cross-
    type differences in the anomaly. Target *moderately* larger: pick `h` by the LOO
    bias-variance criterion, with `h > σ_ij` only as a lower-bound sanity check.
  - σ_ij belongs to the soon-to-be-retired soft-neighborhood loss, but the scale
    concept carries into the **type∧phase contrastive loss** (its positive/negative
    type-neighborhood scale); the ordering applies against whatever that becomes.
    Reconcile the two scales onto one z_type metric when specing Steps 4–5.

**Diagnostics for Step 1:**
- **Smoothness** — leave-one-out μ/σ stability vs. bandwidth; pick the smoothest
  setting that still tracks known type differences.
- **Per-EVT coverage** — mature reference count per EVT class, and the fraction of
  query pixels with ≥ k mature neighbors. Flags types where the mature baseline is
  under-supported (rare types, chronically disturbed types).

---

## Step 2 — Anomaly input transform (spec)

Goal: turn the bindings-defined phase feature `x_it` into the rethought phase
encoder input — a **type-conditional anomaly** plus temporal-difference channels
that make abruptness explicit.

**The transform.**
- Per-channel anomaly `a_it = (x_it − μ_i) / σ_i`, with **μ_i, σ_i constant over
  time** for a pixel (indexed by that pixel's `z_type`, broadcast across all T
  timesteps). Mature ⇒ `a_it ≈ 0`; disturbances ⇒ large excursions that relax back.
- **Pass-through channels** (e.g. `temporal_position`, and anything the bindings
  spec marks non-spectral) are *not* anomaly-normalized — they keep their bindings
  handling and are concatenated as-is.
- **Correctness constraint (critical):** μ_i/σ_i must be estimated on the **same
  representation of `x_it`** that the transform later subtracts from. Whatever units
  the bindings feature builder emits for `x_it` (currently z-scored), the Step-1
  reservoir must store `x_it` in *those same units*. Otherwise the subtraction
  is in the wrong space. This couples Step 1's bank construction to the bindings
  output — build them against the same feature.

**Temporal-difference channels (abruptness).**
- Append **Δa_it = a_it − a_i,t−1** and optionally **Δ²a_it** to the input. Computed
  on the **anomaly**, not raw `x` — abruptness of the *departure from type baseline*
  is the disturbance-onset signal; raw Δ is dominated by type-level seasonal/
  interannual structure.
- Boundary + masked timesteps: Δ at t=0 (no predecessor) and across invalid
  timesteps handled via the existing mask machinery (zero + mark invalid); the TCN
  is already mask-aware.
- These distinguish **fast** ejecta (harvest/fire → large |Δ|) from **slow** ejecta
  (insect → small |Δ| but sustained nonzero `a`), which is exactly the fast/slow
  distinction we need the encoder to see.

**Where the transform lives — one live path.**
- The transform is applied in `process_batch` at anchor pixels. After the z_type
  forward, query the Step-1 **reservoir estimator** at the anchors' current z_type
  for μ/σ, then form `a` and Δ. Both the query anchors *and* the reservoir contents
  are embedded in the same forward passes, so there is no frozen/online split — μ/σ
  are always in step with the current z_type (up to reservoir freshness).
- The phase feature is temporal `[C,T,H,W]` and is consumed only at ~100–300 anchor
  pixels, so it must be built via **`build_feature_at_locations`** at anchor coords
  (per the CLAUDE.md temporal-feature rule — never full-grid, to avoid OOM). μ/σ are
  evaluated at those same anchor coords.
- μ/σ are **stop-grad** (see Step 1): the phase loss shapes the TCN, not z_type,
  through the baseline.
- *(Optional offline sanity prototype only:* precompute per-pixel μ/σ from a frozen
  exp034 z_type and apply as a static normalization, to test "does the anomaly input
  help" before wiring the live reservoir. Not the production path.)

**Config / plumbing notes.**
- `phase_in_channels` grows: anomaly channels + Δ (+ Δ²) + pass-through. Update the
  model config and the phase encoder input wiring.
- Keep it out of the worker `precompute_features` list (that's spatial-only; this is
  temporal and anchor-built).
- FiLM stays at the output and is complementary: anomaly removes type at the input
  (mature ≈ 0 pre-FiLM for every type), FiLM `beta` re-places each type's manifold
  in the shared space (point 4). No change to FiLM here.

**Open decisions (to resolve before implementing):**
- **Δ² or Δ-only.** Δ captures onset abruptness; Δ² adds curvature (distinguishes a
  sharp V-bottom from a rounded L-turn) at the cost of noise amplification. Lean
  Δ-only first, add Δ² if the fast/slow separation (diagnostic C/D) is weak.
- **Scaling of the Δ channels.** Δa is already in σ_i units (since `a` is); decide
  whether Δ needs its own robust rescale so its dynamic range matches `a` for the
  TCN, or whether GroupNorm inside the TCN makes that moot.

---

## Build vs. retire

**Build**
1. Type-local mature-baseline estimator μ(z_type), σ(z_type) — smoothed kNN/kernel
   regression over `ysfc > threshold` pixels.
2. Anomaly input transform `(x−μ)/σ` (+ Δ / Δ² channels).
3. Slow-feature / temporal-smoothness loss (continuity within no-disturbance runs;
   jumps allowed at disturbance) — the term that builds the attractor geometry.
4. Type∧phase contrastive loss (similar-type-similar-state positives; same-type/
   other-state and same-state/other-type negatives); "state" bootstrapped from the
   anomaly trajectory, not CCDC.
5. Eval harness for Step 0 (A–E), split-disciplined, baselined to exp034.

**Retire / demote**
- `soft_neighborhood_phase` KL rank-matching (scale-equivariant — the collapse culprit).
- `phase_recovery_discrimination_loss` hard ysfc-bucket margin.
- Phase VICReg on flattened timesteps (wrong population).
- ysfc bucket loss → optional within-recovery monotonicity prior only.

---

## Checklist (ordered — de-risking first)

- [ ] 0. Eval harness (A–E above), fit on train / report test, baselined to exp034.
- [ ] 1. Live μ/σ estimator of the mature conditional moments `N(μ(z_type),σ(z_type)²)`. Start with **(a) kNN** over a `ReservoirSampler` of `(z_type, mature x_it, ysfc)` on standardized z_type (mature = `ysfc > mature_ysfc_threshold`, param ~10–12 East / ~20–25 West; refresh cadence reset-vs-decay; bandwidth `h > σ_ij`) as the robust oracle; then **(b) parametric readout** (RBF / Lipschitz-MLP, heteroscedastic-NLL, detached z_type, small replay buffer) cross-checked against (a). Check smoothness + per-EVT coverage. *(Optional: frozen-exp034 offline prototype as a sanity check.)*
- [ ] 2. Anomaly input builder `(x−μ)/σ` + Δ/Δ² at anchors via `build_feature_at_locations`; verify mature≈0, fast vs slow disturbances distinct.
- [ ] 3. Turn on anomaly input after warmup; confirm μ/σ settle and "does the input alone help?" vs exp034. Extend the phase-loss warmup as needed for μ/σ settling.
- [ ] 4. Slow-feature/smoothness loss; confirm drift-to-basin + jump-at-disturbance geometry.
- [ ] 5. Type∧phase contrastive loss; retire soft_neighborhood + recovery-disc + phase VICReg; reconcile σ_ij onto the z_type metric; re-run eval.
- [ ] 6. (optional) Encoder A/B: biGRU vs TCN for "where am I in the trajectory," with Δ channels.
- [ ] 7. Consolidate: CLAUDE.md (architecture + loss table + ysfc-as-selector reframing), diagnostics, config plumbing.

## Open data dependencies

- **Change-agent labels** for diagnostic D — source is **LCMS**; deferred, add as a
  bindings source and join to anchors later.
- **FIA attribute join + plot-location extraction** for diagnostic E — machinery
  exists in `frl/analysis/` and `embed_locations.py`; confirm current-model wiring.
- **`mature_ysfc_threshold`** — region-dependent input parameter (~10–12 East /
  ~20–25 West); sweep it, and eventually set per-region.
