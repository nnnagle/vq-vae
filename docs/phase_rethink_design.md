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

μ()/σ() must be **smooth** functions of z_type (the parametric readout must not be
too wiggly), and the fit depends on the *learned, drifting* z_type → circularity.
Mitigation: an explicit smoothness constraint (RBF width / Lipschitz bound), and a
**warmup** that lets both z_type and the readout settle before the phase losses
turn on.

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
pathway. The estimator is a small **parametric readout** of z_type, fit **online
during training** by heteroscedastic Gaussian NLL on mature samples (details below).
**No reservoir, no kNN** — μ/σ are a function of z_type, trained alongside the model
and evaluated with one forward per batch.

- **`x_it` is the phase input exactly as defined in the bindings YAML** (currently
  `phase_ccdc`; whatever it evolves to). The estimator is agnostic to the specific
  channel list — it normalizes the bindings-defined phase feature. Any pass-through
  channels (e.g. `temporal_position`) are handled by the bindings spec, not here.
- **Estimator = online heteroscedastic Gaussian NLL readout.** μ/σ are a small
  **parametric map** of standardized `z_type`, fit to the conditional moments of
  mature spectral:
  `x_it_mature ~ N(μ_θ(z_type), σ_θ(z_type)²)`, trained by Gaussian NLL, with σ via
  `softplus`/`exp` to stay positive. **No reservoir, no kNN.** μ_θ = conditional
  mean, σ_θ = conditional std — the parametric estimator of the same quantity a kNN
  would approximate non-parametrically. z_type is unconstrained in magnitude
  (per CLAUDE.md), so **standardize z_type first**. μ/σ are fed **detached z_type**
  and are **stop-grad** w.r.t. the phase loss (mirrors FiLM stop-grad; keeps
  type/phase separated); z_type is shaped only by the type losses, never by the
  baseline readout.
- **Form.** Prefer a small **RBF network** (smooth by construction, explicit kernel
  width = interpretable bandwidth) or a **Lipschitz-controlled MLP** (spectral-norm /
  gradient penalty + smooth activations + weight decay). The Lipschitz constant is
  the bandwidth analog, `L ≈ 1/h`, so the scale-hierarchy constraint `h > σ_ij`
  becomes `L < 1/σ_ij`.
- **Training data — the current batch's mature anchors, no buffer.** Fit the readout
  online on the mature timesteps (`ysfc > mature_ysfc_threshold`) of the anchor
  pixels **already embedded each batch** — one small NLL step per batch alongside the
  main loss. Per-timestep mature samples make **σ_i pooled** automatically (temporal
  wiggle across mature years + between-pixel spread of mature type-i forests both
  enter the NLL). Data is Virginia (all-East) for now → single threshold ≈ 10–12;
  region-tagged readouts come with the East/West generalization. *(If per-step
  variance is a problem, an optional small **FIFO** replay buffer of recent
  `(z_type, mature x_it)` pairs — a plain queue, not an Algorithm-R reservoir — can
  smooth it; default is no buffer.)*
- **Staleness = parameter lag, handled by warmup.** The readout's weights trail a
  drifting z_type rather than storing stale vectors, and continuously re-fit to
  fresh pairs, so they track a slowly-moving z_type. The existing **phase-loss
  curriculum warmup** now does double duty: it lets the readout **settle** before the
  anomaly-input phase pathway starts learning, and likely needs to be **longer** than
  the current FiLM-stability warmup (μ/σ settling is stricter than z_type merely
  being non-degenerate).
- **Smoothness guard.** μ_θ/σ_θ must stay **slowly-varying functions of z_type**
  (they feed every downstream input; a wiggly μ makes same-type pixels' anomalies
  incomparable and confuses estimator jitter with real anomalies). Knobs: RBF kernel
  width, or the MLP's Lipschitz bound / weight decay / capacity / smooth activations.
  Select by **held-out predictive NLL** on mature pixels (too flexible → overfits
  noise; too stiff → types blur); report the effective Lipschitz constant.
- **Failure-mode watch (the price of going parametric).**
  - *Extrapolation* — the readout can be confidently wrong in low-density z_type
    regions (rare / chronically-disturbed EVT types). Mitigate with low capacity,
    **shrinkage toward a coarse prior** (EVT-class mean, else global mean), or an
    uncertainty head. The **per-EVT coverage diagnostic** watches this.
  - *Silent collapse* — σ_θ→0 or μ_θ ignoring type. Guard by validating **predictive
    NLL / R² on held-out mature pixels** and a σ floor.
- **Scale hierarchy vs. the pairwise comparison scale σ_ij.** The readout's
  characteristic length scale `h` (RBF width, or `1/L` for a Lipschitz-bounded MLP)
  should be **coarser (larger) than the type-similarity comparison scale σ_ij** used
  to weight loss pairs — i.e. `L < 1/σ_ij`. Why: (1) the loss treats pixels within
  σ_ij as same-type/comparable, so the baseline must be ≈ constant across a σ_ij
  neighborhood, else "same-type" pixels get different baselines and their anomalies
  stop being comparable; (2) it gives the smoothness guard headroom. Two caveats:
  - **Different metric spaces today.** The current pair weight is
    `w_ij = exp(−‖spec_i − spec_j‖₂ / sigma)`, `sigma = 5.0`
    (`frl/losses/phase_pairs.py`), defined in **Mahalanobis-whitened spectral**
    units — *not* z_type units. `h` is in **standardized z_type** units. So
    `h > σ_ij` is a principle about relative coarseness **on a shared ruler**, not a
    literal comparison to 5.0. To operationalize it, define the loss's type-
    similarity scale in the **same standardized-z_type metric** as `h` (which we
    want for consistency anyway).
  - **Not too large.** `h ≫ σ_ij` nearly-globalizes the baseline and leaves cross-
    type differences in the anomaly. Target *moderately* larger: pick `h` by the
    held-out-NLL criterion, with `h > σ_ij` only as a lower-bound sanity check.
  - σ_ij belongs to the soon-to-be-retired soft-neighborhood loss, but the scale
    concept carries into the **type∧phase contrastive loss** (its positive/negative
    type-neighborhood scale); the ordering applies against whatever that becomes.
    Reconcile the two scales onto one z_type metric when specing Steps 4–5.

**Diagnostics for Step 1:**
- **Smoothness** — held-out predictive NLL vs. the smoothness knob (RBF width /
  Lipschitz bound); pick the smoothest setting that still tracks known type
  differences; report the effective Lipschitz constant.
- **Per-EVT coverage** — mature-sample count per EVT class and predictive NLL/R²
  stratified by EVT. Flags types where the mature baseline is under-supported (rare
  types, chronically disturbed types) and the readout may be extrapolating.

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
  the bindings feature builder emits for `x_it` (currently z-scored), the Step-1 NLL
  readout must be **fit on `x_it` in those same units**. Otherwise the subtraction
  is in the wrong space. This couples Step 1's fit to the bindings
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
  forward, query the Step-1 **NLL readout** at the anchors' current (detached)
  z_type for μ/σ, then form `a` and Δ. One readout forward per batch — no reservoir,
  no frozen/online split; μ/σ track the current z_type up to parameter lag.
- The phase feature is temporal `[C,T,H,W]` and is consumed only at ~100–300 anchor
  pixels, so it must be built via **`build_feature_at_locations`** at anchor coords
  (per the CLAUDE.md temporal-feature rule — never full-grid, to avoid OOM). μ/σ are
  evaluated at those same anchor coords.
- μ/σ are **stop-grad** (see Step 1): the phase loss shapes the TCN, not z_type,
  through the baseline.

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

## Step 3 — Turn on the anomaly input (procedural)

No new design. Swap the phase encoder input to the Step-2 anomaly + Δ channels,
extend the phase-loss warmup so the Step-1 readout settles first (per "Staleness"),
train, and check against exp034 with the Step-0 eval: "does the anomaly input alone
help?" — *before* adding the new losses. This isolates the input change from the
loss changes so we know which moved the needle.

---

## Step 4 — Slow-feature loss (spec)

Goal: build the **attractor geometry** — within a pixel, z_phase drifts **smoothly**
through stable periods and is allowed to **jump** at disturbance; mature becomes the
dense **basin** that trajectories relax back into. This is the loss that turns
"per-timestep points that move" into a coherent dynamical picture, and it is the
term the old scale-equivariant losses could never provide.

**Core term — a robust / gated temporal-difference penalty on z_phase.**
- Base form: penalize `‖z_phase[t] − z_phase[t−1]‖` within a pixel across time.
- A *bare* squared penalty would forbid the very jumps we want (and see Collapse
  below), so it must be **robust**: either a saturating/Huber ρ that stops growing
  for large steps, **or gated by input abruptness** — weight each transition by
  `w[t] = exp(−‖Δa[t]‖ / τ)` so the smoothness penalty relaxes exactly where the
  **input anomaly** jumps. The gated form is preferred: it is **ysfc-free** (the
  input Δ channel already localizes disturbance onset), and it puts the right
  division of labor in place — the *input* decides **where** a jump happens, the
  *loss* merely **permits** it there and enforces smoothness everywhere else.
- Optional supervision: an `ysfc == 0` mask could instead/also mark jump-allowed
  transitions. This is a *light* ysfc use (a transition selector, like the mature
  selector), not ysfc-as-target — but leans on CCDC's disturbance calls, so keep it
  secondary to the data-driven gate.

**Collapse — slow-feature only makes sense paired with cross-pixel spread.**
`‖Δz‖ → 0` is trivially minimized by a constant embedding (all timesteps, all pixels
equal). Classic SFA adds a unit-variance constraint; here the **Step-5 type∧phase
contrastive loss is the primary anti-collapse** (it repels different types/states),
so Step 4 and Step 5 are complementary and must land together:
- **Slow-feature (Step 4)** = *attract in time* — within-pixel temporal continuity.
- **Contrastive (Step 5)** = *repel across type/state* + align matched states across
  pixels — the spread that prevents collapse and makes mature-A ≈ mature-B for the
  same type.
- If Step 5 lands later, Step 4 needs a temporary explicit variance floor — but on
  the **right population** (across pixels / across the disturbance–recovery axis),
  **not** the flattened N×T timesteps that made the old phase VICReg ineffective.

**What creates the basin (and what does *not*).** We do **not** add an explicit
"pull mature to the origin" term. The mature basin should **emerge**: the Step-2
anomaly makes mature inputs ≈ 0 (dense, similar), disturbance inputs are rare and
distinct, so with temporal smoothness + contrastive alignment the mature timesteps
naturally form the dense attractor and ejecta sit apart. Cross-pixel coincidence of
mature states ("mature-oak-A ≈ mature-oak-B") is delivered by Step 5's same-type/
same-state positives, not by Step 4. Keeping the origin implicit avoids fighting the
FiLM `beta` that places each type's manifold.

**Space & mechanics.**
- Penalize on **post-FiLM z_phase** (the deliverable space). Note FiLM γ,β are
  constant over time for a pixel (z_type is atemporal), so `Δz_phase = γ ⊙ Δh` — β
  cancels in the difference and γ just scales the penalty per type. Acceptable; flag
  if γ-scaling skews the penalty across types.
- Respect the temporal validity mask (skip transitions spanning invalid timesteps).

**Optional companion — within-recovery monotonicity.** After a disturbance,
distance-from-type-origin could be asked to **decrease** with time-since-trough
("wander back"). This is the "ysfc bucket loss → monotonicity prior" from the retire
list — a light **ysfc-as-direction** use (ordering, not target). Adds an explicit
"relax toward maturity" pressure the smoothness term alone doesn't. Keep optional;
add only if trajectories drift but don't reliably return.

**Open decisions (to resolve before implementing):**
- **Gating mechanism** — input-Δ gate (preferred, ysfc-free) vs. robust ρ vs.
  `ysfc==0` mask, or a combination.
- **Anti-collapse** — rely on the Step-5 contrastive spread only, or add an explicit
  correct-population variance floor as a safety net during co-development.
- **Monotonicity companion** — include from the start or hold in reserve.

**Diagnostics (ties to Step 0).** Jump-at-disturbance separation (diagnostic C),
mature-basin formation (within-type mature variance ↓, ejecta separation ↑), and a
collapse check (per-dim z_phase variance on the correct population stays bounded
away from 0).

---

## Build vs. retire

**Build**
1. Type-local mature-baseline μ(z_type), σ(z_type) — an online heteroscedastic
   Gaussian-NLL readout (RBF / Lipschitz-MLP) fit on mature (`ysfc > threshold`)
   timesteps. No reservoir/kNN.
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
- [ ] 1. μ/σ NLL readout: small RBF / Lipschitz-MLP on standardized detached z_type, fit online by heteroscedastic Gaussian NLL on the current batch's mature timesteps (`ysfc > mature_ysfc_threshold`, param ~10–12 East / ~20–25 West); σ via softplus + floor; smoothness knob set by held-out NLL with `L < 1/σ_ij`; check per-EVT coverage / extrapolation. No reservoir.
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
