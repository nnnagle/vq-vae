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
- **Output**: **per-timestep** `z_phase[t] = f(a, Δa)` — a **pure-phase** coordinate. A
  point that drifts toward the **single shared mature origin** (mature `a≈0` anchored to
  `0`) and is kicked out by disturbance. `‖z_phase‖` = departure-from-maturity;
  direction = which tube. **No clustering loss** on disturbance regions (density
  structure is a hypothesis we will *measure*, not impose).
- **Type modulation → none (no FiLM).** Type already enters via the type-conditional
  `μ/σ` input normalization, so output modulation is redundant (`γ` a testable
  fallback). `z_phase` is pure phase; **type lives in `z_type`**; retrieval is on
  **`[z_type, z_phase]`**. (This supersedes the earlier "joint type×phase / FiLM"
  framing.)
- **ysfc**: demoted from loss *target* → mature-set *selector* (and, optionally, a
  weak soft within-recovery drift; see the OU-drift in Step 4).
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
probe **`z_phase[t]`** (pure phase) and the concatenation **`[z_type, z_phase]`** (the
retrieval key); probe **`z_type`** alone for contrast.

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
  and are **stop-grad** w.r.t. the phase loss (the phase loss shapes the encoder, not
  z_type, through the baseline; keeps
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
- **No FiLM** (see Encoder architecture): the anomaly is already type-conditional, so
  `z_phase = f(a, Δa)` is type-conditioned via the input; output modulation is dropped.
  Mature `a≈0 → z_phase ≈ 0` (single shared origin, gauge-anchored).

**Open decisions (to resolve before implementing):**
- **Δ² or Δ-only.** Δ captures onset abruptness; Δ² adds curvature (distinguishes a
  sharp V-bottom from a rounded L-turn) at the cost of noise amplification. Lean
  Δ-only first, add Δ² if the fast/slow separation (diagnostic C/D) is weak.
- **Scaling of the Δ channels.** Δa is already in σ_i units (since `a` is); decide
  whether Δ needs its own robust rescale so its dynamic range matches `a` for the
  TCN, or whether GroupNorm inside the TCN makes that moot.

---

## Encoder architecture — the refactor forces a magnitude-preserving pass

The loss/geometry changes have concrete implications for the *encoder*, not just the
losses. The phase encoder is **`a → TCN → 1×1 bottleneck → z_phase`** — **no FiLM**
(see "No FiLM" below). `z_type` enters the phase pathway **only** through the `μ/σ`
input normalization (Step 2), which is already stop-grad/detached.

**TCN = the lifting map.** Its job is now sharply defined: it convolves over a
**temporal window**, so `z_phase[t]` is a function of a *stretch* of the `(a, Δa)`
trajectory, not one instant. That windowing **is** the Takens lift that makes the
state injective / the flow non-crossing (`s` invertible). Requirements: stay
**bidirectional/acausal** (must see both the pre-disturbance baseline and the forward
recovery to tell "descending" from "recovering"), and the **receptive field must span
enough of the trajectory** to resolve crossings (current `[1,2,4]`+k3 → RF≈15≈T is a
starting point; validate with `Var(Δ|z_phase)`). biGRU is the Step-6 A/B alternative
(its hidden state accumulates the same history).

**Magnitude must survive — depth→radius is now first-class.** Severity = trough depth
= `‖a‖` must become the radius; the encoder must **preserve magnitude
(monotonically/injectively — not necessarily linearly)**. The σ-normalized anomaly is
already the *right*, globally-meaningful normalization (mature ≈ O(1), disturbance
many σ); the encoder must not re-normalize it away. Concrete changes:
- **Remove the (old pre-FiLM) L2-norm.** It puts every timestep on the unit sphere →
  every timestep at the same radius → destroys the radius/progress axis. Mandatory.
- **Drop/replace per-sample normalization.** The current **GroupNorm** on `[N,C,T]`
  normalizes over (channels × **T**) **per sample**, dividing each pixel by its own
  temporal std → a **deep V and a shallow V of the same signature renormalize to the
  same thing** (depth-independent-of-signature is lost; division by a per-sample scale
  is non-invertible). Prefer **no internal norm** (the input is already O(1)-scaled;
  rely on residual connections + init) or a **global/running (batch-statistic) norm**
  that preserves each pixel's relative magnitude. Avoid per-sample/per-timestep norms
  (GroupNorm-over-T, InstanceNorm, LayerNorm-over-T, L2).
- **Optional belt-and-suspenders:** route `‖a[t]‖` (and maybe raw `a[t]`) as a
  **norm-bypassing skip** to the bottleneck, so depth reaches `z_phase` regardless of
  internal normalization. Cheap and robust.

**No FiLM (default) — type conditioning already happens at the input.** FiLM existed
in the original design because the phase input was *globally* z-scored (type-blind), so
type had to be injected at the *output*. Now the input is the **type-conditional
anomaly** `a = (x−μ(z_type))/σ(z_type)`, so `z_phase = f(a, Δa)` is **already a function
of `z_type` through the input**. FiLM would be a *second*, redundant conditioning.
- **`β` (offset) — drop it.** Its job (place types apart so `z_phase` encodes type) is
  gone: `z_phase` is now **pure phase** with a single shared mature origin; type lives
  in `z_type`; retrieval is on `[z_type, z_phase]`.
- **`γ` (per-type reshape) — drop by default; testable fallback.** The type-specific
  recovery *signature* is already in the type-normalized input, so a type-agnostic
  encoder captures it. `μ/σ` normalize only moments 1–2, so if recovery *dynamics* are
  type-specific beyond what the windowed `(a,Δa)` carries, add `γ(z_type)` back (cheap;
  learns ≈1 if unneeded). Test: does the optimal phase geometry vary by type after input
  normalization? Default **no**.
- **Anchor mature to the origin.** A light **anchor loss** `λ·‖z_phase‖²` on mature
  timesteps (`a≈0`, or `ysfc > threshold`) plus a roughly **bias-free** bottleneck
  gauge-fixes mature `→ 0` for all types. This makes `‖z_phase‖` = departure-from-
  maturity and gives the single shared basin.

**Broader lesson:** every per-sample normalization in the phase encoder must be
audited against "does depth/magnitude survive to become radius?" These were harmless
under the old *relative* losses and are harmful now.

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

**Framing: a smoothness constraint on path evolution.** Treat `z_phase[n,·]` as a
**path through embedding space**; this loss penalizes the *kinetic energy* of that
path (how fast the state moves), so the trajectory evolves slowly and continuously.
It is made **edge-preserving in time** — smooth within stable stretches, but preserve
the "edges" that are disturbances — which is the **temporal analog of the spatial
`EdgeAwareSmoothingConv2D`** already used in the type pathway (smooth within regions,
preserve edges/corners). Purpose (see the "why" for each in the discussion): a tight
mature **basin** (noise suppressed), **connected, ordered recovery pathways** (a
usable "how far along" coordinate), and the **relaxation dynamics** ("wander back").

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

**Recommended mathematical form.** Gated kinetic energy of the path. For pixel `n`,
transition `t = 1…T−1`:

```
velocity    v[n,t] = z_phase[n,t] − z_phase[n,t−1]
input jump  g[n,t] = ‖a[n,t] − a[n,t−1]‖₂            # over anomaly channels (Step-2 Δ)
gate        w[n,t] = exp(−(g[n,t]/τ)²) · valid[n,t]·valid[n,t−1]   ∈ (0,1]

L_sf = ( Σ_{n,t} w[n,t] · ‖v[n,t]‖² ) / ( Σ_{n,t} w[n,t] )
```

`w ≈ 1` in stable stretches (smooth hard), `w → 0` at disturbance onset (jump free).
Choices, with rationale:
- **L2 (kinetic energy), not L1 (total variation).** L2 spreads change over many
  small steps → *gradual* recovery (the relaxation limb we want); TV-L1 concentrates
  change into sparse jumps + flat plateaus, which would collapse recovery into one
  step. The **sharp onset is handled by the gate**, not the penalty shape, so L1's
  edge-seeking is unnecessary.
- **First-order (velocity), not second-order (acceleration).** We want "prefer to
  stay put in the basin" = small velocity; a curvature penalty would instead permit
  constant-velocity drift.
- **Soft exp gate, not a hard threshold** — differentiable, partial credit for
  medium changes. `τ` = "how big an input change counts as an event"; set from a
  robust scale of `g` (e.g. median `‖Δa‖` over stable transitions), or tune.
- **Optional robust safety net** — replace `‖v‖²` with a *saturating* `ρ(‖v‖)`
  (Welsch/Cauchy — bounded; not Huber, which keeps growing) so a jump the gate
  *misses* isn't over-penalized. Likely unneeded with a good input gate; keep in
  reserve.

**Directed refinement — the OU-drift (recommended, promoted from reserve).** The
smoothness term above is *undirected* (small step, any direction). Add a **soft,
gated, directed** term: on non-disturbance transitions, discourage the radius from
*increasing*:
```
L_drift = mean over gated transitions of  softplus( ‖z_phase[n,t]‖ − ‖z_phase[n,t−1]‖ )
```
(low weight; same gate `w`). This is the explicit OU inward-drift, and it is the
**primary guard for the recovery ORDERING** (fresh→mid→mature = far→near), *robust to
`k_flow` mis-scaling* (see the worked example). Keep it **soft** (a tendency, not a
pin) so non-monotone recovery is allowed. Division of labor: the **anchor loss** pins
mature at the origin; **`‖a‖` + magnitude-preserving encoder** set how far out the
disturbance kick lands (severity); the **drift** orders the return; **`k_flow`
scaling** is now *secondary* (a mistake is recoverable, not catastrophic). It must be
paired with the outward forces (kicks + contrastive) or it collapses everything to `0`.

**Worked example (fresh-burn C / slow-recovery B / mature A, same type).** With
`A=(a≈0,Δa≈0)`, `C=(a large,Δa≈0)`, `B=(a moderate,Δa<0)`, we want `d(A,C)` largest (B
between A and C). Note `Δa≈0` at *both* A and C, so a `Δa`-dominant metric would
collapse A≈C and push B out — the inversion. What prevents it: **`‖a‖` makes A,C the
far-apart extremes** (magnitude-preserving encoder; `a` weighted ≥ `Δa`), and the
**OU-drift + continuity chain** thread the ordered ray C→B→A so B sits between them.
Distinguishing recovering-B from mature-A when their instantaneous states nearly
coincide is the **window's** job (it still sees the burn C came through — "the only way
from A to C is through B"). The tube **dissolves into the basin** once B recovers
enough that the burn leaves the window. So the ordering rides on **drift + magnitude +
window**, with `k_flow` scaling secondary.

**Collapse — slow-feature only makes sense paired with cross-pixel spread.**
`‖Δz‖ → 0` is trivially minimized by a constant embedding (all timesteps, all pixels
equal). Classic SFA adds a unit-variance constraint; here the **Step-5 type∧phase
contrastive loss is the primary anti-collapse** (it repels different types/states),
so Step 4 and Step 5 are complementary and must land together:
- **Slow-feature (Step 4)** = *attract in time* — within-pixel temporal continuity.
- **Contrastive (Step 5)** = *repel across state* + align matched *disturbed* states
  across pixels — the spread that prevents collapse. (Mature coincidence is handled by
  the anchor loss: all mature → the shared origin.)
- If Step 5 lands later, Step 4 needs a temporary explicit variance floor — but on
  the **right population** (across pixels / across the disturbance–recovery axis),
  **not** the flattened N×T timesteps that made the old phase VICReg ineffective.

**What creates the single mature origin.** The **anchor loss** (`λ·‖z_phase‖²` on
mature timesteps; see Encoder architecture) gauge-fixes mature → `0`. Beyond that the
basin *fills in*: the Step-2 anomaly makes mature inputs ≈ 0 (dense, similar), so with
temporal smoothness + the OU-drift + contrastive alignment the mature timesteps cluster
tightly at the origin and ejecta sit far out on rays. Cross-pixel coincidence of mature
states is delivered by the anchor (all mature → `0`) plus Step-5 positives, not by
Step-4 smoothness alone.

**Space & mechanics.**
- Penalize on **`z_phase`** (there is no FiLM; `z_phase` is the bottleneck output).
- Respect the temporal validity mask (skip transitions spanning invalid timesteps).

**The monotonicity companion is now the OU-drift** (above), promoted from reserve and
kept **soft**. No separate hard monotonic term — a hard "radius→0" would kill
non-monotone recovery.

**Open decisions (to resolve before implementing):**
- **Gating mechanism** — input-Δ gate (preferred, ysfc-free) vs. robust ρ vs.
  `ysfc==0` mask, or a combination.
- **Anti-collapse** — rely on the Step-5 contrastive spread only, or add an explicit
  correct-population variance floor as a safety net during co-development.
- **OU-drift weight** — the soft inward-drift is now **in** (not reserve); the open
  choice is its weight relative to the smoothness term (strong enough to order the
  recovery, soft enough to allow setbacks).
- **Robust safety net** — whether to wrap the velocity term in a *saturating*
  `ρ(‖v‖)` (Welsch/Cauchy — bounded; not Huber) as a backstop for disturbance jumps
  the input gate misses, or rely on the gate alone. Default: gate alone, add the
  saturating `ρ` only if un-gated jumps show up as over-penalized (recovery limbs
  looking artificially smeared, or the loss spiking at mislabeled/undetected
  disturbances). Costs one scale hyperparameter (the saturation radius).

**Diagnostics (ties to Step 0).** Jump-at-disturbance separation (diagnostic C),
mature-basin formation (within-type mature variance ↓, ejecta separation ↑), and a
collapse check (per-dim z_phase variance on the correct population stays bounded
away from 0).

---

## Step 5 — Type∧phase contrastive loss (spec)

Goal: the metric itself. A sample is `(pixel n, timestep t) → z_phase[n,t]` (pure
phase; the retrieval key is `[z_type, z_phase]`). Two samples should be **close iff
they match on *both* type and phase** — this is what makes kNN in `[z_type, z_phase]`
return "same kind of forest, same recovery stage." This loss also **sets z_phase's
scale** and so is the **anti-collapse partner** for Step 4 (which is scale-free on its
own).

### The crystallized geometry (what z_phase *is*)

Think of recovery as a **dynamical system**. Every pixel-time sits either in the
**mature attractor basin** or on one of several **tubes** (recovery pathways ≈
disturbance agents) that a disturbance kicked it onto and down which it relaxes back
to the basin.

> **ARCHITECTURE (current):** `z_phase = f(a, Δa)` — a **type-agnostic encoder** on
> the **type-conditioned** input. **No FiLM** (see "Encoder architecture"): type
> already enters via the `μ/σ` input normalization, so output modulation is redundant.
> `z_phase` is **pure phase** with a **single shared mature origin** (mature anchored
> to `0`); type is carried by `z_type`; retrieval is on **`[z_type, z_phase]`**. This
> supersedes the earlier "joint type×phase / per-type FiLM basins" framing.

- **z_phase is a *lifted, linearizing* phase space.** We choose coordinates in which
  the (curved, channel-space) recovery flow becomes a simple **radial contraction
  toward the origin** (mature): `E[z_{t+1}] ≈ ρ·z_t`, `ρ<1`. All the
  differential-channel-rate **curvature is absorbed into the encoder**; in z_phase the
  tubes are approximately **straight rays out of the origin**.
- **A single shared mature origin — mature `a≈0` is gauge-anchored to `0`.** All types'
  maturity coincides at the origin (a light **anchor loss** pulls mature `z_phase → 0`;
  see Encoder architecture). So `z_phase` is **pure phase**: two pixels of *different*
  types at the *same* type-normalized state map to the *same* `z_phase` — "same phase,"
  with type distinguished by `z_type`. This gives the **tightest mature cluster** (one
  point, not a per-type manifold) and makes `‖z_phase‖` mean exactly "departure from
  maturity." Type is continuous, but it lives in `z_type`, **not** as structure in
  `z_phase` (see the tight-maturity note below).
- **Read-out:** **direction** of `z_phase` = **which tube** (globally valid because the
  ray is straight); **radius** `‖z_phase‖` = **progress/severity** along it; **type** =
  `z_type`. Retrieve on `[z_type, z_phase]` (weight type vs. phase as the query needs).
- **Same-phase-different-type coincidence is fine.** Because tubes fan from one origin,
  a fresh-burn-pine and fresh-burn-oak at the same type-normalized state land near each
  other in `z_phase` (both "fresh burn") — correctly, for *phase*; `z_type` separates
  them. Track same-vs-different-type behavior via the retrieval diagnostics.
- **Non-crossing = the sufficiency criterion.** A flow is a well-defined function
  only if trajectories don't cross; they *do* cross in raw channel space (same anomaly
  can be descending vs. recovering vs. a different tube), so instantaneous `a` is an
  insufficient state. The fix is to **lift the state** with velocity/history until it
  self-unfolds (Takens delay-embedding). Practical target/diagnostic:
  **`Var(Δ | z_phase-neighborhood)` should be small** — same-state points share a
  future ⇒ the tubes are resolved ⇒ z_phase carries enough context/dimension.
  (`Var` large ⇒ z_phase is still overlaying distinct trajectories; add context/dim.)
- **The kicks are exogenous.** Disturbance onsets are stochastic shocks, *not* part
  of the drift field — allowed discontinuities, held out of the flow by the Step-4
  gate. Tube identity properly **dissolves into the basin** near maturity (a recovered
  forest looks mature regardless of past agent), matching reality.

**Attractor profile — OU-like, mean-reverting to the origin.** The radius should pull
**strongly toward `0` far out** but only **weakly near the origin**, where residual
fluctuation takes over — a mean-reverting (Ornstein–Uhlenbeck) process.
- **Radial contraction is the drift.** `E[z_{t+1}] ≈ ρ·z_t` (`ρ<1`) makes the
  *absolute* inward step `(1−ρ)·‖z‖` large at large radius and vanishing near the
  origin — "strong at first, gentle near the center."
- **Made explicit as a loss (the OU-drift, promoted from "reserve").** On **gated
  non-disturbance transitions**, softly encourage the radius *not to increase* —
  `softplus(‖z_{t+1}‖ − ‖z_t‖)`, low weight, same gate as Step-4 smoothness. This is
  the **primary guard for the recovery ORDERING** (fresh→mid→mature = far→near), robust
  to `k_flow` mis-scaling (see the worked example); it replaces "hope the observable
  distance orders the tube." Keep it **soft** (a tendency, not a pin) so non-monotone
  recovery is allowed; the **anchor loss** (not the drift) pins mature at the origin.
- **Anti-collapse.** Pure inward drift is minimized by `z ≡ 0`; the **disturbance kicks
  + contrastive repulsion** are the outward forces that balance it (OU = inward drift ×
  outward kicks/noise). Ship the drift only with them.
- **Maturity stays a tight point, not a fuzzy ball — by choice.** Earlier we let the
  σ-noise floor make the basin a fuzzy ~1σ ball; the tight-maturity decision (below)
  moves the within-mature *structural* spread into `z_type`, so what's left at the
  origin is just denoised weather wiggle. So the basin is **as tight as `z_type` is
  complete** — see the tight-maturity note.

### Tight maturity — structural spread lives in `z_type`, not `z_phase`

We want a **tight** mature cluster for stratification, *not* a fuzzy ball — much of the
mature spread is ecological noise we don't want influencing the phase ordering.
Resolution: the two flavors of mature spread go to different places.
- **Structural/ecological diversity** (dense vs. sparse mature, site) → **`z_type`**.
  The type-conditional `μ/σ` are the mature baseline *for that type*, so a structurally
  off-mean mature pixel still has `a≈0` (departure from its *own* baseline) → it lands
  at the origin. The variation isn't discarded — it's **relocated to `z_type`** (and
  available via `[z_type, z_phase]` retrieval). This is the right home for it: dense vs.
  sparse mature is a *type* distinction, not a *phase* one.
- **Temporal weather wiggle** → **suppressed** by Step-4 smoothness.
- So the mature basin is **tight**, and its residual size is a **diagnostic on `z_type`
  completeness**: a fuzzy mature basin means `z_type` is under-capturing structure that
  is leaking into `a`.
- **Knobs (primary → secondary):** `z_type` expressiveness (the real lever); Step-4
  smoothness; the anchor loss + `σ_flow` at the basin. Because the structure you'd
  "erode" is safely in `z_type`, you may contract mature **hard** (symmetric contraction
  / larger `σ_flow` near the origin is *fine* here — the earlier "one-sided only /
  don't collapse the ball" caution is lifted, since we now *want* a tight point).
- **Don't over-tighten past the noise floor:** the σ-normalization means departures
  *smaller* than mature ecological variation are undetectable — appropriate (you can't
  see a disturbance smaller than natural variation). A basin tighter than that is
  meaningless.

### CRITICAL: closeness-for-the-loss ≠ the emergent geometry

The radial direction=tube / radius=progress structure is what z_phase **emerges
into**; it is **not** what we compute closeness from (using z_phase's own direction to
define its positives is circular and collapse-prone). **Loss closeness is computed
from *observables*:** the type embedding and the **observable flow-state**
`s = (a, Δa)` (anomaly level + velocity) — the observable shadow of the radial state.
`(a, Δa)` is precisely the lifted state the non-crossing argument demands. The
encoder's job is to reproduce that observable closeness as a clean, absolute, radial
Euclidean metric.

**Tubes are built by chaining, not by a single term.** `k_flow` rewards *same
location on the same flowline* (same `a`, same `Δa`); "same tube, different progress"
is **not** a direct positive (different `a`). It emerges transitively: cross-pixel
local positives (`B@p1 ≈ A@p1`) + Step-4 within-pixel continuity (`A@p1 ≈ A@p2`) ⇒
`B@p1 ≈ A@p2`. This is why **Steps 4 and 5 are inseparable** — contrastive supplies
the cross-pixel rungs, continuity the vertical rails; only together do they forge the
tube along which "same-tube > different-tube" holds.

**Form — soft-supervised InfoNCE, Euclidean, fixed temperature.** One kernel drives
everything:

```
d_type(i,j)  = ‖ẑ_type_i − ẑ_type_j‖         # standardized z_type (per-pixel)
s_i          = ( a[n,t], Δa[n,t] )            # OBSERVABLE flow-state: level + velocity; NOT ysfc, NOT z_phase
d_flow(i,j)  = ‖s_i − s_j‖                    # same place, moving the same way
p_ij = exp(−d_type²/2σ_type²) · exp(−d_flow²/2σ_flow²)        # the AND (type ∧ flow-state)

ℓ_ij = −‖z_phase_i − z_phase_j‖² / τ_phase   # Euclidean, FIXED τ
L_i  = − Σ_{j∈pos(i)}  log[ exp(ℓ_ij) / ( exp(ℓ_ij) + Σ_{k∈neg(i)} w_neg(ik) · exp(ℓ_ik) ) ]
```

### Why InfoNCE and not the soft-neighborhood loss used before (honest version)

`KL(p‖q)` and InfoNCE cross-entropy share the *same gradient* given the same target
and candidate set, so "KL vs InfoNCE" is **not** the real distinction. What gives a
loss collapse-resistance and an *absolute, shippable metric* is three properties:
**(1) explicit repulsion** (real negatives pushed apart), **(2) a peaked target**
("the positive is strictly closest", not "reproduce a graded neighborhood"), and
**(3) a fixed temperature as an absolute ruler**. The old `soft_neighborhood`
matched a *diffuse, normalized* target with no explicit repulsion, so a globally
**contracted** embedding satisfies it (all `d→ε` ⇒ `q→uniform ≈` a diffuse `p`) and
absolute distance is never supervised — which is exactly why
`phase_recovery_discrimination_loss` (an absolute margin) had to be bolted on. We
ship a **metric** (kNN distances for post-stratification), so absolute distance must
be pinned; only (1)+(2)+(3) do that.

**Caveat that this forces on us:** soft-supervised InfoNCE is only collapse-resistant
if the **positive target stays peaked and the negatives stay real**. If `σ_type`,
`σ_flow` are too large, the positive set becomes diffuse and this degenerates back
into soft-neighborhood. So the three ingredients below are non-negotiable.

### Getting the three ingredients

**(1) Peakedness — few, confident, AND-gated positives.**
- Positives = candidates that are type-close **∩** state-close: top-k / mutual-kNN by
  `p_ij`, capped at a few per anchor (reuses the spectral mutual-kNN positive logic
  with a two-factor gate). Selecting a *small set* guarantees peakedness rather than
  hoping a soft kernel stays sharp.
- A few reliable cross-pixel positives per step, over many steps, align the whole
  same-state population by transitive closure — peaked ≠ under-aligned.
- **Free positives: within-pixel adjacent timesteps** (same pixel ⇒ same type;
  adjacent stable ⇒ same state). Reliable but **must not dominate**: weight
  cross-pixel positives **≥** within-pixel, or the loss is minimized by making each
  pixel a smooth *private* curve with no shared basin — the old "encodes pixel
  identity" failure.
- `σ_type`, `σ_flow` are the peakedness knobs; calibrate from the distance
  distributions so "same" is genuinely narrow.

**(2) Negatives — the same-type/different-state set is the whole game.**
- Base pool: random cross-batch samples (reuse `cross_phase_*`) → general spread.
- **Hard negatives = same type, different state** — the set that makes recovery
  stages metrically separable *within* a type (the `phase_recovery_discrimination`
  job, done natively). **Must be actively mined/quota'd**: same-type samples are a
  minority of a random pool, so without mining, trivial different-type negatives
  swamp the gradient and phases quietly re-compress. Guarantee each anchor's negative
  set includes its type-neighbors (high `k_type`) that are state-far (low `k_flow`).
- **False-negative suppression** via the same kernel: `w_neg(ik) = (1 − p_ik)`
  clamped (your spectral `1 − exp(−d/σ)` idiom) — an accidental same-type/same-state
  pair gets ≈0 negative weight. One kernel gives positives (high `p`) and negative
  weights (`1 − p`).

**(3) Fixed ruler.**
- **Fixed `τ_phase`** (never learned) = the absolute ruler. Calibrate via a new
  **"Phase sims gap/T"** diagnostic (pos vs neg similarity gap over τ), kept in the
  healthy `~2–3` band, mirroring the spectral calibration. Collapse → uniform
  softmax → loss `= log K` (the maximum), so fixed τ forces a characteristic
  positive-vs-negative distance.
- **Euclidean-squared** similarity, not cosine — the ruler must measure *radial*
  distance (from the shared origin) so severity = `‖z_phase‖` survives. Consistent
  with Step 4 (`‖Δz‖²`).
- **Variance-floor backstop** — light VICReg-style per-dim std hinge on the
  **correct population** (across pixels / disturbance–recovery axis, *not* flattened
  N×T). Bites only below the floor; insurance against τ mis-calibration.

### Scale reconciliation (closes the earlier debt)
`σ_type` is the type-similarity scale = the `σ_ij` we owed, now on the standardized
z_type metric. All scales land on one ruler: readout bandwidth `h` (coarsest) >
`σ_type` — i.e. `h > σ_ij`.

### Mechanics / reuse
Head-free directly on z_phase (deliverable key; consistent with the head-free z_type
stance). Reuses cross-batch phase pooling, mutual-kNN positive selection, distance-
weighted negatives, and the `gap/T` temperature calibration already in the codebase.

### Open decisions
- **Flow-state `s`** — settled as the **observable flow-state `(a, Δa)`** (the lifted,
  non-crossing state). Remaining knob: window length — just `(a, Δa)`, or add `Δ²a` /
  a slightly longer window for a more robust flow tangent; long enough to resolve
  crossings (small `Var(Δ|z_phase)`), short enough to stay local. Type-vs-flow weight
  (`σ_type` vs `σ_flow`).
- **Similarity** — negative-squared-Euclidean (recommended, radial) vs. dot-product
  (existing z_type idiom).
- **Anti-collapse** — rely on fixed-τ InfoNCE alone vs. always-on variance floor.
- **Positive kernel sharpness** — soft top-k weights vs. hard positive selection;
  and the cross-pixel-vs-within-pixel positive weighting.
- **Hard-negative mining quota** — how many same-type/different-state negatives per
  anchor to guarantee.
- **Fallback** — if the soft product-kernel is fiddly, discretize into
  (type-cluster × coarse-anomaly-state) buckets and run plain SupCon (cruder;
  bucketing reintroduces a whiff of ysfc-style discretization).

### Diagnostics (ties to Step 0)
Same-stage retrieval AUC (diagnostic A/E), within-type phase separability (can a
probe read recovery stage off z_phase — the metric the old model failed), the
"Phase sims gap/T" calibration line, the collapse check shared with Step 4, and the
**flow-functional / non-crossing check `Var(Δ | z_phase-neighborhood)`** — small ⇒
tubes resolved and z_phase carries enough context; large ⇒ distinct trajectories are
being overlaid, needs more history/dimension.

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
6. Magnitude-preserving, FiLM-free encoder pass: **remove FiLM** (`z_phase = f(a,Δa)`;
   `γ` a testable fallback), remove the old L2-norm, drop/replace per-sample GroupNorm,
   optional `‖a‖` norm-bypass skip; keep TCN bidirectional + RF ≥ window.
7. Anchor loss `λ·‖z_phase‖²` on mature timesteps → single shared origin; `[z_type,
   z_phase]` as the retrieval key.
8. OU-drift: soft, gated, one-sided radius-non-increase term (Step-4 directed
   refinement) — primary recovery-ordering guard.

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
- [ ] 3b. FiLM-free, magnitude-preserving encoder (see "Encoder architecture"): **remove FiLM** (`z_phase = f(a,Δa)`), add the **anchor loss** `λ·‖z_phase‖²` on mature → single shared origin, remove old L2-norm, drop/replace per-sample GroupNorm (→ none/global), optional `‖a‖` norm-bypass skip; verify depth→radius survives (deep vs shallow V separable in `‖z_phase‖`) and mature→origin. Alongside Step 3.
- [ ] 4. Slow-feature/smoothness loss **+ soft gated OU-drift** (radius-non-increase); confirm drift-to-basin, ordered recovery ray, jump-at-disturbance.
- [ ] 5. Type∧phase contrastive loss on `[z_type, z_phase]`; retire soft_neighborhood + recovery-disc + phase VICReg; reconcile σ_ij onto the z_type metric; re-run eval.
- [ ] 6. (optional) Encoder A/B: biGRU vs TCN for "where am I in the trajectory," with Δ channels.
- [ ] 7. Consolidate: CLAUDE.md (architecture + loss table + ysfc-as-selector reframing), diagnostics, config plumbing.

## Open data dependencies

- **Change-agent labels** for diagnostic D — source is **LCMS**; deferred, add as a
  bindings source and join to anchors later.
- **FIA attribute join + plot-location extraction** for diagnostic E — machinery
  exists in `frl/analysis/` and `embed_locations.py`; confirm current-model wiring.
- **`mature_ysfc_threshold`** — region-dependent input parameter (~10–12 East /
  ~20–25 West); sweep it, and eventually set per-region.
