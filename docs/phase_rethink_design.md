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
- [ ] 1. Offline mature-baseline estimator μ/σ using a *frozen* exp034 z_type, mature set = `ysfc > mature_ysfc_threshold` (param; ~10–12 East / ~20–25 West); check smoothness + per-EVT coverage.
- [ ] 2. Anomaly input builder `(x−μ)/σ` + Δ/Δ²; verify mature≈0, fast vs slow disturbances distinct.
- [ ] 3. Stage-A retrain (phase-only, frozen z_type) on anomaly input; "does the input alone help?"
- [ ] 4. Slow-feature/smoothness loss; confirm drift-to-basin + jump-at-disturbance geometry.
- [ ] 5. Type∧phase contrastive loss; retire soft_neighborhood + recovery-disc + phase VICReg; re-run eval.
- [ ] 6. (optional) Encoder A/B: biGRU vs TCN for "where am I in the trajectory," with Δ channels.
- [ ] 7. (only if warranted) Co-trained EMA/target-network μ/σ with smoothness regularization.
- [ ] 8. Consolidate: CLAUDE.md (architecture + loss table + ysfc-as-selector reframing), diagnostics, config plumbing.

## Open data dependencies

- **Change-agent labels** for diagnostic D — source is **LCMS**; deferred, add as a
  bindings source and join to anchors later.
- **FIA attribute join + plot-location extraction** for diagnostic E — machinery
  exists in `frl/analysis/` and `embed_locations.py`; confirm current-model wiring.
- **`mature_ysfc_threshold`** — region-dependent input parameter (~10–12 East /
  ~20–25 West); sweep it, and eventually set per-region.
