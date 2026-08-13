# Step-0 eval findings (exp035) and next steps

This records the Step-0 phase-eval results on **exp035** (checkpoint
`encoder_best_1_epoch_380.pt`) and what the numbers tell the phase-pathway rethink
(`phase_rethink_design.md`, Steps 1–5). The harness lives in
`frl/training/phase_eval/`; run/compare commands are in `CLAUDE.md`
(Step-0 Phase-Pathway Eval Harness).

exp035 is the first run on the **fixed feature-normalization pipeline** (the
log/sqrt-transform bug that had killed `spectral_velocity`, `variance_*`, etc. —
see `feature_builder._apply_normalization` + `test_feature_normalization.py`). The
Step-0 numbers below are also the first computed with the **fixed ridge scaling**
and the **variance-weighted within-R²**, with `temporal_position` excluded from
Diagnostic A.

## Important eval caveats (read before trusting old numbers)

Three harness bugs were found and fixed *while* exp035 was being evaluated. Any
metrics.json produced before these fixes is not comparable:

1. **Feature normalization** (training + eval): log/sqrt channels were z-scored
   against transform-scale stats → collapsed to a constant. Fixed. Requires
   **retraining** to benefit (exp035 is the first fixed run).
2. **Ridge λ not scaled by M** (eval only): the λ sweep was flat because the
   summed normal equations swamped `λ≤1`. Pre-fix within-R² numbers were
   *unregularized OLS*, which badly overfits the small within-pixel slice and
   reports spuriously **negative** within-R². The "z_phase within-R² = −0.28"
   from the first exp035 eval was this artifact; properly regularized it is
   **positive**.
3. **Unweighted mean of per-channel within-R²** (eval only): dominated by
   near-dead low-variance channels posting wild negatives. Replaced by the
   variance-weighted aggregate.

**Consequence:** the exp034 baseline metrics.json on hand was produced under the
old (unregularized, unweighted, temporal_position-in) harness. For a clean
apples-to-apples comparison, **re-run Step-0 on the exp034 checkpoint** with the
current harness before quoting exp034→exp035 deltas.

## Diagnostic A — reconstruction (the headline)

Weighted within-pixel R² on **test**, 12 spectral/forest channels
(`temporal_position` excluded):

| source | within-R² (ridge) | within-R² (MLP) | note |
|---|---|---|---|
| z_type (control) | 0.000 | 0.000 | atemporal → exactly 0 by construction ✓ |
| **z_phase** (post-FiLM) | **~0.28** | **~0.37** | the shipped embedding |
| **h** (pre-FiLM bottleneck) | **~0.66** | **~0.75** | the FiLM-free ceiling |

(`temporal_position` included, for reference: z_phase 0.336/0.418, h 0.661/0.764.
It reconstructs at ~0.92 in z_phase — trivially, it is the year index — so it was
inflating z_phase's aggregate by ~22%. Excluded going forward.)

**The central result: the h → z_phase within-R² gap is ~0.38.** h and z_phase are
**both 8-dim** (FiLM is a per-channel affine `γ⊙h+β`, so no dimensionality change).
The pre-FiLM trunk holds ~0.66 of the within-pixel temporal signal; the post-FiLM
embedding keeps only ~0.28. FiLM's **type-conditional gain** (γ→~3.5, varies by
z_type) scrambles the h→x mapping so no single linear map from z_phase recovers it.
That is the entanglement the rethink exists to fix — now measured, not asserted.

Per-channel (ridge within-R², h → z_phase):

- **spectral_velocity: 0.606 → 0.605** — the derivative channel **survives FiLM
  intact**.
- Raw bands get compressed ~½–¾: nbr 0.83→0.39, ndmi 0.80→0.46, swir2 0.80→0.32,
  ndvi 0.82→0.24, red 0.72→0.08.
- **nir fails in BOTH** (h −0.39, z_phase −0.44) — upstream/trunk, **not** a FiLM
  problem.
- `seas_amp_*` fail broadly but are low-within-variance, so the weighted aggregate
  correctly discounts them.

MLP ≈ ridge (+~0.08–0.10 only): the within-pixel signal is **mostly linearly
accessible**, so the gap is real information loss, not a probe-capacity artifact.

## Diagnostic B — recovery curves

- **type-phase** (`[z_type, z_phase]`) is clearly better: mean shape-agreement
  **0.975** vs phase-only **0.834** (20 EVTs). z_type supplies the type-specific
  recovery **baseline** that z_phase alone lacks (answers the earlier "curves too
  high / missing intercept" — the fix is a z_type-conditioned intercept, not a
  per-EVT one; EVT stays diagnostic-only).
- ⚠️ This exp035 run predates the **B λ-scaling fix** (B fits its own ridge; it was
  flat at λ=0 too). Re-run B; the phase-only design especially may improve once
  ridge engages, and the phase-only↔type-phase gap may narrow.

## Diagnostic C — ejection

⚠️ **C is known-suspect in this run — do not cite these numbers.** Rerun pending
(a labeling/units issue in the disturbance-year jump). Recompute before use.

## What the next steps should target

The properly-measured picture is more optimistic and more actionable than the
pre-fix "z_phase collapsed" story:

- **z_phase does carry real within-pixel temporal signal (~0.28 weighted).** Not a
  collapse. But it sheds ~half of what the trunk holds.
- **The ceiling is h ≈ 0.66.** Closing the ~0.38 h→z_phase gap is the concrete
  Step-4/5 objective: recover the **band-trajectory** within-signal (nbr/ndmi/swir)
  that h has at ~0.8 and z_phase drops to ~0.35. spectral_velocity already
  survives — the *change/derivative* geometry is intact; the *level trajectory* is
  what FiLM compresses.
- **The mechanism is FiLM's type-conditional multiplicative gain**, not a capacity
  bottleneck (same 8 dims). Steps 4–5 (slow-feature loss + type∧phase contrastive,
  retiring soft_neighborhood/recovery-disc/phase-VICReg) should be judged by
  whether post-FiLM z_phase within-R² moves toward h's ~0.66 while the recovery
  shape-agreement (B) and ejection separation (C) hold or improve.
- **nir** within-pixel reconstruction fails pre-FiLM too — a trunk/encoder issue
  worth a separate look (Step-6 encoder A/B, biGRU vs TCN, is the natural place).

### Concrete before starting Step 1
1. **Re-run Step-0 on the exp034 checkpoint** with the current harness → the real
   baseline. Then re-run exp035 B (and C once fixed) so all of A/B/C are on the
   fixed harness.
2. Diagnostic **A anomaly target** (`AnomalyTargetProvider("mature_baseline")`) is
   the **Step-1 seam** — it raises `NotImplementedError` until the μ/σ estimator
   exists. Wire it when Step 1 lands.
3. Diagnostic **D** (change-agent clustering) is scaffolded (`lcms_agents.py`) but
   needs the LCMS change layer validated in the v2 cube (now built into
   `zarr_v2`). Diagnostic **E** (FIA kNN) reuses `embed_locations.py` + the R
   notebooks.
