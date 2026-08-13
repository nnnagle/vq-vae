"""Step-0 phase-pathway evaluation harness (diagnostics A–C).

See ``docs/phase_rethink_design.md`` (Step 0). Diagnostics:

* **A** — reconstruction probes (``reconstruction.py``): total *and* within-pixel
  variance explained for ``z_phase`` (contrast: pre-FiLM ``h``, ``z_type``).
* **B** — type-conditional recovery curves (``recovery_curves.py``).
* **C** — disturbance-produces-ejection (``ejection.py``).

``run_eval.py`` produces one ``metrics.json`` per checkpoint; ``compare_eval.py``
diffs a new model against the exp034 baseline. Diagnostics D (change-agent
clustering; needs LCMS in the zarr) and E (FIA kNN) are deferred follow-ups.
"""
