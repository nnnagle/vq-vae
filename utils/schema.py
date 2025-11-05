"""
utils/schema.py
----------------
Build and persist the feature schema used by the loader and trainer:
exposure-aware categorical vocabularies and continuous normalization stats.

Purpose
    Provide a single source of truth (“schema”) that the Dataset uses to:
      • collapse categorical raw codes to dense, consecutive IDs per feature
      • record exposure policy (batch_size, steps_per_epoch, min_hits, coverage, caps)
      • attach continuous stats (mean/std and q01/q99) for normalization/denorm
    The schema depends on training knobs, not on Zarr structure. Keep it small,
    explicit, and reproducible.

Exposed API
    build_categorical_schema(feature_meta, batch_size, steps_per_epoch,
                             min_hits_per_epoch=100, mass_coverage=0.999,
                             vocab_cap=5000) -> dict
        - Constructs per-feature vocabularies with exposure thresholds and optional
          mass coverage/cap. IDs are dense and per-feature: 0=MISS, 1=UNK, 2..=kept codes.

    attach_continuous_stats(schema, feature_meta) -> dict
        - Adds per-feature continuous stats: {mean, std, q01, q99}.

    save_schema(schema, path) -> None
    load_schema(path) -> dict

Schema shape (selected keys)
    {
      "policy": {
        "batch_size", "steps_per_epoch", "min_hits_per_epoch",
        "mass_coverage", "vocab_cap", "miss_id", "unk_id", "N_epoch"
      },
      "categorical": {
        "<feature>": {
          "num_ids", "kept_codes", "code2id", "id2code",
          "min_count", "coverage", "counts_per_id"
        }, ...
      },
      "continuous": {
        "<feature>": {"mean","std","q01","q99"}, ...
      }
    }

Conventions
    • MISS_ID = 0, UNK_ID = 1 (reserved across the codebase).
    • Categorical IDs are dense per feature; do not share mappings between features.
    • Continuous normalization elsewhere uses clip[x ∈ (q01,q99)] then z-score.
      Denormalization uses x ≈ z*std + mean (no re-clip).

Design notes
    • Exposure threshold: keep codes expected to appear ≥ min_hits_per_epoch given
      N_epoch = batch_size * steps_per_epoch.
    • Deterministic: kept codes sorted numerically; mappings are reproducible.
    • Counts are stored aligned to IDs (index 0..num_ids-1) for fast weighting.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple, Optional, Any

MISS_ID, UNK_ID = 0, 1


def _exposure_min_count(total: int,
                        batch_size: int,
                        steps_per_epoch: int,
                        min_hits_per_epoch: int) -> int:
    """
    Convert a per-epoch exposure target into an absolute count threshold.

    Let N_epoch = batch_size * steps_per_epoch be the number of samples the
    model sees per epoch. If we want a category to appear at least r times per
    epoch on average, its empirical proportion p must satisfy:
        p >= r / N_epoch
    => min_count = ceil(p * total) = ceil( (r / N_epoch) * total ).
    """
    if total <= 0:
        return 0
    n_epoch = max(1, batch_size * steps_per_epoch)
    p_min = min_hits_per_epoch / n_epoch
    return int(math.ceil(p_min * total))


def build_categorical_schema(
    feature_meta: Dict[str, Any],
    batch_size: int,
    steps_per_epoch: int,
    min_hits_per_epoch: int = 100,
    mass_coverage: Optional[float] = 0.999,
    vocab_cap: Optional[int] = 5000,
) -> Dict[str, Any]:
    """
    Create per-feature categorical vocabularies using exposure-aware collapse.

    Inputs
      feature_meta: dict from Zarr attrs["feature_meta"]
        - for 'cat' features: {"name", "kind":"cat", "classes":[{"code","count"}]}
      batch_size, steps_per_epoch: define N_epoch for exposure math
      min_hits_per_epoch: keep codes expected to appear at least this many times/epoch
      mass_coverage: optional, keep additional head codes until coverage ≥ this fraction
      vocab_cap: optional hard cap on kept codes per feature

    Output schema (selected keys)
      {
        "policy": {batch_size, steps_per_epoch, min_hits_per_epoch, mass_coverage,
                   vocab_cap, miss_id, unk_id, N_epoch},
        "categorical": {
          "<feature_name>": {
            "num_ids": int,              # 2 + len(kept_codes)
            "kept_codes": [raw_code,...],
            "code2id": {raw:int,...},    # MISS=0, UNK=1, kept→2..N (dense, per-feature)
            "id2code": {int:raw,...},
            "min_count": int,            # exposure threshold used
            "coverage": float,           # kept mass / total
            "counts_per_id": [float,...] # aligned with IDs (0..num_ids-1); 0 for MISS/UNK
          }, ...
        }
      }
    """
    n_epoch = max(1, batch_size * steps_per_epoch)

    out = {
        "policy": {
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "min_hits_per_epoch": min_hits_per_epoch,
            "mass_coverage": mass_coverage,
            "vocab_cap": vocab_cap,
            "miss_id": MISS_ID,
            "unk_id": UNK_ID,
            "N_epoch": n_epoch,
        },
        "categorical": {}
    }

    feats = feature_meta.get("features", [])
    by_name = {f["name"]: f for f in feats}

    for name, f in by_name.items():
        if f.get("kind") != "cat":
            continue

        classes: List[Tuple[int, int]] = [
            (int(c["code"]), int(c["count"])) for c in f.get("classes", [])
        ]
        total = sum(n for _, n in classes)
        # Exposure-derived min_count
        min_count = _exposure_min_count(total, batch_size, steps_per_epoch, min_hits_per_epoch)

        # 1) Exposure filter
        kept = [(c, n) for c, n in classes if n >= min_count]
        if not kept and classes:
            # ensure at least one class survives if anything exists
            kept = [max(classes, key=lambda t: t[1])]

        # 2) Mass coverage (optional)
        if mass_coverage is not None and total > 0:
            kept_sorted = sorted(kept, key=lambda t: t[1], reverse=True)
            acc, target = 0, mass_coverage * total
            kept2: List[Tuple[int, int]] = []
            for c, n in kept_sorted:
                kept2.append((c, n))
                acc += n
                if acc >= target:
                    break
            kept = kept2

        # 3) Cap (optional)
        if vocab_cap is not None and len(kept) > vocab_cap:
            kept = sorted(kept, key=lambda t: t[1], reverse=True)[:vocab_cap]

        # Stable code ordering (numeric ascending) for reproducibility
        kept_codes = [c for c, _ in sorted(kept, key=lambda t: t[0])]
        code2id = {c: 2 + i for i, c in enumerate(kept_codes)}
        id2code = {v: k for k, v in code2id.items()}

        # Build counts_per_id aligned to IDs (0..num_ids-1)
        counts_map = {c: n for c, n in classes}
        counts_per_id = [0.0, 0.0]  # MISS, UNK
        counts_per_id.extend([float(counts_map.get(c, 0)) for c in kept_codes])

        coverage = (sum(counts_map.get(c, 0) for c in kept_codes) / total) if total > 0 else 1.0

        out["categorical"][name] = {
            "num_ids": 2 + len(kept_codes),
            "kept_codes": kept_codes,
            "code2id": code2id,
            "id2code": id2code,
            "min_count": int(min_count),
            "coverage": float(coverage),
            "counts_per_id": counts_per_id,
        }

    return out


def attach_continuous_stats(schema: Dict[str, Any],
                            feature_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add continuous feature normalization stats to the schema.

    Output schema gains:
      "continuous": {
        "<feature_name>": {"mean":..., "std":..., "q01":..., "q99":...}, ...
      }
    """
    out = dict(schema)  # shallow copy
    cont_map: Dict[str, Any] = {}

    for f in feature_meta.get("features", []):
        if f.get("kind") != "int":
            continue
        name = f.get("name")
        stats = f.get("stats") or {}
        cont_map[name] = {
            "mean": float(stats.get("mean")) if stats.get("mean") is not None else None,
            "std": float(stats.get("std")) if stats.get("std") is not None else None,
            "q01": float(stats.get("q01")) if stats.get("q01") is not None else None,
            "q99": float(stats.get("q99")) if stats.get("q99") is not None else None,
        }

    out["continuous"] = cont_map
    return out


def save_schema(schema: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
