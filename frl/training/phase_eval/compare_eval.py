#!/usr/bin/env python3
"""Compare two eval-harness ``metrics.json`` files (new model vs exp034 baseline).

Every Step-0 metric is reported *relative to the exp034 baseline* so "better" is
defined. This flattens the nested metrics dicts, joins on the shared metric keys,
and prints/writes a table of ``baseline``, ``new``, ``delta``.

Usage::

    PYTHONPATH=frl python -m training.phase_eval.compare_eval \\
        --new      runs/<new>/phase_eval/metrics.json \\
        --baseline runs/frl_v0_exp034/phase_eval/metrics.json \\
        --output   runs/<new>/phase_eval/compare_vs_exp034
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("phase_eval.compare")


def _flatten(d, prefix: str = "") -> Dict[str, float]:
    """Flatten nested dicts to ``a.b.c → scalar``; keep only numeric leaves."""
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = float(d)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two phase-eval metrics.json files")
    p.add_argument("--new", required=True, help="new-model metrics.json")
    p.add_argument("--baseline", required=True, help="exp034 baseline metrics.json")
    p.add_argument("--output", default=None, help="output path stem (.md and .csv written)")
    args = p.parse_args()

    with open(args.new) as f:
        new = _flatten(json.load(f).get("diagnostics", {}))
    with open(args.baseline) as f:
        base = _flatten(json.load(f).get("diagnostics", {}))

    keys = sorted(set(new) | set(base))
    rows = []
    for k in keys:
        b = base.get(k)
        n = new.get(k)
        delta = (n - b) if (b is not None and n is not None) else None
        rows.append({"metric": k, "baseline": b, "new": n, "delta": delta})
    df = pd.DataFrame(rows)

    md = _to_markdown(rows)
    print(md)

    if args.output:
        stem = Path(args.output)
        stem.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(stem.with_suffix(".csv"), index=False)
        with open(stem.with_suffix(".md"), "w") as f:
            f.write("# Phase-eval comparison: new vs baseline\n\n")
            f.write(f"- new:      `{args.new}`\n- baseline: `{args.baseline}`\n\n")
            f.write(md + "\n")
        logger.info(f"Wrote {stem.with_suffix('.csv')} and {stem.with_suffix('.md')}")


def _fmt(v) -> str:
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _to_markdown(rows: list[dict]) -> str:
    """Render rows as a GitHub markdown table (no ``tabulate`` dependency)."""
    cols = ["metric", "baseline", "new", "delta"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
