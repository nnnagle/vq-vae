#!/usr/bin/env python3
"""Step-0 eval-harness runner: A/B/C for one checkpoint → ``metrics.json``.

Fits probes on **train**, tunes ridge λ on **val**, reports on **test** — the
split discipline the spec requires — using the checkerboard split inside
``ForestDatasetV2``. Run once per checkpoint (new model and the exp034 baseline);
diff the two ``metrics.json`` with ``compare_eval.py``.

Usage (on the HPC, where the zarr + checkpoints live)::

    PYTHONPATH=frl python -m training.phase_eval.run_eval \\
        --checkpoint runs/<exp>/encoder_last.pt \\
        --training   config/frl_training_v1.yaml \\
        --bindings   config/frl_binding_v1.yaml \\
        --evt-map    ../data/LF2024_EVT.csv \\
        --output-dir runs/<exp>/phase_eval/

Quick smoke test: add ``--max-batches 2 --max-pixels-per-sample 500 --no-mlp``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from training.phase_eval.common import build_context, make_loader, write_metrics_json
from training.phase_eval.ejection import run_ejection
from training.phase_eval.reconstruction import FEATURE_SOURCES, run_reconstruction
from training.phase_eval.recovery_curves import run_recovery_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("phase_eval.run")


def _load_evt_crosswalk(path: str | None) -> Dict[int, str]:
    if not path or not Path(path).exists():
        return {}
    xwalk = pd.read_csv(path)
    cols = xwalk.columns.tolist()
    code_col = "VALUE" if "VALUE" in cols else "Value"
    name_col = "EVT_NAME" if "EVT_NAME" in cols else "Description"
    return {
        int(r[code_col]): str(r[name_col])
        for _, r in xwalk.iterrows()
        if pd.notna(r[code_col]) and int(r[code_col]) > 0
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Step-0 phase-pathway eval harness (A/B/C)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--training", default="config/frl_training_v1.yaml")
    p.add_argument("--bindings", default="config/frl_binding_v1.yaml")
    p.add_argument("--evt-map", default="../data/LF2024_EVT.csv")
    p.add_argument("--output-dir", default=None,
                   help="default: <checkpoint dir>/phase_eval/")
    p.add_argument("--which", default="A,B,C",
                   help="comma list of diagnostics to run (subset of A,B,C)")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap batches per split (0 = all); use small for smoke tests")
    p.add_argument("--max-pixels-per-sample", type=int, default=2000,
                   help="cap valid pixels sampled per patch (0 = all)")
    p.add_argument("--sources", default=",".join(FEATURE_SOURCES),
                   help="diagnostic-A feature sources (subset of "
                        "z_phase,h,z_type,type-phase-bilinear)")
    p.add_argument("--anomaly-kinds", default="raw",
                   help="diagnostic-A targets: 'raw' now; 'mature_baseline' after Step 1")
    p.add_argument("--top-k-evt", type=int, default=20)
    p.add_argument("--no-mlp", action="store_true", help="skip the diagnostic-A MLP ceiling")
    args = p.parse_args()

    # Give the main process its CPU cores back for the float64 ridge algebra
    # (launch scripts pin *_NUM_THREADS=1 for the dataloader workers).
    _n = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    torch.set_num_threads(max(1, _n))

    which = {w.strip().upper() for w in args.which.split(",") if w.strip()}
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "phase_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(
        args.bindings, args.training, args.checkpoint,
        device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    train_loader = make_loader(ctx, "train")
    val_loader = make_loader(ctx, "val")
    test_loader = make_loader(ctx, "test")

    metrics: dict = {
        "checkpoint": str(args.checkpoint),
        "z_type_dim": ctx["model"].z_type_dim,
        "z_phase_dim": ctx["model"].z_phase_dim,
        "halo": args.halo,
        "max_batches": args.max_batches,
        "max_pixels_per_sample": args.max_pixels_per_sample,
        "diagnostics": {},
    }

    if "A" in which:
        logger.info("=== Diagnostic A: reconstruction probes ===")
        metrics["diagnostics"]["A_reconstruction"] = run_reconstruction(
            ctx, train_loader, val_loader, test_loader,
            sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
            anomaly_kinds=tuple(k.strip() for k in args.anomaly_kinds.split(",") if k.strip()),
            halo=args.halo, max_pixels_per_sample=args.max_pixels_per_sample,
            max_batches=args.max_batches, fit_mlp=not args.no_mlp,
        )

    if "B" in which:
        logger.info("=== Diagnostic B: type-conditional recovery curves ===")
        evt_xwalk = _load_evt_crosswalk(args.evt_map)
        metrics["diagnostics"]["B_recovery_curves"] = run_recovery_curves(
            ctx, train_loader, val_loader, test_loader,
            evt_code_to_label=evt_xwalk, top_k_evt=args.top_k_evt,
            halo=args.halo, max_pixels_per_sample=args.max_pixels_per_sample,
            max_batches=args.max_batches, output_dir=out_dir,
        )

    if "C" in which:
        logger.info("=== Diagnostic C: disturbance produces ejection ===")
        metrics["diagnostics"]["C_ejection"] = run_ejection(
            ctx, test_loader, halo=args.halo,
            max_pixels_per_sample=args.max_pixels_per_sample, max_batches=args.max_batches,
        )

    write_metrics_json(out_dir / "metrics.json", metrics)
    logger.info("Done.")


if __name__ == "__main__":
    main()
