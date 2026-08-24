#!/usr/bin/env python3
"""Phase-0 CLI: per-EVT classical AR(1)+noise recovery fit on the anomaly.

Grounds the differentiable Kalman filter (``losses.kalman_filter``) *before* any
retraining. For a checkpoint, it extracts per-pixel anomaly recovery series,
groups by EVT, and reports the de-attenuated ρ̂(EVT), the naive (attenuated)
lag-1, the reliability ratio, and an AR(1)-adequacy verdict — using the
unit-tested estimator in ``analysis.ar1_recovery``.

The recovery signal is the **anomaly magnitude** ‖a_t‖ (a = (x−μ(z_type))/σ,
built from the checkpoint's mature-baseline readout): it is ≈0 at maturity,
large just after disturbance, and decays back at rate ρ. Segments are cut at
disturbance resets (ysfc==0 or a ysfc decrease) so pairs never cross an outward
jump — the same gating the filter uses.

Runs on the HPC (needs the zarr + a checkpoint). CPU is fine::

    PYTHONPATH=frl python -m analysis.run_ar1_recovery \\
        --checkpoint runs/frl_v0_exp039/checkpoints/encoder_best_1_epoch_381.pt \\
        --training config/frl_training_v1.yaml --evt-map ../data/LF2024_EVT.csv \\
        --device cpu

This is model-light (only the readout μ/σ is used); with ``--raw-input`` it runs
on the normalized phase-encoder input magnitude instead, needing no readout.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from training.phase_eval.common import build_context, make_loader, iter_batches, extract_pixel_series
from training.phase_eval.run_eval import _load_evt_crosswalk
from analysis.ar1_recovery import summarize_ar1, ar1_adequacy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ar1_recovery")


def _reset_from_ysfc(ysfc: np.ndarray) -> np.ndarray:
    """Segment start (reset) mask [N,T]: disturbance year (ysfc==0) or a ysfc
    decrease (a new recovery sequence begins)."""
    N, T = ysfc.shape
    reset = np.zeros((N, T), dtype=bool)
    reset[:, 0] = True
    reset[:, 1:] = (ysfc[:, 1:] == 0) | (ysfc[:, 1:] < ysfc[:, :-1])
    return reset


def _anomaly_magnitude(batch_series: dict, ctx: dict, raw_input: bool) -> np.ndarray:
    """Per-pixel scalar recovery series ‖a_t‖ (or ‖x_t‖ if raw_input) → [N, T]."""
    x = batch_series["x"].to(ctx["device"])                       # [N, Cx, T]
    if raw_input:
        return x.norm(dim=1).cpu().numpy()                        # [N, T]
    model = ctx["model"]
    z_type = batch_series["z_type"].to(ctx["device"])             # [N, dt]
    mu, sigma = model.mature_baseline.predict(z_type)             # [N, C]
    feats, _ = model.anomaly_transform(x, mu, sigma)              # [N, 2C, T]
    C = x.shape[1]
    a = feats[:, :C, :]                                           # [N, C, T] anomaly block
    return a.norm(dim=1).cpu().numpy()                            # [N, T]


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-0 per-EVT AR(1)+noise recovery fit")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--training", default="config/frl_training_v1.yaml")
    p.add_argument("--bindings", default="config/frl_binding_v1.yaml")
    p.add_argument("--evt-map", default="../data/LF2024_EVT.csv")
    p.add_argument("--split", default="train")
    p.add_argument("--device", default=None)
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--max-pixels-per-sample", type=int, default=2000)
    p.add_argument("--top-k-evt", type=int, default=20)
    p.add_argument("--min-pairs", type=int, default=500,
                   help="skip an EVT with fewer pooled lag-1 recovery pairs")
    p.add_argument("--raw-input", action="store_true",
                   help="use ‖normalized phase input‖ instead of the anomaly (no readout)")
    args = p.parse_args()

    ctx = build_context(args.bindings, args.training, args.checkpoint, device=args.device)
    loader = make_loader(ctx, args.split)
    evt_xwalk = _load_evt_crosswalk(args.evt_map)

    # Pool per-EVT recovery series across the split.
    series: dict[int, list[np.ndarray]] = {}
    resets: dict[int, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in iter_batches(loader, args.max_batches):
            bs = extract_pixel_series(batch, ctx, args.halo,
                                      max_pixels_per_sample=args.max_pixels_per_sample)
            if bs is None:
                continue
            mag = _anomaly_magnitude(bs, ctx, args.raw_input)        # [N, T]
            ysfc = bs["ysfc"].cpu().numpy()                          # [N, T]
            reset = _reset_from_ysfc(ysfc)
            evt = bs["evt"]                                          # [N]
            for code in np.unique(evt):
                m = evt == code
                series.setdefault(int(code), []).append(mag[m])
                resets.setdefault(int(code), []).append(reset[m])

    # Rank EVTs by pixel count, report the top-K.
    counts = {c: sum(s.shape[0] for s in v) for c, v in series.items()}
    top = sorted(counts, key=counts.get, reverse=True)[:args.top_k_evt]

    signal = "‖phase input‖" if args.raw_input else "‖anomaly‖"
    logger.info("=== Phase-0 AR(1)+noise recovery fit  (signal=%s, split=%s) ===",
                signal, args.split)
    logger.info("%-6s %-34s %7s %8s %8s %8s %6s %5s",
                "EVT", "name", "npix", "rho_hat", "rho_naiv", "reliab", "npair", "AR1?")
    rows = []
    for code in top:
        x = np.concatenate(series[code], axis=0)
        r = np.concatenate(resets[code], axis=0)
        s = summarize_ar1(x, reset=r)
        if s["n_pairs_lag1"] < args.min_pairs:
            continue
        ok = ar1_adequacy(s)
        name = evt_xwalk.get(code, "")[:33]
        logger.info("%-6d %-34s %7d %8.3f %8.3f %8.3f %6d %5s",
                    code, name, counts[code], s["rho_ratio"], s["rho_naive"],
                    s["reliability"], s["n_pairs_lag1"], "yes" if ok else "NO")
        rows.append((code, s, ok))

    if rows:
        rhos = np.array([r[1]["rho_ratio"] for r in rows])
        rhos = rhos[np.isfinite(rhos)]
        n_ar1 = sum(1 for r in rows if r[2])
        logger.info("--- summary: %d EVTs | ρ̂ range %.3f–%.3f (mean %.3f) | "
                    "AR(1) adequate in %d/%d ---",
                    len(rows), float(rhos.min()), float(rhos.max()), float(rhos.mean()),
                    n_ar1, len(rows))
        logger.info("Seed ρ(z_type) near these; if AR(1) fails widely, add AR(2)/complex modes.")
    else:
        logger.warning("No EVT met --min-pairs=%d; lower it or raise --max-batches.",
                        args.min_pairs)


if __name__ == "__main__":
    main()
