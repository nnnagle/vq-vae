#!/usr/bin/env python3
"""
Interactive check for EVT tau_ref calibration.

Run from the frl/ directory:
    python examples/check_evt_tau.py

Shows H_ref as a fraction of maximum entropy for a range of tau values,
so you can pick a tau_ref that gives ~30-60% of max entropy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from frl/ or repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.evt_soft_neighborhood import EvtDiffusionMetric
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser


def main():
    bindings = DatasetBindingsParser.from_yaml('config/frl_binding_v1.yaml')

    # --- load stats for code counts ---
    stats_path = bindings.stats.file
    print(f"Loading stats from {stats_path} ...")
    with open(stats_path) as f:
        stats = json.load(f)

    code_counts = (
        stats.get("evt_class", {})
             .get("static_categorical.evt", {})
             .get("counts", {})
    )
    if not code_counts:
        sys.exit("ERROR: EVT counts not found in stats file. Run example_compute_stats.py first.")

    evt_loss_cfg = bindings.get_loss('soft_neighborhood_evt')
    confusion_csv = evt_loss_cfg.confusion_matrix_path
    min_count     = evt_loss_cfg.min_count or 100

    print(f"Confusion matrix : {confusion_csv}")
    print(f"min_count        : {min_count}")

    metric = EvtDiffusionMetric(
        confusion_csv=confusion_csv,
        code_counts=code_counts,
        min_count=min_count,
        diffusion_steps=evt_loss_cfg.diffusion_steps or 2,
    )
    K = metric.n_codes
    print(f"Valid codes      : {K}")
    print()

    # Build code-level distance matrix (K x K)
    d = 1.0 - metric._S          # [K, K], values in [0, 1]
    mask = ~torch.eye(K, dtype=torch.bool)
    log_max = float(torch.log(torch.tensor(float(K - 1))))

    print(f"{'tau_ref':>8}  {'H_ref':>6}  {'% of max':>9}  {'assessment'}")
    print("-" * 50)

    taus = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
    for tau in taus:
        logits = torch.where(mask, -d / tau, torch.tensor(-1e9))
        p = logits.softmax(dim=1)
        H = -(p * p.clamp(min=1e-9).log()).sum(dim=1).mean().item()
        pct = 100.0 * H / log_max
        if pct < 20:
            assessment = "too peaked — loss ignores most neighbours"
        elif pct < 35:
            assessment = "sharp — focused on nearest neighbours"
        elif pct <= 65:
            assessment = "good range"
        elif pct <= 80:
            assessment = "soft — moderate structure"
        else:
            assessment = "too flat — near-uniform, little structure"
        print(f"{tau:>8.2f}  {H:>6.3f}  {pct:>8.1f}%  {assessment}")

    print()
    # Also show the distribution of off-diagonal d_ref values for context
    off_diag = d[mask]
    print("d_ref distribution (off-diagonal, code x code):")
    for q, label in [(0.0,'min'), (0.1,'p10'), (0.25,'p25'),
                     (0.5,'median'), (0.75,'p75'), (0.9,'p90'), (1.0,'max')]:
        print(f"  {label:>6}: {torch.quantile(off_diag, q).item():.4f}")


if __name__ == '__main__':
    main()
