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
    bindings = DatasetBindingsParser('config/frl_binding_v1.yaml').parse()

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
    confusion_csv          = evt_loss_cfg.confusion_matrix_path
    min_count              = evt_loss_cfg.min_count or 100
    min_confusion_samples  = evt_loss_cfg.min_confusion_samples or 30
    diffusion_steps        = evt_loss_cfg.diffusion_steps or 2
    laplace_smoothing      = evt_loss_cfg.laplace_smoothing or 0.0

    print(f"Confusion matrix       : {confusion_csv}")
    print(f"min_count              : {min_count}")
    print(f"min_confusion_samples  : {min_confusion_samples}")
    print(f"diffusion_steps        : {diffusion_steps}")
    print(f"laplace_smoothing      : {laplace_smoothing}")

    # Show how many codes survive each filter independently
    import pandas as pd
    conf_raw = pd.read_csv(confusion_csv, index_col=0)
    for col in ["Row Totals", "Percent Row Agreement"]:
        if col in conf_raw.columns:
            conf_raw = conf_raw.drop(columns=[col])
    for row in ["Column Totals", "Percent Column Agreement"]:
        if row in conf_raw.index:
            conf_raw = conf_raw.drop(index=[row])
    conf_raw = conf_raw[conf_raw.index.notna()]
    conf_raw.index = conf_raw.index.astype(int)
    conf_raw.columns = conf_raw.columns.astype(int)
    conf_raw = conf_raw.astype(float)

    int_counts = {int(k): float(v) for k, v in code_counts.items()}
    n_in_table = len(conf_raw.index)
    n_pass_min_count = sum(
        1 for c in conf_raw.index if int_counts.get(c, 0) >= min_count
    )
    row_sums = conf_raw.sum(axis=1)
    n_pass_confusion = sum(
        1 for c in conf_raw.index
        if int_counts.get(c, 0) >= min_count and row_sums.get(c, 0) >= min_confusion_samples
    )
    print(f"\nCode survival:")
    print(f"  In confusion table          : {n_in_table}")
    print(f"  After min_count filter      : {n_pass_min_count}")
    print(f"  After min_confusion_samples : {n_pass_confusion}")

    metric = EvtDiffusionMetric(
        confusion_csv=confusion_csv,
        code_counts=code_counts,
        min_count=min_count,
        min_confusion_samples=min_confusion_samples,
        diffusion_steps=diffusion_steps,
        laplace_smoothing=laplace_smoothing,
    )
    K = metric.n_codes
    print(f"  In metric (final)           : {K}")
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

    # ------------------------------------------------------------------
    # Histogram 1: raw confusion table row sums
    # Informs choice of min_confusion_samples.
    # ------------------------------------------------------------------
    print()
    print("── Row sums of raw confusion table (field samples per code) ──")
    keep_codes = sorted(metric.valid_codes)
    rs = np.array([row_sums.get(c, 0.0) for c in keep_codes])
    _ascii_hist(rs, n_bins=10, label="row sum", integer=True)

    # ------------------------------------------------------------------
    # Histogram 2: off-diagonal similarity values, P^1 vs P^k
    # Shows how diffusion fills in transitive similarities.
    # ------------------------------------------------------------------
    print()
    print("── Off-diagonal similarity: direct confusion (P^1) vs diffused (P^k) ──")

    metric_p1 = EvtDiffusionMetric(
        confusion_csv=confusion_csv,
        code_counts=code_counts,
        min_count=min_count,
        min_confusion_samples=min_confusion_samples,
        diffusion_steps=1,
        laplace_smoothing=laplace_smoothing,
    )
    S1 = metric_p1._S
    Sk = metric._S
    eye = torch.eye(K, dtype=torch.bool)

    s1_off = S1[~eye].numpy()
    sk_off = Sk[~eye].numpy()

    # Exclude exact zeros separately so the histogram shows the non-zero structure
    nz1 = s1_off[s1_off > 0]
    nzk = sk_off[sk_off > 0]

    pct_nz1 = 100.0 * len(nz1) / len(s1_off)
    pct_nzk = 100.0 * len(nzk) / len(sk_off)

    print(f"  P^1  non-zero off-diagonal: {len(nz1):5d} / {len(s1_off)} ({pct_nz1:.1f}%)")
    _ascii_hist(nz1, n_bins=8, label="similarity", lo=0.0, hi=1.0)
    print()
    print(f"  P^{diffusion_steps}  non-zero off-diagonal: {len(nzk):5d} / {len(sk_off)} ({pct_nzk:.1f}%)")
    _ascii_hist(nzk, n_bins=8, label="similarity", lo=0.0, hi=1.0)


def _ascii_hist(
    values: np.ndarray,
    n_bins: int = 10,
    label: str = "value",
    lo: float | None = None,
    hi: float | None = None,
    bar_width: int = 40,
    integer: bool = False,
) -> None:
    """Print a simple ASCII histogram to stdout."""
    if len(values) == 0:
        print("  (no data)")
        return
    lo = float(values.min()) if lo is None else lo
    hi = float(values.max()) if hi is None else hi
    if lo == hi:
        print(f"  all values = {lo}")
        return
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    max_count = counts.max() or 1
    for i in range(n_bins):
        bar = int(bar_width * counts[i] / max_count)
        if integer:
            range_str = f"{int(edges[i]):>6} – {int(edges[i+1]):>6}"
        else:
            range_str = f"{edges[i]:>5.3f} – {edges[i+1]:>5.3f}"
        print(f"  {range_str} │{'█' * bar:<{bar_width}}│ {counts[i]}")


if __name__ == '__main__':
    main()
