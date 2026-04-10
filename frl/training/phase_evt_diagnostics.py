#!/usr/bin/env python3
"""
EVT-stratified phase signal diagnostics.

Iterates over all patches in a dataset split, extracts FiLM gamma values and
z_phase temporal variance per pixel, and accumulates these statistics per
EVT (Existing Vegetation Type) class.  Optionally applies a pre-fitted phase
linear probe to compute per-EVT prediction accuracy (R²).

Outputs (all in --output-dir):
  gamma_by_evt.csv          Per-EVT mean FiLM gamma (per channel + aggregate)
  temporal_frac_by_evt.csv  Per-EVT z_phase temporal variance fraction
  probe_r2_by_evt.csv       Per-EVT phase probe R²  [requires --probe]
  gamma_heatmap.png         Heatmap: top-K EVT × 12 z_phase channels
  temporal_frac_heatmap.png Heatmap: top-K EVT × 12 z_phase channels
  probe_r2_heatmap.png      Heatmap: top-K EVT × target channels  [if --probe]
  summary.json              All stats + run metadata

FiLM gamma is extracted from model.phase_film(z_type); it measures how much
the type embedding amplifies each phase channel.  Gamma ~ 1.0 at init; values
> 1 after training indicate that the type encoder actively amplifies that
phase dimension.  Dynamic forest types (deciduous, early-successional) are
hypothesised to receive higher gamma.

Temporal variance fraction is computed per pixel as:

    frac = E[Var(z_phase | pixel)] / Var(z_phase)

A high fraction means most of z_phase variance is within-pixel across time
(temporal signal).  A low fraction means z_phase mainly encodes between-pixel
spatial differences.

Usage:
    python frl/training/phase_evt_diagnostics.py \\
        --checkpoint runs/.../encoder_epoch.pt \\
        --training   frl/config/frl_training_v1.yaml \\
        --bindings   frl/config/frl_binding_v1.yaml \\
        --evt-map    path/to/evt_crosswalk.csv \\
        --probe      runs/.../phase_linear_probe.pt \\
        --top-k-evt  20 \\
        --split      train \\
        --halo       16 \\
        --output-dir runs/.../evt_phase_diagnostics/
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from models import RepresentationModel

from training.fit_phase_linear_probe import (
    PHASE_INPUT_FEATURE,
    PHASE_TARGET_FEATURE,
    ProbePreprocessor,
    _get_target_channels,
    _halo_mask,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("phase_evt_diagnostics")


# ---------------------------------------------------------------------------
# Per-EVT streaming accumulators
# ---------------------------------------------------------------------------

class EvtAccumulators:
    """Streaming per-EVT-class accumulators for phase diagnostics.

    All tensors are stored as CPU float64 for numerical precision.

    Gamma and temporal-variance stats are accumulated once per *pixel*.
    Probe stats (when enabled) are accumulated once per *(pixel, timestep)*.
    """

    def __init__(self, d_phase: int, n_target_ch: Optional[int] = None) -> None:
        self.d_phase = d_phase
        self.n_target_ch = n_target_ch  # None → no probe

        # pixel-level
        self.n_pixels: Dict[int, int] = defaultdict(int)
        self.sum_gamma: Dict[int, torch.Tensor] = defaultdict(
            lambda: torch.zeros(d_phase, dtype=torch.float64)
        )
        self.sum_gamma2: Dict[int, torch.Tensor] = defaultdict(
            lambda: torch.zeros(d_phase, dtype=torch.float64)
        )
        # for temporal variance decomposition of z_phase
        self.sum_within_zp: Dict[int, torch.Tensor] = defaultdict(
            lambda: torch.zeros(d_phase, dtype=torch.float64)
        )
        self.sum_pixmean_zp: Dict[int, torch.Tensor] = defaultdict(
            lambda: torch.zeros(d_phase, dtype=torch.float64)
        )
        self.sum_pixmean2_zp: Dict[int, torch.Tensor] = defaultdict(
            lambda: torch.zeros(d_phase, dtype=torch.float64)
        )

        # (pixel × timestep) level — only when probe is active
        if n_target_ch is not None:
            self.n_obs: Dict[int, int] = defaultdict(int)
            self.sum_ssres: Dict[int, torch.Tensor] = defaultdict(
                lambda: torch.zeros(n_target_ch, dtype=torch.float64)
            )
            self.sum_tgt: Dict[int, torch.Tensor] = defaultdict(
                lambda: torch.zeros(n_target_ch, dtype=torch.float64)
            )
            self.sum_tgt2: Dict[int, torch.Tensor] = defaultdict(
                lambda: torch.zeros(n_target_ch, dtype=torch.float64)
            )
        else:
            self.n_obs = None
            self.sum_ssres = None
            self.sum_tgt = None
            self.sum_tgt2 = None

    def add_pixels(
        self,
        evt_codes: np.ndarray,        # [N] int32
        gamma: torch.Tensor,          # [N, d_phase] float64 CPU
        z_phase_flat: torch.Tensor,   # [N, d_phase, T] float64 CPU
    ) -> None:
        """Accumulate pixel-level gamma + temporal-variance stats per EVT class."""
        unique_evts = np.unique(evt_codes)
        for e in unique_evts:
            idx = evt_codes == e
            n = int(idx.sum())
            gm = gamma[idx]         # [n, d_phase]
            zp = z_phase_flat[idx]  # [n, d_phase, T]

            self.n_pixels[int(e)] += n
            self.sum_gamma[int(e)] += gm.sum(dim=0)
            self.sum_gamma2[int(e)] += (gm ** 2).sum(dim=0)

            pix_mean = zp.mean(dim=2)              # [n, d_phase]
            pix_var = zp.var(dim=2, correction=0)  # [n, d_phase]
            self.sum_within_zp[int(e)] += pix_var.sum(dim=0)
            self.sum_pixmean_zp[int(e)] += pix_mean.sum(dim=0)
            self.sum_pixmean2_zp[int(e)] += (pix_mean ** 2).sum(dim=0)

    def add_probe(
        self,
        evt_codes_obs: np.ndarray,  # [N*T] int32
        pred: torch.Tensor,          # [N*T, n_target_ch] float64 CPU
        target: torch.Tensor,        # [N*T, n_target_ch] float64 CPU
    ) -> None:
        """Accumulate (pixel × timestep) level probe stats per EVT class."""
        if self.n_obs is None:
            return
        unique_evts = np.unique(evt_codes_obs)
        for e in unique_evts:
            idx = evt_codes_obs == e
            n = int(idx.sum())
            p = pred[idx]    # [n, C]
            t = target[idx]  # [n, C]

            self.n_obs[int(e)] += n
            self.sum_ssres[int(e)] += ((p - t) ** 2).sum(dim=0)
            self.sum_tgt[int(e)] += t.sum(dim=0)
            self.sum_tgt2[int(e)] += (t ** 2).sum(dim=0)

    # -------------------------------------------------------------------------
    # Statistics computation
    # -------------------------------------------------------------------------

    def all_evt_codes(self) -> List[int]:
        return sorted(self.n_pixels.keys())

    def gamma_stats(self, e: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean_gamma [d_phase], std_gamma [d_phase]) for EVT class e."""
        n = self.n_pixels[e]
        if n == 0:
            z = torch.zeros(self.d_phase, dtype=torch.float64)
            return z, z
        mean = self.sum_gamma[e] / n
        var = (self.sum_gamma2[e] / n - mean ** 2).clamp(min=0)
        return mean, var.sqrt()

    def temporal_frac(self, e: int) -> torch.Tensor:
        """Return temporal variance fraction [d_phase] for EVT class e.

        Fraction = E[Var(z|pixel)] / (E[Var(z|pixel)] + Var(E[z|pixel]))
        """
        n = self.n_pixels[e]
        if n == 0:
            return torch.zeros(self.d_phase, dtype=torch.float64)
        within = self.sum_within_zp[e] / n
        mean_of_means = self.sum_pixmean_zp[e] / n
        between = (self.sum_pixmean2_zp[e] / n - mean_of_means ** 2).clamp(min=0)
        total = (within + between).clamp(min=1e-12)
        return within / total

    def probe_r2(self, e: int) -> Optional[torch.Tensor]:
        """Return per-channel R² [n_target_ch] in normalized space, or None."""
        if self.n_obs is None or self.n_obs.get(e, 0) == 0:
            return None
        n = self.n_obs[e]
        ssres = self.sum_ssres[e]
        mean_tgt = self.sum_tgt[e] / n
        sstot = (self.sum_tgt2[e] - n * mean_tgt ** 2).clamp(min=1e-12)
        return 1.0 - ssres / sstot


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    accumulators: EvtAccumulators,
    device: torch.device,
    halo: int,
    enc_feature_name: str,
    probe_W: Optional[torch.Tensor] = None,      # [D, C] float64 CPU
    probe_b: Optional[torch.Tensor] = None,      # [C]    float64 CPU
    preprocessor: Optional[ProbePreprocessor] = None,
) -> None:
    """Run a single batched forward pass for all samples, then accumulate per-EVT stats."""
    d_phase = accumulators.d_phase
    batch_size = len(batch["metadata"])
    H = W = T = None

    # --- Collect per-sample features and masks (CPU) ---
    enc_list: List[torch.Tensor] = []
    phase_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    evt_grids: List[np.ndarray] = []
    probe_mask_list: List[torch.Tensor] = []
    tgt_list: List[torch.Tensor] = []

    for i in range(batch_size):
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != "metadata"
        }
        sample["metadata"] = batch["metadata"][i]

        enc_f = feature_builder.build_feature(enc_feature_name, sample)
        phase_f = feature_builder.build_feature(PHASE_INPUT_FEATURE, sample)
        evt_f = feature_builder.build_feature("evt_class", sample)

        if H is None:
            H, W = enc_f.data.shape[-2], enc_f.data.shape[-1]
            T = phase_f.data.shape[1]

        sm_names = sample["metadata"]["channel_names"]["static_mask"]
        sm_data = sample["static_mask"]
        aoi_mask = torch.from_numpy(sm_data[sm_names.index("aoi")]).bool()
        forest_mask = torch.from_numpy(sm_data[sm_names.index("forest")]).bool()
        enc_mask = torch.from_numpy(enc_f.mask).bool()
        phase_mask_spatial = torch.from_numpy(phase_f.mask).all(dim=0)

        evt_grid = evt_f.data[0].astype(np.float32)
        evt_valid = torch.from_numpy(np.isfinite(evt_grid) & (evt_grid > 0))

        halo_m = _halo_mask(H, W, halo, torch.device("cpu"))
        combined_mask = (
            enc_mask & aoi_mask & forest_mask & phase_mask_spatial & evt_valid & halo_m
        )

        enc_list.append(torch.from_numpy(enc_f.data).float())
        phase_list.append(torch.from_numpy(phase_f.data).float())
        mask_list.append(combined_mask)
        evt_grids.append(evt_grid)

        if probe_W is not None and preprocessor is not None:
            tgt_f = feature_builder.build_feature(PHASE_TARGET_FEATURE, sample)
            tgt_mask_spatial = torch.from_numpy(tgt_f.mask).all(dim=0)
            probe_mask_list.append(combined_mask & tgt_mask_spatial)
            tgt_list.append(torch.from_numpy(tgt_f.data).float())  # [C_tgt, T, H, W]
        else:
            probe_mask_list.append(combined_mask)
            tgt_list.append(torch.empty(0))

    if not any(m.any() for m in mask_list):
        return

    # --- Single batched forward pass ---
    Ximg = torch.stack(enc_list).to(device)      # [B, C, H, W]
    Xphase = torch.stack(phase_list).to(device)  # [B, C_phase, T, H, W]

    with torch.no_grad():
        z_type = model(Ximg)                               # [B, 64, H, W]
        z_type_det = z_type.detach()
        z_phase = model.forward_phase(Xphase, z_type_det)  # [B, 12, T, H, W]
        gamma_b, _ = model.phase_film(z_type_det)           # [B, 12, H, W]

    # Move everything to CPU once
    z_type_cpu = z_type.cpu()    # [B, 64, H, W]
    z_phase_cpu = z_phase.cpu()  # [B, 12, T, H, W]
    gamma_cpu = gamma_b.cpu()    # [B, 12, H, W]

    # --- Per-sample accumulation ---
    for i in range(batch_size):
        combined_mask = mask_list[i]
        if not combined_mask.any():
            continue

        mask_flat = combined_mask.reshape(-1)
        evt_grid = evt_grids[i]
        evt_codes_valid = evt_grid.reshape(-1)[mask_flat.numpy()].astype(np.int32)

        # [N, 12] and [N, 12, T]
        gm_valid = gamma_cpu[i].permute(1, 2, 0).reshape(-1, d_phase)[mask_flat].double()
        zp_valid = z_phase_cpu[i].permute(2, 3, 0, 1).reshape(-1, d_phase, T)[mask_flat].double()

        if gm_valid.shape[0] == 0:
            continue

        accumulators.add_pixels(evt_codes_valid, gm_valid, zp_valid)

        # --- optional probe evaluation ---
        if probe_W is None or preprocessor is None:
            continue

        probe_mask = probe_mask_list[i]
        if not probe_mask.any():
            continue

        probe_mask_flat = probe_mask.reshape(-1)
        evt_codes_probe = evt_grid.reshape(-1)[probe_mask_flat.numpy()].astype(np.int32)

        zt_flat_px = z_type_cpu[i].permute(1, 2, 0).reshape(-1, model.z_type_dim)[probe_mask_flat]
        zp_flat_px = z_phase_cpu[i].permute(2, 3, 0, 1).reshape(-1, d_phase, T)[probe_mask_flat]

        tgt_tensor = tgt_list[i]  # [C_tgt, T, H, W]
        C_tgt = tgt_tensor.shape[0]
        tgt_hwt = tgt_tensor.permute(2, 3, 1, 0).reshape(-1, T, C_tgt)
        tgt_valid = tgt_hwt[probe_mask_flat]  # [N2, T, C_tgt]

        pred_parts: List[torch.Tensor] = []
        tgt_parts: List[torch.Tensor] = []
        evt_parts: List[np.ndarray] = []

        for t in range(T):
            zp_t = zp_flat_px[:, :, t]
            X_t = preprocessor.transform(zt_flat_px, zp_t)
            pred_t = (X_t.double() @ probe_W) + probe_b
            pred_parts.append(pred_t)
            tgt_parts.append(tgt_valid[:, t, :].double())
            evt_parts.append(evt_codes_probe)

        accumulators.add_probe(
            np.concatenate(evt_parts),
            torch.cat(pred_parts, dim=0),
            torch.cat(tgt_parts, dim=0),
        )


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
) -> None:
    """Save a labelled heatmap PNG."""
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not available; skipping heatmap %s", out_path.name)
        return

    n_rows, n_cols = matrix.shape
    fig_h = max(4, n_rows * 0.5)
    fig_w = max(8, n_cols * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    annotate = n_rows <= 30 and n_cols <= 20
    short_rows = [(r[:45] + "\u2026") if len(r) > 45 else r for r in row_labels]

    kwargs: dict = dict(
        ax=ax,
        xticklabels=col_labels,
        yticklabels=short_rows,
        cmap=cmap,
        linewidths=0.3,
        linecolor="lightgrey",
        annot=annotate,
        fmt=".2f",
    )
    if center is not None:
        kwargs["center"] = center
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax

    sns.heatmap(matrix, **kwargs)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap: {out_path}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _evt_row_label(code: int, name: str) -> str:
    """Format a heatmap row label as 'CODE: Short Name'."""
    short = (name[:35] + "\u2026") if len(name) > 35 else name
    return f"{code}: {short}"


def save_outputs(
    accumulators: EvtAccumulators,
    top_evt_codes: List[int],
    evt_code_to_label: Dict[int, str],
    out_dir: Path,
    target_channel_names: Optional[List[str]],
) -> dict:
    """Write CSVs, heatmaps; return summary dict."""
    d_phase = accumulators.d_phase
    all_codes = accumulators.all_evt_codes()
    phase_ch_names = [f"ch_{i}" for i in range(d_phase)]

    # ----- gamma_by_evt.csv -----
    gamma_rows = []
    for e in all_codes:
        mean_g, std_g = accumulators.gamma_stats(e)
        name = evt_code_to_label.get(e, f"EVT_{e}")
        row: dict = {
            "evt_code": e,
            "evt_name": name,
            "n_pixels": accumulators.n_pixels[e],
            "gamma_mean": float(mean_g.mean()),
            "gamma_std": float(std_g.mean()),
        }
        for ch in range(d_phase):
            row[f"gamma_ch_{ch}"] = float(mean_g[ch])
        gamma_rows.append(row)
    gamma_df = pd.DataFrame(gamma_rows).sort_values("n_pixels", ascending=False)
    gamma_df.to_csv(out_dir / "gamma_by_evt.csv", index=False)
    logger.info(f"Saved gamma_by_evt.csv ({len(gamma_rows)} EVT classes)")

    # ----- temporal_frac_by_evt.csv -----
    tfrac_rows = []
    for e in all_codes:
        tf = accumulators.temporal_frac(e)
        name = evt_code_to_label.get(e, f"EVT_{e}")
        row = {
            "evt_code": e,
            "evt_name": name,
            "n_pixels": accumulators.n_pixels[e],
            "temporal_frac_mean": float(tf.mean()),
        }
        for ch in range(d_phase):
            row[f"temporal_frac_ch_{ch}"] = float(tf[ch])
        tfrac_rows.append(row)
    tfrac_df = pd.DataFrame(tfrac_rows).sort_values("n_pixels", ascending=False)
    tfrac_df.to_csv(out_dir / "temporal_frac_by_evt.csv", index=False)
    logger.info(f"Saved temporal_frac_by_evt.csv ({len(tfrac_rows)} EVT classes)")

    # ----- probe_r2_by_evt.csv (optional) -----
    r2_rows = []
    short_tgt_names: List[str] = []
    if accumulators.n_obs is not None and target_channel_names:
        short_tgt_names = [c.replace("annual.", "") for c in target_channel_names]
        for e in all_codes:
            r2 = accumulators.probe_r2(e)
            if r2 is None:
                continue
            name = evt_code_to_label.get(e, f"EVT_{e}")
            row = {
                "evt_code": e,
                "evt_name": name,
                "n_obs": accumulators.n_obs.get(e, 0),
                "n_pixels": accumulators.n_pixels[e],
                "r2_mean": float(r2.mean()),
            }
            for ch_idx, ch in enumerate(short_tgt_names):
                row[f"r2_{ch}"] = float(r2[ch_idx])
            r2_rows.append(row)
        if r2_rows:
            r2_df = pd.DataFrame(r2_rows).sort_values("n_pixels", ascending=False)
            r2_df.to_csv(out_dir / "probe_r2_by_evt.csv", index=False)
            logger.info(f"Saved probe_r2_by_evt.csv ({len(r2_rows)} EVT classes)")

    # ----- heatmaps for top-K EVTs -----
    top_labels = [
        _evt_row_label(e, evt_code_to_label.get(e, f"EVT_{e}"))
        for e in top_evt_codes
    ]

    # gamma heatmap [top_k × d_phase]
    gamma_matrix = np.array(
        [[float(accumulators.gamma_stats(e)[0][ch]) for ch in range(d_phase)]
         for e in top_evt_codes],
        dtype=np.float32,
    )
    plot_heatmap(
        gamma_matrix,
        row_labels=top_labels,
        col_labels=phase_ch_names,
        title=(
            f"Mean FiLM Gamma by EVT class (top {len(top_evt_codes)})\n"
            "Columns = z_phase channels 0\u201311  |  center = 1.0 (init value)"
        ),
        out_path=out_dir / "gamma_heatmap.png",
        cmap="RdBu_r",
        center=1.0,
    )

    # temporal fraction heatmap [top_k × d_phase]
    tfrac_matrix = np.array(
        [[float(accumulators.temporal_frac(e)[ch]) for ch in range(d_phase)]
         for e in top_evt_codes],
        dtype=np.float32,
    )
    plot_heatmap(
        tfrac_matrix,
        row_labels=top_labels,
        col_labels=phase_ch_names,
        title=(
            f"z_phase Temporal Variance Fraction by EVT class (top {len(top_evt_codes)})\n"
            "Columns = z_phase channels 0\u201311  |  1.0 = fully temporal"
        ),
        out_path=out_dir / "temporal_frac_heatmap.png",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

    # probe R² heatmap [top_k × n_target_ch]
    if r2_rows and target_channel_names:
        r2_matrix = np.full(
            (len(top_evt_codes), len(target_channel_names)), np.nan, dtype=np.float32,
        )
        for i, e in enumerate(top_evt_codes):
            r2 = accumulators.probe_r2(e)
            if r2 is not None:
                for ch_idx in range(len(target_channel_names)):
                    r2_matrix[i, ch_idx] = float(r2[ch_idx])
        plot_heatmap(
            r2_matrix,
            row_labels=top_labels,
            col_labels=short_tgt_names,
            title=(
                f"Phase Probe R\u00b2 by EVT class (top {len(top_evt_codes)})\n"
                "Columns = phase target channels  |  normalized space"
            ),
            out_path=out_dir / "probe_r2_heatmap.png",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

    # ----- summary.json -----
    summary: dict = {
        "n_evt_classes_observed": len(all_codes),
        "n_evt_classes_top_k": len(top_evt_codes),
        "total_pixels": int(sum(accumulators.n_pixels.values())),
        "by_evt": {},
    }
    for e in all_codes:
        mean_g, std_g = accumulators.gamma_stats(e)
        tf = accumulators.temporal_frac(e)
        entry: dict = {
            "evt_code": e,
            "evt_name": evt_code_to_label.get(e, f"EVT_{e}"),
            "n_pixels": accumulators.n_pixels[e],
            "gamma_mean_avg": float(mean_g.mean()),
            "gamma_std_avg": float(std_g.mean()),
            "gamma_per_channel": [float(v) for v in mean_g],
            "temporal_frac_avg": float(tf.mean()),
            "temporal_frac_per_channel": [float(v) for v in tf],
        }
        r2 = accumulators.probe_r2(e)
        if r2 is not None:
            entry["probe_r2_mean"] = float(r2.mean())
            entry["probe_r2_per_channel"] = [float(v) for v in r2]
            entry["n_obs"] = accumulators.n_obs.get(e, 0)
        summary["by_evt"][str(e)] = entry

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved summary.json")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EVT-stratified phase diagnostics: FiLM gamma distribution and "
            "z_phase temporal variance fraction per vegetation type."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--training", default="config/frl_training_v1.yaml")
    parser.add_argument("--bindings", default="config/frl_binding_v1.yaml")
    parser.add_argument(
        "--evt-map", default=None,
        help="CSV crosswalk with columns Value,Color,Description (LANDFIRE format)",
    )
    parser.add_argument(
        "--probe", default=None,
        help="Path to phase_linear_probe.pt checkpoint (enables per-EVT R²)",
    )
    parser.add_argument(
        "--top-k-evt", type=int, default=20,
        help="Number of top EVT classes (by pixel count) to show in heatmaps (default: 20)",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: train)",
    )
    parser.add_argument(
        "--halo", type=int, default=16,
        help="Pixels from patch edge to exclude (default: 16)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-batches", type=int, default=0,
        help="Cap number of batches processed (0 = all; useful for quick tests)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # --- Config ---
    logger.info(f"Loading bindings: {args.bindings}")
    bindings_config = DatasetBindingsParser(args.bindings).parse()
    logger.info(f"Loading training config: {args.training}")
    training_config = TrainingConfigParser(args.training).parse()

    device_str = args.device or training_config.hardware.device
    device = torch.device(device_str)
    batch_size = args.batch_size or training_config.training.batch_size
    patch_size = training_config.sampling.patch_size
    logger.info(f"Device: {device}  |  batch_size: {batch_size}")

    # --- EVT crosswalk ---
    evt_code_to_label: Dict[int, str] = {}
    if args.evt_map:
        xwalk = pd.read_csv(args.evt_map, dtype={"Value": int})
        evt_code_to_label = {
            int(row["Value"]): str(row["Description"])
            for _, row in xwalk.iterrows()
        }
        logger.info(f"Loaded {len(evt_code_to_label)} EVT codes from {args.evt_map}")

    # --- Output directory ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.checkpoint).parent / "evt_phase_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # --- Dataset ---
    logger.info(f"Creating {args.split} dataset...")
    dataset = ForestDatasetV2(
        bindings_config,
        split=args.split,
        patch_size=patch_size,
        min_aoi_fraction=0.3,
    )
    logger.info(f"{args.split} dataset: {len(dataset)} patches")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_config.hardware.num_workers,
        pin_memory=training_config.hardware.pin_memory,
        collate_fn=collate_fn,
    )

    feature_builder = FeatureBuilder(bindings_config)
    enc_feature_name = training_config.model_input.type_encoder_feature
    logger.info(f"Using encoder feature: '{enc_feature_name}'")

    # --- Model ---
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = RepresentationModel.from_checkpoint(args.checkpoint, device=device, freeze=True)
    d_phase = model.z_phase_dim
    logger.info(f"z_type_dim={model.z_type_dim}  z_phase_dim={d_phase}")

    # --- Optional probe ---
    probe_W: Optional[torch.Tensor] = None
    probe_b: Optional[torch.Tensor] = None
    preprocessor: Optional[ProbePreprocessor] = None
    target_channel_names: Optional[List[str]] = None
    n_target_ch: Optional[int] = None

    if args.probe:
        logger.info(f"Loading phase probe: {args.probe}")
        probe_ckpt = torch.load(args.probe, map_location="cpu", weights_only=False)
        probe_W = probe_ckpt["W"].to(torch.float64)
        probe_b = probe_ckpt["b"].to(torch.float64)
        preprocessor = ProbePreprocessor.from_dict(probe_ckpt)
        target_channel_names = _get_target_channels(feature_builder)
        n_target_ch = len(target_channel_names)
        logger.info(
            f"Probe: design={preprocessor.design}  "
            f"D={preprocessor.output_dim}  n_target_ch={n_target_ch}"
        )

    # --- Accumulators ---
    accumulators = EvtAccumulators(d_phase=d_phase, n_target_ch=n_target_ch)

    # --- Streaming pass ---
    n_batches = len(loader)
    max_batches = args.max_batches

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        process_batch(
            batch, feature_builder, model, accumulators,
            device=device,
            halo=args.halo,
            enc_feature_name=enc_feature_name,
            probe_W=probe_W,
            probe_b=probe_b,
            preprocessor=preprocessor,
        )

        if batch_idx % 5 == 0:
            total_px = sum(accumulators.n_pixels.values())
            logger.info(
                f"  Batch {batch_idx:4d}/{n_batches}  |  "
                f"EVT classes: {len(accumulators.all_evt_codes()):4d}  |  "
                f"pixels: {total_px:,}"
            )

    total_pixels = int(sum(accumulators.n_pixels.values()))
    logger.info(
        f"Streaming complete: {total_pixels:,} valid pixels  |  "
        f"{len(accumulators.all_evt_codes())} EVT classes observed"
    )

    if total_pixels == 0:
        raise RuntimeError("No valid pixels found — check masks and split.")

    # --- EVT frequency ranking ---
    sorted_by_count = sorted(
        accumulators.n_pixels.items(), key=lambda x: x[1], reverse=True
    )
    top_k = min(args.top_k_evt, len(sorted_by_count))
    top_evt_codes = [code for code, _ in sorted_by_count[:top_k]]

    logger.info(f"\nTop {top_k} EVT classes by pixel count:")
    for code, n in sorted_by_count[:top_k]:
        label = evt_code_to_label.get(code, f"EVT_{code}")
        mean_g, _ = accumulators.gamma_stats(code)
        tf = accumulators.temporal_frac(code)
        r2 = accumulators.probe_r2(code)
        r2_str = f"  r2={float(r2.mean()):.3f}" if r2 is not None else ""
        logger.info(
            f"  {code:6d}  {n:9,}  ({100*n/total_pixels:.1f}%)  "
            f"gamma={float(mean_g.mean()):.3f}  "
            f"temp_frac={float(tf.mean()):.3f}"
            f"{r2_str}  {label}"
        )

    # --- Save outputs ---
    save_outputs(
        accumulators,
        top_evt_codes=top_evt_codes,
        evt_code_to_label=evt_code_to_label,
        out_dir=out_dir,
        target_channel_names=target_channel_names,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
