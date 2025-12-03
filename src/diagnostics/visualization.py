# src/diagnostics/visualization.py
#
# Utility functions for writing quick-look spatial/temporal reconstruction
# diagnostics for VAE models. These figures help verify that the model is
# producing sensible reconstructions across both space (H, W) and time (T).
#
# The key design goals:
#   - Keep the function simple and explicit. No hidden magic.
#   - Assume tensors are in physical units (i.e., already unnormalized).
#   - Assume layout [B, T, C, H, W].
#   - Visualize a single continuous feature channel at a time.
#   - Arrange patches and timesteps in a compact grid so artifacts are obvious.
#
# The function below is intentionally minimal and should be considered a
# first-pass diagnostic. If you discover you need more elaborate layouts
# (multiple features, categorical overlays, etc.), create additional
# functions rather than overloading this one.


from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Force a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")  # renders directly to files, never touches a display
import matplotlib.pyplot as plt


def save_spacetime_recon_grid(
    x_phys: torch.Tensor,
    recon_phys: torch.Tensor,
    aoi: Optional[torch.Tensor],
    out_path: Path,
    feature_idx: int = 0,
    feature_name: Optional[str] = None,
    max_patches: int = 4,
    select_most_dynamic: bool = False,
    show_deltas: bool = False,
):
    """
    Save a grid figure comparing input vs. reconstruction for a single
    continuous feature across time.

    Parameters
    ----------
    x_phys : torch.Tensor
        Input data in physical units with shape [B, T, C, H, W].

    recon_phys : torch.Tensor
        Model reconstruction in physical units with shape [B, T, C, H, W].

    aoi : Optional[torch.Tensor]
        AOI mask with shape [B, H, W], where True indicates valid area.
        If provided, regions outside the AOI are masked (set to NaN) in plots.

    out_path : Path
        File path where the figure will be saved. Parent directories are
        created as needed.

    feature_idx : int
        Index of the continuous feature channel to visualize.

    feature_name : Optional[str]
        Optional human-readable name for the feature; used in the figure
        title if provided.

    max_patches : int
        Maximum number of patches (examples from the batch) to include in
        the visualization. Helps keep figures readable when batch size is large.

    select_most_dynamic : bool
        If True, select up to `max_patches` patches with the highest temporal
        variance for this feature (based on x_phys), rather than the first
        `max_patches` in the batch.

    show_deltas : bool
        If True, add extra rows per patch showing temporal differences:
        x(t) - x(0) and recon(t) - recon(0). This makes temporal changes
        visually obvious, especially when the base field is mostly stable.

    Notes
    -----
    Base layout (show_deltas=False):
        - Rows: for each selected patch, first row = input, second = recon.
        - Columns: time dimension.

    With show_deltas=True:
        - Rows per patch:
            0: input
            1: recon
            2: input delta (x(t) - x(0))
            3: recon delta (recon(t) - recon(0))

    Each subplot uses a shared vmin/vmax across the whole figure for the
    raw values, and a separate symmetric color range for the deltas to
    highlight change.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Move tensors to numpy for plotting.
    x_np = x_phys.numpy()
    r_np = recon_phys.numpy()
    B, T, C, H, W = x_np.shape

    # Extract the single feature channel: [B, T, H, W]
    x_feat = x_np[:, :, feature_idx]
    r_feat = r_np[:, :, feature_idx]

    # AOI mask, if present.
    if aoi is not None:
        aoi_np = aoi.numpy()  # shape [B, H, W]
    else:
        aoi_np = None

    # Determine which patches to plot.
    if select_most_dynamic and B > 1:
        # Temporal variance for each patch: mean over H,W of var over T.
        # x_feat: [B, T, H, W] -> var over T -> [B, H, W] -> mean over H,W -> [B]
        var_t = np.var(x_feat, axis=1)             # [B, H, W]
        patch_scores = var_t.mean(axis=(1, 2))     # [B]
        order = np.argsort(patch_scores)[::-1]     # descending
    else:
        # Default: keep original batch order.
        order = np.arange(B)

    b_sel = min(B, max_patches)
    sel_indices = order[:b_sel]

    x_sel = x_feat[sel_indices]        # [b_sel, T, H, W]
    r_sel = r_feat[sel_indices]        # [b_sel, T, H, W]
    if aoi_np is not None:
        aoi_sel = aoi_np[sel_indices]  # [b_sel, H, W]
    else:
        aoi_sel = None

    # Shared color limits across all raw-value subplots.
    # Using min/max of inputs makes differences more visible.
    vmin = np.nanmin(x_sel)
    vmax = np.nanmax(x_sel)

    # If we are plotting deltas, precompute them now.
    if show_deltas:
        # x(t) - x(0) and recon(t) - recon(0)
        x0 = x_sel[:, 0:1, :, :]          # [b_sel, 1, H, W]
        r0 = r_sel[:, 0:1, :, :]

        dx = x_sel - x0                   # [b_sel, T, H, W]
        dr = r_sel - r0

        # Shared symmetric color range across both input and recon deltas.
        max_abs = np.nanmax(
            np.abs(np.concatenate([dx.reshape(-1), dr.reshape(-1)], axis=0))
        )
        if max_abs == 0 or not np.isfinite(max_abs):
            dvmin, dvmax = -1.0, 1.0
        else:
            dvmin, dvmax = -max_abs, max_abs

        rows_per_patch = 4
    else:
        dx = dr = None
        dvmin = dvmax = None
        rows_per_patch = 2

    nrows = b_sel * rows_per_patch
    ncols = T

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

    # Ensure axes is always 2D for uniform indexing.
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    # Populate the subplot grid.
    for i, b in enumerate(sel_indices):
        # Index in the selected arrays
        bi = i

        for t in range(T):
            # Row offsets for this patch
            base_row = i * rows_per_patch

            # -----------------------------
            # Input subplot
            # -----------------------------
            ax_in = axes[base_row, t]
            img_in = x_sel[bi, t].copy()

            # -----------------------------
            # Recon subplot
            # -----------------------------
            ax_re = axes[base_row + 1, t]
            img_re = r_sel[bi, t].copy()

            # Mask out regions outside AOI (if provided).
            if aoi_sel is not None:
                mask = ~aoi_sel[bi]  # True where outside AOI
                img_in[mask] = np.nan
                img_re[mask] = np.nan

            # Plot input
            ax_in.imshow(img_in, vmin=vmin, vmax=vmax)
            ax_in.set_xticks([])
            ax_in.set_yticks([])

            # Plot reconstruction
            ax_re.imshow(img_re, vmin=vmin, vmax=vmax)
            ax_re.set_xticks([])
            ax_re.set_yticks([])

            # Titles on the top patch only.
            if i == 0:
                ax_in.set_title(f"t={t}")

            # Row labels on the first column only.
            if t == 0:
                ax_in.set_ylabel(f"patch {b} in", rotation=90)
                ax_re.set_ylabel(f"patch {b} recon", rotation=90)

            # -----------------------------
            # Optional delta subplots
            # -----------------------------
            if show_deltas:
                ax_din = axes[base_row + 2, t]
                ax_dre = axes[base_row + 3, t]

                img_din = dx[bi, t].copy()
                img_dre = dr[bi, t].copy()

                if aoi_sel is not None:
                    mask = ~aoi_sel[bi]
                    img_din[mask] = np.nan
                    img_dre[mask] = np.nan

                ax_din.imshow(img_din, vmin=dvmin, vmax=dvmax)
                ax_din.set_xticks([])
                ax_din.set_yticks([])

                ax_dre.imshow(img_dre, vmin=dvmin, vmax=dvmax)
                ax_dre.set_xticks([])
                ax_dre.set_yticks([])

                if t == 0:
                    ax_din.set_ylabel(f"patch {b} Δ in", rotation=90)
                    ax_dre.set_ylabel(f"patch {b} Δ recon", rotation=90)

    # Figure title: include feature name if available.
    feat_label = feature_name if feature_name is not None else f"feature_{feature_idx}"
    dynamic_note = " (most dynamic patches)" if select_most_dynamic else ""
    delta_note = " + deltas" if show_deltas else ""
    fig.suptitle(
        f"Spacetime reconstruction: {feat_label}{dynamic_note}{delta_note}",
        fontsize=10,
    )

    # Make layout tight and save the figure.
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
