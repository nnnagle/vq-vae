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


    Notes
    -----
    The layout is:
        - Rows: for each selected patch, first row = input, second = recon.
        - Columns: time dimension.
    Each subplot uses a shared vmin/vmax across the whole figure to ensure
    visual comparability between inputs and reconstructions.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Move tensors to numpy for plotting.
    x_np = x_phys.numpy()
    r_np = recon_phys.numpy()
    B, T, C, H, W = x_np.shape

    # Limit to the requested subset of patches.
    b_sel = min(B, max_patches)

    # Extract the single feature channel.
    x_np = x_np[:b_sel, :, feature_idx]  # -> [b_sel, T, H, W]
    r_np = r_np[:b_sel, :, feature_idx]

    # AOI mask, if present.
    if aoi is not None:
        aoi_np = aoi[:b_sel].numpy()  # shape [b_sel, H, W]
    else:
        aoi_np = None

    # Shared color limits across all subplots.
    # Using min/max of inputs makes differences more visible.
    vmin = np.nanmin(x_np)
    vmax = np.nanmax(x_np)

    # Number of rows: two per patch (input/recon).
    nrows = b_sel * 2
    ncols = T

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

    # Ensure axes is always 2D for uniform indexing.
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    # Populate the subplot grid.
    for b in range(b_sel):
        for t in range(T):
            # Input subplot
            ax_in = axes[2 * b, t]
            img_in = x_np[b, t].copy()

            # Recon subplot
            ax_re = axes[2 * b + 1, t]
            img_re = r_np[b, t].copy()

            # Mask out regions outside AOI (if provided).
            if aoi_np is not None:
                mask = ~aoi_np[b]  # True where outside AOI
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

            # Titles and labels to help navigation.
            if b == 0:
                ax_in.set_title(f"t={t}")

            if t == 0:
                ax_in.set_ylabel(f"patch {b} in", rotation=90)
                ax_re.set_ylabel(f"patch {b} recon", rotation=90)

    # Figure title: include feature name if available.
    feat_label = feature_name if feature_name is not None else f"feature_{feature_idx}"
    fig.suptitle(f"Spacetime reconstruction: {feat_label}", fontsize=10)

    # Make layout tight and save the figure.
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
