# train_vae_v0.py
#
# Minimal end-to-end training script for ConvVAE + PatchDataset.
#
# Pipeline:
#   Zarr cube  ->  PatchDataset  ->  DataLoader  ->  ConvVAE  ->  VAE loss
#
# This script is deliberately simple:
#   - Uses only the continuous group (via PatchDataset v0).
#   - Treats time × channels as a single big channel dimension.
#   - Runs a short training loop just to confirm:
#       * data flows correctly,
#       * model compiles and runs,
#       * loss is finite and roughly decreases.
#
# Once this works, *then* you start adding:
#   - normalization,
#   - masks,
#   - categorical features,
#   - better architecture,
#   - logging/metrics, etc.


from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import xarray as xr

from src.data.normalization import build_normalizer_from_zarr
from src.data.patch_dataset import PatchDataset
from src.models.vae_v0 import ConvVAE


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    """
    Standard VAE loss = reconstruction loss + β * KL divergence.

    Args
    ----
    recon : torch.Tensor
        Reconstructed input [B, C_in, H, W].
    x : torch.Tensor
        Original input [B, C_in, H, W].
    mu : torch.Tensor
        Latent mean [B, latent_dim].
    logvar : torch.Tensor
        Latent log-variance [B, latent_dim].
    beta : float
        Weight on the KL term. beta=1.0 → standard VAE;
        beta < 1.0 → softer regularization.

    Returns
    -------
    total_loss : torch.Tensor (scalar)
    recon_loss : torch.Tensor (scalar)
    kl : torch.Tensor (scalar)
    """
    # Pixel-wise reconstruction error (MSE here; you can try L1 later)
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    # KL divergence term for N(mu, sigma^2) vs N(0, 1)
    # KL = -0.5 * Σ (1 + log σ^2 - μ^2 - σ^2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl
    return total, recon_loss, kl

def aoi_masked_vae_loss(recon, x, mu, logvar, aoi_mask, beta: float = 1.0):
    """
    recon, x:   [B, C_in, H, W]
    aoi_mask:   [B, H, W] or [B, 1, H, W], True = inside AOI
    """
    # Ensure aoi_mask is [B, 1, H, W] and boolean
    if aoi_mask.dim() == 3:
        aoi_mask = aoi_mask.unsqueeze(1)  # [B, 1, H, W]
    aoi_mask = aoi_mask.bool()

    # Broadcast AOI to all channels
    # x: [B, C_in, H, W], mask: [B, 1, H, W] -> [B, C_in, H, W]
    mask = aoi_mask.expand(-1, x.size(1), -1, -1)

    diff = (recon - x) * mask
    valid = mask.sum()

    if valid == 0:
        recon_loss = torch.tensor(0.0, device=x.device)
    else:
        recon_loss = (diff ** 2).sum() / valid

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


def main():
    # ------------------------------------------------------------------
    # 1. Data: PatchDataset + DataLoader
    # ------------------------------------------------------------------
    # Point this at your built Zarr cube
    zarr_path = Path("/data/VA/va_cube.zarr")

    # Build normalizer once from Zarr metadata
    ds = xr.open_zarr(zarr_path)
    normalizer = build_normalizer_from_zarr(
        ds,
        group="continuous",            # adjust if your group name differs
        feature_dim="feature_continuous",  # adjust to your coord name
        enable=True,
    )

    # Dataset: each item is [T, C_cont, 256, 256]
    dataset = PatchDataset(zarr_path, patch_size=256)

    # DataLoader: small batch size, single worker for v0 debugging
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Peek at one batch to determine input shape
    x0, aoi0 = next(iter(dataloader))        # [B, T, C, H, W]
    B, T, C, H, W = x0.shape
    in_channels = T * C               # flatten time × channels for v0
    print(f"[INFO] Batch shape: {x0.shape}  -> in_channels={in_channels}")

    # ------------------------------------------------------------------
    # 2. Model + optimizer setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = ConvVAE(in_channels=in_channels, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ------------------------------------------------------------------
    # 3. Tiny training loop (smoke test)
    # ------------------------------------------------------------------
    # This is not “real training” yet — just enough iterations to see
    # whether the loss behaves and gradients flow.
    max_steps = 100
    beta = 0.1  # smaller KL weight to start

    step = 0
    while step < max_steps:
        for batch in dataloader:
            # batch: [B, T, C, H, W]
            x, aoi = batch
            B, T, C, H, W = x.shape

            # Collapse time and channels into a single channel dimension
            # so the ConvVAE sees a standard 2D image:

            x = x.to(device)  # [B, T, H, W]
            aoi = aoi.to(device)                         # [B, H, W]

            x_bt = x.view(B * T, C, H, W)      # [B*T, C, H, W]
            x_bt_norm = normalizer(x_bt)       # [B*T, C, H, W]
            x_norm = x_bt_norm.view(B, T, C, H, W)
            x_flat = x_norm.view(B, T * C, H, W)   # [B, T*C, H, W]

            if aoi.dim() == 3:
              aoi = aoi.unsqueeze(1)  # [B, 1, H, W]

            optimizer.zero_grad()

            # Forward pass through the VAE
            recon, mu, logvar = model(x_flat)

            # Compute VAE loss
            #loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta=beta)
            loss, recon_loss, kl = aoi_masked_vae_loss(
                recon, x_flat, mu, logvar, aoi, beta=beta
            )

            # Backprop + update
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"[step {step:04d}] "
                    f"loss={loss.item():.4f}  "
                    f"recon={recon_loss.item():.4f}  "
                    f"kl={kl.item():.4f}"
                )

            step += 1
            if step >= max_steps:
                break


    model.eval()
    with torch.no_grad():
        x, aoi = next(iter(dataloader))       # [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.to(device)

        # --- forward preprocessing (same as in training) ---
        x_bt = x.view(B * T, C, H, W)        # [B*T, C, H, W]
        x_bt_norm = normalizer(x_bt)         # normalized
        x_norm = x_bt_norm.view(B, T, C, H, W)
        x_flat = x_norm.view(B, T * C, H, W) # [B, T*C, H, W]

        recon, mu, logvar = model(x_flat)    # recon is normalized space

        # --- back-transform reconstructions to physical space ---
        recon_bt = recon.view(B * T, C, H, W)         # [B*T, C, H, W]
        recon_phys_bt = normalizer.unnormalize(recon_bt)
        recon_phys = recon_phys_bt.view(B, T, C, H, W)

        x_phys_bt = normalizer.unnormalize(x_bt_norm) # or x_bt_norm
        x_phys = x_phys_bt.view(B, T, C, H, W)

    # For sanity, look at one example & one feature over time flattened
    x0_norm = x_flat[0].cpu().numpy()             # normalized, [T*C, H, W] flattened earlier
    r0_norm = recon[0].cpu().numpy()

    # Back-transformed:
    x0_phys = x_phys[0].cpu().numpy()            # [T, C, H, W]
    r0_phys = recon_phys[0].cpu().numpy()

    print("NORMALIZED:")
    print("  input min/max:", x0_norm.min(), x0_norm.max())
    print("  recon min/max:", r0_norm.min(), r0_norm.max())
    print("  input mean/std:", x0_norm.mean(), x0_norm.std())
    print("  recon mean/std:", r0_norm.mean(), r0_norm.std())

    print("\nPHYSICAL:")
    print("  input min/max:", x0_phys.min(), x0_phys.max())
    print("  recon min/max:", r0_phys.min(), r0_phys.max())
    print("  input mean/std:", x0_phys.mean(), x0_phys.std())
    print("  recon mean/std:", r0_phys.mean(), r0_phys.std())
if __name__ == "__main__":
    main()
