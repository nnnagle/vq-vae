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


def main():
    # ------------------------------------------------------------------
    # 1. Data: PatchDataset + DataLoader
    # ------------------------------------------------------------------
    # Point this at your built Zarr cube
    zarr_path = Path("/data/VA/zarr/out.zarr")

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
    x0 = next(iter(dataloader))        # [B, T, C, H, W]
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
            x = batch
            B, T, C, H, W = x.shape

            # Collapse time and channels into a single channel dimension
            # so the ConvVAE sees a standard 2D image:
            #   [B, T, C, H, W] -> [B, T*C, H, W]
            x = x.view(B, T * C, H, W).to(device)

            optimizer.zero_grad()

            # Forward pass through the VAE
            recon, mu, logvar = model(x)

            # Compute VAE loss
            loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta=beta)

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


if __name__ == "__main__":
    main()
