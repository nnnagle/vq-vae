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
from src.data.categorical_meta import infer_categorical_meta_from_zarr

from src.models.categorical_embedding import CategoricalEmbeddingEncoder
from src.models.vae_v0 import ConvVAE
from src.training.losses import (
    aoi_masked_vae_loss,
    categorical_recon_loss_from_embeddings,
)
from src.training.categorical_eval import print_categorical_histograms  # NEW


def main():
    # ------------------------------------------------------------------
    # 1. Data: PatchDataset + DataLoader
    # ------------------------------------------------------------------
    # Point this at your built Zarr cube
    zarr_path = Path("/data/VA/zarr/va_cube.zarr")

    # Build normalizer once from Zarr metadata
    ds = xr.open_zarr(zarr_path)
    normalizer = build_normalizer_from_zarr(
        ds,
        group="continuous",            # adjust if your group name differs
        feature_dim="feature_continuous",  # adjust to your coord name
        enable=True,
    )

    # Categorical embedding encoder (optional)
    if "categorical" in ds:
        feature_ids, num_classes, emb_dims = infer_categorical_meta_from_zarr(ds)
        cat_encoder = CategoricalEmbeddingEncoder(
            feature_ids=feature_ids,
            num_classes=num_classes,
            emb_dims=emb_dims,
        )
        print("[INFO] categorical features:", feature_ids)
        print("[INFO] categorical embedding out_channels:", cat_encoder.out_channels)
    else:
        cat_encoder = None
        print("[INFO] no categorical group; running continuous-only.")
        

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
    batch0 = next(iter(dataloader))
    x_cont0 = batch0["x_cont"]   # [B, T, C_cont, H, W]
    x_cat0  = batch0["x_cat"]     # [B, T, C_cat,  H, W] or None
    aoi0    = batch0["aoi"]      # [B, H, W] (after collation)
    B, T, C_cont, H, W = x_cont0.shape
    # How many embedding channels will we add?
    C_emb = cat_encoder.out_channels if (cat_encoder is not None and x_cat0 is not None) else 0

    C_all = C_cont + C_emb
    in_channels = T * C_all
    print(f"[INFO] Batch shape: {x_cont0.shape}  -> in_channels={in_channels}")
    print(f"[INFO] C_cont={C_cont}, C_emb={C_emb}, T={T} -> in_channels={in_channels}")

    # ------------------------------------------------------------------
    # 2. Model + optimizer setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = ConvVAE(in_channels=in_channels, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if cat_encoder is not None:
        cat_encoder = cat_encoder.to(device)
        
    # ------------------------------------------------------------------
    # 3. Tiny training loop (smoke test)
    # ------------------------------------------------------------------
    # This is not “real training” yet — just enough iterations to see
    # whether the loss behaves and gradients flow.
    max_steps = 100
    beta = 0.1  # smaller KL weight to start
    lambda_cat = 1.0 # weight on categorical loss

    step = 0
    while step < max_steps:
        for batch in dataloader:
            # batch: [B, T, C, H, W]
            x_cont = batch["x_cont"]
            x_cat  = batch.get("x_cat", None)   # [B, T, C_cat, H, W] or None
            x_mask = batch.get("x_mask", None)  # [B, T, C_mask, H, W] or None
            aoi    = batch.get("aoi")

            B, T, C_cont, H, W = x_cont.shape

            # Collapse time and channels into a single channel dimension
            # so the ConvVAE sees a standard 2D image:

            x_cont = x_cont.to(device, non_blocking=True)
            if x_cat is not None:
                x_cat = x_cat.to(device, non_blocking=True)
            aoi = aoi.to(device, non_blocking=True)     # [B, H, W]

            x_bt = x_cont.view(B * T, C_cont, H, W)      # [B*T, C, H, W]
            x_bt_norm = normalizer(x_bt)       # [B*T, C, H, W]
            x_cont_norm = x_bt_norm.view(B, T, C_cont, H, W)

            if cat_encoder is not None and x_cat is not None:
                x_cat_emb = cat_encoder(x_cat)         # [B, T, C_cat_oh, H, W]
                x_all = torch.cat([x_cont_norm, x_cat_emb], dim=2)
            else:
                x_all = x_cont_norm
            
            
            B, T, C_all, H, W = x_all.shape
            # Flatten time × channels for ConvVAE
            x_flat_all = x_all.view(B, T * C_all, H, W)      # [B, T*C_all, H, W]
            # continuous-only flattened (for recon loss)
            x_flat_cont = x_cont_norm.view(B, T * C_cont, H, W)


            if aoi.dim() == 3:
              aoi = aoi.unsqueeze(1)  # [B, 1, H, W]

            optimizer.zero_grad()

            # Forward pass through the VAE
            recon, mu, logvar = model(x_flat_all)

            # Compute VAE loss
            #loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, beta=beta)
            recon_full = recon.view(B, T, C_all, H, W)
            recon_cont = recon_full[:, :, :C_cont, ...]              # [B,T,C_cont,H,W]
            recon_cont_flat = recon_cont.reshape(B, T * C_cont, H, W)   # [B, T*C_cont, H, W]
            vae_total, cont_recon_loss, kl = aoi_masked_vae_loss(
                recon_cont_flat, x_flat_cont, mu, logvar, aoi, beta=beta
            )
            cat_loss = categorical_recon_loss_from_embeddings(
                recon_full=recon_full,
                x_cat=x_cat,
                cat_encoder=cat_encoder,
                C_cont=C_cont,
            )
            if not torch.isfinite(cat_loss):
                print(f"[DEBUG] non-finite cat_loss at step={step}, setting to zero.")
                cat_loss = torch.tensor(0.0, device=cat_loss.device)
            
            loss = vae_total + lambda_cat * cat_loss

            # Backprop + update
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"[step {step:04d}] "
                    f"loss={loss.item():.4f}  "
                    f"cont_recon_loss={cont_recon_loss.item():.4f}  "
                    f"cat_loss={cat_loss.item():.4f} "
                    f"kl={kl.item():.4f} "
                )

            step += 1
            if step >= max_steps:
                break
    # ------------------------------------------------------------------
    # 4. Model evaluation
    # ------------------------------------------------------------------
    # Tiny model evaluation
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))

        x_cont = batch["x_cont"]    # [B, T, C_cont, H, W]
        x_cat  = batch["x_cat"]     # [B, T, C_cat, H, W] or None
        aoi    = batch["aoi"]       # [B, H, W] or [B, 1, H, W]

        x_cont = x_cont.to(device)
        if x_cat is not None:
            x_cat = x_cat.to(device)
        aoi    = aoi.to(device)

        B, T, C_cont, H, W = x_cont.shape

       
        # --- forward preprocessing (same as in training) ---
        x_bt = x_cont.view(B * T, C_cont, H, W)        # [B*T, C, H, W]
        x_bt_norm = normalizer(x_bt)         # normalized
        x_cont_norm = x_bt_norm.view(B, T, C_cont, H, W)
        
        # categorical → embeddings
        if cat_encoder is not None and x_cat is not None:
            x_cat_emb = cat_encoder(x_cat)
            x_all = torch.cat([x_cont_norm, x_cat_emb], dim=2)
        else:
            x_all = x_cont_norm
        
        B, T, C_all, H, W = x_all.shape
        x_flat = x_all.view(B, T * C_all, H, W) # [B, T*C, H, W]

        recon, mu, logvar = model(x_flat)    # recon is normalized space

        recon_full = recon.view(B, T, C_all, H, W)
        # --- back-transform reconstructions to physical space ---
        recon_cont = recon_full[:, :, :C_cont, ...]   #[B, T, C_Cont, H, W]
        recon_bt = recon_cont.reshape(B * T, C_cont, H, W)         # [B*T, C, H, W]
        recon_phys_bt = normalizer.unnormalize(recon_bt)
        recon_phys = recon_phys_bt.view(B, T, C_cont, H, W)

        x_phys_bt = normalizer.unnormalize(x_bt_norm) # or x_bt_norm
        x_phys = x_phys_bt.view(B, T, C_cont, H, W)

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
    
    if (cat_encoder is not None) and (x_cat is not None) and (num_classes is not None):
        print_categorical_histograms(
            recon_full=recon_full,
            x_cat=x_cat,
            cat_encoder=cat_encoder,
            num_classes=num_classes,
            C_cont=C_cont,
        )
    else:
        print("\n[INFO] skipping categorical histograms (no categorical encoder/data).")
    

if __name__ == "__main__":
    main()
