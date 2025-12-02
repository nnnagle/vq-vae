# train_vae_v0.py
#
# Minimal end-to-end training script for ConvVAE + PatchDataset.
#
# Now uses:
#   - spatial train/val/test splits from PatchDataset(split=...),
#   - a debug window option for fast experiments,
#   - epoch-based train/val loops (from src.training.loops),
#   - simple checkpointing + reconstruction diagnostics.
#

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import optim
import xarray as xr

from src.data.normalization import build_normalizer_from_zarr
from src.data.patch_dataset import PatchDataset
from src.data.categorical_meta import infer_categorical_meta_from_zarr

from src.models.categorical_embedding import CategoricalEmbeddingEncoder
from src.models.vae_v0 import ConvVAE
from src.training.categorical_eval import print_categorical_histograms
from src.training.loops import train_one_epoch, eval_one_epoch   # <-- NEW import

# ----------------------------------------------------------------------
# Debug controls
# ----------------------------------------------------------------------

DEBUG_WINDOW = True     # Flip to False for full-domain training

# Debug window parameters (used only if DEBUG_WINDOW=True)
DEBUG_WINDOW_ORIGIN = (256 * 10, 256 * 20)   # (y0, x0)
DEBUG_WINDOW_SIZE   = (1024, 1024)           # (height, width)
DEBUG_BLOCK_WIDTH   = 1
DEBUG_BLOCK_HEIGHT  = 1

# Default block size when not debugging
DEFAULT_BLOCK_WIDTH  = 7
DEFAULT_BLOCK_HEIGHT = 7


def main():
    patch_size = 256

    # ----------------------------------------------
    # Debug window vs full domain
    # ----------------------------------------------
    if DEBUG_WINDOW:
        window_origin = DEBUG_WINDOW_ORIGIN
        window_size   = DEBUG_WINDOW_SIZE
        block_width   = DEBUG_BLOCK_WIDTH
        block_height  = DEBUG_BLOCK_HEIGHT

        print(f"[INFO] DEBUG WINDOW ENABLED @ origin={window_origin}, size={window_size}")
    else:
        window_origin = None
        window_size   = None
        block_width   = DEFAULT_BLOCK_WIDTH
        block_height  = DEFAULT_BLOCK_HEIGHT

        print("[INFO] DEBUG WINDOW DISABLED — using full domain")

    # ------------------------------------------------------------------
    # 1. Data: PatchDataset + DataLoader
    # ------------------------------------------------------------------
    zarr_path = Path("/data/VA/zarr/va_cube.zarr")

    # Build normalizer once from Zarr metadata
    ds = xr.open_zarr(zarr_path)
    normalizer = build_normalizer_from_zarr(
        ds,
        group="continuous",
        feature_dim="feature_continuous",
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
        feature_ids = None
        num_classes = None
        print("[INFO] no categorical group; running continuous-only.")

    # Build separate train/val/test datasets using spatial split (+ optional window)
    train_dataset = PatchDataset(
        zarr_path,
        patch_size=patch_size,
        split="train",
        block_width=block_width,
        block_height=block_height,
        window_origin=window_origin,
        window_size=window_size,
    )

    val_dataset = PatchDataset(
        zarr_path,
        patch_size=patch_size,
        split="val",
        block_width=block_width,
        block_height=block_height,
        window_origin=window_origin,
        window_size=window_size,
    )

    test_dataset = PatchDataset(
        zarr_path,
        patch_size=patch_size,
        split="test",
        block_width=block_width,
        block_height=block_height,
        window_origin=window_origin,
        window_size=window_size,
    )

    print(
        f"[INFO] dataset sizes: "
        f"train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Peek at one TRAIN batch to determine input shape
    batch0 = next(iter(train_loader))
    x_cont0 = batch0["x_cont"]   # [B, T, C_cont, H, W]
    x_cat0  = batch0["x_cat"]    # [B, T, C_cat,  H, W] or None
    B, T, C_cont, H, W = x_cont0.shape

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
    # 3. Epoch-based training loop
    # ------------------------------------------------------------------
    num_epochs = 5      # still small for now
    beta = 0.1
    lambda_cat = 1.0

    best_val_loss = float("inf")

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            normalizer=normalizer,
            cat_encoder=cat_encoder,
            beta=beta,
            lambda_cat=lambda_cat,
        )

        val_metrics = eval_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            normalizer=normalizer,
            cat_encoder=cat_encoder,
            beta=beta,
            lambda_cat=lambda_cat,
        )

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_cont={train_metrics['cont_recon']:.4f}  "
            f"train_cat={train_metrics['cat_loss']:.4f}  "
            f"train_kl={train_metrics['kl']:.4f}  |  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_cont={val_metrics['cont_recon']:.4f}  "
            f"val_cat={val_metrics['cat_loss']:.4f}  "
            f"val_kl={val_metrics['kl']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = ckpt_dir / "vae_v0_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "beta": beta,
                    "lambda_cat": lambda_cat,
                },
                ckpt_path,
            )
            print(f"[INFO] Saved new best model at epoch {epoch} (val_loss={best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # 4. Final reconstruction diagnostics on a single VAL batch
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        try:
            val_batch = next(iter(val_loader))
        except StopIteration:
            print("[WARN] val_loader is empty; skipping final reconstruction diagnostics.")
            return

        x_cont = val_batch["x_cont"]    # [B, T, C_cont, H, W]
        x_cat  = val_batch["x_cat"]     # [B, T, C_cat, H, W] or None

        x_cont = x_cont.to(device)
        if x_cat is not None:
            x_cat = x_cat.to(device)

        B, T, C_cont, H, W = x_cont.shape

        # forward preprocessing (same as training)
        x_bt = x_cont.view(B * T, C_cont, H, W)
        x_bt_norm = normalizer(x_bt)
        x_cont_norm = x_bt_norm.view(B, T, C_cont, H, W)

        if cat_encoder is not None and x_cat is not None:
            x_cat_emb = cat_encoder(x_cat)
            x_all = torch.cat([x_cont_norm, x_cat_emb], dim=2)
        else:
            x_all = x_cont_norm

        B, T, C_all, H, W = x_all.shape
        x_flat = x_all.view(B, T * C_all, H, W)

        recon, mu, logvar = model(x_flat)
        recon_full = recon.view(B, T, C_all, H, W)

        # continuous back-transform
        recon_cont = recon_full[:, :, :C_cont, ...]
        recon_bt = recon_cont.reshape(B * T, C_cont, H, W)
        recon_phys_bt = normalizer.unnormalize(recon_bt)
        recon_phys = recon_phys_bt.view(B, T, C_cont, H, W)

        x_phys_bt = normalizer.unnormalize(x_bt_norm)
        x_phys = x_phys_bt.view(B, T, C_cont, H, W)

    x0_norm = x_flat[0].cpu().numpy()
    r0_norm = recon[0].cpu().numpy()

    x0_phys = x_phys[0].cpu().numpy()
    r0_phys = recon_phys[0].cpu().numpy()

    print("NORMALIZED (VAL example):")
    print("  input min/max:", x0_norm.min(), x0_norm.max())
    print("  recon min/max:", r0_norm.min(), r0_norm.max())
    print("  input mean/std:", x0_norm.mean(), x0_norm.std())
    print("  recon mean/std:", r0_norm.mean(), r0_norm.std())

    print("\nPHYSICAL (VAL example):")
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
