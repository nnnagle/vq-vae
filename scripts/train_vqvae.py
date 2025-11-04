#!/usr/bin/env python3
# =============================================================================
# scripts/train_vqvae.py — End-to-end training for mixed geospatial VQ-VAE
#
# Workflow
#   1) Read feature_meta from the Zarr → build exposure-aware schema → save
#   2) Create VQVAEDataset/DataLoader using that schema
#   3) Define model: MixedInputEncoder → VectorQuantizer → MixedDecoder heads
#        - Decoder predicts: continuous features, categorical logits, canopy scalar (final T)
#   4) Train with AMP, class-weighted CE (cats), MSE (cont), canopy MSE, VQ loss
#
# CLI (example)
#   python train_vqvae.py \
#     --zarr data/cube.zarr --run_dir runs/exp_001 \
#     --batch_size 64 --steps_per_epoch 10000 --epochs 5 \
#     --min_hits_per_epoch 100 --mass_coverage 0.999 --vocab_cap 5000 \
#     --codebook_size 256 --emb_dim 128 --beta 0.25
# =============================================================================

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import xarray as xr

# ---- project utils (expected layout under utils/)
from utils.schema import (
    build_categorical_schema,
    attach_continuous_stats,
    save_schema,
)

from utils.train_debug_utils import StepTimers, print_device_summary, maybe_sync_cuda


# IGNORE_INDEX fallback in case the loader doesn't export it
try:
    from utils.loader import VQVAEDataset, default_collate_fn, IGNORE_INDEX
except Exception:  # pragma: no cover
    from utils.loader import VQVAEDataset, default_collate_fn
    IGNORE_INDEX = -100

from utils.weights import cat_class_weights
from utils.argyaml import parse_args_with_yaml
from utils.samplers import ChunkBatchSampler

# -------------------------------------------------------------------
# Enable optimized matrix math on Ada / Ampere GPUs (like RTX 4060)
# -------------------------------------------------------------------
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
# Optional: you can DROP torch.set_float32_matmul_precision entirely.


# ------------------------------- Helpers --------------------------------

def make_run_dir(run_dir: str) -> Path:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_feature_meta_from_zarr(zarr_path: str) -> Dict[str, Any]:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    raw = ds.attrs.get("feature_meta", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"feature_meta not found or invalid: {e}")

def build_and_save_schema(zarr_path: str, run_dir: Path,
                          batch_size: int, steps_per_epoch: int,
                          min_hits_per_epoch: int, mass_coverage: Optional[float],
                          vocab_cap: Optional[int]) -> Path:
    fm = read_feature_meta_from_zarr(zarr_path)
    cat_schema = build_categorical_schema(
        fm, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
        min_hits_per_epoch=min_hits_per_epoch,
        mass_coverage=mass_coverage, vocab_cap=vocab_cap
    )
    schema = attach_continuous_stats(cat_schema, fm)
    schema_path = run_dir / "schema.json"
    save_schema(schema, str(schema_path))
    # also save raw feature_meta for provenance
    with open(run_dir / "feature_meta.json", "w") as f:
        json.dump(fm, f, indent=2)
    return schema_path

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mse_ignore_nan(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE that ignores NaNs in target."""
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    diff = (pred - torch.nan_to_num(target, nan=0.0)) ** 2
    return diff[mask].mean()


# ------------------------------- Model ----------------------------------

class MixedInputEncoder(nn.Module):
    """
    Inputs:
      - cont: [B,T,C_cont] (z-scored; NaNs possible)
      - cat:  [B,T,C_cat]  (dense IDs; used by embeddings)
      - naip: [B,Bn,3,3]   (single-band canopy height patch; Bn==1)
    Output:
      - z_e:  [B,T,D]      (pre-quantization latent)
    """
    def __init__(self, cont_dim: int, cat_dims: List[int], naip_bands: int,
                 emb_dim: int, cat_emb_dim: int = 6, hidden: int = 128):
        super().__init__()
        # per-categorical Embedding tables
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(v, cat_emb_dim) if v > 0 else None for v in cat_dims]
        )

        # tiny CNN for NAIP patch (bands-first; here naip_bands==1)
        self.naip_cnn = nn.Sequential(
            nn.Conv2d(in_channels=naip_bands, out_channels=hidden, kernel_size=3),  # -> [B,hidden,1,1]
            nn.ReLU(),
        )
        self.naip_proj = nn.Linear(hidden, hidden)

        # projections for other inputs
        self.cont_proj = nn.Linear(max(cont_dim, 1), hidden) if cont_dim > 0 else None
        self.cat_dim_total = sum(cat_emb_dim for v in cat_dims if v > 0)
        self.cat_proj = nn.Linear(max(self.cat_dim_total, 1), hidden) if self.cat_dim_total > 0 else None

        # fuse to latent
        fuse_in = hidden + (hidden if self.cont_proj else 0) + (hidden if self.cat_proj else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # T inferred from cont if present; else from cat
        if batch["cont"].ndim == 3:
            B, T, _ = batch["cont"].shape
        else:
            B, T, _ = batch["cat"].shape

        # NAIP context (shared across T)
        naip = torch.nan_to_num(batch["naip"], nan=0.0, posinf=0.0, neginf=0.0)
        naip_feat = self.naip_cnn(naip).flatten(1)  # [B,hidden]
        naip_feat = self.naip_proj(naip_feat)
        naip_rep = naip_feat.unsqueeze(1).expand(-1, T, -1) # [B,T,hidden]

        # Continuous path
        if self.cont_proj is not None:
            cont = torch.nan_to_num(batch["cont"], nan=0.0, posinf=0.0, neginf=0.0)
            cont_h = self.cont_proj(cont)                    # [B,T,hidden]
        else:
            cont_h = torch.zeros(B, T, 0, device=naip_rep.device)

        # Categorical path
        if self.cat_proj is not None and self.cat_dim_total > 0:
            cat_ids = batch["cat"]                           # [B,T,K]
            cat_embs = []
            col = 0
            for emb in self.cat_embs:
                if emb is None:
                    continue
                cat_embs.append(emb(cat_ids[..., col]))     # [B,T,cat_emb_dim]
                col += 1
            if cat_embs:
                cat_concat = torch.cat(cat_embs, dim=-1)
            else:
                cat_concat = torch.zeros(B, T, 0, device=naip_rep.device)
            cat_h = self.cat_proj(cat_concat)               # [B,T,hidden]
        else:
            cat_h = torch.zeros(B, T, 0, device=naip_rep.device)

        fused = torch.cat([naip_rep, cont_h, cat_h], dim=-1)
        z_e = self.fuse(fused)                               # [B,T,D]
        return z_e


class VectorQuantizer(nn.Module):
    """Straight-Through Vector Quantizer (ST-VQ)."""
    def __init__(self, codebook_size: int, emb_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.codebook = nn.Parameter(torch.randn(codebook_size, emb_dim) * 0.05)

    def forward(self, z_e: torch.Tensor):
        # flatten
        z = z_e.reshape(-1, self.emb_dim)                    # [N,D]
        z_sq = (z ** 2).sum(dim=1, keepdim=True)             # [N,1]
        e_sq = (self.codebook ** 2).sum(dim=1)               # [K]
        distances = z_sq + e_sq.unsqueeze(0) - 2 * (z @ self.codebook.t())
        indices = torch.argmin(distances, dim=1)             # [N]
        z_q = F.embedding(indices, self.codebook).view_as(z_e)

        # VQ losses
        commitment = F.mse_loss(z_e.detach(), z_q, reduction="mean")
        codebook = F.mse_loss(z_e, z_q.detach(), reduction="mean")
        z_q_st = z_e + (z_q - z_e).detach()
        loss_vq = codebook + self.beta * commitment

        with torch.no_grad():
            one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
            avg_probs = one_hot.mean(dim=0)
            perplexity = torch.exp(- (avg_probs * (avg_probs + 1e-12).log()).sum())

        return z_q_st, indices.view(z_e.shape[:-1]), loss_vq, perplexity


class MixedDecoder(nn.Module):
    """
    Decodes latent [B,T,D] into:
      - continuous regression: [B,T,C_cont]
      - categorical logits: dict(name -> [B,T,num_ids])
      - canopy scalar: [B] using final time step only
    """
    def __init__(self, emb_dim: int, cont_dim: int, cat_vocab_sizes: Dict[str, int], hidden: int = 128):
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_names = list(cat_vocab_sizes.keys())
        self.cat_vocab_sizes = cat_vocab_sizes

        self.backbone = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.cont_head = nn.Linear(hidden, cont_dim) if cont_dim > 0 else None
        self.cat_heads = nn.ModuleDict({name: nn.Linear(hidden, v) for name, v in cat_vocab_sizes.items()})
        self.canopy_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z: torch.Tensor):
        h = self.backbone(z)  # [B,T,H]
        cont = self.cont_head(h) if self.cont_head is not None else None
        cat_logits = {name: head(h) for name, head in self.cat_heads.items()}

        # canopy from final time step
        h_last = h[:, -1, :]                          # [B,H]
        canopy_pred = self.canopy_head(h_last).squeeze(-1)  # [B]
        return cont, cat_logits, canopy_pred


class VQVAE(nn.Module):
    def __init__(self,
                 cont_dim: int,
                 cat_vocab_sizes: Dict[str, int],
                 naip_bands: int,
                 emb_dim: int,
                 codebook_size: int,
                 beta: float = 0.25,
                 hidden: int = 128,
                 cat_emb_dim: int = 6):
        super().__init__()
        self.encoder = MixedInputEncoder(
            cont_dim=cont_dim,
            cat_dims=list(cat_vocab_sizes.values()),
            naip_bands=naip_bands,
            emb_dim=emb_dim,
            cat_emb_dim=cat_emb_dim,
            hidden=hidden,
        )
        self.quant = VectorQuantizer(codebook_size=codebook_size, emb_dim=emb_dim, beta=beta)
        self.decoder = MixedDecoder(emb_dim=emb_dim, cont_dim=cont_dim,
                                    cat_vocab_sizes=cat_vocab_sizes, hidden=hidden)

    def forward(self, batch):
        z_e = self.encoder(batch)
        z_q, _, vq_loss, perplexity = self.quant(z_e)
        cont_pred, cat_logits, canopy_pred = self.decoder(z_q)
        return cont_pred, cat_logits, canopy_pred, vq_loss, perplexity


# ------------------------------ Training --------------------------------

def maybe_compute_canopy_target_from_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Robust canopy target:
      - If 'canopy' exists in batch, use it.
      - Else compute a **NaN-safe** mean over the 3x3 NAIP patch.
        Prefer weighted mean using 'naip_nan_mask' when present; otherwise use torch.nanmean.
    """
    if "canopy" in batch:
        return batch["canopy"]

    naip = batch["naip"]  # [B,Bn,3,3] or [B,Bn,H,W]
    if "naip_nan_mask" in batch:
        # mask: 1 where NaN in original; valid weights = 1 - mask
        mask = batch["naip_nan_mask"].to(naip.dtype)
        w = (1.0 - mask)
        num = (naip * w).sum(dim=(-1, -2))              # sum over spatial
        den = w.sum(dim=(-1, -2)).clamp_min(1.0)
        canopy = (num / den)                             # [B,Bn]
    else:
        canopy = torch.nanmean(naip, dim=(-1, -2))      # [B,Bn]

    # pick band 0 if multi-band
    if canopy.ndim == 2 and canopy.size(1) >= 1:
        canopy = canopy[:, 0]
    return canopy  # [B]

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dir = make_run_dir(args.run_dir)

    # 1) Build & save schema (exposure-aware collapse)
    schema_path = build_and_save_schema(
        zarr_path=args.zarr,
        run_dir=run_dir,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        min_hits_per_epoch=args.min_hits_per_epoch,
        mass_coverage=args.mass_coverage,
        vocab_cap=args.vocab_cap,
    )

    # 2) Dataset / DataLoader
    ds = VQVAEDataset(args.zarr, str(schema_path), eager=args.eager, ignore_unk_in_loss=True)
    print(f"[debug] dataset length = {len(ds)}")
    if len(ds) == 0:
      raise RuntimeError("[debug] Dataset is empty. Check your mask and indexing.")
    batch_sampler = ChunkBatchSampler(
      ds.xy_by_chunk,
      batch_size=args.batch_size,
      drop_last=False,
      replacement_within_chunk=False,   # flip to True if strong class imbalance
      seed=42
    )
    # loader = DataLoader(
    #     ds, 
    #     batch_size=args.batch_size, 
    #     shuffle=True,
    #     #num_workers=0, # No workers to debug
    #     num_workers=args.num_workers, 
    #     pin_memory=True, # not needed on CPU 
    #     collate_fn=default_collate_fn,
    #     persistent_workers=True,
    #     prefetch_factor=4,
    #     drop_last=False,
    #     multiprocessing_context="spawn",
    #     timeout=300 # no timeout while debugging
    # )
    loader = DataLoader(
      ds,
      batch_sampler=batch_sampler,
      num_workers=args.num_workers,
      pin_memory=True,
      collate_fn=default_collate_fn,
      persistent_workers=True,
      prefetch_factor=8,
      multiprocessing_context="spawn",
      timeout=300,
    )

    # cat vocab sizes (dict name -> num_ids) in the same order as ds.cat_names
    cat_vocab_sizes: Dict[str, int] = {}
    for name in ds.cat_names:
        entry = ds.schema_cat.get(name)
        if entry is not None:
            cat_vocab_sizes[name] = int(entry["num_ids"])

    naip_bands = int(ds.naip.shape[-1])           # krow,kcol,band → bands
    cont_dim = len(ds.cont_names)

    # 3) Model
    model = VQVAE(
        cont_dim=cont_dim,
        cat_vocab_sizes=cat_vocab_sizes,
        naip_bands=naip_bands,
        emb_dim=args.emb_dim,
        codebook_size=args.codebook_size,
        beta=args.beta,
        hidden=args.hidden,
        cat_emb_dim=args.cat_emb_dim,
    ).to(device)
    print(f"Model params: {count_params(model)/1e6:.2f}M")
    print(f"\n[Device Summary]")
    if device.type == "cuda":
      print(f"  → Using GPU: {torch.cuda.get_device_name(device)}")
      print(f"  → CUDA version: {torch.version.cuda}")
      print(f"  → Memory allocated: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
      print(f"  → Memory reserved:  {torch.cuda.memory_reserved(device)/1e6:.2f} MB")
    else:
      print("  → Using CPU (no CUDA detected or --cpu flag used).")

    # Sanity check: ensure model is on GPU when expected
    first_param_device = next(model.parameters()).device
    if device.type == "cuda" and first_param_device.type != "cuda":
      print("⚠️  Warning: model parameters are still on CPU despite CUDA being available!")
    elif device.type == "cpu" and first_param_device.type == "cuda":
      print("⚠️  Warning: model parameters are on GPU but --cpu flag was used!")
    
    # class weights for categorical heads
    class_weights = {name: ds.class_weights_by_cat_name(name).to(device) for name in ds.cat_names}

    # Optimizer / AMP scaler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    #scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp)) # Deprecated
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    # cosine schedule over total steps
    total_steps = args.epochs * args.steps_per_epoch
    def lr_at(step):
        if total_steps <= 1: return args.lr
        cos = 0.5 * (1 + math.cos(math.pi * step / total_steps))
        return args.min_lr + (args.lr - args.min_lr) * cos
    
    timers = StepTimers()
    model.train()
    step = 0
    best_loss = float("inf")
    model.train()

    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch: {epoch}")
        running = {"recon_cont": 0.0, "recon_cat": 0.0, "canopy": 0.0, "vq": 0.0, "total": 0.0}
        count = 0

        for i,batch in enumerate(loader):
            if i >= args.steps_per_epoch:
              break
            timers.mark_load()
            # LR schedule
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            # to device (move masks too if present)
            for k in ("cont", "cat", "cat_target", "naip", "years", "yx", "cont_nan_mask", "naip_nan_mask"):
                if k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

            # canopy scalar (compute if loader didn't add it), NaN-safe
            canopy_target = maybe_compute_canopy_target_from_batch(batch).to(device)

            opt.zero_grad(set_to_none=True)
            with maybe_sync_cuda(device):
              with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16 if args.bf16 else torch.float16,
                                enabled=(device.type == "cuda" and not args.no_amp)):
                cont_pred, cat_logits, canopy_pred, vq_loss, perplexity = model(batch)
                timers.mark_fwd()

                # Sanitize outputs before loss (belt-and-suspenders)
                if cont_pred is not None:
                    cont_pred = torch.nan_to_num(cont_pred, nan=0.0, posinf=0.0, neginf=0.0)
                canopy_pred = torch.nan_to_num(canopy_pred, nan=0.0, posinf=0.0, neginf=0.0)
                cat_logits = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in cat_logits.items()}

                # Continuous recon loss (already NaN-safe via target mask)
                if cont_pred is not None and batch["cont"].shape[-1] > 0:
                    loss_cont = mse_ignore_nan(cont_pred, batch["cont"])
                else:
                    loss_cont = cont_pred.new_tensor(0.0) if cont_pred is not None else torch.tensor(0.0, device=device)

                # Categorical recon loss (sum over features) with IGNORE_INDEX
                loss_cat = 0.0
                for j, name in enumerate(ds.cat_names):
                    logits = cat_logits[name]                         # [B,T,num_ids]
                    target = batch["cat_target"][..., j]              # [B,T]
                    w = class_weights[name]                           # [num_ids]
                    ce = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target.reshape(-1),
                        weight=w,
                        ignore_index=IGNORE_INDEX,
                        reduction="mean"
                    )
                    loss_cat = loss_cat + ce

                # Canopy scalar MSE (final time step latent used by head), NaN-safe
                loss_canopy = mse_ignore_nan(canopy_pred, canopy_target)

                # VQ
                loss_vq = vq_loss

                # Total
                loss = (
                    args.lambda_cont * loss_cont
                    + args.lambda_cat * loss_cat
                    + args.lambda_canopy * loss_canopy
                    + args.lambda_vq * loss_vq
                )

            scaler.scale(loss).backward()
            timers.mark_bwd()
            
            if args.clip_grad is not None and args.clip_grad > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt)
            scaler.update()
            timers.mark_opt()
            

            # stats
            running["recon_cont"] += float(loss_cont.detach())
            running["recon_cat"]  += float(loss_cat.detach())
            running["canopy"]     += float(loss_canopy.detach())
            running["vq"]         += float(loss_vq.detach())
            running["total"]      += float(loss.detach())
            count += 1
            step += 1

            if step % args.log_every == 0:
                avg = {k: v / max(1, count) for k, v in running.items()}
                t_load, t_fwd, t_bwd, t_opt = timers.consume()
                print(f"[epoch {epoch} step {step}] "
                      f"total={avg['total']:.4f} cont={avg['recon_cont']:.4f} "
                      f"cat={avg['recon_cat']:.4f} canopy={avg['canopy']:.4f} vq={avg['vq']:.4f} "
                      f"pplx={float(perplexity):.2f} lr={opt.param_groups[0]['lr']:.2e} "
                      f"t_load={t_load:.2f}s t_fwd={t_fwd:.2f}s t_bwd={t_bwd:.2f}s t_opt={t_opt:.2f}s")
                timers.last = time.time()
                running = {"recon_cont": 0.0, "recon_cat": 0.0, "canopy": 0.0, "vq": 0.0, "total": 0.0}
                count = 0

            if step >= total_steps:
                break

        # checkpoint each epoch
        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "step": step,
        }
        torch.save(ckpt, str(Path(args.run_dir) / f"ckpt_epoch{epoch:03d}.pt"))

        # Simple early best
        if step % args.log_every == 0:
            epoch_loss = avg["total"]
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                torch.save(ckpt, str(Path(args.run_dir) / "ckpt_best.pt"))

        if step >= total_steps:
            break

    print("Training done.")


# ------------------------------- CLI ------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Train a mixed-input VQ-VAE on a Zarr cube (with canopy scalar head).")
    p.add_argument("--config", type=str, help="Path to YAML config file (with a 'train_vqvae' section).")
    p.add_argument("--zarr", required=True, help="Path to consolidated Zarr dataset")
    p.add_argument("--run_dir", required=True, help="Directory to write schema/checkpoints/logs")

    # data/loader
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--steps_per_epoch", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eager", action="store_true")

    # schema (exposure-aware collapse)
    p.add_argument("--min_hits_per_epoch", type=int, default=100)
    p.add_argument("--mass_coverage", type=float, default=0.999)
    p.add_argument("--vocab_cap", type=int, default=5000)

    # model
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--cat_emb_dim", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.25)

    # optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--lambda_cont", type=float, default=1.0)
    p.add_argument("--lambda_cat", type=float, default=1.0)
    p.add_argument("--lambda_canopy", type=float, default=1.0)
    p.add_argument("--lambda_vq", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=1)

    # precision/runtime
    p.add_argument("--no_amp", action="store_true", help="Disable AMP (half precision)")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 in autocast if available")
    p.add_argument("--cpu", action="store_true")

    return parse_args_with_yaml(p, section="train_vqvae")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = parse_args()
    train(args)
