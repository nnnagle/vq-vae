"""
scripts/train_vqvae.py
----------------------
End-to-end training for a mixed-input geospatial VQ-VAE (continuous + categorical + NAIP).

Purpose
    Orchestrate the full training loop:
      1) Read feature_meta from the Zarr and build an exposure-aware schema
         (dense categorical IDs, continuous stats) → persist to run_dir.
      2) Construct VQVAEDataset/DataLoader (chunk-aware sampling, masks).
      3) Build the model: MixedInputEncoder → VectorQuantizer → MixedDecoder heads.
      4) Train with AMP, class-weighted CE (categoricals), MSE (continuous),
         canopy scalar MSE, and VQ loss; checkpoint and log progress.

Key I/O
    Inputs:
      • --zarr: consolidated Zarr cube with attrs_raw, mask, years, (optional) naip_patch
      • --config: YAML with a 'train_vqvae' section (overrides CLI where present)
    Outputs (under --run_dir):
      • schema.json            (categorical vocab + continuous stats)
      • feature_meta.json      (raw feature meta for provenance)
      • ckpt_epochXXX.pt       (epoch checkpoints)
      • ckpt_best.pt           (best-so-far by running loss)

Trainer contract
    Dataset provides:
      • tensors: cont, cat, cat_target, naip, naip_nan_mask, years, yx
      • schema_cat (name → num_ids), cont_names, cat_names
      • default_collate_fn and IGNORE_INDEX for CE masking
    Model returns:
      • cont_pred [B,T,C_cont], cat_logits {name: [B,T,num_ids]}, canopy_pred [B]
      • vq_loss (scalar), perplexity (monitoring)

Losses
    total = λ_cont * MSE(cont) + λ_cat * Σ CE(cat, class_weights, ignore_index)
            + λ_canopy * MSE(canopy) + λ_vq * VQ
    NaN-safe handling for continuous/NAIP via masks and torch.nan_to_num.

Performance notes
    • Chunk-locked BatchSampler to reduce Zarr I/O thrash.
    • CUDA autocast (fp16 or bf16) + GradScaler on GPU.
    • Cosine LR schedule from --lr → --min_lr over total steps.
    • TF32 enabled on Ampere/Ada for fast matmul/conv.

Quick start (CLI)
    python -m scripts.train_vqvae \\
      --zarr out/cube.zarr --run_dir runs/exp_001 \\
      --batch_size 64 --steps_per_epoch 10000 --epochs 5 \\
      --min_hits_per_epoch 100 --mass_coverage 0.999 --vocab_cap 5000 \\
      --codebook_size 256 --emb_dim 128 --beta 0.25

Design rules
    • Keep schema the single source of truth for vocab/stats (not zarr_info).
    • Deterministic mappings and logging; fail loudly on empty datasets.
    • Prefer spawn multiprocessing for DataLoader workers.
"""

from __future__ import annotations

import argparse
import math
import os
import time

from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import xarray as xr

# ---- project utils (expected layout under utils/)


from vqvae.train_debug_utils import StepTimers, print_device_summary, maybe_sync_cuda


# IGNORE_INDEX fallback in case the loader doesn't export it
try:
    from vqvae.loader import VQVAEDataset, default_collate_fn, IGNORE_INDEX
except Exception:  # pragma: no cover
    from vqvae.loader import VQVAEDataset, default_collate_fn
    IGNORE_INDEX = -100

from utils.weights import cat_class_weights
from utils.argyaml import parse_args_with_yaml
from utils.samplers import ChunkBatchSampler
from vqvae.model import VQVAE
from vqvae.preprocess import (
  build_and_save_schema, 
  read_feature_meta_from_zarr,
  maybe_compute_canopy_target_from_batch,
  )
from vqvae.annealers import LossWeightScheduler, AnnealConfig, load_scheduler

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

# ------------------------------ Training --------------------------------


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
    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    codebook_params = [p for n,p in model.named_parameters()
                       if ".quant.codebook" in n or n.endswith("quant.codebook")]
    other_params    = [p for n,p in model.named_parameters()
                       if (".quant.codebook" not in n and not n.endswith("quant.codebook"))]
    opt = torch.optim.AdamW([
      {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay, "betas": (0.9, 0.95)},
      {"params": codebook_params, "lr": args.lr, "weight_decay": 0.0,            "betas": (0.9, 0.95)},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    #scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp)) # Deprecated
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    # cosine schedule over total steps
    total_steps = args.epochs * args.steps_per_epoch
    
    # Optional annealing for loss weights (e.g., lambda_vq)
    base_weights = {
        "vq": args.lambda_vq,
        "cont": args.lambda_cont,
        "cat": args.lambda_cat,
        "canopy": args.lambda_canopy,
    }
    # avoid passing None for ceil/final into the builder
    _flat = vars(args).copy()
    if _flat.get("anneal_vq_ceil") is None:
        _flat.pop("anneal_vq_ceil", None)
    if _flat.get("anneal_vq_final") is None:
        _flat.pop("anneal_vq_final", None)
    lambda_vq = load_scheduler(base_weights, _flat)
    
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
                    + lambda_vq(step=step)["vq"] * loss_vq
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

    # Annealer parameters
    # annealing (optional; disabled unless --anneal_vq_enable is set)
    p.add_argument("--anneal_vq_enable", action="store_true")
    p.add_argument("--anneal_vq_schedule", type=str, default="warmup_hold_decay",
                   choices=["constant", "linear", "cosine", "exponential", "stepwise", "warmup_hold_decay"])
    p.add_argument("--anneal_vq_start", type=int, default=0)
    p.add_argument("--anneal_vq_duration", type=int, default=0)
    p.add_argument("--anneal_vq_floor", type=float, default=0.0)
    # leave None to inherit args.lambda_vq as the target automatically
    p.add_argument("--anneal_vq_ceil", type=float, default=0.1)
    p.add_argument("--anneal_vq_k", type=float, default=5.0) # Steepness of Exponential annealer
    # warmup/hold/decay for the common 3-phase profile
    p.add_argument("--anneal_vq_warmup", type=int, default=10000)
    p.add_argument("--anneal_vq_hold", type=int, default=15000)
    p.add_argument("--anneal_vq_decay", type=int, default=5000)
    # leave None to decay back to floor; set to keep some weight after decay
    p.add_argument("--anneal_vq_final", type=float, default=.08)
    # e.g. --anneal_vq_milestones 1000:0.01 8000:0.1
    p.add_argument("--anneal_vq_milestones", type=str, nargs="*", default=None) # Stepwise annealer

    return parse_args_with_yaml(p, section="train_vqvae")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = parse_args()
    train(args)
