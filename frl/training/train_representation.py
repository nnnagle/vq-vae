#!/usr/bin/env python3
"""
Minimal training script for representation learning.

This script trains a simple encoder using InfoNCE contrastive loss:
- Input: features.type_encoder_input
- Encoder: Conv2D kernel=1, two layers [128, 64]
- Output: z_type (64-dim embedding per pixel)
- Loss: InfoNCE with auxiliary distance-based pair selection

Usage:
    python scripts/train_representation.py

    # Or with custom config paths:
    python scripts/train_representation.py \
        --bindings config/frl_binding_v1.yaml \
        --training config/frl_training_v1.yaml
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# FRL imports
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn
from data.loaders.builders.feature_builder import FeatureBuilder
from models import RepresentationModel
from training.representation.checkpointing import (
    CheckpointManager,
    resume_from_checkpoint,
)
from training.representation.config_builders import (
    build_evt,
    build_loss_config,
    build_phase_config,
    build_recovery_disc_config,
    build_spatial_pair_config,
    build_spread_config,
)
from training.representation.curriculum import (
    compute_input_dropout_rate,
    compute_smoothing_min_gate,
)
from training.representation.epoch_logging import log_epoch
from training.representation.loops import train_epoch, validate_epoch
from training.representation.profiling import is_profiling, set_profile
from training.representation.scheduler import build_scheduler

logger = logging.getLogger(__name__)


def _dataloader_worker_init(worker_id: int) -> None:
    """Pin each DataLoader worker to a single compute thread.

    With many workers, an uncapped BLAS/torch thread pool per worker
    oversubscribes the cores: each worker's numpy/whitening ops slow down, so
    adding workers stops improving (or even worsens) throughput. The launch
    scripts set OPENBLAS/MKL/NUMEXPR/OMP _NUM_THREADS=1 before import so the
    forked workers inherit single-threaded BLAS; this additionally pins torch
    intra-op threads inside each worker.
    """
    torch.set_num_threads(1)


def _count_blocks(dataset) -> int:
    bh, bw = dataset.split_block_size
    ps = dataset.patch_size
    blocks = set()
    for w in dataset.patches:
        blocks.add((w.row_start // ps // bh, w.col_start // ps // bw))
    return len(blocks)


def main():
    parser = argparse.ArgumentParser(description='Train representation encoder')
    parser.add_argument(
        '--bindings',
        type=str,
        default=None,
        help='Path to bindings config (overrides training config bindings_path)'
    )
    parser.add_argument(
        '--training',
        type=str,
        default='config/frl_training_v1.yaml',
        help='Path to training config'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default=None,
        help='Path to model architecture config (overrides training config model_path)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (overrides config)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints (overrides config)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing experiment directory if it exists'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='CKPT',
        help='Path to a checkpoint to resume training from. Loads model weights and '
             'optimizer state; the scheduler is reinitialized as a fresh cosine decay '
             'from the checkpoint LR over the remaining epochs (num_epochs - start_epoch).'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        default=False,
        help='Disable automatic resume from encoder_last.pt even if it exists in the '
             'experiment directory. Starts training fresh from epoch 0.'
    )
    parser.add_argument(
        '--phase-start-epoch',
        type=int,
        default=None,
        help='Override curriculum_start_epoch for all phase losses (useful for timing/debug runs)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of DataLoader worker processes (overrides config)'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Stop each epoch after this many batches (useful for timing/debug runs)'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        default=False,
        help='Enable performance profiling: per-epoch dataloader wait/step split and '
             'per-component step breakdown. Adds cuda.synchronize() timing barriers, so '
             'leave off for production runs (default off = no profiling overhead).'
    )
    args = parser.parse_args()

    set_profile(args.profile)
    if is_profiling():
        logger.info("Profiling enabled (--profile): per-epoch wait/step + step breakdown will be logged.")

    # Parse configs first to get defaults
    logger.info(f"Loading training config from {args.training}")
    _training_parser = TrainingConfigParser(args.training)
    training_config = _training_parser.parse()
    raw_training_config: dict = _training_parser.raw_config or {}

    bindings_path = args.bindings or training_config.config_paths.get('bindings_path')
    if bindings_path is None:
        raise ValueError("No bindings path found. Set config.bindings_path in the training YAML or pass --bindings.")
    logger.info(f"Loading bindings config from {bindings_path}")
    bindings_config = DatasetBindingsParser(bindings_path).parse()

    model_config_path = args.model_config or training_config.config_paths.get('model_path')
    if model_config_path is None:
        raise ValueError("No model config path found. Set config.model_path in the training YAML or pass --model-config.")
    logger.info(f"Loading model config from {model_config_path}")
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    # Apply config values with command-line overrides
    num_epochs = args.epochs or training_config.training.num_epochs
    batch_size = args.batch_size or training_config.training.batch_size
    lr = args.lr or training_config.optimizer.lr
    weight_decay = training_config.optimizer.weight_decay
    device_str = args.device or training_config.hardware.device

    # Build run directory structure: {run_root}/{experiment_name}/{ckpt_dir|log_dir}
    run_root = Path(training_config.run.run_root)
    experiment_name = training_config.run.experiment_name
    experiment_dir = run_root / experiment_name

    # Check if experiment directory already exists
    if experiment_dir.exists():
        if args.overwrite:
            logger.info(f"Overwriting existing experiment directory: {experiment_dir}")
            shutil.rmtree(experiment_dir)
        elif args.no_resume:
            logger.error(
                f"Experiment directory already exists: {experiment_dir}. "
                f"Use --overwrite to replace it, or remove --no-resume to auto-resume."
            )
            raise SystemExit(1)
        else:
            logger.info(
                f"Experiment directory already exists: {experiment_dir}. "
                f"Will auto-resume if checkpoint found."
            )

    checkpoint_dir = args.checkpoint_dir or str(experiment_dir / training_config.run.ckpt_dir)
    log_dir = experiment_dir / training_config.run.log_dir

    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Create feature builder early so datasets can precompute features in workers
    logger.info("Creating feature builder...")
    feature_builder = FeatureBuilder(bindings_config)

    # Collect all feature names that process_batch() will need so workers can
    # precompute them.  Only include features that are used over the full
    # spatial grid (H×W) — temporal features like ysfc and phase_encoder_feature
    # are accessed only at ~100–200 anchor pixels, so precomputing and stacking
    # the full [C, T, H, W] array for every sample in the batch would cause OOM.
    # Those features are built on-demand in process_batch() via the fallback path.
    _type_enc_feat = training_config.model_input.type_encoder_feature
    _phase_enc_feat = training_config.model_input.phase_encoder_feature
    precompute_features: list[str] = [
        _type_enc_feat,
        'infonce_type_spectral',
        'evt_class',
    ]
    # Drop any names the bindings config doesn't actually define (conditional
    # features like evt_class may not exist in all configs).
    precompute_features = [
        n for n in precompute_features
        if bindings_config.get_feature(n) is not None
    ]
    logger.info(f"Features precomputed in DataLoader workers: {precompute_features}")

    # Spatial pair config — the spatial-InfoNCE parameters process_batch() uses
    # (see build_loss_config) so DataLoader workers can precompute anchors, spatial
    # pairs, and spectral weights. Values must match loss_config's spatial section.
    spatial_pair_config = build_spatial_pair_config(bindings_config, _type_enc_feat)

    # Create train dataset
    logger.info("Creating train dataset...")
    patch_size = training_config.sampling.patch_size
    spatial = training_config.spatial_domain
    _bg = spatial.full_domain.block_grid if spatial.full_domain and spatial.full_domain.block_grid else [4, 4]
    split_block_size = (int(_bg[0]), int(_bg[1]))
    debug_window = None
    if spatial.debug_mode and spatial.debug_window is not None:
        w = spatial.debug_window
        debug_window = ((w.origin[0], w.origin[1]), (w.size[0], w.size[1]))
        logger.info(f"Debug mode: spatial window origin={w.origin} size={w.size}")

    epoch_cfg = training_config.training.epoch
    train_dataset = ForestDatasetV2(
        bindings_config,
        split='train',
        patch_size=patch_size,
        min_aoi_fraction=0.3,
        debug_window=debug_window,
        epoch_mode=epoch_cfg.mode,
        sample_frac=epoch_cfg.sample_frac,
        sample_number=epoch_cfg.sample_number,
        feature_builder=feature_builder,
        precompute_features=precompute_features,
        spatial_pair_config=spatial_pair_config,
        training=True,
        split_block_size=split_block_size,
    )
    logger.info(
        f"Train dataset: {len(train_dataset.patches)} patches "
        f"({_count_blocks(train_dataset)} blocks of {split_block_size[0]}×{split_block_size[1]} patches) "
        f"| epoch_mode={epoch_cfg.mode}, patches/epoch={len(train_dataset)}"
    )

    n_workers = args.num_workers if args.num_workers is not None else training_config.hardware.num_workers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        prefetch_factor=training_config.hardware.prefetch_factor if n_workers > 0 else None,
        pin_memory=training_config.hardware.pin_memory,
        persistent_workers=n_workers > 0,
        worker_init_fn=_dataloader_worker_init if n_workers > 0 else None,
        collate_fn=collate_fn,
    )

    # Create validation dataset
    logger.info("Creating validation dataset...")
    val_dataset = ForestDatasetV2(
        bindings_config,
        split='val',
        patch_size=patch_size,
        min_aoi_fraction=0.3,
        debug_window=debug_window,
        feature_builder=feature_builder,
        precompute_features=precompute_features,
        spatial_pair_config=spatial_pair_config,
        training=False,
        split_block_size=split_block_size,
    )
    logger.info(
        f"Validation dataset: {len(val_dataset.patches)} patches "
        f"({_count_blocks(val_dataset)} blocks of {split_block_size[0]}×{split_block_size[1]} patches)"
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        prefetch_factor=training_config.hardware.prefetch_factor if n_workers > 0 else None,
        pin_memory=training_config.hardware.pin_memory,
        persistent_workers=n_workers > 0,
        worker_init_fn=_dataloader_worker_init if n_workers > 0 else None,
        collate_fn=collate_fn,
    )

    # Read feature dimensions from bindings config
    type_enc_feature = _type_enc_feat
    phase_enc_feature = _phase_enc_feat
    type_in_channels = len(bindings_config.get_feature(type_enc_feature).channels)
    phase_in_channels = len(bindings_config.get_feature(phase_enc_feature).channels)
    logger.info(
        f"Feature dimensions from config: "
        f"{type_enc_feature}={type_in_channels}, {phase_enc_feature}={phase_in_channels}"
    )

    # Create model
    logger.info("Creating representation model...")
    model = RepresentationModel.from_config(
        model_config,
        type_in_channels=type_in_channels,
        phase_in_channels=phase_in_channels,
    ).to(device)
    logger.info(f"Model (v{RepresentationModel.VERSION}): {model}")

    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    spatial_params = sum(p.numel() for p in model.spatial_conv.parameters() if p.requires_grad)
    phase_tcn_params = sum(p.numel() for p in model.phase_tcn.parameters() if p.requires_grad)
    phase_film_params = sum(p.numel() for p in model.phase_film.parameters() if p.requires_grad)
    phase_head_params = sum(p.numel() for p in model.phase_head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Trainable parameters: type(encoder={encoder_params:,}, spatial={spatial_params:,}), "
        f"phase(tcn={phase_tcn_params:,}, film={phase_film_params:,}, head={phase_head_params:,}), "
        f"total={total_params:,}"
    )

    # Create optimizer
    logger.info(f"Creating optimizer: lr={lr}, weight_decay={weight_decay}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Checkpoint manager owns the top-k (monitor_val, path) list and all per-epoch
    # saves (last / periodic / best). Serialization is injected so the manager stays
    # torch-agnostic and unit-testable.
    ckpt_manager = CheckpointManager(
        checkpoint_dir,
        training_config.run.checkpoint,
        logger,
        save_fn=torch.save,
        load_fn=lambda p: torch.load(p, map_location='cpu', weights_only=False),
    )

    # --- Checkpoint resume ---
    # Auto-resume: if encoder_last.pt exists in the experiment dir, resume from it
    # unless --no-resume or an explicit --resume path was given.
    # Manual --resume: load the specified checkpoint and rebuild scheduler as a fresh
    # cosine from the checkpoint LR (original behavior).
    start_epoch, resume_lr, _auto_resume_scheduler_state = resume_from_checkpoint(
        args, model, optimizer, checkpoint_dir, device, lr, ckpt_manager, logger,
    )

    # Scheduler is created after phase_config below, so it can condition on whether
    # phase loss is active (needed for the two-phase LR schedule).

    # Loss and training config — sourced from parsed bindings YAML (incl. optional
    # variance-covariance type + phase-VCR sub-configs). EVT keys added below.
    loss_config = build_loss_config(bindings_config, training_config, logger)

    # Pass spatial pair config to datasets so workers precompute pairs.
    # Workers are not spawned until the training loop starts, so setting
    # these attributes after DataLoader construction is safe.
    _spatial_pair_config = {k: loss_config[k] for k in [
        'stride', 'border', 'jitter_radius', 'supplement_n',
        'spatial_positive_k', 'spatial_positive_max_dist',
        'spatial_negative_min_dist', 'spatial_negative_max_dist',
        'spatial_negatives_per_anchor', 'spatial_spectral_tau', 'spatial_min_w',
    ]}
    _spatial_pair_config['type_encoder_feature'] = loss_config['type_encoder_feature']
    train_dataset.spatial_pair_config = _spatial_pair_config
    train_dataset.training = True
    val_dataset.spatial_pair_config = _spatial_pair_config
    val_dataset.training = False
    logger.info(
        f"Spatial pair worker precompute: "
        f"train precompute_features={train_dataset.precompute_features}, "
        f"spatial_pair_config_keys={list(_spatial_pair_config.keys())}"
    )

    # --- Phase loss / spread / recovery-discrimination setup ---
    phase_sampler, phase_config = build_phase_config(bindings_config, logger)
    spread_config = build_spread_config(bindings_config, logger)
    recovery_disc_config = build_recovery_disc_config(bindings_config, logger)

    # Apply --phase-start-epoch override to all phase curriculum configs
    if args.phase_start_epoch is not None:
        for cfg in [phase_config, spread_config, recovery_disc_config]:
            if cfg is not None:
                cfg['curriculum_start_epoch'] = args.phase_start_epoch
        logger.info(f"CLI override: all phase curriculum_start_epoch set to {args.phase_start_epoch}")

    # --- EVT soft neighbourhood loss setup ---
    # Builds the EVT-stratified sampler + diffusion metric and adds evt_* keys to
    # loss_config when the EVT loss is active.
    evt_metric, evt_sampler = build_evt(
        bindings_config, feature_builder, loss_config, device, logger,
    )

    # Create scheduler with optional warmup.
    # Must be after phase_config is built so the two-phase branch can read
    # curriculum_start_epoch from it. Also restores scheduler state on auto-resume.
    scheduler = build_scheduler(
        optimizer, training_config, num_epochs, len(train_dataloader),
        lr, start_epoch, resume_lr, phase_config,
        _auto_resume_scheduler_state, logger,
    )

    # Create output directories
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to both console and file
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_dir / 'training.log', mode='a')
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    logger.info(f"Checkpoint dir: {ckpt_dir}")
    logger.info(f"Log dir: {log_dir}")

    # Save experiment artifacts for reproducibility (skip if already present — resume case)
    for src, dst in [
        (bindings_path, experiment_dir / Path(bindings_path).name),
        (args.training, experiment_dir / Path(args.training).name),
        (model_config_path, experiment_dir / Path(model_config_path).name),
        (RepresentationModel.source_file(), experiment_dir / "representation.py"),
    ]:
        if not Path(dst).exists():
            shutil.copy2(src, dst)
    logger.info(f"Saved config and model source to {experiment_dir}")

    # Pre-extract input dropout schedule config (scalar or dict) for the epoch loop.
    input_dropout_schedule_cfg = model_config.get("type_encoder", {}).get("input_dropout", 0.0)

    # Pre-extract spatial smoothing curriculum config.
    smoothing_cur_cfg = raw_training_config.get("spatial_smoothing_curriculum", {})
    smoothing_cur_enabled = smoothing_cur_cfg.get("enabled", False)
    smoothing_freeze_until = int(smoothing_cur_cfg.get("freeze_until_epoch", 20))
    smoothing_ramp_epochs = int(smoothing_cur_cfg.get("ramp_epochs", 30))
    if smoothing_cur_enabled:
        logger.info(
            f"Spatial smoothing curriculum: gate=1.0 (identity) for epochs 0–{smoothing_freeze_until - 1}, "
            f"linearly released to 0.0 over epochs {smoothing_freeze_until}–"
            f"{smoothing_freeze_until + smoothing_ramp_epochs - 1}"
        )

    # Log git commit so checkpoints can be traced back to exact code.
    try:
        import subprocess
        _git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
        _git_dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL
        ).decode().strip()
        _dirty_marker = ' (dirty)' if _git_dirty else ''
        logger.info(f"Git commit: {_git_hash}{_dirty_marker}")
    except Exception:
        logger.info("Git commit: unavailable")

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})...")
    for epoch in range(start_epoch, num_epochs):
        train_dataset.on_epoch_start()  # Reshuffle patches

        # Apply scheduled input dropout rate for this epoch.
        input_dropout_rate = compute_input_dropout_rate(
            input_dropout_schedule_cfg, epoch, num_epochs
        )
        model.set_input_dropout_rate(input_dropout_rate)
        if input_dropout_rate > 0.0:
            logger.debug(f"Epoch {epoch}: input_dropout_rate={input_dropout_rate:.4f}")

        # Apply spatial smoothing curriculum: lock gate open early so the model
        # can't use smoothing as a shortcut before z_type develops spectral structure.
        if smoothing_cur_enabled:
            min_gate = compute_smoothing_min_gate(
                epoch, smoothing_freeze_until, smoothing_ramp_epochs
            )
            model.set_spatial_min_gate(min_gate)
            logger.debug(f"Epoch {epoch}: spatial min_gate={min_gate:.3f}")

        train_stats = train_epoch(
          train_dataloader, feature_builder, model,
          optimizer, scheduler, device, loss_config, epoch, num_epochs,
          phase_sampler=phase_sampler, phase_config=phase_config,
          spread_config=spread_config, recovery_disc_config=recovery_disc_config,
          evt_metric=evt_metric, evt_sampler=evt_sampler,
          max_batches=args.max_batches,
        )

        val_stats = validate_epoch(
          val_dataloader, feature_builder, model,
          device, loss_config,
          phase_sampler=phase_sampler, phase_config=phase_config,
          spread_config=spread_config, recovery_disc_config=recovery_disc_config,
          epoch=epoch, evt_metric=evt_metric, evt_sampler=evt_sampler,
          max_batches=args.max_batches,
        )

        log_epoch(
            logger, epoch, num_epochs, train_stats, val_stats,
            loss_config, phase_config, evt_metric,
        )

        # Flat metrics dict — keys match the monitor strings used in the YAML.
        epoch_metrics = {
            "train/loss_total":         train_stats['loss'],
            "train/loss_spectral":      train_stats['spectral_loss'],
            "train/loss_spatial":       train_stats['spatial_loss'],
            "train/loss_phase":              train_stats['phase_loss'],
            "train/loss_phase_spread":       train_stats.get('phase_spread_loss', 0.0),
            "train/loss_phase_recovery_disc": train_stats.get('phase_recovery_disc_loss', 0.0),
            "train/loss_phase_leakage":      train_stats.get('phase_leakage_loss', 0.0),
            "train/loss_vcr":                train_stats['vcr_loss'],
            "train/loss_phase_vcr":          train_stats['phase_vcr_loss'],
            "val/loss_total":                val_stats['loss'],
            "val/loss_spectral":             val_stats['spectral_loss'],
            "val/loss_spatial":              val_stats['spatial_loss'],
            "val/loss_phase":                val_stats['phase_loss'],
            "val/loss_phase_spread":         val_stats.get('phase_spread_loss', 0.0),
            "val/loss_phase_recovery_disc":  val_stats.get('phase_recovery_disc_loss', 0.0),
            "val/loss_phase_leakage":        val_stats.get('phase_leakage_loss', 0.0),
            "val/loss_vcr":                  val_stats['vcr_loss'],
            "val/loss_phase_vcr":            val_stats['phase_vcr_loss'],
        }

        # Type-leakage metrics (train only — val computed separately below)
        for prefix, stats in [("train", train_stats), ("val", val_stats)]:
            tls = stats.get('type_leakage_stats')
            if tls is not None:
                epoch_metrics[f"{prefix}/phase_type_leakage_cov"]  = tls['cross_cov_frob']
                epoch_metrics[f"{prefix}/phase_type_leakage_r2_mean"] = tls['r2_mean']
                epoch_metrics[f"{prefix}/phase_type_leakage_r2_max"]  = tls['r2_max']

        # Checkpoint state dict (shared by periodic and last saves).
        ckpt_state = {
            'epoch': epoch + 1,
            'model_version': RepresentationModel.VERSION,
            'model_config': model_config,
            'type_in_channels': model.type_in_channels,
            'phase_in_channels': model.phase_in_channels,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            **{k: v for k, v in epoch_metrics.items()},
        }

        ckpt_manager.save(epoch, ckpt_state, epoch_metrics)

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
