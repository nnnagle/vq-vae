"""Checkpoint save/restore policy for representation training.

``CheckpointManager`` owns the checkpoint bookkeeping extracted from the epoch
loop of ``train_representation.py``:

- always-save ``encoder_last.pt`` (overwritten each epoch),
- periodic ``encoder_epoch_NNN.pt`` saves (never pruned),
- top-k ``encoder_best_RANK_epoch_NNN.pt`` tracking with NaN-safe ranking,
  pruning beyond ``save_top_k``, and rank-encoded renaming.

``resume_from_checkpoint`` restores model / optimizer / epoch state for both
``--resume`` (manual) and auto-resume, and for the auto path repopulates the
manager's top-k list and returns the scheduler state to restore.

The manager is deliberately torch-agnostic: serialization is injected as
``save_fn`` / ``load_fn`` (``torch.save`` / ``torch.load`` in production). This
keeps the risky top-k prune/rename logic unit-testable without importing torch.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable


class CheckpointManager:
    """Owns the (monitor_val, path) top-k list and all per-epoch checkpoint saves.

    Args:
        ckpt_dir: Directory checkpoints are written to.
        ckpt_cfg: Parsed ``run.checkpoint`` config — reads ``.monitor``, ``.mode``,
            ``.save_last``, ``.save_every_n_epochs``, ``.save_top_k``,
            ``.monitor_start_epoch``.
        logger: Caller's logger (keeps ``%(name)s`` identical to the old inline code).
        save_fn: ``(state_dict, path) -> None`` serializer (``torch.save``).
        load_fn: ``(path) -> dict`` loader used to read a saved checkpoint's metric
            during ``restore_top_k`` (``torch.load`` with ``map_location='cpu'``).
    """

    def __init__(
        self,
        ckpt_dir,
        ckpt_cfg,
        logger: logging.Logger,
        save_fn: Callable,
        load_fn: Callable,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.cfg = ckpt_cfg
        self.logger = logger
        self.save_fn = save_fn
        self.load_fn = load_fn
        # Tracks (monitor_val, path) for top-k checkpoint pruning, sorted best-first.
        self.saved_ckpts: list = []

    def restore_top_k(self) -> None:
        """Rebuild ``saved_ckpts`` from ``encoder_best_*.pt`` on disk (auto-resume).

        Keeps top-k pruning aware of checkpoints written before a crash.
        """
        monitor_key = self.cfg.monitor
        for p in sorted(self.ckpt_dir.glob("encoder_best_*.pt")):
            try:
                c = self.load_fn(p)
                val = c.get(monitor_key, float('nan'))
                self.saved_ckpts.append((val, p))
                self.logger.info(
                    f"  Restored top-k entry: {p.name} ({monitor_key}={val:.4f})"
                )
            except Exception as e:
                self.logger.warning(f"  Could not load {p.name} for top-k restore: {e}")

    def save(self, epoch: int, ckpt_state: dict, epoch_metrics: dict) -> None:
        """Perform all per-epoch checkpoint saves for a completed epoch.

        ``ckpt_state`` is the opaque serializable dict (built by the caller);
        ``epoch_metrics`` supplies the monitored value for top-k selection.
        """
        cfg = self.cfg
        logger = self.logger
        ckpt_dir = self.ckpt_dir

        monitor_key = cfg.monitor
        if monitor_key not in epoch_metrics:
            raise KeyError(
                f"Checkpoint monitor '{monitor_key}' not found in epoch_metrics. "
                f"Available keys: {list(epoch_metrics.keys())}"
            )
        monitor_val = epoch_metrics[monitor_key]

        # Always save 'last' checkpoint (overwrites each epoch).
        if cfg.save_last:
            last_path = ckpt_dir / "encoder_last.pt"
            self.save_fn(ckpt_state, last_path)
            logger.info(f"Saved last checkpoint to {last_path}")

        # Periodic save (every nth epoch, never pruned).
        if (epoch + 1) % cfg.save_every_n_epochs == 0:
            ckpt_path = ckpt_dir / f"encoder_epoch_{epoch+1:03d}.pt"
            self.save_fn(ckpt_state, ckpt_path)
            logger.info(
                f"Saved periodic checkpoint to {ckpt_path} "
                f"({monitor_key}={monitor_val:.4f})"
            )

        # Top-k save: only track best after monitor_start_epoch so pre-curriculum
        # epochs (where phase loss is zero) can't be selected as the best checkpoint.
        # saved_ckpts is sorted best-first (index 0 = rank 1).
        # NaN-safe sort: map non-finite values to the worst possible sentinel so they
        # sort to the tail and can be displaced by any finite value.
        reverse = (cfg.mode == "max")
        nan_sentinel = float('-inf') if reverse else float('inf')
        self.saved_ckpts.sort(
            key=lambda x: x[0] if math.isfinite(x[0]) else nan_sentinel, reverse=reverse
        )
        worst_val_in_top_k = (
            self.saved_ckpts[-1][0] if len(self.saved_ckpts) >= cfg.save_top_k else None
        )
        if worst_val_in_top_k is not None and not math.isfinite(worst_val_in_top_k):
            worst_val_in_top_k = nan_sentinel
        is_better = (
            math.isfinite(monitor_val)
            and (
                worst_val_in_top_k is None
                or (cfg.mode == "min" and monitor_val < worst_val_in_top_k)
                or (cfg.mode == "max" and monitor_val > worst_val_in_top_k)
            )
        )
        if is_better and epoch >= cfg.monitor_start_epoch:
            # Save under a temporary name; will be renamed with rank below.
            tmp_path = ckpt_dir / f"encoder_best_epoch_{epoch+1:03d}.pt"
            self.save_fn(ckpt_state, tmp_path)
            self.saved_ckpts.append((monitor_val, tmp_path))
            self.saved_ckpts.sort(key=lambda x: x[0], reverse=reverse)

            # Prune worst entry if over top-k.
            while len(self.saved_ckpts) > cfg.save_top_k:
                worst_val, worst_path = self.saved_ckpts.pop()
                if worst_path.exists():
                    worst_path.unlink()
                    logger.info(
                        f"Removed checkpoint {worst_path.name} "
                        f"({monitor_key}={worst_val:.4f}, outside top-{cfg.save_top_k})"
                    )

            # Rename all top-k files to reflect current rank (rank 1 = best).
            # Use temp names first to avoid collisions during rename.
            tmp_renames = []
            for rank, (val, old_path) in enumerate(self.saved_ckpts, 1):
                ep = old_path.stem.split("_")[-1]  # e.g. '042'
                new_name = ckpt_dir / f"encoder_best_{rank}_epoch_{ep}.pt"
                tmp_name = ckpt_dir / f"_tmp_rank_{rank}_{ep}.pt"
                old_path.rename(tmp_name)
                tmp_renames.append((rank, val, tmp_name, new_name))
            self.saved_ckpts = []
            for rank, val, tmp_name, new_name in tmp_renames:
                tmp_name.rename(new_name)
                self.saved_ckpts.append((val, new_name))
            logger.info(f"Updated top-{cfg.save_top_k} checkpoints:")
            for rank, (val, path) in enumerate(self.saved_ckpts, 1):
                logger.info(f"  #{rank}: {path.name} ({monitor_key}={val:.4f})")


def resume_from_checkpoint(
    args,
    model,
    optimizer,
    checkpoint_dir,
    device,
    lr: float,
    ckpt_manager: CheckpointManager,
    logger: logging.Logger,
):
    """Restore model / optimizer / epoch state for ``--resume`` or auto-resume.

    Manual ``--resume`` loads the given checkpoint and lets the caller rebuild a
    fresh cosine schedule from the checkpoint LR. Auto-resume (``encoder_last.pt``
    present, no ``--no-resume``) additionally repopulates ``ckpt_manager``'s top-k
    list and returns the saved scheduler state for exact LR continuation.

    Returns:
        (start_epoch, resume_lr, scheduler_state) — ``scheduler_state`` is None
        unless auto-resume captured one.
    """
    import torch

    start_epoch = 0
    resume_lr = lr  # used only if manual --resume triggers the fresh-cosine path
    scheduler_state = None
    auto_resume_path = Path(checkpoint_dir) / "encoder_last.pt"

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']  # saved as epoch+1 (the next epoch to run)
        resume_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Resumed: start_epoch={start_epoch}, resume_lr={resume_lr:.3e}")

    elif not args.no_resume and auto_resume_path.exists():
        logger.info(f"Auto-resuming from {auto_resume_path}")
        ckpt = torch.load(auto_resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        logger.info(f"Auto-resumed: start_epoch={start_epoch}, "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}")
        # Rebuild saved_ckpts from existing best checkpoints on disk so top-k
        # pruning is aware of what was saved before the crash.
        ckpt_manager.restore_top_k()
        # scheduler state is restored by the caller after scheduler creation
        scheduler_state = ckpt.get('scheduler_state_dict')

    return start_epoch, resume_lr, scheduler_state
