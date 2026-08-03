"""Learning-rate scheduler construction for representation training.

``build_scheduler`` reproduces the scheduler logic formerly inline in ``main()``:

- Resumed run (``start_epoch > 0``): a fresh cosine from the resume LR to eta_min
  over the remaining epochs.
- Two-phase cosine (warmup enabled + ``phase_warmup`` enabled + phase loss active):
  initial warmup → first cosine → phase re-warmup at the phase-loss entry →
  second cosine. Accommodates AdamW's cold v_t for phase params when the phase
  loss curriculum switches on.
- Standard single-phase: linear warmup + cosine annealing.
- No warmup: plain ``CosineAnnealingLR``.

For auto-resume, a captured ``scheduler_state`` is loaded so the LR continues
exactly where it left off.
"""

from __future__ import annotations

import logging

import numpy as np
import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config,
    num_epochs: int,
    steps_per_epoch: int,
    lr: float,
    start_epoch: int,
    resume_lr: float,
    phase_config: dict | None,
    scheduler_state: dict | None,
    logger: logging.Logger,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build (and, for auto-resume, restore) the LR scheduler.

    Must be called after ``phase_config`` is built so the two-phase branch can
    read ``curriculum_start_epoch`` from it. ``steps_per_epoch`` is
    ``len(train_dataloader)``.
    """
    scheduler_config = training_config.scheduler
    total_steps = num_epochs * steps_per_epoch
    eta_min_factor = scheduler_config.eta_min / lr  # express eta_min as a multiplier on peak lr

    def _cosine(start_val, end_val, progress):
        """Cosine interpolation from start_val to end_val over [0, 1]."""
        return end_val + (start_val - end_val) * 0.5 * (1.0 + np.cos(np.pi * progress))

    if start_epoch > 0:
        # Resumed run: fresh cosine from resume_lr → eta_min over remaining epochs.
        # No warmup or phase re-warmup needed — the model is already well-trained.
        remaining_steps = (num_epochs - start_epoch) * steps_per_epoch
        eta_min_factor_resume = scheduler_config.eta_min / resume_lr
        logger.info(
            f"Resume scheduler: cosine from lr={resume_lr:.3e} to "
            f"eta_min={scheduler_config.eta_min:.1e} over "
            f"{num_epochs - start_epoch} epochs ({remaining_steps} steps)"
        )

        def lr_lambda(step):
            progress = step / max(remaining_steps, 1)
            return _cosine(1.0, eta_min_factor_resume, min(progress, 1.0))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_config.warmup.enabled:
        warmup_steps = scheduler_config.warmup.epochs * steps_per_epoch
        phase_warmup_cfg = getattr(scheduler_config, 'phase_warmup', None)

        if (
            phase_warmup_cfg is not None
            and phase_warmup_cfg.enabled
            and phase_config is not None
        ):
            # Two-phase LR schedule to accommodate the phase-loss curriculum:
            #
            #  Segment 1 — initial warmup (0 → warmup_steps):
            #    LR rises linearly 0 → peak. Prevents large updates while weights
            #    are uninitialized.
            #
            #  Segment 2 — first cosine (warmup_steps → phase_start_step):
            #    LR decays along the full-range cosine (as if running to total_steps).
            #    Phase loss is zero during this window; spectral/spatial losses train
            #    freely.
            #
            #  Segment 3 — phase re-warmup (phase_start_step → phase_warmup_end_step):
            #    Phase loss enters. AdamW's variance estimates (v_t) for phase
            #    parameters are zero at this point — bias correction makes the first
            #    update a unit-norm step regardless of gradient magnitude, so a high
            #    LR here causes overshooting. To counteract this:
            #      a) LR drops immediately to start_factor × lr at phase_start_step,
            #         giving low-LR steps to let v_t accumulate accurate estimates.
            #      b) LR then ramps linearly to peak_factor × lr over phase_warmup.epochs,
            #         mirroring the initial warmup but scoped to the phase-loss entry.
            #
            #  Segment 4 — second cosine (phase_warmup_end_step → total_steps):
            #    LR decays from peak_factor × lr down to eta_min over the remainder
            #    of training (~165 epochs for a 200-epoch run).
            #
            # curriculum_start_epoch is the epoch where cw is first evaluated, but
            # cw = (epoch - start_epoch) / ramp = 0 when epoch == start_epoch exactly.
            # Phase gradients first appear at start_epoch + 1, so that is where the
            # LR drop must land.
            phase_start_epoch = phase_config['curriculum_start_epoch'] + 1
            phase_start_step = phase_start_epoch * steps_per_epoch
            phase_warmup_end_step = (
                phase_start_step + phase_warmup_cfg.epochs * steps_per_epoch
            )
            start_factor = phase_warmup_cfg.start_factor  # immediate drop on phase entry
            second_peak = phase_warmup_cfg.peak_factor    # ramp target, as multiplier on peak lr

            logger.info(
                f"Using two-phase cosine schedule: "
                f"warmup={scheduler_config.warmup.epochs} epochs, "
                f"phase re-warmup at epoch {phase_start_epoch} "
                f"for {phase_warmup_cfg.epochs} epochs "
                f"(start_factor={start_factor}, peak_factor={second_peak})"
            )

            def lr_lambda(step):
                if step < warmup_steps:
                    # Segment 1: linear warmup 0 → peak
                    return max(step / warmup_steps, 1e-8)
                elif step < phase_start_step:
                    # Segment 2: cosine decay (full-range) while phase loss is silent
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return _cosine(1.0, eta_min_factor, progress)
                elif step < phase_warmup_end_step:
                    # Segment 3: immediate drop to start_factor, then linear ramp to
                    # peak_factor. The low starting LR lets AdamW's v_t accumulate
                    # before taking large steps with the new phase gradients.
                    ramp_progress = (step - phase_start_step) / (phase_warmup_end_step - phase_start_step)
                    return start_factor + (second_peak - start_factor) * ramp_progress
                else:
                    # Segment 4: cosine decay from peak_factor to eta_min
                    progress = (step - phase_warmup_end_step) / (total_steps - phase_warmup_end_step)
                    return _cosine(second_peak, eta_min_factor, progress)

        else:
            # Standard single-phase: linear warmup + cosine annealing
            logger.info(f"Using cosine scheduler with {warmup_steps} warmup steps")

            def lr_lambda(step):
                if step < warmup_steps:
                    return max(step / warmup_steps, 1e-8)
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return _cosine(1.0, eta_min_factor, progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        logger.info(f"Using cosine annealing scheduler: eta_min={scheduler_config.eta_min}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=scheduler_config.eta_min,
        )

    # For auto-resume: restore scheduler state so LR continues exactly where it left off.
    # (Manual --resume intentionally rebuilds a fresh cosine; auto-resume is a crash recovery.)
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
            logger.info("Restored scheduler state from auto-resume checkpoint")
        except Exception as e:
            logger.warning(f"Could not restore scheduler state: {e}; continuing with rebuilt schedule")

    return scheduler
