"""Curriculum schedules for representation training.

Pure functions that map an epoch index to a scalar schedule value. Kept free of
any torch / model state so they can be unit-tested in isolation:

- ``compute_input_dropout_rate`` — input-dropout probability (constant / linear / cosine).
- ``ramp_weight`` — the shared 0→1 curriculum ramp used by every phase loss.
- ``compute_smoothing_min_gate`` — spatial-smoothing gate lock (1→0 release).
"""

from __future__ import annotations

import math


def compute_input_dropout_rate(
    schedule_cfg: float | dict,
    epoch: int,
    total_epochs: int,
) -> float:
    """Return the input dropout rate for the current epoch.

    Args:
        schedule_cfg: Either a scalar float (constant rate) or a dict with keys:
            - schedule: 'constant' | 'linear' | 'cosine'
            - For 'constant': rate (float)
            - For 'linear' / 'cosine': start, end, epochs (ramp length)
        epoch: Current epoch index (0-based).
        total_epochs: Total training epochs (used as ramp length fallback).

    Returns:
        Dropout probability for this epoch.
    """
    if isinstance(schedule_cfg, (int, float)):
        return float(schedule_cfg)

    schedule = schedule_cfg.get("schedule", "constant")

    if schedule == "constant":
        return float(schedule_cfg.get("rate", 0.0))

    start = float(schedule_cfg.get("start", 0.0))
    end = float(schedule_cfg.get("end", 0.1))
    ramp_epochs = int(schedule_cfg.get("epochs", total_epochs))
    t = min(epoch / max(ramp_epochs, 1), 1.0)

    if schedule == "linear":
        return start + t * (end - start)
    elif schedule == "cosine":
        return start + (end - start) * (1 - math.cos(math.pi * t)) / 2
    else:
        raise ValueError(f"Unknown input_dropout schedule: {schedule!r}")


def ramp_weight(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    """Linear 0→1 curriculum ramp.

    Returns 0.0 before ``start_epoch``, 1.0 at/after ``start_epoch + ramp_epochs``,
    and a linear interpolation in between. This is the shared schedule for the
    phase neighborhood, spread-ranking, and recovery-discrimination losses.

    Note that at ``epoch == start_epoch`` the weight is exactly 0.0, so the first
    epoch with non-zero gradient is ``start_epoch + 1`` (the scheduler's phase
    re-warmup relies on this).
    """
    if epoch < start_epoch:
        return 0.0
    if epoch >= start_epoch + ramp_epochs:
        return 1.0
    return (epoch - start_epoch) / ramp_epochs


def compute_smoothing_min_gate(
    epoch: int, freeze_until_epoch: int, ramp_epochs: int
) -> float:
    """Return the minimum smoothing gate value for the spatial-smoothing curriculum.

    The gate is locked open (1.0 = identity / no smoothing) for the first
    ``freeze_until_epoch`` epochs so the model cannot use smoothing as a shortcut
    before ``z_type`` develops spectral structure, then linearly released to 0.0
    over the following ``ramp_epochs``. This is the complement of ``ramp_weight``.
    """
    return 1.0 - ramp_weight(epoch, freeze_until_epoch, ramp_epochs)
