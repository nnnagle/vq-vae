"""Process-wide profiling flag for representation training.

Replaces the former module-level ``_PROFILE`` global in ``train_representation.py``.
When profiling is off (default/production) every timing ``cuda.synchronize()`` and
the per-epoch perf logging are skipped, so training runs with no profiling
overhead. ``--profile`` turns on the dataloader wait/step split and the
per-component step breakdown.

A tiny module holding shared state so that the training step (``step.py``), the
epoch loops (``loops.py``), and ``main()`` all read one flag across module
boundaries — a plain ``global`` no longer reaches across modules.
"""

from __future__ import annotations

_PROFILE = False


def set_profile(value: bool) -> None:
    """Set the process-wide profiling flag (called once from ``main`` per ``--profile``)."""
    global _PROFILE
    _PROFILE = bool(value)


def is_profiling() -> bool:
    """Return whether profiling (timing barriers + perf logging) is enabled."""
    return _PROFILE
