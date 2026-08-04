#!/usr/bin/env python3
"""LCMS annual Change product — change-agent class codes, names, and groupings.

Source: the LCMS "Change" product with agent attribution, mosaicked into
``/data/VA/lcms/lcms_chg_1985_2024.vrt`` and loaded into the zarr as the
``lcms_chg_class`` annual layer (bindings feature ``change_agent``). Used by
phase-rethink diagnostic D (docs/phase_rethink_design.md §D) to test whether
disturbance ejecta / recovery trajectories in z_phase organize by agent.

Codes 1–13 are disturbance agents; 14–15 are non-disturbance states (growth,
stable); 0 and 16 (Non-Processing Area Mask) are missing/background.
"""

from __future__ import annotations

# code → (name, hex color)
LCMS_CHANGE_CLASSES: dict[int, tuple[str, str]] = {
    1:  ("Wind", "#ff09f3"),
    2:  ("Hurricane", "#541aff"),
    3:  ("Snow or Ice Transition", "#e4f5fd"),
    4:  ("Desiccation", "#cc982e"),
    5:  ("Inundation", "#0adaff"),
    6:  ("Prescribed Fire", "#a10018"),
    7:  ("Wildfire", "#d54309"),
    8:  ("Mechanical Land Transformation", "#fafa4b"),
    9:  ("Tree Removal", "#afde1c"),
    10: ("Defoliation", "#ffc80d"),
    11: ("Southern Pine Beetle", "#a64c28"),
    12: ("Insect, Disease, or Drought Stress", "#f39268"),
    13: ("Other Loss", "#c291d5"),
    14: ("Vegetation Successional Growth", "#00a398"),
    15: ("Stable", "#3d4551"),
    16: ("Non-Processing Area Mask", "#1b1716"),
}

# Missing / background codes — excluded from all agent analysis.
MISSING_CODES: frozenset[int] = frozenset({0, 16})

# Disturbance agents (the codes diagnostic D groups ejecta by).
DISTURBANCE_AGENTS: tuple[int, ...] = tuple(range(1, 14))  # 1..13

# Present but non-disturbance states.
NON_DISTURBANCE: tuple[int, ...] = (14, 15)


def agent_name(code: int) -> str:
    """Human-readable class name for a code (``'unknown(<code>)'`` if unmapped)."""
    entry = LCMS_CHANGE_CLASSES.get(int(code))
    return entry[0] if entry else f"unknown({int(code)})"


def agent_color(code: int) -> str:
    """Hex color for a code (grey if unmapped)."""
    entry = LCMS_CHANGE_CLASSES.get(int(code))
    return entry[1] if entry else "#808080"


def is_missing(code: int) -> bool:
    """True for the missing/background codes (0 and 16)."""
    return int(code) in MISSING_CODES


def is_disturbance(code: int) -> bool:
    """True for disturbance-agent codes (1–13)."""
    return int(code) in DISTURBANCE_AGENTS
