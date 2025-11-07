"""
Configurable loss-weight annealing utilities.

Goal
-----
Make loss (or any scalar) weights schedulable via config — **no hardcoding** in the
training loop. Drop-in, framework-agnostic (pure Python), PyTorch-friendly.

Key features
------------
- Enable/disable per weight via config.
- Multiple schedule types: constant, linear, cosine, exponential, stepwise, and
  warmup+hold+decay.
- Works with global step or any monotonically increasing counter.
- Safe bounds: every schedule clamps to [floor, ceil].
- Minimal surface: one call returns a dict of effective weights.

Typical usage
-------------
>>> base = {"vq": 0.10, "aux": 1.0}
>>> cfgs = {
...   "vq": AnnealConfig(enable=True, schedule="linear", start=0, duration=20000,
...                       floor=0.0, ceil=0.10),
... }
>>> sched = LossWeightScheduler(base, cfgs)
>>> eff = sched(step=1234)["vq"]

In your loss:
    loss = recon + eff["vq"] * loss_vq

You can add CLI glue (argparse) or load YAML/JSON into `cfgs` — see helpers at
bottom for a simple adapter from flat dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Callable, Any
import math


# --------------------------- schedules --------------------------- #

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def constant(_: int, *, floor: float, ceil: float, **__) -> float:
    # Return ceil (the target value). Floors/ceils still clamp.
    return _clamp(ceil, floor, ceil)


def linear(step: int, *, start: int, duration: int, floor: float, ceil: float, **__) -> float:
    if step <= start:
        return floor
    if duration <= 0:
        return ceil
    t = (step - start) / float(duration)
    t = _clamp(t, 0.0, 1.0)
    return floor + (ceil - floor) * t


def cosine(step: int, *, start: int, duration: int, floor: float, ceil: float, **__) -> float:
    if step <= start:
        return floor
    if duration <= 0:
        return ceil
    t = (step - start) / float(duration)
    t = _clamp(t, 0.0, 1.0)
    # cosine warmup from floor->ceil
    return floor + (ceil - floor) * 0.5 * (1 - math.cos(math.pi * t))


def exponential(step: int, *, start: int, duration: int, floor: float, ceil: float, k: float = 5.0, **__) -> float:
    """Exponential rise from floor to ceil.
    k controls the steepness; larger -> sharper near the end.
    """
    if step <= start:
        return floor
    if duration <= 0:
        return ceil
    t = (step - start) / float(duration)
    t = _clamp(t, 0.0, 1.0)
    # (1 - exp(-k t)) / (1 - exp(-k)) normalized 0->1
    denom = 1.0 - math.exp(-k)
    w = (1.0 - math.exp(-k * t)) / denom if denom > 0 else t
    return floor + (ceil - floor) * w


def stepwise(step: int, *, milestones: List[Tuple[int, float]], floor: float, ceil: float, **__) -> float:
    """Piecewise-constant schedule.
    milestones: list of (at_step, value) sorted by at_step.
    Values are clamped to [floor, ceil]. If no milestone reached yet, returns floor.
    """
    v = floor
    for s, val in milestones:
        if step >= s:
            v = val
        else:
            break
    return _clamp(v, floor, ceil)


def warmup_hold_decay(step: int, *, start: int, warmup: int, hold: int, decay: int,
                      floor: float, ceil: float, final: Optional[float] = None, **__) -> float:
    """Three-phase: linear warmup -> hold -> cosine decay.
    final defaults to floor if not provided.
    """
    final_val = floor if final is None else final
    if step <= start:
        return floor
    # phase boundaries
    w_end = start + max(0, warmup)
    h_end = w_end + max(0, hold)
    d_end = h_end + max(1, decay)

    if step <= w_end:
        # linear warmup
        t = 0.0 if warmup <= 0 else (step - start) / float(warmup)
        t = _clamp(t, 0.0, 1.0)
        return floor + (ceil - floor) * t
    elif step <= h_end:
        return ceil
    else:
        # cosine decay from ceil -> final_val
        t = (step - h_end) / float(max(1, decay))
        t = _clamp(t, 0.0, 1.0)
        return final_val + (ceil - final_val) * 0.5 * (1 + math.cos(math.pi * t))


_SCHEDULES: Dict[str, Callable[..., float]] = {
    "constant": constant,
    "linear": linear,
    "cosine": cosine,
    "exponential": exponential,
    "stepwise": stepwise,
    "warmup_hold_decay": warmup_hold_decay,
}


# --------------------------- config types --------------------------- #

@dataclass
class AnnealConfig:
    enable: bool = False
    schedule: str = "linear"  # one of _SCHEDULES keys
    # common params
    start: int = 0
    duration: int = 0          # used by linear/cosine/exponential
    floor: float = 0.0         # starting value
    ceil: float = 1.0          # target plateau value
    # optional knobs
    k: float = 5.0             # exponential steepness
    milestones: Optional[List[Tuple[int, float]]] = None  # stepwise
    # warmup/hold/decay params
    warmup: int = 0
    hold: int = 0
    decay: int = 0
    final: Optional[float] = None

    def validate(self) -> None:
        if self.schedule not in _SCHEDULES:
            raise ValueError(f"Unknown schedule '{self.schedule}'. Choices: {list(_SCHEDULES)}")
        if self.floor > self.ceil:
            raise ValueError("floor must be <= ceil")
        if self.schedule == "stepwise" and not self.milestones:
            raise ValueError("stepwise requires non-empty milestones")


# --------------------------- main API --------------------------- #

class LossWeightScheduler:
    """Manages effective weights for multiple named scalars.

    base: dict of {name: base_value}. If a name has an AnnealConfig with enable=True,
    we schedule from `floor` to `ceil` (or according to the schedule) independent of
    base_value. If enable=False or no config for a name, we fall back to base_value.

    You can choose either semantics:
      (A) "absolute" scheduling: set floor/ceil to absolute values. (default)
      (B) "scale base" scheduling: pass floor/ceil as multipliers and set
          mode="scale" to multiply the base value.
    """

    def __init__(self,
                 base: Dict[str, float],
                 configs: Optional[Dict[str, AnnealConfig]] = None,
                 *,
                 mode: str = "absolute") -> None:
        assert mode in {"absolute", "scale"}
        self.base = dict(base)
        self.configs = configs or {}
        self.mode = mode
        # validate configs
        for k, c in self.configs.items():
            c.validate()

    def value(self, name: str, step: int) -> float:
        base_v = self.base.get(name, 0.0)
        cfg = self.configs.get(name)
        if not cfg or not cfg.enable:
            return base_v
        fn = _SCHEDULES[cfg.schedule]
        if cfg.schedule == "stepwise":
            val = fn(step, milestones=cfg.milestones or [], floor=cfg.floor, ceil=cfg.ceil)
        elif cfg.schedule == "warmup_hold_decay":
            val = fn(step, start=cfg.start, warmup=cfg.warmup, hold=cfg.hold, decay=cfg.decay,
                     floor=cfg.floor, ceil=cfg.ceil, final=cfg.final)
        elif cfg.schedule == "exponential":
            val = fn(step, start=cfg.start, duration=cfg.duration, floor=cfg.floor,
                     ceil=cfg.ceil, k=cfg.k)
        else:
            val = fn(step, start=cfg.start, duration=cfg.duration, floor=cfg.floor, ceil=cfg.ceil)

        if self.mode == "absolute":
            return val
        else:  # scale base
            return base_v * val

    def __call__(self, step: int) -> Dict[str, float]:
        return {name: self.value(name, step) for name in self.base.keys()}

    # Convenience: merge new configs at runtime
    def update_configs(self, updates: Dict[str, AnnealConfig]) -> None:
        for k, v in updates.items():
            v.validate()
            self.configs[k] = v


# --------------------------- tiny adapters --------------------------- #

def load_scheduler(base: Dict[str, float], flat: Dict[str, Any], *, prefix: str = "anneal_") -> LossWeightScheduler:
    """Build scheduler from a flat dict of arguments (e.g., argparse Namespace.__dict__).

    Expected keys (examples for name="vq"):
      anneal_vq_enable: bool
      anneal_vq_schedule: str
      anneal_vq_start: int
      anneal_vq_duration: int
      anneal_vq_floor: float
      anneal_vq_ceil: float
      anneal_vq_k: float
      anneal_vq_warmup / _hold / _decay / _final
      anneal_vq_milestones: list of "step:value" strings (e.g., ["1000:0.01","5000:0.1"]).

    Any missing fields fall back to AnnealConfig defaults.
    """
    # collect per-name shards by scanning keys
    per_name: Dict[str, Dict[str, Any]] = {}
    for k, v in flat.items():
        if not k.startswith(prefix):
            continue
        tail = k[len(prefix):]  # e.g., "vq_enable"
        if "_" not in tail:
            continue
        name, field = tail.split("_", 1)
        shard = per_name.setdefault(name, {})
        shard[field] = v

    cfgs: Dict[str, AnnealConfig] = {}
    for name, shard in per_name.items():
        # milestones parsing
        milestones_raw = shard.get("milestones")
        milestones: Optional[List[Tuple[int, float]]] = None
        if milestones_raw:
            milestones = []
            for item in milestones_raw:
                if isinstance(item, str) and ":" in item:
                    s, val = item.split(":", 1)
                    milestones.append((int(s), float(val)))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    milestones.append((int(item[0]), float(item[1])))
        cfg = AnnealConfig(
            enable=bool(shard.get("enable", False)),
            schedule=str(shard.get("schedule", "linear")),
            start=int(shard.get("start", 0)),
            duration=int(shard.get("duration", 0)),
            floor=float(shard.get("floor", 0.0)),
            ceil=float(shard.get("ceil", base.get(name, 1.0))),
            k=float(shard.get("k", 5.0)),
            milestones=milestones,
            warmup=int(shard.get("warmup", 0)),
            hold=int(shard.get("hold", 0)),
            decay=int(shard.get("decay", 0)),
            final=float(shard["final"]) if "final" in shard and shard["final"] is not None else None,
        )
        cfg.validate()
        cfgs[name] = cfg

    return LossWeightScheduler(base, cfgs)


# --------------------------- tiny demo --------------------------- #
if __name__ == "__main__":
    # quick sanity check
    base = {"vq": 0.1, "aux": 1.0}
    cfgs = {
        "vq": AnnealConfig(enable=True, schedule="warmup_hold_decay", start=0,
                            warmup=1000, hold=4000, decay=5000,
                            floor=0.0, ceil=0.1, final=0.08),
    }
    sched = LossWeightScheduler(base, cfgs)
    for s in [0, 500, 1000, 2000, 6000, 10000]:
        print(s, sched.value("vq", s))
