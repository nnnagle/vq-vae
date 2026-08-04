#!/usr/bin/env python3
"""Shared plumbing for the Step-0 phase-pathway eval harness.

This module centralizes the pieces every diagnostic (A/B/C) needs so they all
agree on masks, pixel selection, and how embeddings are produced:

* dataset / dataloader / model / feature-builder construction (mirrors
  :func:`fit_phase_linear_probe.main`);
* a single per-pixel time-series extractor
  (:func:`extract_pixel_series`) that returns, for a batch of patches, the
  per-pixel arrays ``z_type``, ``z_phase`` (post-FiLM), optional pre-FiLM ``h``,
  the phase-encoder input ``x``, ``ysfc``, ``evt``, per-timestep validity, and
  any requested extra target features — all gathered at the *same* valid pixels;
* small helpers: within/between variance decomposition, JSON metric writing, and
  the deferred anomaly-target seam (:class:`AnomalyTargetProvider`).

Design notes
------------
Embeddings are produced with :meth:`RepresentationModel.forward_phase_at_locations`
(anchor-only), which is bit-identical to the dense ``forward_phase`` at the same
pixels but avoids materializing the full ``[C,T,H,W]`` grid (per the CLAUDE.md
temporal-feature rule). z_type is stop-grad'd before the phase pathway, matching
training and inference convention.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# NOTE: the heavy, GDAL/zarr-bound data + model imports are deferred into the
# functions that need them (``build_context`` / ``make_loader``) so that the pure
# metric helpers in this module — and the diagnostics that import them — can be
# used (and unit-tested) without the full runtime stack installed.

logger = logging.getLogger("phase_eval")


def _halo_mask(H: int, W: int, halo: int, device: torch.device) -> torch.Tensor:
    """``[H, W]`` bool mask, True for pixels ≥ ``halo`` from every edge.

    Kept identical to ``fit_phase_linear_probe._halo_mask`` so all eval scripts
    agree on which edge pixels the spatial conv contaminates.
    """
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    if 2 * halo >= H or 2 * halo >= W:
        return mask
    mask[halo:H - halo, halo:W - halo] = True
    return mask

# Bindings feature names shared across diagnostics. The phase-encoder input is
# the model's temporal input; NBR (diagnostic B) comes from the phase target
# feature to match the existing recovery-curve convention.
PHASE_INPUT_FEATURE = "phase_ccdc"
PHASE_TARGET_FEATURE = "soft_neighborhood_phase_target"
YSFC_FEATURE = "ysfc"
EVT_FEATURE = "evt_class"


# ---------------------------------------------------------------------------
# Config / model / loaders
# ---------------------------------------------------------------------------

def build_context(
    bindings_path: str,
    training_path: str,
    checkpoint: str,
    device: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict:
    """Parse configs, load the frozen model + feature builder, and record wiring.

    Returns a dict with keys: ``bindings_config``, ``training_config``,
    ``feature_builder``, ``model``, ``device``, ``batch_size``, ``patch_size``,
    ``num_workers``, ``pin_memory``, ``enc_feature_name``.
    """
    from data.loaders.builders.feature_builder import FeatureBuilder
    from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
    from data.loaders.config.training_config_parser import TrainingConfigParser
    from models import RepresentationModel

    bindings_config = DatasetBindingsParser(bindings_path).parse()
    training_config = TrainingConfigParser(training_path).parse()

    device_str = device or training_config.hardware.device
    dev = torch.device(device_str)
    bs = batch_size or training_config.training.batch_size
    nw = num_workers if num_workers is not None else training_config.hardware.num_workers

    feature_builder = FeatureBuilder(bindings_config)
    model = RepresentationModel.from_checkpoint(checkpoint, device=dev, freeze=True)

    # Per the CLAUDE.md rule: never hard-code the encoder feature name.
    enc_feature_name = training_config.model_input.type_encoder_feature

    logger.info(
        f"Loaded checkpoint {checkpoint} | device={dev} | batch_size={bs} | "
        f"z_type_dim={model.z_type_dim} z_phase_dim={model.z_phase_dim} | "
        f"enc_feature='{enc_feature_name}'"
    )
    return {
        "bindings_config": bindings_config,
        "training_config": training_config,
        "feature_builder": feature_builder,
        "model": model,
        "device": dev,
        "batch_size": bs,
        "patch_size": training_config.sampling.patch_size,
        "num_workers": nw,
        "pin_memory": training_config.hardware.pin_memory,
        "enc_feature_name": enc_feature_name,
    }


def make_loader(ctx: dict, split: str):
    """Build a non-shuffled DataLoader for a checkerboard split."""
    from torch.utils.data import DataLoader
    from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2, collate_fn

    dataset = ForestDatasetV2(
        ctx["bindings_config"],
        split=split,
        patch_size=ctx["patch_size"],
        min_aoi_fraction=0.3,
    )
    logger.info(f"{split} dataset: {len(dataset)} patches")
    return DataLoader(
        dataset,
        batch_size=ctx["batch_size"],
        shuffle=False,
        num_workers=ctx["num_workers"],
        pin_memory=ctx["pin_memory"],
        collate_fn=collate_fn,
    )


def iter_batches(loader: DataLoader, max_batches: int):
    """Iterate with an optional cap (``max_batches=0`` → no cap)."""
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        yield batch


# ---------------------------------------------------------------------------
# Per-pixel time-series extraction
# ---------------------------------------------------------------------------

def _sample_to_numpy(batch: dict, i: int) -> dict:
    """Slice batch item ``i`` into the per-sample dict the FeatureBuilder wants."""
    sample = {
        key: (val[i].numpy() if isinstance(val, torch.Tensor) else val[i])
        for key, val in batch.items()
        if key != "metadata"
    }
    sample["metadata"] = batch["metadata"][i]
    return sample


def _spatial_valid_mask(
    sample: dict,
    feature_builder: FeatureBuilder,
    enc_feature_name: str,
    phase_input_feature: str,
    halo: int,
    require_evt: bool,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Build the ``[H, W]`` spatial validity mask shared by all diagnostics.

    Valid = inside AOI ∧ forest ∧ encoder-valid ∧ phase-valid (all timesteps)
    ∧ ≥ ``halo`` from every edge, and (optionally) a finite positive EVT code.

    Returns the boolean mask and the ``[H, W]`` EVT-code grid (int32).
    """
    enc_f = feature_builder.build_feature(enc_feature_name, sample)
    phase_f = feature_builder.build_feature(phase_input_feature, sample)
    H, W = enc_f.data.shape[-2], enc_f.data.shape[-1]

    sm_names = sample["metadata"]["channel_names"]["static_mask"]
    sm_data = sample["static_mask"]
    aoi = torch.from_numpy(sm_data[sm_names.index("aoi")]).bool()
    forest = torch.from_numpy(sm_data[sm_names.index("forest")]).bool()
    enc_mask = torch.from_numpy(enc_f.mask).bool()
    phase_spatial = torch.from_numpy(phase_f.mask).all(dim=0)
    halo_m = _halo_mask(H, W, halo, torch.device("cpu"))

    mask = aoi & forest & enc_mask & phase_spatial & halo_m

    evt_f = feature_builder.build_feature(EVT_FEATURE, sample)
    evt_grid = evt_f.data[0].astype(np.int32)
    if require_evt:
        evt_valid = torch.from_numpy(np.isfinite(evt_f.data[0]) & (evt_f.data[0] > 0))
        mask = mask & evt_valid
    return mask, evt_grid


def extract_pixel_series(
    batch: dict,
    ctx: dict,
    halo: int,
    max_pixels_per_sample: int = 0,
    need_pre_film: bool = False,
    extra_target_features: Sequence[str] = (),
    require_evt: bool = True,
    chunk_size: int = 200_000,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Extract per-pixel time series for a batch of patches.

    For every valid pixel (see :func:`_spatial_valid_mask`) this gathers the full
    ``T``-length time series and runs the phase pathway anchor-only. Pixels are
    optionally subsampled per patch to ``max_pixels_per_sample`` (0 = keep all).

    Returns a dict of CPU tensors concatenated across the batch, or ``None`` if
    no valid pixels were found:

    ============  ==============================  =====================================
    key           shape                           meaning
    ============  ==============================  =====================================
    ``z_type``    ``[N, dt]``                     type embedding per pixel
    ``z_phase``   ``[N, T, zp]``                  post-FiLM phase embedding
    ``h``         ``[N, T, zp]`` (if requested)   pre-FiLM bottleneck (contrast probe)
    ``x``         ``[N, Cx, T]``                  phase-encoder input (normalized)
    ``ysfc``      ``[N, T]``                      years since fire/change (raw years)
    ``evt``       ``[N]`` int                     EVT class code
    ``valid_tp``  ``[N, T]`` bool                 per-timestep validity
    ``targets``   {name: ``[N, Cf, T]``}          extra target features, gathered
    ============  ==============================  =====================================
    """
    model: RepresentationModel = ctx["model"]
    device = ctx["device"]
    fb: FeatureBuilder = ctx["feature_builder"]
    enc_feature_name = ctx["enc_feature_name"]
    if rng is None:
        rng = np.random.default_rng(0)

    out_z_type: List[torch.Tensor] = []
    out_z_phase: List[torch.Tensor] = []
    out_h: List[torch.Tensor] = []
    out_x: List[torch.Tensor] = []
    out_ysfc: List[torch.Tensor] = []
    out_evt: List[np.ndarray] = []
    out_valid: List[torch.Tensor] = []
    out_targets: Dict[str, List[torch.Tensor]] = {n: [] for n in extra_target_features}

    for i in range(len(batch["metadata"])):
        sample = _sample_to_numpy(batch, i)
        mask, evt_grid = _spatial_valid_mask(
            sample, fb, enc_feature_name, PHASE_INPUT_FEATURE, halo, require_evt,
        )
        if not mask.any():
            continue

        enc_f = fb.build_feature(enc_feature_name, sample)
        phase_f = fb.build_feature(PHASE_INPUT_FEATURE, sample)
        ysfc_f = fb.build_feature(YSFC_FEATURE, sample)

        H, W = mask.shape
        T = phase_f.data.shape[1]
        Cx = phase_f.data.shape[0]

        flat_mask = mask.reshape(-1)
        coords = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)  # [Nsel]
        if max_pixels_per_sample and coords.numel() > max_pixels_per_sample:
            sel = rng.choice(coords.numel(), size=max_pixels_per_sample, replace=False)
            coords = coords[torch.from_numpy(np.sort(sel))]
        n = coords.numel()
        if n == 0:
            continue

        # Dense z_type (single patch, cheap), gathered at pixel coords.
        enc_tensor = torch.from_numpy(enc_f.data).float().unsqueeze(0).to(device)
        with torch.no_grad():
            z_type = model(enc_tensor)  # [1, dt, H, W]
        dt = model.z_type_dim
        z_type_flat = z_type.squeeze(0).permute(1, 2, 0).reshape(-1, dt)  # [H*W, dt]
        z_type_px = z_type_flat[coords.to(device)]  # [n, dt]

        # Phase input gathered at pixel coords: [Cx, T, H, W] -> [n, Cx, T].
        x_grid = torch.from_numpy(phase_f.data).float()  # [Cx, T, H, W]
        x_flat = x_grid.reshape(Cx, T, -1)               # [Cx, T, H*W]
        x_px = x_flat[:, :, coords].permute(2, 0, 1).contiguous()  # [n, Cx, T]

        # Phase pathway anchor-only forward, in chunks to bound memory.
        zp = model.z_phase_dim
        z_phase_px = torch.empty(n, T, zp)
        h_px = torch.empty(n, T, zp) if need_pre_film else None
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            xin = x_px[s:e].to(device)
            zin = z_type_px[s:e].detach()
            with torch.no_grad():
                if need_pre_film:
                    z_chunk, h_chunk = model.forward_phase_at_locations(
                        xin, zin, return_pre_film=True,
                    )
                    h_px[s:e] = h_chunk.permute(0, 2, 1).cpu()  # [N, zp, T] -> [N, T, zp]
                else:
                    z_chunk = model.forward_phase_at_locations(xin, zin)
            z_phase_px[s:e] = z_chunk.cpu()

        # ysfc + per-timestep validity at pixel coords.
        ysfc_grid = torch.from_numpy(ysfc_f.data[0]).float()          # [T, H, W]
        ysfc_valid = torch.from_numpy(ysfc_f.mask[0]).bool()          # [T, H, W]
        phase_valid_tp = torch.from_numpy(phase_f.mask).bool()        # [T, H, W]
        ysfc_px = ysfc_grid.reshape(T, -1)[:, coords].permute(1, 0).contiguous()  # [n, T]
        valid_px = (ysfc_valid & phase_valid_tp).reshape(T, -1)[:, coords].permute(1, 0).contiguous()

        evt_px = evt_grid.reshape(-1)[coords.numpy()]

        out_z_type.append(z_type_px.cpu())
        out_z_phase.append(z_phase_px)
        if need_pre_film:
            out_h.append(h_px)
        out_x.append(x_px)
        out_ysfc.append(ysfc_px)
        out_valid.append(valid_px)
        out_evt.append(evt_px)

        for name in extra_target_features:
            tf = fb.build_feature(name, sample)
            Cf = tf.data.shape[0]
            t_flat = torch.from_numpy(tf.data).float().reshape(Cf, T, -1)  # [Cf, T, H*W]
            out_targets[name].append(t_flat[:, :, coords].permute(2, 0, 1).contiguous())

    if not out_z_type:
        return None

    result = {
        "z_type": torch.cat(out_z_type, dim=0),
        "z_phase": torch.cat(out_z_phase, dim=0),
        "x": torch.cat(out_x, dim=0),
        "ysfc": torch.cat(out_ysfc, dim=0),
        "evt": np.concatenate(out_evt, axis=0),
        "valid_tp": torch.cat(out_valid, dim=0),
        "targets": {n: torch.cat(v, dim=0) for n, v in out_targets.items()},
    }
    if need_pre_film:
        result["h"] = torch.cat(out_h, dim=0)
    return result


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def variance_decompose(
    values: torch.Tensor, valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Within- and between-pixel variance per channel via the law of total variance.

    Args:
        values: ``[N, T, C]`` per-pixel time series.
        valid:  ``[N, T]`` boolean validity mask.

    Returns:
        ``(within, between)`` each ``[C]`` — E[Var(X|pixel)] and Var(E[X|pixel]),
        computed over pixels with ≥ 2 valid timesteps.
    """
    N, T, C = values.shape
    v = valid.unsqueeze(-1).float()                        # [N, T, 1]
    cnt = v.sum(dim=1)                                     # [N, 1]
    has = (cnt.squeeze(-1) >= 2)                           # [N]
    if has.sum() == 0:
        z = torch.zeros(C, dtype=torch.float64)
        return z, z
    vals = values[has]
    vv = v[has]
    cc = cnt[has].clamp(min=1)
    pix_mean = (vals * vv).sum(dim=1) / cc                 # [Nh, C]
    centered = (vals - pix_mean.unsqueeze(1)) * vv         # [Nh, T, C]
    pix_var = (centered ** 2).sum(dim=1) / cc              # [Nh, C]
    within = pix_var.mean(dim=0).double()                  # E[Var(X|pixel)]
    between = pix_mean.var(dim=0, correction=0).double()   # Var(E[X|pixel])
    return within, between


def write_metrics_json(path: str | Path, payload: dict) -> None:
    """Write a metrics payload to JSON (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    logger.info(f"Wrote metrics to {p}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Not JSON-serializable: {type(o)}")


# ---------------------------------------------------------------------------
# Deferred anomaly-target seam (Step 1)
# ---------------------------------------------------------------------------

class AnomalyTargetProvider:
    """Maps the phase input ``x`` to the diagnostic-A regression target.

    ``raw`` (live now) returns ``x`` unchanged — the model-agnostic reconstruction
    target usable on exp034 and the new model alike. ``mature_baseline`` is the
    Step-1 seam: it will return ``(x − μ(z_type)) / σ(z_type)`` once the online
    heteroscedastic Gaussian-NLL readout exists; until then it raises so the
    coupling is explicit rather than silently wrong.
    """

    def __init__(self, kind: str = "raw"):
        if kind not in ("raw", "mature_baseline"):
            raise ValueError(f"unknown anomaly target kind: {kind!r}")
        self.kind = kind

    def __call__(self, x: torch.Tensor, z_type: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` ``[N, Cx, T]``, ``z_type`` ``[N, dt]`` → target ``[N, Cx, T]``."""
        if self.kind == "raw":
            return x
        raise NotImplementedError(
            "mature_baseline anomaly target needs the Step-1 μ/σ readout "
            "(see docs/phase_rethink_design.md, Step 1); not yet implemented."
        )
