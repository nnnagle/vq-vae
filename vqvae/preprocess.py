# vqvae/preprocess.py

import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import xarray as xr

from vqvae.schema import (
    build_categorical_schema,
    attach_continuous_stats,
    save_schema,
)

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
  
def read_feature_meta_from_zarr(zarr_path: str) -> Dict[str, Any]:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    raw = ds.attrs.get("cube_meta", ds.attrs.get("feature_meta", "{}"))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"feature_meta not found or invalid: {e}")
      
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

