#!/usr/bin/env python3
"""
export_codebook.py
------------------------
CLI to decode a trained VQ-VAE's codebook into original data space and
write flat arrays for downstream analysis.

Outputs (saved to a single NPZ by default):
  - cont_KT : [K*T, C_cont] float32 (continuous, original units)
  - cats_KT : [K*T, C_cat] float32 (categorical, original raw codes; NaN for MISS/UNK)
  - code_id : [K*T] int32 (code index 0..K-1 for each row in cont_KT/cats_KT)
  - year    : [K*T] int32 (year values aligned to cont_KT/cats_KT rows)
  - codes_K3: [K,3] float32 (columns: code_id, code_usage, canopy)
  - meta    : JSON string with names and shapes

Optionally, write CSVs for human inspection with headers.

Usage:
  python scripts/export_codebook.py \
      --zarr path/to/cube.zarr \
      --ckpt path/to/ckpt_best.pt \
      --out runs/exp001/decoded \
      [--csv]

Notes
-----
- Runs entirely on CPU. K<1000 should be trivial in memory/time.
- The temporal contract is honoured: we decode [K,T,D] and then reshape to [K*T,*].
- No batching or device flags; add later only if needed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Project imports
from vqvae.postprocess import (
    load_model_and_ds,
    decode_codebook_sequences,
    denorm_continuous_KTC,
    decode_cats_KTC,
    flatten_to_KT,
    code_summary,
    extract_code_usage_from_state,
)


def _extract_years_array(ds) -> np.ndarray:
    """Return years as a 1-D np.ndarray[T]. Tries common access patterns.

    Contract: raises RuntimeError if T cannot be determined.
    """
    # Preferred: xarray Dataset-like access
    try:
        years = ds.ds["years"].values
        if years.ndim != 1:
            years = years.reshape(-1)
        return years
    except Exception:
        pass

    # Fallback: attribute or list
    for attr in ("years", "time", "times"):
        if hasattr(ds, attr):
            arr = getattr(ds, attr)
            arr = np.asarray(arr)
            return arr.reshape(-1)

    raise RuntimeError("Could not locate a 1-D years vector in dataset.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode codebook to original data scale and export arrays.")
    ap.add_argument("--zarr", required=True, help="Path to consolidated Zarr cube used during training")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="Output prefix (directory or file prefix, without extension)")
    ap.add_argument("--csv", action="store_true", help="Also write CSVs with headers for cont/cats and summary")
    args = ap.parse_args()

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1) Restore model/dataset
    model, ds = load_model_and_ds(args.zarr, args.ckpt)

    # 2) Decode sequences at true T
    years = _extract_years_array(ds)
    T = int(years.shape[0])
    cont_pred, cat_logits, canopy = decode_codebook_sequences(model, T)

    # 3) Post-process heads
    # Continuous
    if cont_pred is not None and cont_pred.shape[-1] > 0:
        cont_KTC = denorm_continuous_KTC(
            cont_pred, cont_names=ds.cont_names, cont_stats=ds.cont_stats
        )  # [K,T,C_cont]
        cont_KT, code_id_cont, year_cont = flatten_to_KT(cont_KTC, years)
    else:
        # Build empty blocks with consistent indexing
        K = int(model.quant.codebook.shape[0])
        cont_KT = np.zeros((K * T, 0), dtype=np.float32)
        code_id_cont = np.repeat(np.arange(K, dtype=np.int32), T)
        year_cont = np.tile(years.astype(np.int32, copy=False), K)

    # Categorical
    if cat_logits and len(ds.cat_names) > 0:
        cats_KTC = decode_cats_KTC(cat_logits, cat_names=ds.cat_names, cat_maps=ds.cat_maps)
        cats_KT, code_id_cats, year_cats = flatten_to_KT(cats_KTC, years)
    else:
        K = int(model.quant.codebook.shape[0])
        cats_KT = np.zeros((K * T, 0), dtype=np.float32)
        code_id_cats = np.repeat(np.arange(K, dtype=np.int32), T)
        year_cats = np.tile(years.astype(np.int32, copy=False), K)

    # 4) Sanity: code_id/year must match across cont/cats; prefer cont's copies
    code_id = code_id_cont if cont_KT.shape[1] >= cats_KT.shape[1] else code_id_cats
    year = year_cont if cont_KT.shape[1] >= cats_KT.shape[1] else year_cats

    # 5) Summary [K,3]
    ckpt = torch.load(args.ckpt, map_location="cpu")
    usage = extract_code_usage_from_state(ckpt.get("model", ckpt))

    # --- Summarize codebook ---
    summary_K3 = code_summary(model, canopy, usage=usage)

    #summary_K3 = code_summary(model, canopy)

    # 6) Save NPZ (single bundle)
    npz_path = out_prefix.with_suffix(".npz")
    meta: Dict[str, Any] = {
        "cont_names": list(ds.cont_names),
        "cat_names": list(ds.cat_names),
        "T": T,
        "K": int(model.quant.codebook.shape[0]),
        "shapes": {
            "cont_KT": list(cont_KT.shape),
            "cats_KT": list(cats_KT.shape),
            "code_id": list(code_id.shape),
            "year": list(year.shape),
            "codes_K3": list(summary_K3.shape),
        },
        "notes": "cats_KT contains original raw codes; NaN denotes MISS/UNK; canopy in codes_K3 is raw model head value.",
    }

    np.savez_compressed(
        npz_path,
        cont_KT=cont_KT,
        cats_KT=cats_KT,
        code_id=code_id,
        year=year,
        codes_K3=summary_K3,
        meta=json.dumps(meta),
    )

    # 7) Optional CSVs
    if args.csv:
        # Continuous
        if cont_KT.shape[1] > 0:
            import pandas as pd
            df_cont = pd.DataFrame(cont_KT, columns=list(ds.cont_names))
            df_cont.insert(0, "year", year.astype(int))
            df_cont.insert(0, "code_id", code_id.astype(int))
            df_cont.to_csv(out_prefix.with_name(out_prefix.name + "_cont_KT.csv"), index=False)
        # Categoricals
        if cats_KT.shape[1] > 0:
            import pandas as pd
            df_cats = pd.DataFrame(cats_KT, columns=list(ds.cat_names))
            df_cats.insert(0, "year", year.astype(int))
            df_cats.insert(0, "code_id", code_id.astype(int))
            df_cats.to_csv(out_prefix.with_name(out_prefix.name + "_cats_KT.csv"), index=False)
        # Summary
        import pandas as pd
        df_sum = pd.DataFrame(summary_K3, columns=["code_id", "code_usage", "canopy"]).astype({"code_id": int})
        df_sum.to_csv(out_prefix.with_name(out_prefix.name + "_codes_K3.csv"), index=False)

    print(f"Wrote {npz_path}")
    if args.csv:
        print("CSV files written alongside the NPZ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
