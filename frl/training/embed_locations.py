"""
embed_locations.py

Given a CSV with lat/lon/year columns, embed each location using the FoR-EST
model and write a new CSV with:
  - pixel_row, pixel_col       — Zarr pixel coordinates
  - split                      — 'train', 'val', or 'test' (checkerboard partition)
  - ysfc                       — years since fire/disturbance at the given year
  - evt                        — Existing Vegetation Type code
  - x_type_0 .. x_type_{C-1}  — normalized type encoder inputs at the pixel
  - x_phase_0 .. x_phase_{C-1}— normalized phase encoder inputs at the pixel for the given year
  - z_type_0 .. z_type_63     — type embedding (64-d)
  - z_phase_0 .. z_phase_11   — phase embedding (12-d) at the given year
  - g_type_0 .. g_type_{P-1}  — projected type embedding (output_dim-d); g(h) in SimCLR notation

Usage:
  python -m training.embed_locations \\
      --csv plots.csv \\
      --checkpoint runs/.../encoder_last.pt \\
      --training config/frl_training_v1.yaml \\
      --zarr-config ../zarr_builder/va_vae_dataset.yaml \\
      --output embeddings.csv \\
      --batch-size 32 --num-workers 4
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pyproj import Transformer
from torch.utils.data import DataLoader, Dataset

from data.loaders.builders.feature_builder import FeatureBuilder
from data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from data.loaders.config.training_config_parser import TrainingConfigParser
from data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2
from data.loaders.readers.windows import SpatialWindow
from models.representation import RepresentationModel

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed lat/lon/year points with FoR-EST model")
    p.add_argument("--csv", required=True, help="Input CSV with lat/lon/year columns")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint .pt path")
    p.add_argument("--training", required=True, help="Training YAML path")
    p.add_argument(
        "--zarr-config",
        default="zarr_builder/va_vae_dataset.yaml",
        help="zarr_builder YAML containing spatial CRS and geotransform",
    )
    p.add_argument("--output", help="Output CSV path (default: <input>_embeddings.csv)")
    p.add_argument("--lat-col", default="LAT_ACTUAL", help="Latitude column name")
    p.add_argument("--lon-col", default="LON_ACTUAL", help="Longitude column name")
    p.add_argument("--year-col", default="MEASYEAR", help="Year column name")
    p.add_argument("--patch-size", type=int, default=64,
                   help="Spatial context patch size for type encoder (pixels)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Model inference batch size")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker processes for parallel Zarr I/O")
    p.add_argument("--device", default="cpu", help="Torch device (cpu or cuda)")
    return p.parse_args()


def load_spatial_config(zarr_config_path: str):
    """Read CRS WKT and GDAL-style geotransform from zarr_builder YAML.

    Returns (crs_wkt, x0, y0, pw, ph) where:
      x0, y0 = top-left corner in projected coordinates
      pw      = pixel width  (positive, e.g. 30 m)
      ph      = pixel height (negative for north-up, e.g. -30 m)
    """
    with open(zarr_config_path) as f:
        cfg = yaml.safe_load(f)
    spatial = cfg["dataset"]["spatial"]
    crs_wkt = spatial["crs"]["wkt"]
    t = spatial["transform"]  # [pw, rx, x0, ry, ph, y0]
    return crs_wkt, float(t[2]), float(t[5]), float(t[0]), float(t[4])


def pixel_split(pix_row: int, pix_col: int, patch_size: int) -> str:
    """Return 'train', 'val', or 'test' using the same checkerboard as ForestDatasetV2."""
    block_height, block_width = 4, 4
    block_row = (pix_row // patch_size) // block_height
    block_col = (pix_col // patch_size) // block_width
    A = (block_row // 2 + block_col // 2) % 2
    B = (block_row + block_col) % 4
    if A == 0 and B == 0:
        return "test"
    if A == 0 and B == 2:
        return "val"
    return "train"


def load_sample_at_window(dataset: ForestDatasetV2, window: SpatialWindow) -> dict:
    """Load all dataset groups for a spatial window using ForestDatasetV2 internals."""
    metadata = {"spatial_window": window, "channel_names": {}, "patch_idx": 0}
    result = {}
    for group_name, group_config in dataset.config.dataset_groups.items():
        data, channel_names = dataset._load_group(group_config, window)
        result[group_name] = data
        metadata["channel_names"][group_name] = channel_names
    result["metadata"] = metadata
    return result


class LocationDataset(Dataset):
    """Loads one patch per (pixel_row, pixel_col, year_idx) record."""

    def __init__(
        self,
        records: list[dict],
        fv2_dataset: ForestDatasetV2,
        feature_builder: FeatureBuilder,
        type_enc_feature: str,
        phase_enc_feature: str,
        patch_size: int,
        full_height: int,
        full_width: int,
        ysfc_channel_idx: int,
        evt_channel_idx: int,
    ):
        self.records = records
        self.dataset = fv2_dataset
        self.feature_builder = feature_builder
        self.type_enc_feature = type_enc_feature
        self.phase_enc_feature = phase_enc_feature
        self.P = patch_size
        self.full_height = full_height
        self.full_width = full_width
        self.ysfc_channel_idx = ysfc_channel_idx
        self.evt_channel_idx = evt_channel_idx

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        pix_row, pix_col = rec["pix_row"], rec["pix_col"]
        year_idx = rec["year_idx"]
        P = self.P

        r0 = max(0, min(pix_row - P // 2, self.full_height - P))
        c0 = max(0, min(pix_col - P // 2, self.full_width - P))
        lr = pix_row - r0
        lc = pix_col - c0

        window = SpatialWindow(row_start=r0, col_start=c0, height=P, width=P)
        sample = load_sample_at_window(self.dataset, window)

        ysfc_val = float(sample["annual"][self.ysfc_channel_idx, year_idx, lr, lc])
        evt_val = int(sample["static_categorical"][self.evt_channel_idx, lr, lc])

        feat_type = self.feature_builder.build_feature(self.type_enc_feature, sample)
        feat_phase = self.feature_builder.build_feature(self.phase_enc_feature, sample)

        return {
            "x_type_patch": torch.from_numpy(feat_type.data).float(),          # [C_type, P, P]
            "x_phase_pixel": torch.from_numpy(feat_phase.data[:, :, lr, lc]).float(),  # [C_phase, T]
            "x_type_center": torch.from_numpy(feat_type.data[:, lr, lc]).float(),      # [C_type]
            "x_phase_center": torch.from_numpy(feat_phase.data[:, year_idx, lr, lc]).float(),  # [C_phase]
            "x_phase_center_dm": torch.from_numpy(
                feat_phase.data[:, :, lr, lc] - feat_phase.data[:, :, lr, lc].mean(axis=1, keepdims=True)
            ).float()[:, year_idx],  # [C_phase] demeaned over T, extracted at year_idx
            "local_row": lr,
            "local_col": lc,
            "year_idx": year_idx,
            "ysfc": ysfc_val,
            "evt": evt_val,
            "orig_idx": rec["orig_idx"],
        }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    device = torch.device(args.device)

    # --- Configs -------------------------------------------------------------
    training_config = TrainingConfigParser(args.training).parse()
    bindings_path = training_config.config_paths.get("bindings_path")
    if not bindings_path:
        raise ValueError("No bindings_path found in training config")
    type_enc_feature = training_config.model_input.type_encoder_feature
    phase_enc_feature = training_config.model_input.phase_encoder_feature
    logger.info("Type encoder feature : %s", type_enc_feature)
    logger.info("Phase encoder feature: %s", phase_enc_feature)

    bindings_config = DatasetBindingsParser(bindings_path).parse()
    feature_builder = FeatureBuilder(bindings_config)
    time_start = bindings_config.time_window.start
    time_end = bindings_config.time_window.end
    n_years = bindings_config.time_window.n_years

    # --- Model ---------------------------------------------------------------
    model = RepresentationModel.from_checkpoint(args.checkpoint, device=device, freeze=True)
    model.eval()

    # --- Spatial reference ---------------------------------------------------
    crs_wkt, x0, y0, pw, ph = load_spatial_config(args.zarr_config)
    transformer = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)

    # --- Dataset + channel indices -------------------------------------------
    P = args.patch_size
    fv2 = ForestDatasetV2(bindings_config, split=None, patch_size=P, min_aoi_fraction=0.0)
    full_height, full_width = fv2.zarr_root["aoi"].shape
    logger.info("Zarr extent: %d rows × %d cols", full_height, full_width)

    # Determine fixed channel indices from one probe load (same for every patch)
    probe = load_sample_at_window(fv2, SpatialWindow(0, 0, P, P))
    annual_names = probe["metadata"]["channel_names"]["annual"]
    cat_names = probe["metadata"]["channel_names"]["static_categorical"]
    ysfc_channel_idx = annual_names.index("ysfc")
    evt_channel_idx = cat_names.index("evt")

    # --- Validate all rows (vectorized projection) ---------------------------
    df = pd.read_csv(args.csv)
    logger.info("Loaded %d rows from %s", len(df), args.csv)

    lats = df[args.lat_col].to_numpy(dtype=float)
    lons = df[args.lon_col].to_numpy(dtype=float)
    years = df[args.year_col].to_numpy(dtype=int)

    # Vectorized WGS84 → projected → pixel
    xs, ys = transformer.transform(lons, lats)
    pix_cols = ((xs - x0) / pw).astype(int)
    pix_rows = ((ys - y0) / ph).astype(int)

    records = []
    orig_indices = list(df.index)
    for i, orig_idx in enumerate(orig_indices):
        year_idx = int(years[i]) - time_start
        if not (0 <= year_idx < n_years):
            logger.warning("Row %s: year %d outside %d-%d; skipping.",
                           orig_idx, years[i], time_start, time_end)
            continue
        pr, pc = int(pix_rows[i]), int(pix_cols[i])
        if not (0 <= pr < full_height and 0 <= pc < full_width):
            logger.warning("Row %s: pixel (%d,%d) out of bounds; skipping.", orig_idx, pr, pc)
            continue
        records.append({"pix_row": pr, "pix_col": pc, "year_idx": year_idx, "orig_idx": orig_idx})

    logger.info("%d of %d rows are within bounds and time window", len(records), len(df))

    # --- DataLoader ----------------------------------------------------------
    loc_ds = LocationDataset(
        records, fv2, feature_builder, type_enc_feature, phase_enc_feature,
        P, full_height, full_width, ysfc_channel_idx, evt_channel_idx,
    )
    loader = DataLoader(
        loc_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    # --- Batched inference ---------------------------------------------------
    out_records = []
    n_done = 0

    for batch in loader:
        B = batch["x_type_patch"].shape[0]
        lr_vec = batch["local_row"]   # [B] int
        lc_vec = batch["local_col"]   # [B] int
        yi_vec = batch["year_idx"]    # [B] int
        bidx = torch.arange(B)

        x_type_batch = batch["x_type_patch"].to(device)    # [B, C_type, P, P]
        x_phase_batch = batch["x_phase_pixel"].to(device)  # [B, C_phase, T]

        with torch.no_grad():
            z_type_patch = model(x_type_batch)                     # [B, 64, P, P]
            z_type_center = z_type_patch[bidx, :, lr_vec, lc_vec]  # [B, 64]
            z_proj_center = model.project_type(z_type_center)       # [B, proj_dim]
            z_phase_all = model.forward_phase_at_locations(
                x_phase_batch, z_type_center
            )                                                       # [B, T, 12]
            z_phase_center = z_phase_all[bidx, yi_vec, :]          # [B, 12]

        z_type_np = z_type_center.cpu().numpy()    # [B, 64]
        z_proj_np = z_proj_center.cpu().numpy()    # [B, proj_dim]
        z_phase_np = z_phase_center.cpu().numpy()  # [B, 12]
        x_type_np = batch["x_type_center"].numpy()          # [B, C_type]
        x_phase_np = batch["x_phase_center"].numpy()         # [B, C_phase]
        x_phase_dm_np = batch["x_phase_center_dm"].numpy()  # [B, C_phase]

        for i in range(B):
            oi = int(batch["orig_idx"][i])
            rec = dict(df.loc[oi])
            pr = records[n_done + i]["pix_row"]
            pc = records[n_done + i]["pix_col"]
            rec["pixel_row"] = pr
            rec["pixel_col"] = pc
            rec["split"] = pixel_split(pr, pc, P)
            rec["ysfc"] = float(batch["ysfc"][i])
            rec["evt"] = int(batch["evt"][i])
            for j, v in enumerate(x_type_np[i]):
                rec[f"x_type_{j}"] = float(v)
            for j, v in enumerate(x_phase_np[i]):
                rec[f"x_phase_{j}"] = float(v)
            for j, v in enumerate(x_phase_dm_np[i]):
                rec[f"x_phase_dm_{j}"] = float(v)
            for j, v in enumerate(z_type_np[i]):
                rec[f"z_type_{j}"] = float(v)
            for j, v in enumerate(z_phase_np[i]):
                rec[f"z_phase_{j}"] = float(v)
            for j, v in enumerate(z_proj_np[i]):
                rec[f"g_type_{j}"] = float(v)
            out_records.append(rec)

        n_done += B
        logger.info("Embedded %d / %d", n_done, len(records))

    # --- Write output --------------------------------------------------------
    out_path = args.output or str(Path(args.csv).stem) + "_embeddings.csv"
    pd.DataFrame(out_records).to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(out_records), out_path)


if __name__ == "__main__":
    main()
