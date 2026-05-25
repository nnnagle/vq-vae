"""
embed_locations.py

Given a CSV with lat/lon/year columns, embed each location using the FoR-EST
model and write a new CSV with:
  - pixel_row, pixel_col       — Zarr pixel coordinates
  - ysfc                       — years since fire/disturbance at the given year
  - evt                        — Existing Vegetation Type code
  - x_type_0 .. x_type_{C-1}  — normalized type encoder inputs at the pixel
  - x_phase_0 .. x_phase_{C-1}— normalized phase encoder inputs at the pixel for the given year
  - z_type_0 .. z_type_63     — type embedding (64-d)
  - z_phase_0 .. z_phase_11   — phase embedding (12-d) at the given year

Usage:
  python frl/training/embed_locations.py \\
      --csv plots.csv \\
      --checkpoint runs/.../encoder_last.pt \\
      --training frl/config/frl_training_v1.yaml \\
      --zarr-config zarr_builder/va_vae_dataset.yaml \\
      --output embeddings.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pyproj import Transformer

from frl.data.loaders.builders.feature_builder import FeatureBuilder
from frl.data.loaders.config.dataset_bindings_parser import DatasetBindingsParser
from frl.data.loaders.config.training_config_parser import TrainingConfigParser
from frl.data.loaders.dataset.forest_dataset_v2 import ForestDatasetV2
from frl.data.loaders.readers.windows import SpatialWindow
from frl.models.representation import RepresentationModel

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
    p.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Spatial context patch size for type encoder (pixels)",
    )
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


def latlon_to_pixel(lat: float, lon: float, transformer: Transformer,
                    x0: float, y0: float, pw: float, ph: float):
    """Project WGS84 (lat, lon) to Zarr integer pixel (row, col).

    ph is expected to be negative (north-up raster).
    """
    x, y = transformer.transform(lon, lat)
    col = int((x - x0) / pw)
    row = int((y - y0) / ph)
    return row, col


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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    device = torch.device(args.device)

    # --- Training config -------------------------------------------------
    training_config = TrainingConfigParser(args.training).parse()
    bindings_path = training_config.config_paths.get("bindings_path")
    if not bindings_path:
        raise ValueError("No bindings_path found in training config")
    type_enc_feature = training_config.model_input.type_encoder_feature
    phase_enc_feature = training_config.model_input.phase_encoder_feature
    logger.info("Type encoder feature : %s", type_enc_feature)
    logger.info("Phase encoder feature: %s", phase_enc_feature)

    # --- Bindings config + FeatureBuilder --------------------------------
    bindings_config = DatasetBindingsParser(bindings_path).parse()
    feature_builder = FeatureBuilder(bindings_config)
    time_start = bindings_config.time_window.start
    time_end = bindings_config.time_window.end
    n_years = bindings_config.time_window.n_years

    # --- Model -----------------------------------------------------------
    model = RepresentationModel.from_checkpoint(
        args.checkpoint, device=device, freeze=True
    )
    model.eval()

    # --- Spatial reference -----------------------------------------------
    crs_wkt, x0, y0, pw, ph = load_spatial_config(args.zarr_config)
    transformer = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)

    # --- Dataset (loads AOI once; gives us _load_group) ------------------
    dataset = ForestDatasetV2(
        bindings_config,
        split=None,
        patch_size=args.patch_size,
        min_aoi_fraction=0.0,
    )
    full_height, full_width = dataset.zarr_root["aoi"].shape
    logger.info("Zarr extent: %d rows × %d cols", full_height, full_width)

    # --- CSV -------------------------------------------------------------
    df = pd.read_csv(args.csv)
    logger.info("Loaded %d rows from %s", len(df), args.csv)

    out_records = []

    for idx, row_data in df.iterrows():
        lat = float(row_data[args.lat_col])
        lon = float(row_data[args.lon_col])
        year = int(row_data[args.year_col])

        # Year → time index
        year_idx = year - time_start
        if not (0 <= year_idx < n_years):
            logger.warning(
                "Row %d: year %d outside time window %d-%d; skipping.",
                idx, year, time_start, time_end,
            )
            continue

        # Lat/lon → pixel
        pix_row, pix_col = latlon_to_pixel(lat, lon, transformer, x0, y0, pw, ph)
        if not (0 <= pix_row < full_height and 0 <= pix_col < full_width):
            logger.warning(
                "Row %d: pixel (%d, %d) out of bounds [0-%d, 0-%d]; skipping.",
                idx, pix_row, pix_col, full_height - 1, full_width - 1,
            )
            continue

        # Patch origin (centered on pixel, clamped to valid extent)
        P = args.patch_size
        r0 = max(0, min(pix_row - P // 2, full_height - P))
        c0 = max(0, min(pix_col - P // 2, full_width - P))
        lr = pix_row - r0  # local row within patch
        lc = pix_col - c0  # local col within patch

        # Load raw data for patch
        window = SpatialWindow(row_start=r0, col_start=c0, height=P, width=P)
        sample = load_sample_at_window(dataset, window)

        # --- ysfc and evt from the raw sample dict ----------------------
        annual_data = sample["annual"]   # [C_annual, T, P, P]
        annual_names = sample["metadata"]["channel_names"]["annual"]
        ysfc_idx = annual_names.index("ysfc")
        ysfc_val = float(annual_data[ysfc_idx, year_idx, lr, lc])

        cat_data = sample["static_categorical"]   # [C_cat, P, P]
        cat_names = sample["metadata"]["channel_names"]["static_categorical"]
        evt_idx = cat_names.index("evt")
        evt_val = int(cat_data[evt_idx, lr, lc])

        # --- Normalized input features -----------------------------------
        feat_type = feature_builder.build_feature(type_enc_feature, sample)
        # feat_type.data: [C_type, P, P]
        feat_phase = feature_builder.build_feature(phase_enc_feature, sample)
        # feat_phase.data: [C_phase, T, P, P]

        x_type = feat_type.data[:, lr, lc]           # [C_type]
        x_phase_year = feat_phase.data[:, year_idx, lr, lc]  # [C_phase]

        # --- Type embedding (needs spatial patch for context) ------------
        x_type_tensor = (
            torch.from_numpy(feat_type.data).float().unsqueeze(0).to(device)
        )  # [1, C_type, P, P]
        with torch.no_grad():
            z_type_patch = model(x_type_tensor)  # [1, 64, P, P]
        z_type = z_type_patch[0, :, lr, lc].cpu().numpy()  # [64]

        # --- Phase embedding at target pixel / all years -----------------
        # x_phase_pixel: [1, C_phase, T]
        x_phase_pixel = (
            torch.from_numpy(feat_phase.data[:, :, lr, lc])
            .float()
            .unsqueeze(0)
            .to(device)
        )
        z_type_pixel = torch.from_numpy(z_type).float().unsqueeze(0).to(device)  # [1, 64]
        with torch.no_grad():
            z_phase_all = model.forward_phase_at_locations(
                x_phase_pixel, z_type_pixel
            )  # [1, T, 12]
        z_phase = z_phase_all[0, year_idx, :].cpu().numpy()  # [12]

        # --- Assemble record --------------------------------------------
        record = dict(row_data)
        record["pixel_row"] = pix_row
        record["pixel_col"] = pix_col
        record["ysfc"] = ysfc_val
        record["evt"] = evt_val
        for i, v in enumerate(x_type):
            record[f"x_type_{i}"] = float(v)
        for i, v in enumerate(x_phase_year):
            record[f"x_phase_{i}"] = float(v)
        for i, v in enumerate(z_type):
            record[f"z_type_{i}"] = float(v)
        for i, v in enumerate(z_phase):
            record[f"z_phase_{i}"] = float(v)

        out_records.append(record)
        logger.info(
            "Row %d: pixel=(%d,%d)  ysfc=%.1f  evt=%d",
            idx, pix_row, pix_col, ysfc_val, evt_val,
        )

    # --- Write output ----------------------------------------------------
    out_path = args.output or str(Path(args.csv).stem) + "_embeddings.csv"
    pd.DataFrame(out_records).to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(out_records), out_path)


if __name__ == "__main__":
    main()
