"""
postprocess.py
---------------
Utilities to decode a trained VQ-VAE's codebook into data space and prepare
flat arrays for downstream analysis.

Design goals
- Keep the compute path pure and side-effect free. No file I/O here.
- Honour the temporal contract: the model operates on [*, T, *] tensors.
- Use the same normalization/encoding conventions as training by delegating to
  the project's loader utilities.

Provided now
- load_model_and_ds: Restore a model and dataset metadata on CPU.
- decode_codebook_sequences: Decode every codebook vector as a length-T sequence.
- denorm_continuous_KTC: Invert z-scores for continuous outputs (row-by-row) using
  training stats.
- decode_cats_KTC: Convert categorical logits to original raw codes using the
  dataset's mapping.
- flatten_to_KT: Reshape [K, T, C] → [K*T, C] and build index columns.
- code_summary: Summarise per-code usage and canopy scalar.

Stubs for future work (API placeholders)
- make_batch_from_raw: Turn raw arrays into a model-ready batch dict.
- encode_quantize_decode: End-to-end encode→quantize→decode for arbitrary batches.
- iter_raster_chunks: Stream large rasters in chunks.
- process_chunk_to_codes: Encode a raster chunk to code indices.

Contracts
- All functions are CPU-first and do not assume a GPU is available.
- Shapes use K for number of codebook entries, T for time steps, Cc/Cg for the
  number of continuous/categorical features, and V for a categorical head's
  vocabulary size.
- Categorical MISS/UNK conventions follow the training loader (0=MISS, 1=UNK).

Example (sketch)
----------------
>>> model, ds = load_model_and_ds("cube.zarr", "ckpt_best.pt")
>>> T = int(ds.ds["years"].sizes.get("time", ds.ds["years"].shape[0]))
>>> cont_pred, cat_logits, canopy = decode_codebook_sequences(model, T)
>>> cont_KTC = denorm_continuous_KTC(cont_pred, cont_names=ds.cont_names, cont_stats=ds.cont_stats)
>>> cats_KTC = decode_cats_KTC(cat_logits, cat_names=ds.cat_names, cat_maps=ds.cat_maps)
>>> cont_KT, code_id, year = flatten_to_KT(cont_KTC, years=ds.ds["years"].values)
>>> cats_KT, _, _ = flatten_to_KT(cats_KTC, years=ds.ds["years"].values)
>>> summary_K3 = code_summary(model, canopy)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Project imports: keep narrow to avoid training-only deps
from vqvae.model import VQVAE
from vqvae.loader import VQVAEDataset
from vqvae.loader_utils import (
    denorm_continuous_row,
    decode_categorical_batch,
    build_id_maps_inverse,
)


# ---------------------------- Core utilities ---------------------------- #
def load_model_and_ds(zarr_path: str, ckpt_path: str):
    """Restore a trained VQ-VAE and a dataset handle (for schema/metadata).

    Parameters
    ----------
    zarr_path : str
        Path to the consolidated Zarr cube used during training.
    ckpt_path : str
        Path to a checkpoint (.pt) that contains a model state_dict and args.

    Returns
    -------
    model : VQVAE
        A CPU-resident model in eval() mode.
    ds : VQVAEDataset
        Dataset handle providing schema (cont_names, cat_names, stats/maps, years).

    Notes
    -----
    - The model architecture is rebuilt from dataset schema (dims) and checkpoint args.
    - This function keeps everything on CPU for simplicity and portability.
    """
    ds = VQVAEDataset(zarr_path, eager=False)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    ck_args = ckpt.get("args", {})

    # Infer quantizer from keys (don’t trust args blindly)
    sd_keys = list(sd.keys())
    has_ema = any(k.startswith("quant.ema_cluster_size") or k.startswith("quant.ema_embed_sum")
                  for k in sd_keys)
    q_from_args = str(ck_args.get("quantizer", "")).strip().lower()
    quantizer = q_from_args if q_from_args in {"ema", "st"} else ("ema" if has_ema else "st")
    if q_from_args == "ema" and not has_ema:
        quantizer = "st"

    cont_dim = len(ds.cont_names)
    cat_vocab_sizes = {}
    for name in ds.cat_names:
        entry = getattr(ds, "schema_cat", {}).get(name)
        cat_vocab_sizes[name] = int(entry["num_ids"]) if entry is not None else (2 + len(ds.cat_maps.get(name, {})))
    naip_bands = int(ds.naip.shape[-1]) if getattr(ds, "naip", None) is not None else 1

    model = VQVAE(
        cont_dim=cont_dim,
        cat_vocab_sizes=cat_vocab_sizes,
        naip_bands=naip_bands,
        emb_dim=int(ck_args.get("emb_dim", 128)),
        codebook_size=int(ck_args.get("codebook_size", 256)),
        beta=float(ck_args.get("beta", 0.25)),
        hidden=int(ck_args.get("hidden", 128)),
        cat_emb_dim=int(ck_args.get("cat_emb_dim", 6)),
        quantizer=quantizer,
        ema_decay=float(ck_args.get("ema_decay", 0.99)),
        ema_eps=float(ck_args.get("ema_eps", 1e-5)),
    )

    # --- critical fix: strip CodebookManager stats before loading ---
    filtered_sd = {k: v for k, v in sd.items() if "codebook_manager" not in k}

    try:
        model.load_state_dict(filtered_sd, strict=True)
    except RuntimeError:
        # tolerate benign diffs (e.g., EMA buffers present/absent)
        model.load_state_dict(filtered_sd, strict=False)

    model.eval()
    return model, ds


def decode_codebook_sequences(
    model: VQVAE,
    T: int,
    *,
    time_features: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """Decode every codebook entry as a length-T sequence.

    Parameters
    ----------
    model : VQVAE
        Model with an exposed quantizer codebook and decoder.
    T : int
        Sequence length (number of time steps). Usually len(ds.ds["years"]).
    time_features : torch.Tensor, optional
        Placeholder for future temporal conditioning. Currently unused.

    Returns
    -------
    cont_pred : torch.Tensor | None
        Shape [K, T, Cc] or None if Cc == 0.
    cat_logits : dict[str, torch.Tensor]
        For each categorical feature name, logits of shape [K, T, V_name].
    canopy : torch.Tensor
        Shape [K], raw canopy head output per code.

    Contract
    --------
    - No gradients; this is decode-only. Caller may .cpu().numpy() results.
    - The decoder is invoked once on a [K, T, D] tensor, not via repetition.
    """
    with torch.inference_mode():
        codebook: torch.Tensor = model.quant.codebook.detach()  # [K, D]
        K, D = codebook.shape
        zq = codebook.unsqueeze(1).expand(K, T, D)  # [K, T, D]
        cont_pred, cat_logits, canopy = model.decode(zq)
        # canopy may be [K] or [K, 1]; standardize to [K]
        if canopy.ndim > 1:
            canopy = canopy.squeeze(-1)
    return cont_pred, cat_logits, canopy


def denorm_continuous_KTC(
    cont_pred: Optional[torch.Tensor],
    *,
    cont_names: List[str],
    cont_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Invert z-scores for continuous outputs on a [K, T, Cc] tensor.

    Parameters
    ----------
    cont_pred : torch.Tensor | None
        Continuous predictions with shape [K, T, Cc], or None if no continuous.
    cont_names : list[str]
        Feature names in the same column order as the model head.
    cont_stats : dict
        Per-feature stats (mean/std/q01/q99) from training.

    Returns
    -------
    cont_KTC : np.ndarray
        Denormalized continuous values with shape [K, T, Cc]. If Cc==0 or
        cont_pred is None, returns an array of shape [K, T, 0].

    Contract
    --------
    - Uses the project helper denorm_continuous_row row-by-row to mirror loader semantics.
    - Does not re-apply clipping to [q01, q99].
    """
    if cont_pred is None:
        # Infer K and T if possible, else return empty
        return np.zeros((0, 0, 0), dtype=np.float32)

    #cont_np = cont_pred.cpu().numpy()  # [K, T, Cc]
    cont_np = cont_pred.detach().cpu().numpy()  # [K, T, Cc]
    K, T, Cc = cont_np.shape
    if Cc == 0:
        return np.zeros((K, T, 0), dtype=np.float32)

    cont_indices = list(range(Cc))
    feature_names = list(cont_names)

    out = np.empty_like(cont_np, dtype=np.float32)
    for k in range(K):
        for t in range(T):
            out[k, t] = denorm_continuous_row(
                cont_np[k, t], cont_indices, feature_names, cont_stats
            )
    return out


def decode_cats_KTC(
    cat_logits: Dict[str, torch.Tensor],
    *,
    cat_names: List[str],
    cat_maps: Dict[str, Dict[int, int]],
) -> np.ndarray:
    """Convert categorical logits to original raw codes on [K, T, Cg].

    Parameters
    ----------
    cat_logits : dict[name -> torch.Tensor]
        Each tensor has shape [K, T, V_name] (logits per class).
    cat_names : list[str]
        Names of categorical features in column order.
    cat_maps : dict
        Forward maps {name: {raw_code: dense_id}} from training. Used to build
        inverse maps (dense_id -> raw_code).

    Returns
    -------
    cats_KTC : np.ndarray
        Raw categorical codes (float32) with shape [K, T, Cg]. MISS/UNK → NaN.

    Contract
    --------
    - Argmax over V converts logits to dense IDs.
    - Inverse mapping uses build_id_maps_inverse and decode_categorical_batch.
    """
    if not cat_names:
        # Infer K and T from any logits tensor if present
        any_tensor = next(iter(cat_logits.values())) if cat_logits else None
        if any_tensor is None:
            return np.zeros((0, 0, 0), dtype=np.float32)
        K, T = any_tensor.shape[:2]
        return np.zeros((K, T, 0), dtype=np.float32)

    # Stack dense IDs in the schema order
    dense_list = []
    for name in cat_names:
        logits = cat_logits[name]  # [K, T, V]
        dense = logits.argmax(dim=-1)  # [K, T]
        dense_list.append(dense)
    dense_ids_KTC = torch.stack(dense_list, dim=-1).detach().cpu().numpy().astype(np.int64)  # [K, T, Cg]

    K, T, Cg = dense_ids_KTC.shape
    id_maps_inv = build_id_maps_inverse(cat_maps)  # {name: {dense_id: raw_code}}

    # Decode per K slice using the vectorized batch helper over T rows
    out = np.empty((K, T, Cg), dtype=np.float32)
    cat_indices = list(range(Cg))
    for k in range(K):
        out[k] = decode_categorical_batch(
            dense_ids_2d=dense_ids_KTC[k],
            cat_indices=cat_indices,
            feature_names=cat_names,
            id_maps_inv=id_maps_inv,
        )
    return out


def flatten_to_KT(matrix_KTC: np.ndarray, years: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten [K, T, C] → [K*T, C] and build index columns.

    Parameters
    ----------
    matrix_KTC : np.ndarray
        A 3D array shaped [K, T, C].
    years : np.ndarray
        Vector of years with shape [T]. Values are copied as-is to the index column.

    Returns
    -------
    flat_KT_C : np.ndarray
        The flattened feature matrix [K*T, C].
    code_id : np.ndarray
        Column vector [K*T] with code indices 0..K-1 repeated by T.
    year : np.ndarray
        Column vector [K*T] with years tiled by K.

    Contract
    --------
    - Pure reshape + index construction. No copies beyond the reshape.
    - Caller is responsible for writing or joining these columns downstream.
    """
    if matrix_KTC.ndim != 3:
        raise ValueError(f"Expected [K,T,C], got shape {matrix_KTC.shape}")
    K, T, C = matrix_KTC.shape
    if years.shape[0] != T:
        raise ValueError(f"years length {years.shape[0]} does not match T={T}")

    flat = matrix_KTC.reshape(K * T, C)
    code_id = np.repeat(np.arange(K, dtype=np.int32), T)
    year = np.tile(years.astype(np.int32, copy=False), K)
    return flat, code_id, year

def extract_code_usage_from_state(sd: dict) -> Optional[np.ndarray]:
    pick = (
        "quant.ema_cluster_size",
        "quant.cluster_size",
        "codebook_manager.usage_ema",
        "quant.codebook_manager.usage_ema",
    )
    for k in pick:
        if k in sd:
            arr = sd[k]
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            return np.asarray(arr, dtype=np.float32).reshape(-1)
    return None


def code_summary(
    model: VQVAE,
    canopy: torch.Tensor | np.ndarray,
    *,
    usage: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a [K,3] summary: (code_id, code_usage, canopy).

    Parameters
    ----------
    model : VQVAE
        The trained model (for K).
    canopy : torch.Tensor | np.ndarray
        Per-code canopy scalar, shape [K].
    usage : np.ndarray, optional
        Optional usage vector per code (e.g., EMA cluster sizes). If None,
        fills with zeros.

    Returns
    -------
    summary : np.ndarray
        Shape [K, 3] with columns [code_id, code_usage, canopy].

    Contract
    --------
    - Does not normalise usage; caller may scale to probabilities if desired.
    - Canopy is copied as float32.
    """
    if isinstance(canopy, torch.Tensor):
        canopy_np = canopy.detach().cpu().numpy().astype(np.float32)
    else:
        canopy_np = np.asarray(canopy, dtype=np.float32)

    K = int(model.quant.codebook.shape[0])
    if usage is None:
        usage_np = np.zeros((K,), dtype=np.float32)
    else:
        usage_np = np.asarray(usage, dtype=np.float32)
        if usage_np.shape[0] != K:
            raise ValueError(f"usage length {usage_np.shape[0]} does not match K={K}")

    code_id = np.arange(K, dtype=np.int32)
    summary = np.stack([code_id.astype(np.float32), usage_np, canopy_np], axis=1)
    return summary


# ---------------------------- Future stubs ----------------------------- #

def make_batch_from_raw(
    raw_cont: Optional[np.ndarray],
    raw_cat: Optional[np.ndarray],
    *,
    cont_stats: Dict[str, Dict[str, float]],
    cat_maps: Dict[str, Dict[int, int]],
    years: np.ndarray,
    naip_context: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """(Stub) Convert raw arrays to a model-ready batch dict.

    Intended future behaviour
    -------------------------
    - raw_cont: shape [N, T, Cc] in original units. Will be normalised using
      the same policy as training (clip→z-score).
    - raw_cat: shape [N, T, Cg] with raw integer codes. Will be mapped to
      dense IDs using cat_maps (0=MISS, 1=UNK reserved).
    - years: shape [T] to populate the batch's temporal axis.
    - naip_context: shape [N, Bn, 3, 3] (single-band CHM) or None (zeros).

    Returns
    -------
    batch : dict[str, torch.Tensor]
        Keys: cont, cat, naip, years (+ masks/targets as needed). Shapes match
        the trainer contract: cont/cat → [N, T, *], naip → [N, Bn, 3, 3].
    """
    raise NotImplementedError("make_batch_from_raw is a planned extension.")


def encode_quantize_decode(
    model: VQVAE,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """(Stub) End-to-end encode → quantize → decode for arbitrary batches.

    Returns (planned)
    -----------------
    indices : [N, T] int tensor of code indices
    cont_pred : [N, T, Cc] or None
    cat_logits : dict[name -> [N, T, V]]
    canopy : [N] tensor
    """
    raise NotImplementedError("encode_quantize_decode is a planned extension.")


def iter_raster_chunks(
    zarr_path: str,
    *,
    chunk_shape: Tuple[int, int, int] = (32, 256, 256),  # (T, Y, X)
    mask: Optional[np.ndarray] = None,
):
    """(Stub) Yield raw data windows from a large Zarr cube.

    Yields (planned)
    ----------------
    coords : dict with slice objects for (time, y, x)
    raw_cont : [Ny, T, Cc] flattened over spatial dims
    raw_cat  : [Ny, T, Cg]
    yx_index : [Ny, 2] integer pixel coordinates
    """
    raise NotImplementedError("iter_raster_chunks is a planned extension.")


def process_chunk_to_codes(
    model: VQVAE,
    raw_cont: np.ndarray,
    raw_cat: np.ndarray,
    *,
    cont_stats: Dict[str, Dict[str, float]],
    cat_maps: Dict[str, Dict[int, int]],
    years: np.ndarray,
    naip_context: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(Stub) Encode a raster chunk to code indices [Ny, T].

    Contract (planned)
    ------------------
    - Stateless w.r.t. I/O; the caller manages reading and writing.
    - Uses the same normalisation/encoding as training via make_batch_from_raw.
    """
    raise NotImplementedError("process_chunk_to_codes is a planned extension.")
