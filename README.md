# VQ-VAE Raster Compression Project

## Purpose

This project implements and trains a **Vector Quantized Variational Autoencoder (VQ-VAE)** for large raster time series stored in **chunked Zarr archives**.  
The core goal is to learn compact, spatially and temporally aware latent representations of geospatial features (e.g., canopy, continuous and categorical variables) derived from NAIP and related datasets.

It is intended to:
- Enable lossy but structure-preserving compression of raster cubes.
- Support downstream modeling (e.g., land-cover transitions, ecological forecasting).
- Handle very large rasters efficiently using chunk-wise sampling.

---

## Data Layout

### Input
- Zarr archive built via `scripts/build_zarr.py` using parameters in `scripts/config.yaml`.
- Each Zarr chunk holds `(time, y, x, feature)` blocks.
- NAIP imagery may be read separately or pre-baked into the Zarr cube.

### Chunking
- Typical chunk sizes: `time=5`, `y=32`, `x=32`, `feature=64`.
- Compression: `LZ4` (fast decompression).
- Ragged chunks occur when raster dimensions aren’t multiples of chunk size; sampler logic handles that.

---

## Training Workflow

1. **Dataset**  
   Implemented in `utils/loader.py` and `utils/loader_utils.py`.  
   Loads samples from Zarr by chunk; supports both random and chunk-aware batching.

2. **Sampler**  
   `utils/samplers.py` provides a *chunk-aware batch sampler*: each batch pulls from one chunk only, improving I/O locality.

3. **Model**  
   `train_vqvae.py` builds and trains the VQ-VAE defined in `weights.py`.  
   Uses loss components: reconstruction (`cont`, `cat`, `canopy`) and quantization (`vq`).

4. **Configuration**  
   All hyperparameters live in `scripts/config.yaml`.  
   Key tuning levers:
   - `chunk size` and `compression` affect data loading.
   - `beta` (commitment cost) and `codebook_size` affect codebook utilization.
   - `batch_size`, `num_workers`, and `prefetch_factor` control I/O performance.

5. **Diagnostics**
   Training logs include:
   - `t_load`, `t_fwd`, `t_bwd`, `t_opt` — time per batch phase.
   - `pplx` — codebook perplexity (diversity of code usage).
   - `total`, `cont`, `cat`, `canopy`, `vq` — loss terms.

---

## Notes for *Future Me*

- **Performance tuning:**  
  The biggest gains came from increasing chunk size along `time`, switching to LZ4, and matching `num_workers` to CPU cores (8–16).  
  Keep `pin_memory=True` and `non_blocking=True` for GPU transfer.

- **Training stability:**  
  When the model collapses to low perplexity (`pplx ≈ 1–2`), try lowering `beta` or re-initializing the codebook.  
  Typical convergence happens around 10 epochs.

- **Samplers & chunking:**  
  If you modify `ChunkBatchSampler`, run `pytest tests/test_samplers.py` first — it catches cross-chunk leakage.

- **Crashes on shutdown:**  
  If you see `terminate called without an active exception`, ensure DataLoader uses `multiprocessing_context="spawn"` and that rasterio/Blosc handles are opened per-worker.

- **GPU utilization:**  
  Aim for `t_load` ≤ `t_fwd + t_bwd + t_opt`. Increase batch size or grad accumulation until GPU is saturated (~80–90% util).

---

## Notes for *Future ChatGPT*

You are assisting in this repository.  
When analyzing or generating code:

1. **Respect directory semantics:**  
   - `scripts/` → entry points (`train_vqvae.py`, `build_zarr.py`, etc.)  
   - `utils/` → supporting logic (`argyaml.py`, `log.py.py`, `data_stack.py`, etc).  
   - `tests/` → unit and integration tests.  
   - `vqvae/` → vqvae specific logic (`model.py`, `chunking.py`, etc)

2. **Understand data flow:**  
   - `build_zarr.py` → produces Zarr archive.  
   - `train_vqvae.py` → orchestrates training.

3. **Assume**: PyTorch training on GPU, PyTorch 2.x, CUDA ≥ 12.  
   Use spawn context for multiprocessing, pinned memory for GPU transfer, and handle per-worker rasterio environments.

4. **When optimizing or refactoring:**
   - Maintain chunk locality in sampling.
   - Preserve deterministic batching for reproducibility (`seed` in samplers).
   - Do not introduce I/O in the training loop; prefetch at DataLoader level.
   - Any proposed rewrite must preserve the shape and dtype conventions of dataset outputs.

5. **When adding new utilities:**  
   - Keep CPU-only preprocessing out of training hot path.  
   - Write fast, side-effect-free unit tests (mock Zarrs instead of creating real ones).

---

## Example Commands

```bash
# Build Zarr archive
python -m scripts.build_zarr --config scripts/config.yaml

# Train the VQ-VAE
python -m scripts.train_vqvae --config scripts/config.yaml

# Run sampler tests
pytest tests/test_samplers.py -v

```

---

## House Style
Use this template for utility headers.  See scripts/data_stack.py for example. 
"""
utils/<module>.py
-----------------
<One-line summary of what this file does.>

Purpose
    - <Bullet 1>
    - <Bullet 2>
    - <Bullet 3>

Used by
    - <script_or_module_a.py> : <very short why>
    - <script_or_module_b.py> : <very short why>

Design notes
    - <Non-obvious constraint or invariant>
    - <Evaluation strategy / performance note>
    - <Shape/dtype conventions if relevant>

Assistant guidance
    When extending:
        - <Actionable rule 1>
        - <Actionable rule 2>
        - <Actionable rule 3>
"""
