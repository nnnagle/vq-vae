# FoR-EST Forest Estimation with Embedded State Trajectories

## Purpose

A Representation Learning model for creating a distance metric between forest pixels for representing with FIA observations.
The core goal is to learn compact, spatially and temporally aware latent representations of geospatial features (e.g., canopy, continuous and categorical variables) derived from geospatial datasets.

The embedding space is segmented into a forest *type* vector, representing atemporal forest type, and a tempral *phase* vector, representing temporal transitions for that forest type.

Also contains associated data for pre-processing the data and building a zarr archive.

---

## Workflow
- zarr_builder
- update yaml in frl/config/
- frl/examples/data/example_compute_stats.py
- frl/training/train_representation.py

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

1. **Configuration**
  `/frl/config/`

2. **Dataset**  
   Implemented in `frl/data/loaders/dataset/forest_dataset_v2.py`.  
   Loads samples from Zarr by chunk; supports deterministic "checkerboard" train/val/test split.

3. **Feature Builder**  
  `frl/data/loaders/builders/feature_builder.py'
  Builds features specified in yaml

4. **Model**
  `frl/models/representation.py` Main encoder logic
  `frl/training/train_representation.py` training loop



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

