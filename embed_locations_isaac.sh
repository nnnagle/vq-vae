#!/usr/bin/env bash
#SBATCH --job-name=frl-embed
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
# Restrict to the two UNIFORM 32 GB nodes (any single GPU is 32 GB), rather than
# --exclusive (idles the 2nd GPU → cancelled for an unused GPU) or plain --gpus=1
# on a mixed node (SLURM may hand us the 16 GB V100 → OOM; GRES mislabels it so we
# can't select by type). Excluded: clrv1101 (16 GB). note:  mixed 32/16 are clrv1103/1105/1201.
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=18
# --mem=500G: enough headroom to hold the v2 zarr in /dev/shm (RAM tmpfs).
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# Embed lat/lon/year points (e.g. FIA plots) with the FoR-EST model:
# writes z_type_*/g_type_* (and phase) embedding columns per location.

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

set -euo pipefail

# BLAS caps keep the dataloader workers single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Mixed/mislabeled GPUs on this partition (some nodes pair a 32 GB V100S with a
# 16 GB V100). Pin to the largest-memory GPU visible to this job.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.total \
    --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -1 | tr -d ' ' | cut -d',' -f1)
echo "Pinned to GPU index $CUDA_VISIBLE_DEVICES (largest memory) via CUDA_VISIBLE_DEVICES"

# Load the v2 zarr into RAM (matches eval_isaac_v2.sh). Single-nest tar extracts to
# /dev/shm/zarr/va_vae_dataset.zarr, so ZARR_ROOT=/dev/shm/zarr.
DATA_DIR=/lustre/isaac24/proj/UTK0496/zarr_v2

# /dev/shm is a node-local tmpfs not cleared between jobs — clear a prior run's
# stale extract, and remove ours on exit.
rm -rf /dev/shm/zarr
trap 'rm -rf /dev/shm/zarr' EXIT

SHM_AVAIL=$(df -B1 --output=avail /dev/shm | tail -1)
NEED=$((300 * 1024 * 1024 * 1024))
if [ "${SHM_AVAIL:-0}" -lt "$NEED" ]; then
    echo "ERROR: /dev/shm on $(hostname) has $(df -h /dev/shm | awk 'NR==2{print $4}')" \
         "free (< ~300 GB) even after cleanup (df -h /dev/shm)." >&2
    exit 1
fi

echo "Extracting v2 Zarr tar to RAM ($(date))..."
tar xf "$DATA_DIR/zarr_v2.tar" -C /dev/shm/
cp "$DATA_DIR/va_vae_dataset_stats.json" /dev/shm/zarr/
echo "Zarr extract complete ($(date))"
export ZARR_ROOT=/dev/shm/zarr

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl

# --- edit these ---------------------------------------------------------------
RUN=runs/frl_v0_exp031
CKPT_DIR="$RUN/checkpoints"
CSV=/nfs/home/nnagle/va_merge.csv    # needs LAT_ACTUAL, LON_ACTUAL, MEASYEAR columns (the defaults)
ZARR_CONFIG=/lustre/isaac24/scratch/nnagle/vq-vae/zarr_builder/va_vae_dataset.yaml
NWORKERS=16
# To pin a specific checkpoint instead of auto-detecting, set CKPT=... here.
# ------------------------------------------------------------------------------

# Auto-detect the best checkpoint: rank-1 top-k file, else encoder_last
# (same logic as post_train_isaac.sh).
if [[ -z "${CKPT:-}" ]]; then
  shopt -s nullglob
  _best=( "$CKPT_DIR"/encoder_best_1_epoch_*.pt )
  shopt -u nullglob
  if (( ${#_best[@]} )); then
    CKPT=$(ls -t "${_best[@]}" | head -n1)
  else
    CKPT="$CKPT_DIR/encoder_last.pt"
    echo "No encoder_best_1_epoch_*.pt found; falling back to encoder_last.pt"
  fi
fi
[[ -f "$CKPT" ]] || { echo "ERROR: no usable checkpoint in $CKPT_DIR" >&2; exit 1; }
echo "Using checkpoint: $CKPT"

OUT="$RUN/$(basename "${CSV%.csv}")_embeddings.csv"
mkdir -p "$RUN"

PYTHONPATH=. python training/embed_locations.py \
  --csv         "$CSV" \
  --checkpoint  "$CKPT" \
  --training    config/frl_training_v1.yaml \
  --zarr-config "$ZARR_CONFIG" \
  --output      "$OUT" \
  --device      cuda \
  --num-workers "$NWORKERS"

echo "Embeddings written to $OUT ($(date))"
