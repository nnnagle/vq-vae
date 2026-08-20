#!/usr/bin/env bash
#SBATCH --job-name=frl-embed
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
# Node choice is constrained, NOT via --exclusive (which would idle the second GPU
# and get the job cancelled). Restrict to the two UNIFORM 32 GB nodes so any single
# GPU we're handed is 32 GB; exclude clrv1101 (16 GB). Mixed 32/16 GB nodes are
# clrv1103/1105/1201 — a plain --gpus=1 there can bind the 16 GB card and OOM.
# /dev/shm is node-wide, so the pre-clean + space guard below handle a co-tenant's
# stale extract and fail fast if too little is free.
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# Embed lat/lon/year points (e.g. FIA plots) with the FoR-EST model:
# writes z_type_*/g_type_* (and z_phase_*) embedding columns per location.
#
# Uses the v2 dataset (zarr_v2, includes the lcms_chg_class change-agent layer that
# the current frl_binding_v1.yaml declares) staged into RAM, following the protocol
# in train_isaac_ram_v2.sh. The CORRECTED single-nest tar layout means the tar's
# members are `zarr/va_vae_dataset.zarr/...`, so extracting to /dev/shm/ yields
# /dev/shm/zarr/va_vae_dataset.zarr and ZARR_ROOT=/dev/shm/zarr (no double nest).
# Data (tar + stats sidecar) is staged on Lustre at:
#   /lustre/isaac24/proj/UTK0496/zarr_v2/{zarr_v2.tar, va_vae_dataset_stats.json}

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

set -euo pipefail

# BLAS caps keep the dataloader workers single-threaded.
export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Mixed/mislabeled GPUs on this partition — pin to the largest-memory GPU so we
# don't bind the 16 GB card and OOM. CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA's
# device indices match nvidia-smi's.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.total \
    --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -1 | tr -d ' ' | cut -d',' -f1)
echo "Pinned to GPU index $CUDA_VISIBLE_DEVICES (largest memory) via CUDA_VISIBLE_DEVICES"

DATA_DIR=/lustre/isaac24/proj/UTK0496/zarr_v2

# /dev/shm is a tmpfs NOT cleared between jobs, so a previous (possibly failed) run
# on this node can leave a stale extract that fills it. Clean our extract dir before
# starting, and remove it on exit so we don't strand ~284 GB in tmpfs.
rm -rf /dev/shm/zarr
trap 'rm -rf /dev/shm/zarr' EXIT

# Fail fast if /dev/shm still can't hold the ~284 GB extract after cleanup.
SHM_AVAIL=$(df -B1 --output=avail /dev/shm | tail -1)
NEED=$((300 * 1024 * 1024 * 1024))
if [ "${SHM_AVAIL:-0}" -lt "$NEED" ]; then
    echo "ERROR: /dev/shm on $(hostname) has $(df -h /dev/shm | awk 'NR==2{print $4}')" \
         "free (< ~300 GB) even after cleanup. Likely a smaller shm mount than the" \
         "50%-of-RAM default, or leftover files owned by another user" \
         "(df -h /dev/shm; ls -la /dev/shm)." >&2
    exit 1
fi

echo "Extracting v2 Zarr tar to RAM ($(date))..."
tar xf "$DATA_DIR/zarr_v2.tar" -C /dev/shm/
cp "$DATA_DIR/va_vae_dataset_stats.json" /dev/shm/zarr/
echo "Zarr extract complete ($(date))"
export ZARR_ROOT=/dev/shm/zarr

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl

# --- edit these ---------------------------------------------------------------
RUN=runs/frl_v0_exp039
CKPT_DIR="$RUN/checkpoints"
CSV=/nfs/home/nnagle/va_merge.csv    # needs LAT_ACTUAL, LON_ACTUAL, MEASYEAR columns (the defaults)
ZARR_CONFIG=/lustre/isaac24/scratch/nnagle/vq-vae/zarr_builder/va_vae_dataset.yaml
NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))
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
  --num-workers "$NUM_WORKERS"

echo "Embeddings written to $OUT ($(date))"
