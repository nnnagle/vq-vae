#!/usr/bin/env bash
#SBATCH --job-name=frl-embed
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
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

# Reads the zarr from Lustre; BLAS caps keep the dataloader workers single-threaded.
export ZARR_ROOT=/lustre/isaac24/scratch/nnagle/zarr
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
