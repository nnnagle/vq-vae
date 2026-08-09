#!/bin/bash
#SBATCH --job-name=frl-eval-v2
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# Step-0 eval harness launcher (diagnostics A/B/C) for the v2 dataset.
#
# Self-contained: extraction to /dev/shm and the eval run happen in THIS job on
# ONE node — /dev/shm is node-local, so you cannot extract in one job and eval in
# another. Mirrors train_isaac_ram_v2.sh (single-nest tar, --exclusive for a full
# /dev/shm + whole node, stale-shm cleanup, largest-GPU pin).
#
# Usage:
#   sbatch eval_isaac_v2.sh /path/to/checkpoint.pt [extra run_eval args...]
# Examples:
#   # smoke test (fast, cheapest diagnostic, no MLP):
#   sbatch eval_isaac_v2.sh runs/frl_v0_exp034/encoder_last.pt \
#          --which C --max-batches 2 --max-pixels-per-sample 500 --no-mlp
#   # full A/B/C baseline (EVT crosswalk is baked in as a default below):
#   sbatch eval_isaac_v2.sh runs/frl_v0_exp034/encoder_last.pt
# Compare two runs afterward with training.phase_eval.compare_eval.

# set -eu (not pipefail): several lines below use `... | head -1`, which SIGPIPEs
# the upstream command; with pipefail that would abort the script.
set -eu

CKPT="${1:?usage: sbatch eval_isaac_v2.sh <checkpoint.pt> [extra run_eval args...]}"
shift || true
# Resolve to an absolute path NOW, while cwd is still the sbatch submit dir. We cd
# into the repo before invoking python, so a relative checkpoint path would
# otherwise be resolved against the wrong directory.
CKPT="$(realpath -m "$CKPT")"
if [ ! -e "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT (resolved from submit dir $(pwd))" >&2
    exit 1
fi

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Mixed/mislabeled GPUs on this partition (some nodes pair a 32 GB V100S with a
# 16 GB V100). Under --exclusive all GPUs are visible; pin to the largest one.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.total \
    --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -1 | tr -d ' ' | cut -d',' -f1)
echo "Pinned to GPU index $CUDA_VISIBLE_DEVICES (largest memory) via CUDA_VISIBLE_DEVICES"

DATA_DIR=/lustre/isaac24/proj/UTK0496/zarr_v2

# /dev/shm is a node-local tmpfs not cleared between jobs — clear a prior run's
# stale extract, and remove ours on exit. Safe under --exclusive.
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

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))

# EVT crosswalk for diagnostic B's recovery-curve labels. Baked in as a default;
# override by passing your own --evt-map after the checkpoint (argparse takes the
# last one).
EVT_MAP=/lustre/isaac24/scratch/nnagle/vq-vae/data/LF2024_EVT.csv

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl
echo "Evaluating checkpoint: $CKPT"
python -m training.phase_eval.run_eval \
    --checkpoint "$CKPT" \
    --training config/frl_training_v1.yaml \
    --evt-map "$EVT_MAP" \
    --num-workers "${NUM_WORKERS}" \
    "$@"
