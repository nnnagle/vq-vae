#!/bin/bash
#SBATCH --job-name=frl-eval-v2
#SBATCH --partition=campus-bigmem
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-bigmem
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# CPU big-memory Step-0 eval launcher (diagnostics A/B/C) for the v2 dataset.
#
# Why CPU/bigmem, not a GPU node: the campus-gpu partitions run a GPU-utilization
# watchdog, and this eval is GPU-light (extract + ridge fitting + streaming are
# CPU-bound), so it gets cancelled "for not using GPU". Running on a CPU bigmem
# partition with --device cpu sidesteps that entirely. The model forward runs on
# CPU; run_eval sets torch threads to SLURM_CPUS_PER_TASK so it uses all cores.
#
# VERIFY the partition/qos names + node RAM first (the docs pin the QOS but not the
# partition name):
#   sinfo -o "%P %c %m %l" | sort -u
#   sinfo -p campus-bigmem -N -o "%N %c %m"
# Adjust --partition/--qos/--mem below to match.
#
# Self-contained (extract + eval in ONE job — /dev/shm is node-local). Avoid
# --exclusive: reserving a whole node but leaving cores/GPUs idle gets the job
# cancelled for holding unused resources. If a co-tenant contends for /dev/shm,
# the pre-clean + guard below fail fast; just resubmit (or raise --mem/--cpus).
#
# Usage:
#   sbatch eval_isaac_bigmem.sh /path/to/checkpoint.pt [extra run_eval args...]
#   # smoke test:
#   sbatch eval_isaac_bigmem.sh runs/frl_v0_exp034/checkpoints/encoder_best_1_epoch_395.pt \
#          --which C --max-batches 2 --max-pixels-per-sample 500 --no-mlp

set -eu

CKPT="${1:?usage: sbatch eval_isaac_bigmem.sh <checkpoint.pt> [extra run_eval args...]}"
shift || true
# Resolve to absolute NOW (cwd is the sbatch submit dir), before we cd into the repo.
CKPT="$(realpath -m "$CKPT")"
if [ ! -e "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT (resolved from submit dir $(pwd))" >&2
    exit 1
fi

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae:$PYTHONPATH
# Keep the many dataloader workers single-threaded (numpy/OpenBLAS); run_eval
# raises torch's own thread count to SLURM_CPUS_PER_TASK for the CPU model
# forward + float64 ridge matmuls in the main process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on node: $(hostname) (CPU eval, no GPU)"

DATA_DIR=/lustre/isaac24/proj/UTK0496/zarr_v2

# /dev/shm is a node-local tmpfs not cleared between jobs — clear a prior run's
# stale extract, and remove ours on exit.
rm -rf /dev/shm/zarr
trap 'rm -rf /dev/shm/zarr' EXIT

SHM_AVAIL=$(df -B1 --output=avail /dev/shm | tail -1)
NEED=$((300 * 1024 * 1024 * 1024))
if [ "${SHM_AVAIL:-0}" -lt "$NEED" ]; then
    echo "ERROR: /dev/shm on $(hostname) has $(df -h /dev/shm | awk 'NR==2{print $4}')" \
         "free (< ~300 GB). A co-tenant is using it; resubmit (df -h /dev/shm)." >&2
    exit 1
fi

echo "Extracting v2 Zarr tar to RAM ($(date))..."
tar xf "$DATA_DIR/zarr_v2.tar" -C /dev/shm/
cp "$DATA_DIR/va_vae_dataset_stats.json" /dev/shm/zarr/
echo "Zarr extract complete ($(date))"
export ZARR_ROOT=/dev/shm/zarr

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))
EVT_MAP=/lustre/isaac24/scratch/nnagle/vq-vae/data/LF2024_EVT.csv

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl
echo "Evaluating checkpoint (CPU): $CKPT"
python -m training.phase_eval.run_eval \
    --checkpoint "$CKPT" \
    --training config/frl_training_v1.yaml \
    --device cpu \
    --evt-map "$EVT_MAP" \
    --num-workers "${NUM_WORKERS}" \
    "$@"
