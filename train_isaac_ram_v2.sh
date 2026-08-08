#!/bin/bash
#SBATCH --job-name=frl-train-v2
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
# /dev/shm is a single node-wide tmpfs shared by every job on the node, and these
# nodes host 2 GPUs → up to 2 jobs. The ~284 GB zarr extract needs ~284 GB of
# /dev/shm; a co-tenant's shm usage can leave too little ("No space left on
# device"). --exclusive reserves the whole node so the full ~385 GB /dev/shm
# (~50% of 770 GB RAM) is ours alone. All campus-gpu-large nodes are 770 GB, so
# node size is not the issue and no --mem bump / node exclude is needed.
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# v2 dataset launcher (zarr_v2, includes the lcms_chg_class change-agent layer).
# Uses the CORRECTED single-nest tar layout: the tar's members are
# `zarr/va_vae_dataset.zarr/...`, so extracting to /dev/shm/ yields
# /dev/shm/zarr/va_vae_dataset.zarr and ZARR_ROOT=/dev/shm/zarr (no double nest).
# Build the tar with (see CLAUDE.md → Tar extraction layout):
#   cd /data/VA/zarr_v2 && tar -cf zarr_v2.tar --transform='s,^,zarr/,' va_vae_dataset.zarr
# Data (tar + stats sidecar) is staged on Lustre at:
#   /lustre/isaac24/proj/UTK0496/zarr_v2/{zarr_v2.tar, va_vae_dataset_stats.json}

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

DATA_DIR=/lustre/isaac24/proj/UTK0496/zarr_v2

# /dev/shm is a tmpfs that is NOT cleared between jobs, so a previous (possibly
# failed) run on this node can leave a stale extract that fills it up. Clean our
# extract dir before starting, and remove it on exit so we don't strand ~284 GB
# in tmpfs for the next job. Safe because --exclusive gives us the whole node.
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

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl
python -m training.train_representation \
    --training config/frl_training_v1.yaml \
    --batch-size 16 \
    --num-workers ${NUM_WORKERS}
