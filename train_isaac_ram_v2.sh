#!/bin/bash
#SBATCH --job-name=frl-train-v2
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
# Exclude the smaller-RAM nodes: the ~284 GB zarr is extracted into /dev/shm,
# which is RAM-backed and capped near 50% of node RAM regardless of --mem. Only
# the 770 GB campus-gpu-large nodes have a big-enough /dev/shm. clrv1101 (small
# GPU) and clrv1205 (V100S-32GB, ~512 GB RAM → ~256 GB /dev/shm) are too small.
#SBATCH --exclude=clrv1101,clrv1205
#SBATCH --cpus-per-task=48
# --mem>=600G only schedules on the 770 GB nodes (their /dev/shm ~385 GB fits the
# 284 GB extract); 512 GB nodes are excluded by this alone.
#SBATCH --mem=600G
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

# Fail fast if this node's /dev/shm is too small for the ~284 GB extract, rather
# than dying partway through. Needs ~300 GB free.
SHM_AVAIL=$(df -B1 --output=avail /dev/shm | tail -1)
NEED=$((300 * 1024 * 1024 * 1024))
if [ "${SHM_AVAIL:-0}" -lt "$NEED" ]; then
    echo "ERROR: /dev/shm on $(hostname) has $(df -h /dev/shm | awk 'NR==2{print $4}')" \
         "free (< ~300 GB). This node is too small; resubmit (the --mem/--exclude" \
         "settings should keep it off small nodes)." >&2
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
