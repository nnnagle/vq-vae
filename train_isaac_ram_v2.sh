#!/bin/bash
#SBATCH --job-name=frl-train-v2
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
# Node choice is constrained, NOT via --exclusive. --exclusive reserves the whole
# 2-GPU node, leaving the second GPU idle → the scheduler cancels the job for
# holding an unused GPU. But plain --gpus=1 on a mixed node can hand us the 16 GB
# V100 (GRES mislabels it as v100s, so we can't select by type, and the
# largest-GPU pin only sees the one GPU we were given) → OOM. Fix: restrict to the
# two UNIFORM 32 GB nodes (clrv1107, clrv1205), where any single GPU is 32 GB and
# the other 32 GB GPU stays free for a co-tenant (no idle-resource penalty).
# Excluded: clrv1101 (16 GB) and the mixed 32/16 GB nodes clrv1103/1105/1201.
# /dev/shm is node-wide, so a co-tenant could still contend for it; the pre-clean
# + space guard below handle stale extracts and fail fast if too little is free.
#SBATCH --exclude=clrv1101,clrv1103,clrv1105,clrv1201
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

# This partition has mixed/mislabeled GPUs — e.g. clrv1103 has a 32 GB V100S AND
# a 16 GB V100 despite the gpu:v100s:2 GRES tag. Under --exclusive the whole
# node's GPUs are visible, so pin training to the largest-memory GPU to avoid
# binding to the 16 GB card and OOMing. CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA's
# device indices match nvidia-smi's.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.total \
    --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -1 | tr -d ' ' | cut -d',' -f1)
echo "Pinned to GPU index $CUDA_VISIBLE_DEVICES (largest memory) via CUDA_VISIBLE_DEVICES"

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
