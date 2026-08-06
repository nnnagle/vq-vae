#!/bin/bash
#SBATCH --job-name=frl-train-v2
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
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
# then stage zarr_v2.tar + the v2 stats sidecars on Lustre at the paths below.

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

echo "Extracting v2 Zarr tar to RAM ($(date))..."
tar xf /lustre/isaac24/scratch/nnagle/zarr_v2.tar -C /dev/shm/
cp /lustre/isaac24/scratch/nnagle/zarr_v2/*.json /dev/shm/zarr/
cp /lustre/isaac24/scratch/nnagle/zarr_v2/*.csv /dev/shm/zarr/
echo "Zarr extract complete ($(date))"
export ZARR_ROOT=/dev/shm/zarr

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl
python -m training.train_representation \
    --training config/frl_training_v1.yaml \
    --batch-size 16 \
    --num-workers ${NUM_WORKERS}
