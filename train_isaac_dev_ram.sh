#!/bin/bash
#SBATCH --job-name=frl-dev-ram
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae-dev/runs/slurm-%j.log

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae-dev:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# CUDA_LAUNCH_BLOCKING left unset: this dev script profiles throughput, and
# serialized launches would distort the timing. Debug with it on if needed.

echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Extracting Zarr tar to RAM ($(date))..."
tar xf /lustre/isaac24/scratch/nnagle/zarr.tar -C /dev/shm/
cp /lustre/isaac24/scratch/nnagle/zarr/*.json /dev/shm/zarr/zarr/
cp /lustre/isaac24/scratch/nnagle/zarr/*.csv /dev/shm/zarr/zarr/
echo "Zarr extract complete ($(date))"
export ZARR_ROOT=/dev/shm/zarr/zarr

NUM_WORKERS=$(( SLURM_CPUS_PER_TASK - 2 ))

cd /lustre/isaac24/scratch/nnagle/vq-vae-dev/frl
python -m training.train_representation \
    --training config/frl_training_v1.yaml \
    --batch-size 16 \
    --num-workers ${NUM_WORKERS} \
    --epochs 3 \
    --max-batches 20 \
    --phase-start-epoch 1 \
    --profile \
    --overwrite
