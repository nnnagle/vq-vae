#!/bin/bash
#SBATCH --job-name=frl-dev
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae-dev/runs/slurm-%j.log

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae-dev:$PYTHONPATH
export ZARR_ROOT=/lustre/isaac24/scratch/nnagle/zarr
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /lustre/isaac24/scratch/nnagle/vq-vae-dev/frl
python -m training.train_representation \
    --training config/frl_training_v1.yaml \
    --batch-size 16 \
    --epochs 3 \
    --max-batches 3 \
    --phase-start-epoch 0 \
    --overwrite
