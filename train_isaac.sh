#!/bin/bash
#SBATCH --job-name=frl-train
#SBATCH --partition=campus-gpu
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=v100s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log

module purge
conda activate /nfs/home/nnagle/.conda/envs/frl

export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae:$PYTHONPATH
export ZARR_ROOT=/lustre/isaac24/scratch/nnagle/zarr

echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl
python -m training.train_representation --training config/frl_training_v1.yaml
