#!/bin/bash
#SBATCH --job-name=frl-smoke-v2
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# v2 SMOKE launcher — same v2 dataset (zarr_v2, with lcms_chg_class) and
# single-nest layout as train_isaac_ram_v2.sh, but runs a tiny 3-epoch /
# 20-batch-per-epoch pass with --overwrite to validate a code change end to end
# on real data without committing to a full run.
#
# --phase-start-epoch 1 is the key: the default phase curriculum starts ~epoch 50,
# so without this the smoke run would only exercise the always-on readout/transform
# path and NEVER run the phase encoder / anchor loss / Δ-scale lock. Setting it to 1
# makes epoch 0 settle the readout + observe Δ, and epoch 1 lock Δ and run the
# encoder + anchor + phase losses. Watch the log for:
#   - "Locked anomaly Δ scale = …" at epoch 1
#   - a finite  anchor=…  term in the train/val loss line
#   - the "Readout coverage: mean leverage …" line

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.total \
    --format=csv,noheader,nounits | sort -t',' -k2 -nr | head -1 | tr -d ' ' | cut -d',' -f1)
echo "Pinned to GPU index $CUDA_VISIBLE_DEVICES (largest memory)"

DATA_DIR=/lustre/isaac24/proj/UTK0496/zarr_v2

rm -rf /dev/shm/zarr
trap 'rm -rf /dev/shm/zarr' EXIT

SHM_AVAIL=$(df -B1 --output=avail /dev/shm | tail -1)
NEED=$((300 * 1024 * 1024 * 1024))
if [ "${SHM_AVAIL:-0}" -lt "$NEED" ]; then
    echo "ERROR: /dev/shm on $(hostname) has $(df -h /dev/shm | awk 'NR==2{print $4}')" \
         "free (< ~300 GB) even after cleanup." >&2
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
    --num-workers ${NUM_WORKERS} \
    --epochs 3 \
    --max-batches 20 \
    --phase-start-epoch 1 \
    --overwrite
