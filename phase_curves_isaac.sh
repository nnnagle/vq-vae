#!/usr/bin/env bash
#SBATCH --job-name=frl-phase-curves
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

set -euo pipefail

# Read the zarr from Lustre (no RAM/NVMe copy needed for a one-off analysis).
export ZARR_ROOT=/lustre/isaac24/scratch/nnagle/zarr
# BLAS thread caps: the loader uses num_workers=46 from the config; without
# these, each worker's BLAS pool would oversubscribe the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl

# --- edit these for the run you want to analyze ------------------------------
RUN=runs/frl_v0_exp031
CKPT="$RUN/checkpoints/encoder_last.pt"          # or a specific best checkpoint: ls "$RUN/checkpoints/"
PROBE="$RUN/checkpoints/phase_linear_probe.pt"   # from fit_phase_linear_probe.py
EVT_MAP=/lustre/isaac24/scratch/nnagle/vq-vae/data/LF2024_EVT.csv   # LANDFIRE crosswalk (VALUE/EVT_NAME)
# -----------------------------------------------------------------------------

# PYTHONPATH=. so `import data...` / `import training...` resolve from the frl/ cwd.
PYTHONPATH=. python training/phase_recovery_curves.py \
  --checkpoint "$CKPT" \
  --probe      "$PROBE" \
  --training   config/frl_training_v1.yaml \
  --bindings   config/frl_binding_v1.yaml \
  --evt-map    "$EVT_MAP" \
  --output-dir "$RUN/recovery_curves"
