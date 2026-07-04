#!/usr/bin/env bash
#SBATCH --job-name=frl-posttrain
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log

# Post-training pipeline: fit the phase linear probe, then run the phase
# diagnostics (per-EVT recovery curves + EVT-stratified FiLM/variance).
# Submit after a training run finishes, or chain it:
#   sbatch --dependency=afterok:<train_jobid> post_train_isaac.sh

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /nfs/home/nnagle/.conda/envs/frl

set -euo pipefail

# Read the zarr from Lustre (no RAM/NVMe copy needed for a one-off analysis).
export ZARR_ROOT=/lustre/isaac24/scratch/nnagle/zarr
# BLAS thread caps: the loaders use num_workers=46 from the config.
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
PROBE="$RUN/checkpoints/phase_linear_probe.pt"   # written by step 1, consumed by steps 2-3
EVT_MAP=/lustre/isaac24/scratch/nnagle/vq-vae/data/LF2024_EVT.csv   # LANDFIRE crosswalk (VALUE/EVT_NAME)
# -----------------------------------------------------------------------------

mkdir -p "$RUN/recovery_curves" "$RUN/evt_diagnostics"

# --- 1. Fit the phase linear probe (critical: steps 2-3 need it) -------------
echo "=== [1/3] fit_phase_linear_probe ($(date)) ==="
PYTHONPATH=. python training/fit_phase_linear_probe.py \
  --checkpoint "$CKPT" \
  --training   config/frl_training_v1.yaml \
  --bindings   config/frl_binding_v1.yaml \
  --output     "$PROBE"

# --- 2. Per-EVT NBR recovery curves vs ysfc ----------------------------------
# Diagnostics are guarded with `|| echo WARN` so a failure in one does not
# block the other (set -e would otherwise abort the whole job).
echo "=== [2/3] phase_recovery_curves ($(date)) ==="
PYTHONPATH=. python training/phase_recovery_curves.py \
  --checkpoint "$CKPT" \
  --probe      "$PROBE" \
  --training   config/frl_training_v1.yaml \
  --bindings   config/frl_binding_v1.yaml \
  --evt-map    "$EVT_MAP" \
  --output-dir "$RUN/recovery_curves" \
  || echo "WARN: phase_recovery_curves failed"

# --- 3. EVT-stratified FiLM gamma + z_phase temporal variance ----------------
echo "=== [3/3] phase_evt_diagnostics ($(date)) ==="
PYTHONPATH=. python training/phase_evt_diagnostics.py \
  --checkpoint "$CKPT" \
  --training   config/frl_training_v1.yaml \
  --bindings   config/frl_binding_v1.yaml \
  --evt-map    "$EVT_MAP" \
  --probe      "$PROBE" \
  --output-dir "$RUN/evt_diagnostics" \
  || echo "WARN: phase_evt_diagnostics failed"

echo "=== post-training pipeline complete ($(date)) ==="
