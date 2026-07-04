#!/usr/bin/env bash
#SBATCH --job-name=frl-posttrain
#SBATCH --partition=campus-gpu-large
#SBATCH --account=acf-utk0011
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --exclude=clrv1101
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

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
# BLAS thread caps: keep each dataloader worker single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /lustre/isaac24/scratch/nnagle/vq-vae/frl

# --- edit these for the run you want to analyze ------------------------------
RUN=runs/frl_v0_exp031
CKPT_DIR="$RUN/checkpoints"
PROBE="$CKPT_DIR/phase_linear_probe.pt"          # written by step 1, consumed by steps 2-3
EVT_MAP=/lustre/isaac24/scratch/nnagle/vq-vae/data/LF2024_EVT.csv   # LANDFIRE crosswalk (VALUE/EVT_NAME)
NWORKERS=16   # dataloader workers for these inference passes. The config's 46 is
              # tuned for training throughput; that many here buffers ~46x2 full
              # patches (+pin_memory) and blows host RAM. 16 keeps the GPU fed.
# To pin a specific checkpoint instead of auto-detecting, set e.g.
#   CKPT="$CKPT_DIR/encoder_best_1_epoch_386.pt"
# -----------------------------------------------------------------------------

# Auto-detect the best checkpoint: the rank-1 top-k file
# (encoder_best_1_epoch_*.pt; rank 1 = best val/loss_total, renamed each time a
# new best is saved). Falls back to encoder_last.pt if top-k tracking hasn't
# started yet (checkpoint.monitor_start_epoch). Honors a pre-set CKPT.
if [[ -z "${CKPT:-}" ]]; then
  shopt -s nullglob
  _best=( "$CKPT_DIR"/encoder_best_1_epoch_*.pt )
  shopt -u nullglob
  if (( ${#_best[@]} )); then
    CKPT=$(ls -t "${_best[@]}" | head -n1)   # newest, in case a stale rank-1 lingers
  else
    CKPT="$CKPT_DIR/encoder_last.pt"
    echo "No encoder_best_1_epoch_*.pt found; falling back to encoder_last.pt"
  fi
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: no usable checkpoint found in $CKPT_DIR" >&2
  exit 1
fi
echo "Using checkpoint: $CKPT"

mkdir -p "$RUN/recovery_curves" "$RUN/evt_diagnostics"

# --- 1. Fit the phase linear probe (critical: steps 2-3 need it) -------------
# Skip the (expensive) fit if the probe already exists — makes the job
# resumable. Set FORCE_REFIT=1 to refit (e.g. after changing CKPT, since the
# probe is fit to a specific checkpoint).
if [[ -f "$PROBE" && "${FORCE_REFIT:-0}" != "1" ]]; then
  echo "=== [1/3] probe exists, skipping fit: $PROBE  (FORCE_REFIT=1 to refit) ==="
else
  echo "=== [1/3] fit_phase_linear_probe ($(date)) ==="
  PYTHONPATH=. python training/fit_phase_linear_probe.py \
    --checkpoint "$CKPT" \
    --training   config/frl_training_v1.yaml \
    --bindings   config/frl_binding_v1.yaml \
    --output      "$PROBE" \
    --num-workers "$NWORKERS"
fi

# --- 2. Per-EVT NBR recovery curves vs ysfc ----------------------------------
# Diagnostics are guarded so a failure in one doesn't block the other; we
# record failures in `fail` and exit non-zero at the end, so a broken step
# triggers SLURM's FAIL email instead of a misleading "completed" one.
fail=0
echo "=== [2/3] phase_recovery_curves ($(date)) ==="
PYTHONPATH=. python training/phase_recovery_curves.py \
  --checkpoint "$CKPT" \
  --probe      "$PROBE" \
  --training   config/frl_training_v1.yaml \
  --bindings   config/frl_binding_v1.yaml \
  --evt-map    "$EVT_MAP" \
  --output-dir "$RUN/recovery_curves" \
  --num-workers "$NWORKERS" \
  || { echo "WARN: phase_recovery_curves failed"; fail=1; }

# --- 3. EVT-stratified FiLM gamma + z_phase temporal variance ----------------
echo "=== [3/3] phase_evt_diagnostics ($(date)) ==="
PYTHONPATH=. python training/phase_evt_diagnostics.py \
  --checkpoint "$CKPT" \
  --training   config/frl_training_v1.yaml \
  --bindings   config/frl_binding_v1.yaml \
  --evt-map    "$EVT_MAP" \
  --probe      "$PROBE" \
  --output-dir "$RUN/evt_diagnostics" \
  --num-workers "$NWORKERS" \
  || { echo "WARN: phase_evt_diagnostics failed"; fail=1; }

if [[ $fail -eq 0 ]]; then
  echo "=== post-training pipeline complete — all steps OK ($(date)) ==="
else
  echo "=== post-training pipeline complete — SOME STEPS FAILED ($(date)) ==="
fi
exit $fail
