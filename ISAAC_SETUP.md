# ISAAC-NG Setup Guide for vq-vae / FoR-EST

## Logging In

```bash
ssh nnagle@login.isaac.utk.edu
```

Authenticate with your UT NetID password + Duo MFA.

---

## Key Paths on ISAAC

| What | Path |
|------|------|
| Home directory | `/nfs/home/nnagle` |
| Scratch (data + code) | `/lustre/isaac24/scratch/nnagle/` |
| Code (repo) | `/lustre/isaac24/scratch/nnagle/vq-vae/` |
| Conda environment | `/nfs/home/nnagle/.conda/envs/frl/` |
| Zarr archive | `/lustre/isaac24/scratch/nnagle/zarr/va_vae_dataset.zarr` |
| Stats file | `/lustre/isaac24/scratch/nnagle/zarr/va_vae_dataset_stats.json` |

---

## Environment Setup (first time only)

```bash
module purge
module load anaconda3
conda create -n frl python=3.11
conda activate /nfs/home/nnagle/.conda/envs/frl

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install zarr numpy pandas scipy matplotlib pyproj PyYAML pytest
```

Add these lines to `~/.bashrc`:

```bash
export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae:$PYTHONPATH
export ZARR_ROOT=/lustre/isaac24/scratch/nnagle/zarr
```

---

## Activating the Environment

Every time you log in:

```bash
module purge
conda activate /nfs/home/nnagle/.conda/envs/frl
```

The `PYTHONPATH` and `ZARR_ROOT` variables are set automatically via `~/.bashrc`.

---

## Running a Training Job (batch — recommended)

Submit the job from the login node:

```bash
sbatch /lustre/isaac24/scratch/nnagle/vq-vae/train_isaac.sh
```

Check job status:

```bash
squeue -u nnagle
```

Watch the log (replace JOBID with the number printed by sbatch):

```bash
tail -f /lustre/isaac24/scratch/nnagle/vq-vae/runs/slurm-JOBID.log
```

---

## Running an Interactive GPU Session (for debugging)

```bash
srun --partition=campus-gpu \
     --account=acf-utk0011 \
     --qos=campus-gpu \
     --gpus=1 \
     --cpus-per-task=4 \
     --mem=32G \
     --pty bash
```

Then activate the environment and run training manually:

```bash
module purge
conda activate /nfs/home/nnagle/.conda/envs/frl
export PYTHONPATH=/lustre/isaac24/scratch/nnagle/vq-vae:$PYTHONPATH
export ZARR_ROOT=/lustre/isaac24/scratch/nnagle/zarr
cd /lustre/isaac24/scratch/nnagle/vq-vae/frl
python -m training.train_representation --training config/frl_training_v1.yaml
```

---

## Compute Statistics (first time only)

```bash
cd /lustre/isaac24/scratch/nnagle/vq-vae/frl
python -m data.examples.example_compute_stats
```

---

## Transferring Data from the Linux Server

Run this on the Linux server (inside `tmux`):

```bash
tmux new -s zarr-transfer
rsync -avP --no-compress /path/to/your.zarr \
    nnagle@dtn1.isaac.utk.edu:/lustre/isaac24/scratch/nnagle/zarr/
```

Authenticate with NetID password + Duo (type `1` for push). Detach with `Ctrl+B, D`.

To resume an interrupted transfer, rerun the same command — rsync skips already-transferred data.

---

## Slurm Account Info

| Field | Value |
|-------|-------|
| Account | `acf-utk0011` |
| GPU partition | `campus-gpu` |
| GPU QOS | `campus-gpu` |
| Available QOS | `campus`, `campus-bigmem`, `campus-gpu`, `long`, `long-bigmem`, `long-gpu`, `short` |
