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
| Scratch (data + venv) | `/lustre/isaac24/scratch/nnagle/` |
| Code (repo) | `/lustre/isaac24/scratch/nnagle/vq-vae/` |
| Python venv | `/lustre/isaac24/scratch/nnagle/envs/frl/` |
| Zarr archive | `/lustre/isaac24/scratch/nnagle/<zarr-name>` |

---

## Starting an Interactive GPU Session

```bash
srun --partition=campus-gpu \
     --account=acf-utk0011 \
     --qos=campus-gpu \
     --gpus=1 \
     --pty bash
```

---

## Activating the Environment

Once on a compute node (or after login), run:

```bash
module load Python/3.9.10-gcc
module load cuda/12.2.0-binary
source /lustre/isaac24/scratch/nnagle/envs/frl/bin/activate
```

Your PYTHONPATH is already set in `~/.bashrc`, so `import frl` will work automatically.

---

## Before Training: Update the Zarr Path

Edit `frl/config/frl_binding_v1.yaml` and update the zarr paths:

```yaml
zarr:
  path: /lustre/isaac24/scratch/nnagle/<zarr-name>.zarr
  file: /lustre/isaac24/scratch/nnagle/<zarr-name>_stats.json
```

---

## Compute Statistics (first time only)

```bash
cd /lustre/isaac24/scratch/nnagle/vq-vae
python frl/examples/data/example_compute_stats.py
```

---

## Training

```bash
cd /lustre/isaac24/scratch/nnagle/vq-vae
python frl/training/train_representation.py \
    --training frl/config/frl_training_v1.yaml
```

---

## Transferring Data from the Linux Server

Run this on the Linux server (inside `tmux`):

```bash
tmux new -s zarr-transfer
rsync -avP --no-compress /path/to/your.zarr \
    nnagle@dtn1.isaac.utk.edu:/lustre/isaac24/scratch/nnagle/
```

Authenticate with NetID password + Duo (type `1` for push). Detach with `Ctrl+B, D`.

To resume an interrupted transfer, just rerun the same command — rsync skips already-transferred data.

---

## Slurm Account Info

| Field | Value |
|-------|-------|
| Account | `acf-utk0011` |
| GPU partition | `campus-gpu` |
| GPU QOS | `campus-gpu` |
| Available QOS | `campus`, `campus-bigmem`, `campus-gpu`, `long`, `long-bigmem`, `long-gpu`, `short` |
