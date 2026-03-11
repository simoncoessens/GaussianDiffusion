#!/bin/bash
#SBATCH --job-name=cifar_sweep
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/cifar10_sweep_%j.out
#SBATCH --error=logs/cifar10_sweep_%j.err

# Sweep sampling hyperparameters on a trained CIFAR-10 checkpoint.
# V100 is sufficient since this only samples (no training).
#
# Submit:
#   CKPT=checkpoints/cifar10/std3_8h384d_lr1e3/best.pt sbatch scripts/cifar10/slurm/sample_sweep.sh
#   CKPT=checkpoints/cifar10/std3_lr5e4_bs128_500ep/best.pt SWEEP=eta sbatch scripts/cifar10/slurm/sample_sweep.sh

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module --ignore_cache load cuda/12.2.1/gcc-11.2.0

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$WORKDIR"
mkdir -p logs

CKPT=${CKPT:-"checkpoints/cifar10/std3_8h384d_lr1e3/best.pt"}
N_SAMPLES=${N_SAMPLES:-1000}
N_REAL=${N_REAL:-10000}
SWEEP=${SWEEP:-"full"}

echo "========================================================"
echo "  Sampling Sweep — $SWEEP"
echo "  SLURM job $SLURM_JOB_ID"
echo "  Checkpoint: $CKPT"
echo "  N_SAMPLES: $N_SAMPLES"
echo "  SWEEP: $SWEEP"
echo "  Started: $(date)"
echo "========================================================"

$PYTHON scripts/cifar10/sample_sweep.py \
    --checkpoint "$CKPT" \
    --n_samples "$N_SAMPLES" \
    --n_real "$N_REAL" \
    --sweep_mode "$SWEEP"

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)."
