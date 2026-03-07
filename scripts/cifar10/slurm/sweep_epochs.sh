#!/bin/bash
#SBATCH --job-name=cifar_sweep_ep
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/cifar10_sweep_ep_%A_%a.out
#SBATCH --error=logs/cifar10_sweep_ep_%A_%a.err

# Epoch sweep: test different training durations
# Epoch values: 1000, 2000, 3000, 5000
# Uses K=200 (default from config)

echo "========================================================"
echo "  SLURM job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "  Host    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Started : $(date)"
echo "========================================================"

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$WORKDIR"
mkdir -p logs reports/cifar10

K=${K_OVERRIDE:-200}
EPOCH_VALUES=(1000 2000 3000 5000)
EPOCHS=${EPOCH_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running epochs=$EPOCHS (K=$K) sweep..."
$PYTHON scripts/cifar10/hparam_sweep.py \
    --K "$K" \
    --epochs "$EPOCHS" \
    --lr 5e-3 \
    --kernel_size 11 \
    --n_images 20 \
    --device cuda

echo "Finished epochs=$EPOCHS at $(date)"
