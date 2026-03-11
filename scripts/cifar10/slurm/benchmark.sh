#!/bin/bash
#SBATCH --job-name=cifar_bench
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/cifar10_benchmark_%j.out
#SBATCH --error=logs/cifar10_benchmark_%j.err

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module --ignore_cache load cuda/12.2.1/gcc-11.2.0

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion
mkdir -p logs

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
$PYTHON scripts/cifar10/benchmark_train.py
