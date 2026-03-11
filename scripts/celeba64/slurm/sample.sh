#!/bin/bash
#SBATCH --job-name=celeba_sample
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/celeba64_sample_%j.out
#SBATCH --error=logs/celeba64_sample_%j.err

# Sample from CelebA-64 checkpoint and compute FID/IS/KID (unconditional).
#
# Env vars:
#   CHECKPOINT=checkpoints/celeba64/baseline/last.pt  (required)
#   N_SAMPLES=10000  SAMPLER=ddpm  DDIM_STEPS=200  DDIM_ETA=0.0  TAG=baseline
#
# Submit:
#   CHECKPOINT=checkpoints/celeba64/6b16h256d/last.pt TAG=6b16h256d sbatch scripts/celeba64/slurm/sample.sh
#   SAMPLER=ddim DDIM_ETA=0.4 CHECKPOINT=... sbatch scripts/celeba64/slurm/sample.sh

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module --ignore_cache load cuda/12.2.1/gcc-11.2.0

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$WORKDIR"
mkdir -p logs

CHECKPOINT=${CHECKPOINT:?"CHECKPOINT env var required"}
N_SAMPLES=${N_SAMPLES:-10000}
SAMPLER=${SAMPLER:-ddpm}
DDIM_STEPS=${DDIM_STEPS:-200}
DDIM_ETA=${DDIM_ETA:-0.0}
TAG=${TAG:-"eval"}
BATCH_SIZE=${BATCH_SIZE:-16}

OUT_DIR="samples/celeba64/${TAG}_${SAMPLER}"
if [ "$SAMPLER" = "ddim" ]; then
    OUT_DIR="${OUT_DIR}_s${DDIM_STEPS}_e${DDIM_ETA}"
fi

DATA_H5="data/celeba64/celeba64_gaussians_K1000.h5"

mkdir -p "$OUT_DIR"

echo "========================================================"
echo "  SLURM job $SLURM_JOB_ID — CelebA-64 sampling"
echo "  Host       : $(hostname)"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  CHECKPOINT : $CHECKPOINT"
echo "  N_SAMPLES  : $N_SAMPLES"
echo "  SAMPLER    : $SAMPLER"
if [ "$SAMPLER" = "ddim" ]; then
echo "  DDIM_STEPS : $DDIM_STEPS"
echo "  DDIM_ETA   : $DDIM_ETA"
fi
echo "  OUT_DIR    : $OUT_DIR"
echo "========================================================"

SAMPLER_ARGS="--sampler $SAMPLER"
if [ "$SAMPLER" = "ddim" ]; then
    SAMPLER_ARGS="$SAMPLER_ARGS --ddim_steps $DDIM_STEPS --ddim_eta $DDIM_ETA"
fi

$PYTHON scripts/celeba64/sample.py \
    --checkpoint    "$CHECKPOINT" \
    --real_data_h5  "$DATA_H5" \
    --n_samples     "$N_SAMPLES" \
    --batch_size    "$BATCH_SIZE" \
    --out_dir       "$OUT_DIR" \
    $SAMPLER_ARGS

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)."
