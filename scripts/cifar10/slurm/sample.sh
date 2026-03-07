#!/bin/bash
#SBATCH --job-name=cifar_sample
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/cifar10_sample_%j.out
#SBATCH --error=logs/cifar10_sample_%j.err

# Sample from CIFAR-10 checkpoint and compute FID/IS/KID.
#
# Env vars:
#   CHECKPOINT=checkpoints/cifar10/baseline/last.pt  (required)
#   N_SAMPLES=10000  CFG_SCALE=1.5  SAMPLER=ddpm
#   DDIM_STEPS=200  DDIM_ETA=0.0  TAG=baseline
#
# Submit:
#   CHECKPOINT=checkpoints/cifar10/6b16h256d/last.pt TAG=6b16h256d sbatch scripts/cifar10/slurm/sample.sh
#   SAMPLER=ddim DDIM_ETA=0.4 CHECKPOINT=... sbatch scripts/cifar10/slurm/sample.sh

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
CFG_SCALE=${CFG_SCALE:-1.5}
SAMPLER=${SAMPLER:-ddpm}
DDIM_STEPS=${DDIM_STEPS:-200}
DDIM_ETA=${DDIM_ETA:-0.0}
TAG=${TAG:-"eval"}
BATCH_SIZE=${BATCH_SIZE:-32}

OUT_DIR="samples/cifar10/${TAG}_${SAMPLER}"
if [ "$SAMPLER" = "ddim" ]; then
    OUT_DIR="${OUT_DIR}_s${DDIM_STEPS}_e${DDIM_ETA}"
fi
OUT_DIR="${OUT_DIR}_w${CFG_SCALE}"

DATA_H5="data/cifar10/cifar10_gaussians_K500.h5"

mkdir -p "$OUT_DIR"

echo "========================================================"
echo "  SLURM job $SLURM_JOB_ID — CIFAR-10 sampling"
echo "  Host       : $(hostname)"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  CHECKPOINT : $CHECKPOINT"
echo "  N_SAMPLES  : $N_SAMPLES"
echo "  SAMPLER    : $SAMPLER"
echo "  CFG_SCALE  : $CFG_SCALE"
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

$PYTHON scripts/cifar10/sample.py \
    --checkpoint    "$CHECKPOINT" \
    --real_data_h5  "$DATA_H5" \
    --n_samples     "$N_SAMPLES" \
    --batch_size    "$BATCH_SIZE" \
    --cfg_scale     "$CFG_SCALE" \
    --out_dir       "$OUT_DIR" \
    $SAMPLER_ARGS

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)."
