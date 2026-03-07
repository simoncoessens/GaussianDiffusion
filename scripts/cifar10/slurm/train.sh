#!/bin/bash
#SBATCH --job-name=cifar_train
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/cifar10_train_%j.out
#SBATCH --error=logs/cifar10_train_%j.err

# Train CIFAR-10 Gaussian diffusion model.
#
# Env vars (all optional, defaults shown):
#   EPOCHS=500  BATCH_SIZE=64  LR=1e-4  HIDDEN_DIM=256  NUM_BLOCKS=6
#   NUM_HEADS=16  CFG_SCALE=1.5  CFG_DROPOUT=0.1  TIMESTEPS=200
#   TAG=baseline  RESUME=  (path to checkpoint)
#
# Submit:
#   sbatch scripts/cifar10/slurm/train.sh
#   EPOCHS=1000 HIDDEN_DIM=256 NUM_BLOCKS=6 TAG=7.4M sbatch scripts/cifar10/slurm/train.sh
#   BATCH_SIZE=128 sbatch -p gpu scripts/cifar10/slurm/train.sh  # V100

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module --ignore_cache load cuda/12.2.1/gcc-11.2.0

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$WORKDIR"
mkdir -p logs

EPOCHS=${EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-256}
NUM_BLOCKS=${NUM_BLOCKS:-6}
NUM_HEADS=${NUM_HEADS:-16}
CFG_SCALE=${CFG_SCALE:-1.5}
CFG_DROPOUT=${CFG_DROPOUT:-0.1}
TIMESTEPS=${TIMESTEPS:-200}
SCHEDULE_S=${SCHEDULE_S:-0.008}
TAG=${TAG:-"${NUM_BLOCKS}b${NUM_HEADS}h${HIDDEN_DIM}d"}
RESUME=${RESUME:-""}

CKPT_DIR="checkpoints/cifar10/${TAG}"
SAMPLE_DIR="samples/cifar10/${TAG}"
DATA_H5="data/cifar10/cifar10_gaussians_K500.h5"

mkdir -p "$CKPT_DIR" "$SAMPLE_DIR"

echo "========================================================"
echo "  SLURM job $SLURM_JOB_ID"
echo "  Host    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Started : $(date)"
echo "========================================================"
echo "Config:"
echo "  TAG         : $TAG"
echo "  EPOCHS      : $EPOCHS"
echo "  BATCH_SIZE  : $BATCH_SIZE"
echo "  LR          : $LR"
echo "  HIDDEN_DIM  : $HIDDEN_DIM"
echo "  NUM_BLOCKS  : $NUM_BLOCKS"
echo "  NUM_HEADS   : $NUM_HEADS"
echo "  CFG_SCALE   : $CFG_SCALE"
echo "  CFG_DROPOUT : $CFG_DROPOUT"
echo "  TIMESTEPS   : $TIMESTEPS"
echo "  RESUME      : ${RESUME:-none}"
echo ""

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

$PYTHON scripts/cifar10/train.py \
    --data_h5       "$DATA_H5" \
    --epochs        "$EPOCHS" \
    --batch_size    "$BATCH_SIZE" \
    --lr            "$LR" \
    --hidden_dim    "$HIDDEN_DIM" \
    --num_blocks    "$NUM_BLOCKS" \
    --num_heads     "$NUM_HEADS" \
    --cfg_scale     "$CFG_SCALE" \
    --cfg_dropout   "$CFG_DROPOUT" \
    --timesteps     "$TIMESTEPS" \
    --schedule_s    "$SCHEDULE_S" \
    --checkpoint_dir "$CKPT_DIR" \
    --sample_dir    "$SAMPLE_DIR" \
    --sample_every  50 \
    --use_ema \
    --amp \
    --compile \
    $RESUME_ARG

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)."
