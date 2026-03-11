#!/bin/bash
#SBATCH --job-name=cifar_scout
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/cifar10_scout_%A_%a.out
#SBATCH --error=logs/cifar10_scout_%A_%a.err

# Tier 1: Scout — SLURM array job for parallel hyperparameter screening.
# Each array task runs 30 epochs on 6k images (1 class), no FID, V100.
# ~4-5 min per config. Signal: val_loss at epoch 30.
#
# Define configs below as parallel arrays, then submit:
#   sbatch --array=0-7 scripts/cifar10/slurm/scout.sh        # all 8 configs
#   sbatch --array=0-3 scripts/cifar10/slurm/scout.sh        # first 4 only
#   sbatch --array=0-7%4 scripts/cifar10/slurm/scout.sh      # 8 configs, max 4 concurrent
#
# Or override for a single config:
#   LR=1e-3 TAG=scout_lr1e3 sbatch scripts/cifar10/slurm/scout.sh   # no --array = single run

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module --ignore_cache load cuda/12.2.1/gcc-11.2.0

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$WORKDIR"
mkdir -p logs

# =====================================================================
# SWEEP CONFIGS — edit these arrays to define your sweep.
# Each index is one config. All arrays must have the same length.
# =====================================================================
SWEEP_TAGS=(   "scout_lr5e4"  "scout_lr1e3"  "scout_lr2e3"  "scout_lr5e3"  "scout_lr1e4"  "scout_lr5e5"  "scout_bs64"   "scout_bs256" )
SWEEP_LRS=(    5e-4           1e-3           2e-3           5e-3           1e-4           5e-5           1e-4           1e-4          )
SWEEP_BS=(     128            128            128            128            128            128            64             256           )
SWEEP_BLOCKS=( 6              6              6              6              6              6              6              6             )
SWEEP_HEADS=(  16             16             16             16             16             16             16             16            )
SWEEP_HDIM=(   256            256            256            256            256            256            256            256           )
SWEEP_WD=(     1e-4           1e-4           1e-4           1e-4           1e-4           1e-4           1e-4           1e-4          )

# =====================================================================
# Select config: from array task ID or from env vars (single run)
# =====================================================================
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    IDX=$SLURM_ARRAY_TASK_ID
    TAG=${SWEEP_TAGS[$IDX]}
    LR=${SWEEP_LRS[$IDX]}
    BATCH_SIZE=${SWEEP_BS[$IDX]}
    NUM_BLOCKS=${SWEEP_BLOCKS[$IDX]}
    NUM_HEADS=${SWEEP_HEADS[$IDX]}
    HIDDEN_DIM=${SWEEP_HDIM[$IDX]}
    WEIGHT_DECAY=${SWEEP_WD[$IDX]}
else
    # Single-run mode: use env vars with defaults
    TAG=${TAG:-"scout_single"}
    LR=${LR:-1e-4}
    BATCH_SIZE=${BATCH_SIZE:-128}
    NUM_BLOCKS=${NUM_BLOCKS:-6}
    NUM_HEADS=${NUM_HEADS:-16}
    HIDDEN_DIM=${HIDDEN_DIM:-256}
    WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
fi

# Fixed params (override via env if needed)
EPOCHS=${EPOCHS:-30}
CFG_SCALE=${CFG_SCALE:-2.0}
CFG_DROPOUT=${CFG_DROPOUT:-0.1}
TIMESTEPS=${TIMESTEPS:-200}
SCHEDULE_S=${SCHEDULE_S:-0.008}
WARMUP_STEPS=${WARMUP_STEPS:-500}
CLASSES=${CLASSES:-0}
FID_EVERY=${FID_EVERY:-0}
SAMPLE_EVERY=${SAMPLE_EVERY:-30}
WANDB=${WANDB:-1}

CKPT_DIR="checkpoints/cifar10/${TAG}"
SAMPLE_DIR="samples/cifar10/${TAG}"
DATA_H5="data/cifar10/cifar10_gaussians_K500.h5"

mkdir -p "$CKPT_DIR" "$SAMPLE_DIR"

echo "========================================================"
echo "  SCOUT — Tier 1 fast screening"
echo "  SLURM job $SLURM_JOB_ID (array task ${SLURM_ARRAY_TASK_ID:-N/A})"
echo "  Host    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  Started : $(date)"
echo "========================================================"
echo "Config:"
echo "  TAG          : $TAG"
echo "  EPOCHS       : $EPOCHS"
echo "  BATCH_SIZE   : $BATCH_SIZE"
echo "  LR           : $LR"
echo "  WEIGHT_DECAY : $WEIGHT_DECAY"
echo "  HIDDEN_DIM   : $HIDDEN_DIM"
echo "  NUM_BLOCKS   : $NUM_BLOCKS"
echo "  NUM_HEADS    : $NUM_HEADS"
echo "  CLASSES      : $CLASSES"
echo "  FID_EVERY    : $FID_EVERY (disabled)"
echo "  WANDB        : $WANDB"
echo ""

WANDB_ARG=""
if [ "$WANDB" = "1" ]; then
    WANDB_ARG="--wandb"
fi

CLASSES_ARG=""
if [ -n "$CLASSES" ]; then
    CLASSES_ARG="--classes $CLASSES"
fi

$PYTHON scripts/cifar10/train.py \
    --data_h5       "$DATA_H5" \
    --epochs        "$EPOCHS" \
    --batch_size    "$BATCH_SIZE" \
    --lr            "$LR" \
    --weight_decay  "$WEIGHT_DECAY" \
    --warmup_steps  "$WARMUP_STEPS" \
    --hidden_dim    "$HIDDEN_DIM" \
    --num_blocks    "$NUM_BLOCKS" \
    --num_heads     "$NUM_HEADS" \
    --cfg_scale     "$CFG_SCALE" \
    --cfg_dropout   "$CFG_DROPOUT" \
    --timesteps     "$TIMESTEPS" \
    --schedule_s    "$SCHEDULE_S" \
    --checkpoint_dir "$CKPT_DIR" \
    --sample_dir    "$SAMPLE_DIR" \
    --sample_every  "$SAMPLE_EVERY" \
    --fid_every     "$FID_EVERY" \
    --tag           "$TAG" \
    $WANDB_ARG \
    $CLASSES_ARG

echo ""
echo "Job $SLURM_JOB_ID (task ${SLURM_ARRAY_TASK_ID:-N/A}) finished at $(date)."
