#!/bin/bash
#SBATCH --job-name=cifar_validate
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/cifar10_validate_%A_%a.out
#SBATCH --error=logs/cifar10_validate_%A_%a.err

# Tier 2: Validate — rank scout winners by FID on full data (SLURM array).
# Runs 75 epochs on all 60k images, FID every 25 epochs, A100.
# ~25 min per config. Signal: FID at epoch 75.
#
# Define winning configs below, then submit:
#   sbatch --array=0-3 scripts/cifar10/slurm/validate.sh       # all 4 winners
#   sbatch --array=0-1 scripts/cifar10/slurm/validate.sh       # first 2 only
#
# Or single run via env vars:
#   LR=1e-3 TAG=val_lr1e3 sbatch scripts/cifar10/slurm/validate.sh

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module --ignore_cache load cuda/12.2.1/gcc-11.2.0

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$WORKDIR"
mkdir -p logs

# =====================================================================
# SWEEP CONFIGS — populate with scout winners to validate.
# Each index is one config. All arrays must have the same length.
# =====================================================================
SWEEP_TAGS=(   "val_lr5e4"  "val_lr1e3"  "val_lr2e3"  "val_lr5e3" )
SWEEP_LRS=(    5e-4         1e-3         2e-3         5e-3        )
SWEEP_BS=(     512          512          512          512         )
SWEEP_BLOCKS=( 6            6            6            6           )
SWEEP_HEADS=(  16           16           16           16          )
SWEEP_HDIM=(   256          256          256          256         )
SWEEP_WD=(     1e-4         1e-4         1e-4         1e-4        )

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
    TAG=${TAG:-"val_single"}
    LR=${LR:-1e-4}
    BATCH_SIZE=${BATCH_SIZE:-512}
    NUM_BLOCKS=${NUM_BLOCKS:-6}
    NUM_HEADS=${NUM_HEADS:-16}
    HIDDEN_DIM=${HIDDEN_DIM:-256}
    WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
fi

# Fixed params (override via env if needed)
EPOCHS=${EPOCHS:-75}
CFG_SCALE=${CFG_SCALE:-2.0}
CFG_DROPOUT=${CFG_DROPOUT:-0.1}
TIMESTEPS=${TIMESTEPS:-200}
SCHEDULE_S=${SCHEDULE_S:-0.008}
WARMUP_STEPS=${WARMUP_STEPS:-500}
RESUME=${RESUME:-""}
FID_EVERY=${FID_EVERY:-25}
FID_N_SAMPLES=${FID_N_SAMPLES:-250}
FID_N_REAL=${FID_N_REAL:-5000}
FID_DDIM_STEPS=${FID_DDIM_STEPS:-25}
WANDB=${WANDB:-1}
SAMPLE_EVERY=${SAMPLE_EVERY:-25}
CLASSES=${CLASSES:-""}

CKPT_DIR="checkpoints/cifar10/${TAG}"
SAMPLE_DIR="samples/cifar10/${TAG}"
DATA_H5="data/cifar10/cifar10_gaussians_K500.h5"

mkdir -p "$CKPT_DIR" "$SAMPLE_DIR"

echo "========================================================"
echo "  VALIDATE — Tier 2 FID ranking"
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
echo "  CFG_SCALE    : $CFG_SCALE"
echo "  FID_EVERY    : $FID_EVERY"
echo "  FID_SAMPLES  : $FID_N_SAMPLES"
echo "  FID_REAL     : $FID_N_REAL"
echo "  FID_DDIM     : $FID_DDIM_STEPS"
echo "  CLASSES      : ${CLASSES:-all}"
echo "  WANDB        : $WANDB"
echo "  RESUME       : ${RESUME:-none}"
echo ""

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

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
    --fid_n_samples "$FID_N_SAMPLES" \
    --fid_n_real    "$FID_N_REAL" \
    --fid_ddim_steps "$FID_DDIM_STEPS" \
    --tag           "$TAG" \
    --use_ema \
    --amp \
    --compile \
    $WANDB_ARG \
    $RESUME_ARG \
    $CLASSES_ARG

echo ""
echo "Job $SLURM_JOB_ID (task ${SLURM_ARRAY_TASK_ID:-N/A}) finished at $(date)."
