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
#   FID_EVERY=25  FID_N_SAMPLES=500
#   WANDB=1  (set to 0 to disable)
#   NGPUS=1  (set to 4 for multi-GPU DDP)
#
# Submit:
#   TAG=my_run sbatch scripts/cifar10/slurm/train.sh
#   EPOCHS=1000 TAG=long sbatch scripts/cifar10/slurm/train.sh
#   BATCH_SIZE=128 sbatch -p gpu scripts/cifar10/slurm/train.sh  # V100
#   NGPUS=4 sbatch --gres=gpu:4 --cpus-per-task=32 --mem=200G scripts/cifar10/slurm/train.sh

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
CFG_SCALE=${CFG_SCALE:-2.0}
CFG_DROPOUT=${CFG_DROPOUT:-0.1}
TIMESTEPS=${TIMESTEPS:-200}
SCHEDULE_S=${SCHEDULE_S:-0.008}
TAG=${TAG:-"${NUM_BLOCKS}b${NUM_HEADS}h${HIDDEN_DIM}d"}
RESUME=${RESUME:-""}
FID_EVERY=${FID_EVERY:-25}
FID_N_SAMPLES=${FID_N_SAMPLES:-500}
FID_N_REAL=${FID_N_REAL:-10000}
FID_DDIM_STEPS=${FID_DDIM_STEPS:-50}
WANDB=${WANDB:-1}
SAMPLE_EVERY=${SAMPLE_EVERY:-10}
CLASSES=${CLASSES:-""}
AMP=${AMP:-1}
COMPILE=${COMPILE:-1}
WARMUP_STEPS=${WARMUP_STEPS:-500}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
LOSS_WEIGHTS=${LOSS_WEIGHTS:-""}
PREDICTION_TYPE=${PREDICTION_TYPE:-"epsilon"}
MIN_SNR_GAMMA=${MIN_SNR_GAMMA:-0}
HFLIP=${HFLIP:-0}
BF16=${BF16:-0}
RESET_SCHEDULER=${RESET_SCHEDULER:-0}
NGPUS=${NGPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}

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
echo "  FID_EVERY   : $FID_EVERY"
echo "  FID_DDIM    : $FID_DDIM_STEPS"
echo "  CLASSES     : ${CLASSES:-all}"
echo "  WANDB       : $WANDB"
echo "  NGPUS       : $NGPUS"
echo "  RESUME      : ${RESUME:-none}"
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

AMP_ARG=""
if [ "$AMP" = "1" ]; then
    AMP_ARG="--amp"
fi

COMPILE_ARG=""
if [ "$COMPILE" = "1" ]; then
    COMPILE_ARG="--compile"
fi

LOSS_WEIGHTS_ARG=""
if [ -n "$LOSS_WEIGHTS" ]; then
    LOSS_WEIGHTS_ARG="--loss_weights $LOSS_WEIGHTS"
fi

PRED_TYPE_ARG=""
if [ "$PREDICTION_TYPE" != "epsilon" ]; then
    PRED_TYPE_ARG="--prediction_type $PREDICTION_TYPE"
fi

MIN_SNR_ARG=""
if [ "$MIN_SNR_GAMMA" != "0" ]; then
    MIN_SNR_ARG="--min_snr_gamma $MIN_SNR_GAMMA"
fi

HFLIP_ARG=""
if [ "$HFLIP" = "1" ]; then
    HFLIP_ARG="--hflip"
fi

BF16_ARG=""
if [ "$BF16" = "1" ]; then
    BF16_ARG="--bf16"
fi

RESET_SCHED_ARG=""
if [ "$RESET_SCHEDULER" = "1" ]; then
    RESET_SCHED_ARG="--reset_scheduler"
fi

# Use torchrun for multi-GPU DDP, plain python for single GPU
if [ "$NGPUS" -gt 1 ]; then
    LAUNCHER="$PYTHON -m torch.distributed.run --nproc_per_node=$NGPUS --master_port=$MASTER_PORT"
else
    LAUNCHER="$PYTHON"
fi

$LAUNCHER scripts/cifar10/train.py \
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
    $AMP_ARG \
    $COMPILE_ARG \
    $WANDB_ARG \
    $RESUME_ARG \
    $CLASSES_ARG \
    $LOSS_WEIGHTS_ARG \
    $PRED_TYPE_ARG \
    $MIN_SNR_ARG \
    $HFLIP_ARG \
    $BF16_ARG \
    $RESET_SCHED_ARG

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)."
