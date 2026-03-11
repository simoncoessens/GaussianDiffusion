#!/bin/bash
#SBATCH --job-name=encode_celeba64
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/encode_celeba64_%A_%a.out
#SBATCH --error=logs/encode_celeba64_%A_%a.err

# Encode all 202,599 CelebA images into 20 interleaved shard HDF5 files.
# Each shard processes ~10k images on one GPU using batched encoding.
# Default batch_size=256 (V100). Use BATCH_SIZE=512 for A100.
#
# Submit (A100, 10 shards, BS=512):
#   BATCH_SIZE=512 sbatch --array=0-9 -p gpua100 scripts/celeba64/slurm/encode.sh
#
# Submit (V100, 10 shards, BS=256):
#   sbatch --array=10-19 -p gpu scripts/celeba64/slurm/encode.sh
#
# Resume after crash:
#   RESUME=1 sbatch --array=2 -p gpua100 scripts/celeba64/slurm/encode.sh

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
mkdir -p logs data/celeba64/shards

SHARD_ID=${SLURM_ARRAY_TASK_ID}

# Encoding hyperparameters (override via env)
N_SHARDS=${N_SHARDS:-20}
K=${K:-1000}
EPOCHS=${EPOCHS:-3000}
LR=${LR:-4e-2}
EARLY_STOP=${EARLY_STOP:-1e-5}
KS=${KS:-32}
DEVICE=${DEVICE:-cuda}
SOFT_CLAMP=${SOFT_CLAMP:-1}
NO_SCHED=${NO_SCHED:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
CROP_SIZE=${CROP_SIZE:-140}

echo "Config:"
echo "  SHARD_ID    : $SHARD_ID / $N_SHARDS"
echo "  K           : $K"
echo "  EPOCHS      : $EPOCHS"
echo "  LR          : $LR"
echo "  EARLY_STOP  : $EARLY_STOP"
echo "  KS          : $KS"
echo "  SOFT_CLAMP  : $SOFT_CLAMP"
echo "  NO_SCHED    : $NO_SCHED"
echo "  BATCH_SIZE  : $BATCH_SIZE"
echo "  CROP_SIZE   : $CROP_SIZE"
echo "  DEVICE      : $DEVICE"
echo ""

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
    echo "  [RESUME MODE]"
fi

EXTRA_ARGS=""
if [ "$SOFT_CLAMP" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --soft_clamp"
fi
if [ "$NO_SCHED" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_scheduler"
fi

$PYTHON scripts/celeba64/encode.py \
    --shard_id     "$SHARD_ID" \
    --n_shards     "$N_SHARDS" \
    --data_dir     data/celeba64/raw \
    --out_dir      data/celeba64/shards \
    --device       "$DEVICE" \
    --K            "$K" \
    --epochs       "$EPOCHS" \
    --lr           "$LR" \
    --early_stop   "$EARLY_STOP" \
    --kernel_size  "$KS" \
    --batch_size   "$BATCH_SIZE" \
    --crop_size    "$CROP_SIZE" \
    $EXTRA_ARGS \
    $RESUME_FLAG

EXIT_CODE=$?
echo ""
echo "========================================================"
echo "  Shard $SHARD_ID finished with exit code $EXIT_CODE"
echo "  Finished: $(date)"
echo "========================================================"
exit $EXIT_CODE
