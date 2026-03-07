#!/bin/bash
#SBATCH --job-name=encode_mnist
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/encode_full_%A_%a.out
#SBATCH --error=logs/encode_full_%A_%a.err

# Encode all 70,000 MNIST images into 4 interleaved shard HDF5 files.
# Each shard runs on one A100 GPU and processes 17,500 images.
#
# Submit (fresh):
#   sbatch --array=0-3 scripts/mnist/slurm/encode.sh
#
# Resume after crash (re-run the failed shards):
#   sbatch --array=2 scripts/mnist/slurm/encode.sh --resume
#   (the --resume flag is forwarded to the Python script via EXTRA_ARGS)
#
# Monitor:
#   tail -f logs/encode_full_<ARRAY_ID>_<TASK_ID>.out
#   tail -f logs/encode_full_mnist_shard<N>.log   (per-shard log file)

echo "========================================================"
echo "  SLURM job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "  Host    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Started : $(date)"
echo "========================================================"

module load gcc/15.1.0/gcc-15.1.0
module load anaconda3/2023.09-0/none-none
module load cuda/12.2.2/none-none

export PYTHONUNBUFFERED=1

# Use the NLS env (has torch 2.6.0+cu124, h5py, torchvision, etc.)
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python

if [ ! -f "$PYTHON" ]; then
    echo "[FATAL] Python not found at: $PYTHON"
    exit 1
fi

echo "  Python  : $PYTHON"
echo "  Version : $($PYTHON --version 2>&1)"
echo ""

WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$WORKDIR"
mkdir -p logs data/mnist/shards

SHARD_ID=${SLURM_ARRAY_TASK_ID}

# Configurable overrides via environment variables
N_SHARDS=${N_SHARDS:-4}
K=${K:-70}
EPOCHS=${EPOCHS:-3000}
LR=${LR:-5e-3}
EARLY_STOP=${EARLY_STOP:-1e-4}
DEVICE=${DEVICE:-cuda}

echo "Config:"
echo "  SHARD_ID    : $SHARD_ID / $N_SHARDS"
echo "  K           : $K"
echo "  EPOCHS      : $EPOCHS"
echo "  LR          : $LR"
echo "  EARLY_STOP  : $EARLY_STOP"
echo "  DEVICE      : $DEVICE"
echo ""

# Pass --resume if this is a restart (set RESUME=1 to enable)
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
    echo "  [RESUME MODE: skipping already-done rows]"
fi

$PYTHON scripts/mnist/encode.py \
    --shard_id     "$SHARD_ID" \
    --n_shards     "$N_SHARDS" \
    --out_dir      data/mnist/shards \
    --device       "$DEVICE" \
    --K            "$K" \
    --epochs       "$EPOCHS" \
    --lr           "$LR" \
    --early_stop   "$EARLY_STOP" \
    --kernel_size  11 \
    --recycle_every 300 \
    --log_every    25 \
    --progress_every 100 \
    --ckpt_log_every 500 \
    $RESUME_FLAG

EXIT_CODE=$?
echo ""
echo "========================================================"
echo "  Shard $SHARD_ID finished with exit code $EXIT_CODE"
echo "  Finished: $(date)"
echo "========================================================"
exit $EXIT_CODE
