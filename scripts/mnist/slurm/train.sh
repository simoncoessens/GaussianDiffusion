#!/bin/bash
#SBATCH --job-name=gauss_diff_v2
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.err

# Train diffusion model on v2 Gaussian dataset (coord-inversion fix applied).
# New data: K=70 Gaussians, ~40 dB PSNR (vs old ~26 dB), 100% alive.
# Submit ONLY after full train split encode is complete (60K images).
#
# Check progress: ls data/mnist_gaussian_representations_v2/train/*.pt | wc -l
# Should be >= 54000 (90% of 60K before starting is fine).

echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion

cd "$WORKDIR"
mkdir -p logs checkpoints/mnist samples/mnist

DATA_DIR=${DATA_DIR:-"data/mnist_gaussian_representations_v2/train/"}
EPOCHS=${EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-4}
TIMESTEPS=${TIMESTEPS:-200}
NUM_GAUSSIANS=${NUM_GAUSSIANS:-70}
SAMPLE_EVERY=${SAMPLE_EVERY:-10}

echo ""
echo "Config:"
echo "  DATA_DIR    : $DATA_DIR"
echo "  EPOCHS      : $EPOCHS"
echo "  BATCH_SIZE  : $BATCH_SIZE"
echo "  LR          : $LR"
echo "  TIMESTEPS   : $TIMESTEPS"
echo "  K (Gaussians): $NUM_GAUSSIANS"
echo ""
echo "Checking data availability:"
N_FILES=$(ls "$DATA_DIR"*.pt 2>/dev/null | wc -l)
echo "  $N_FILES .pt files in $DATA_DIR"
if [ "$N_FILES" -lt 1000 ]; then
    echo "ERROR: Too few .pt files ($N_FILES < 1000). Run encoding first."
    exit 1
fi
echo ""

$PYTHON -m src.train \
    --config        mnist \
    --data_dir      "$DATA_DIR" \
    --epochs        "$EPOCHS" \
    --batch_size    "$BATCH_SIZE" \
    --lr            "$LR" \
    --timesteps     "$TIMESTEPS" \
    --num_gaussians "$NUM_GAUSSIANS" \
    --checkpoint_dir checkpoints/mnist/ \
    --sample_dir    samples/mnist/ \
    --sample_every  "$SAMPLE_EVERY"

echo ""
echo "Job $SLURM_JOB_ID finished."
