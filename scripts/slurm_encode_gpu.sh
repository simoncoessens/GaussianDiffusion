#!/bin/bash
#SBATCH --job-name=encode_mnist_gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9            # 10 GPU workers, each handles ~6K train or ~1K test images
#SBATCH --time=8:00:00
#SBATCH --output=logs/encode_gpu_%A_%a.out
#SBATCH --error=logs/encode_gpu_%A_%a.err

# Re-encode MNIST with the fixed encoder (coordinate inversion bug fixed).
# Uses GPU for fast encoding. Expected: 40+ dB PSNR per image (vs 26-29 dB old).
#
# Coord-inversion fix: _init_gaussians, _dead_mask, _recycle now correctly
# account for the renderer inverting coordinates. Gaussians start at correct
# rendered positions → 100% alive, fast convergence (avg 790 epochs for K=70).
#
# Environment variables:
#   SPLIT       — "train" or "test"  (default: train)
#   NUM_CHUNKS  — number of parallel workers (default: 10)
#   OUT_DIR     — output directory for .pt files
#   K           — number of Gaussians (default: 70)
#   EPOCHS      — max epochs per image (default: 3000, early stop kicks in)
#   EARLY_STOP  — MSE threshold: 0.0001=40dB (default), 0.001=30dB (faster)

SPLIT=${SPLIT:-"train"}
NUM_CHUNKS=${NUM_CHUNKS:-10}
K=${K:-70}
EPOCHS=${EPOCHS:-3000}
EARLY_STOP=${EARLY_STOP:-0.0001}

# Output directory: new location to preserve the old dataset
OUT_DIR=${OUT_DIR:-"data/mnist_gaussian_representations_v2/${SPLIT}/"}

export PYTHONUNBUFFERED=1
PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
WORKDIR=/gpfs/workdir/coessenss/GaussianDiffusion

cd "$WORKDIR"
mkdir -p logs "$OUT_DIR"

echo "Starting job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Config: K=$K, epochs=$EPOCHS, early_stop=$EARLY_STOP, split=$SPLIT"
echo "Output: $OUT_DIR"
echo "Chunk: $SLURM_ARRAY_TASK_ID / $NUM_CHUNKS"
echo ""

$PYTHON scripts/encode_mnist.py \
    --split        "$SPLIT" \
    --out_dir      "$OUT_DIR" \
    --chunk_id     "$SLURM_ARRAY_TASK_ID" \
    --num_chunks   "$NUM_CHUNKS" \
    --K            "$K" \
    --epochs       "$EPOCHS" \
    --lr           0.005 \
    --kernel_size  11 \
    --early_stop   "$EARLY_STOP" \
    --device       cuda \
    --data_root    "$WORKDIR/data"

echo ""
echo "Job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID finished."

# Submission commands:
#
# Train split (60k images, 10 GPU workers, ~6k per worker, ~6h):
#   SPLIT=train sbatch --array=0-9 scripts/slurm_encode_gpu.sh
#
# Test split (10k images, 10 GPU workers, ~1k per worker, ~1h):
#   SPLIT=test sbatch --array=0-9 scripts/slurm_encode_gpu.sh
#
# Fast variant (30 dB, ~1.5h train):
#   SPLIT=train EARLY_STOP=0.001 sbatch --array=0-9 scripts/slurm_encode_gpu.sh
