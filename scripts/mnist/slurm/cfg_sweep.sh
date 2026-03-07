#!/bin/bash
#SBATCH --job-name=cfg_sweep
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/cfg_sweep_%j.out

# ============================================================
# CFG (Classifier-Free Guidance) training sweep
# 14M model (12B/16H/256), AMP, compile, EMA, on A100
#
# Usage:
#   EPOCHS=500 CFG_SCALE=3.0 sbatch scripts/mnist/slurm/cfg_sweep.sh
# ============================================================

module load gcc/11.2.0/gcc-4.8.5 2>/dev/null
module load anaconda3/2022.10/gcc-11.2.0 2>/dev/null
module load cuda/12.2.1/gcc-11.2.0 2>/dev/null

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
PROJECT=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$PROJECT"
mkdir -p logs

set -euo pipefail

EP=${EPOCHS:-500}
BS=${BS:-1024}
CFG_S=${CFG_SCALE:-3.0}
CFG_D=${CFG_DROPOUT:-0.1}

TAG="cfg_${EP}ep_w${CFG_S}"

echo "=========================================="
echo "  CFG Sweep: ${EP} epochs, bs=${BS}"
echo "  CFG scale=${CFG_S}, dropout=${CFG_D}"
echo "  $(date)"
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')
echo "  GPU: $GPU_NAME"
echo "=========================================="

MODEL_ARGS="--num_blocks 12 --num_heads 16 --hidden_dim 256"

# Train with EMA + warmup + CFG
$PYTHON -m src.train \
    --config mnist \
    --data_h5 data/mnist/mnist_gaussians_partial.h5 \
    --epochs $EP \
    --batch_size $BS \
    --lr 3e-4 \
    --timesteps 200 \
    --amp \
    --compile \
    --use_ema \
    --ema_decay 0.9999 \
    --warmup_steps 500 \
    --num_workers 8 \
    --num_classes 10 \
    --cfg_dropout $CFG_D \
    --cfg_scale $CFG_S \
    $MODEL_ARGS \
    --sample_every 50 \
    --checkpoint_dir checkpoints/mnist/$TAG/ \
    --sample_dir samples/mnist/$TAG/

# Sample + metrics
echo ""
echo "--- Sampling + metrics for ${TAG} ---"
$PYTHON -m src.sample \
    --config mnist \
    --checkpoint checkpoints/mnist/$TAG/best.pt \
    --n_samples 5000 \
    --cfg_scale $CFG_S \
    --real_data_h5 data/mnist/mnist_gaussians_partial.h5 \
    --out_dir samples/mnist/$TAG/

echo ""
echo "=========================================="
echo "  Results: ${TAG}"
echo "=========================================="
cat samples/mnist/$TAG/metrics.json 2>/dev/null || echo "(not found)"
echo ""
echo "Done — $(date)"
