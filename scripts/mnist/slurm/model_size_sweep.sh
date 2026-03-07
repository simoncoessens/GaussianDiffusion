#!/bin/bash
#SBATCH --job-name=model_size
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/model_size_%j.out

# ============================================================
# Model size sweep with CFG on regular GPUs (V100)
#
# Usage:
#   BLOCKS=6 HEADS=8 DIM=128 sbatch scripts/mnist/slurm/model_size_sweep.sh
#   DATA=data/mnist/mnist_gaussians_partial.h5 sbatch scripts/mnist/slurm/model_size_sweep.sh  # partial data
# ============================================================

module load gcc/11.2.0/gcc-4.8.5 2>/dev/null
module load anaconda3/2022.10/gcc-11.2.0 2>/dev/null
module load cuda/12.2.1/gcc-11.2.0 2>/dev/null

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
PROJECT=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$PROJECT"
mkdir -p logs

set -euo pipefail

NB=${BLOCKS:-6}
NH=${HEADS:-8}
HD=${DIM:-128}
EP=${EPOCHS:-500}
BS=${BS:-512}
CFG_S=${CFG_SCALE:-2.0}
SCHED_S=${SCHED_S:-0.008}
DATA=${DATA:-data/mnist/mnist_gaussians_K70.h5}

# Auto-detect tag prefix from data file
if [[ "$DATA" == *"partial"* ]]; then
  TAG="size_${NB}b${NH}h${HD}d_${EP}ep_w${CFG_S}"
else
  TAG="full_size_${NB}b${NH}h${HD}d_${EP}ep_w${CFG_S}"
fi
[[ "$SCHED_S" != "0.008" ]] && TAG="${TAG}_s${SCHED_S}"
TAG=${EXPTAG:-$TAG}

echo "=========================================="
echo "  Model Size Sweep"
echo "  ${NB} blocks, ${NH} heads, dim=${HD}"
echo "  ${EP} epochs, bs=${BS}, CFG w=${CFG_S}, data=${DATA}"
echo "  $(date)"
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')
echo "  GPU: $GPU_NAME"
echo "=========================================="

# Train with EMA + warmup + CFG (no compile on V100 to avoid OOM)
$PYTHON -m src.train \
    --config mnist \
    --data_h5 $DATA \
    --epochs $EP \
    --batch_size $BS \
    --lr 3e-4 \
    --timesteps 200 \
    --amp \
    --use_ema \
    --ema_decay 0.9999 \
    --warmup_steps 500 \
    --num_workers 4 \
    --num_classes 10 \
    --cfg_dropout 0.1 \
    --cfg_scale $CFG_S \
    --num_blocks $NB \
    --num_heads $NH \
    --hidden_dim $HD \
    --schedule_s $SCHED_S \
    --sample_every 100 \
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
    --real_data_h5 $DATA \
    --out_dir samples/mnist/$TAG/

echo ""
echo "=========================================="
echo "  Results: ${TAG}"
echo "=========================================="
cat samples/mnist/$TAG/metrics.json 2>/dev/null || echo "(not found)"
echo ""
echo "Done — $(date)"
