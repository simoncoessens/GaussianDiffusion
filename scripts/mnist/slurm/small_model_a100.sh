#!/bin/bash
#SBATCH --job-name=small_mdl
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/small_model_%j.out

# ============================================================
# Small model sweep on A100 — optimized for speed
#
# Usage:
#   BLOCKS=4 HEADS=4 DIM=96 EPOCHS=2000 sbatch scripts/mnist/slurm/small_model_a100.sh
#   DATA=data/mnist/mnist_gaussians_partial.h5 sbatch scripts/mnist/slurm/small_model_a100.sh  # partial data
# ============================================================

module load gcc/11.2.0/gcc-4.8.5 2>/dev/null
module load anaconda3/2022.10/gcc-11.2.0 2>/dev/null
module load cuda/12.2.1/gcc-11.2.0 2>/dev/null

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
PROJECT=/gpfs/workdir/coessenss/GaussianDiffusion
cd "$PROJECT"
mkdir -p logs

set -euo pipefail

NB=${BLOCKS:-4}
NH=${HEADS:-4}
HD=${DIM:-96}
EP=${EPOCHS:-2000}
BS=${BS:-256}
CFG_S=${CFG_SCALE:-2.0}
LR=${LR:-3e-4}
WU=${WARMUP:-500}
SCHED_S=${SCHED_S:-0.008}
DATA=${DATA:-data/mnist/mnist_gaussians_K70.h5}
RESUME=${RESUME:-}

# Auto-detect tag prefix from data file
if [[ "$DATA" == *"partial"* ]]; then
  TAG="small_${NB}b${NH}h${HD}d_${EP}ep_bs${BS}_w${CFG_S}"
else
  TAG="full_${NB}b${NH}h${HD}d_${EP}ep_bs${BS}_w${CFG_S}"
fi
[[ "$SCHED_S" != "0.008" ]] && TAG="${TAG}_s${SCHED_S}"
# Allow explicit tag override
TAG=${EXPTAG:-$TAG}

echo "=========================================="
echo "  Small Model Sweep (A100 optimized)"
echo "  ${NB}B/${NH}H/${HD}D, ${EP} epochs, bs=${BS}"
echo "  CFG w=${CFG_S}, lr=${LR}, data=${DATA}"
echo "  $(date)"
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')
echo "  GPU: $GPU_NAME"
echo "=========================================="

RESUME_ARG=""
[[ -n "$RESUME" ]] && RESUME_ARG="--resume $RESUME"

# Train with all A100 optimizations
$PYTHON -m src.train \
    --config mnist \
    --data_h5 $DATA \
    --epochs $EP \
    --batch_size $BS \
    --lr $LR \
    --timesteps 200 \
    --amp \
    --compile \
    --use_ema \
    --ema_decay 0.9999 \
    --warmup_steps $WU \
    --num_workers 8 \
    --num_classes 10 \
    --cfg_dropout 0.1 \
    --cfg_scale $CFG_S \
    --num_blocks $NB \
    --num_heads $NH \
    --hidden_dim $HD \
    --schedule_s $SCHED_S \
    --sample_every 200 \
    --checkpoint_dir checkpoints/mnist/$TAG/ \
    --sample_dir samples/mnist/$TAG/ \
    $RESUME_ARG

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
