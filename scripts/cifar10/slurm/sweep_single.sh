#!/bin/bash
#SBATCH --job-name=cifar_sweep
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/cifar10_sweep_%j.out
#SBATCH --error=logs/cifar10_sweep_%j.err

# Generic single-experiment sweep script.
# Submit with environment variables:
#   K=500 EPOCHS=3000 LR=5e-3 KS=11 N_IMAGES=20 sbatch -p gpua100 scripts/cifar10/slurm/sweep_single.sh

echo "========================================================"
echo "  SLURM job ${SLURM_JOB_ID}"
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
mkdir -p logs reports/cifar10

# Defaults (override via env)
K=${K:-200}
EPOCHS=${EPOCHS:-3000}
LR=${LR:-5e-3}
KS=${KS:-11}
N_IMAGES=${N_IMAGES:-20}
RECYCLE=${RECYCLE:-300}
EARLY_STOP=${EARLY_STOP:-1e-4}
SIGMA_ACT=${SIGMA_ACT:-sigmoid}
INIT_MODE=${INIT_MODE:-brightness}
SOFT_CLAMP=${SOFT_CLAMP:-0}
NO_SCHED=${NO_SCHED:-0}

echo "Config: K=$K  EPOCHS=$EPOCHS  LR=$LR  KS=$KS  N=$N_IMAGES  RECYCLE=$RECYCLE"
echo "  sigma_act=$SIGMA_ACT  init=$INIT_MODE  soft_clamp=$SOFT_CLAMP  no_sched=$NO_SCHED"
echo ""

EXTRA_ARGS=""
if [ "$SOFT_CLAMP" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --soft_clamp"
fi
if [ "$NO_SCHED" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no_scheduler"
fi

$PYTHON scripts/cifar10/hparam_sweep.py \
    --K "$K" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --kernel_size "$KS" \
    --n_images "$N_IMAGES" \
    --recycle_every "$RECYCLE" \
    --early_stop "$EARLY_STOP" \
    --sigma_activation "$SIGMA_ACT" \
    --init_mode "$INIT_MODE" \
    $EXTRA_ARGS \
    --device cuda

echo "Finished at $(date)"
