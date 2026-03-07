#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/metrics_%j.out

module load gcc/11.2.0/gcc-4.8.5 2>/dev/null
module load anaconda3/2022.10/gcc-11.2.0 2>/dev/null
module load cuda/12.2.1/gcc-11.2.0 2>/dev/null

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion

CKPT=${CKPT:-checkpoints/mnist/small_8b4h96d_2000ep_bs256/best.pt}
OUTDIR=${OUTDIR:-samples/mnist/small_8b4h96d_2000ep_bs256}
CFG_S=${CFG_SCALE:-2.0}
DATA=${DATA:-data/mnist/mnist_gaussians_K70.h5}
SAMPLER=${SAMPLER:-ddpm}
DDIM_STEPS=${DDIM_STEPS:-50}
DDIM_ETA=${DDIM_ETA:-0.0}
CLIP_X0=${CLIP_X0:-0}
CLIP_RANGE=${CLIP_RANGE:-5.0}

echo "Metrics-only run: CKPT=$CKPT CFG=$CFG_S DATA=$DATA SAMPLER=$SAMPLER CLIP_X0=$CLIP_X0"

CLIP_ARGS=""
[[ "$CLIP_X0" == "1" ]] && CLIP_ARGS="--clip_x0 --clip_range $CLIP_RANGE"

$PYTHON -m src.sample \
    --config mnist \
    --checkpoint $CKPT \
    --n_samples 5000 \
    --cfg_scale $CFG_S \
    --real_data_h5 $DATA \
    --sampler $SAMPLER \
    --ddim_steps $DDIM_STEPS \
    --ddim_eta $DDIM_ETA \
    $CLIP_ARGS \
    --out_dir $OUTDIR

cat $OUTDIR/metrics.json 2>/dev/null || echo "(not found)"
echo "Done — $(date)"
