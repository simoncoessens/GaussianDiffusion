#!/bin/bash
#SBATCH --job-name=sample_best
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/sample_best_%j.out

# Sample from the best MNIST model (FID=6.66)
# Usage: sbatch scripts/mnist/slurm/sample_best.sh

module load gcc/11.2.0/gcc-4.8.5 2>/dev/null
module load anaconda3/2022.10/gcc-11.2.0 2>/dev/null
module load cuda/12.2.1/gcc-11.2.0 2>/dev/null

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion
mkdir -p logs

N=${N_SAMPLES:-5000}

echo "Sampling $N images from best MNIST model …"
echo "$(date)"

$PYTHON -m src.sample \
    --config mnist \
    --checkpoint checkpoints/mnist/full_size_6b16h256d_1500ep_w1.5/best.pt \
    --n_samples $N \
    --cfg_scale 1.5 \
    --real_data_h5 data/mnist/mnist_gaussians_K70.h5 \
    --sampler ddim \
    --ddim_steps 200 \
    --ddim_eta 0.4 \
    --out_dir samples/mnist/best

cat samples/mnist/best/metrics.json 2>/dev/null || echo "(not found)"
echo "Done — $(date)"
