#!/bin/bash
# Sample from the best MNIST model (FID=6.66)
# Usage: bash scripts/mnist/sample_best.sh [N_SAMPLES]

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion

N=${1:-5000}

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
