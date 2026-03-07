#!/bin/bash
#SBATCH --job-name=bench_old
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_old_%j.out
#SBATCH --error=logs/bench_old_%j.err

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion
mkdir -p logs

echo "=== OLD vs NEW renderer, production config, V100 ==="
$PYTHON -c "
import sys, time, math, pickle
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path('.')))
from src.encode import encode_image
import src.utils.gaussian_to_image as renderer_mod

# Load real CIFAR-10 images
cifar_dir = Path('data/cifar10/raw/cifar-10-batches-py')
with open(cifar_dir / 'data_batch_1', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
images = batch[b'data'].reshape(-1, 3, 32, 32)

# === Test 1: New direct eval path (ks=32 triggers it) ===
print('=== NEW (direct eval, ks=32) ===')
N = 5
times_new = []
for i in range(N):
    img = torch.from_numpy(images[i]).float() / 255.0
    t0 = time.time()
    W, loss = encode_image(img, K=500, epochs=3000, lr=0.04, kernel_size=32,
                           early_stop_threshold=1e-5, device='cuda',
                           soft_clamp=True, use_scheduler=False)
    elapsed = time.time() - t0
    psnr = 10*math.log10(1.0/loss) if loss > 1e-10 else 100.0
    times_new.append(elapsed)
    print(f'  [{i+1}/{N}] {elapsed:.2f}s  PSNR={psnr:.1f}dB')

# === Test 2: Force old path by temporarily removing _generate_direct ===
print()
print('=== OLD (affine_grid+grid_sample, ks=31 to force old path) ===')
times_old = []
for i in range(N):
    img = torch.from_numpy(images[i]).float() / 255.0
    t0 = time.time()
    W, loss = encode_image(img, K=500, epochs=3000, lr=0.04, kernel_size=31,
                           early_stop_threshold=1e-5, device='cuda',
                           soft_clamp=True, use_scheduler=False)
    elapsed = time.time() - t0
    psnr = 10*math.log10(1.0/loss) if loss > 1e-10 else 100.0
    times_old.append(elapsed)
    print(f'  [{i+1}/{N}] {elapsed:.2f}s  PSNR={psnr:.1f}dB')

mean_new = sum(times_new) / len(times_new)
mean_old = sum(times_old) / len(times_old)
print(f'')
print(f'SUMMARY (V100, no compile):')
print(f'  NEW (ks=32, direct): {mean_new:.2f}s/img')
print(f'  OLD (ks=31, affine): {mean_old:.2f}s/img')
print(f'  Speedup: {mean_old/mean_new:.2f}x')
"
