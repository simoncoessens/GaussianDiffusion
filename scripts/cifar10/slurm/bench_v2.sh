#!/bin/bash
#SBATCH --job-name=bench_v2
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_v2_%j.out
#SBATCH --error=logs/bench_v2_%j.err

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion
mkdir -p logs

$PYTHON -c "
import sys, time, math, pickle
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path('.')))
from src.encode import encode_image
import src.encode as _enc_mod
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

# Load real CIFAR-10 images
cifar_dir = Path('data/cifar10/raw/cifar-10-batches-py')
with open(cifar_dir / 'data_batch_1', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
images = batch[b'data'].reshape(-1, 3, 32, 32)

# Compile
_compiled_fn = torch.compile(generate_2D_gaussian_splatting, mode='reduce-overhead')
_original_render = _enc_mod._render
def _render_compiled(p, kernel_size, image_size, device, channels=1, soft_clamp=False):
    coords = torch.stack([p['x'], p['y']], dim=1)
    return _compiled_fn(kernel_size=kernel_size, sigma_x=p['sigma_x'], sigma_y=p['sigma_y'],
        rho=p['rho'], coords=coords, colours=p['colours'],
        image_size=image_size, channels=channels, device=device, soft_clamp=soft_clamp)
_enc_mod._render = _render_compiled

# Warmup
print('Warmup...')
img_w = torch.from_numpy(images[0]).float() / 255.0
encode_image(img_w, K=500, epochs=100, lr=0.04, kernel_size=32,
             early_stop_threshold=1e-5, device='cuda', soft_clamp=True, use_scheduler=False)
print('Warmup done.\n')

N = 10

# Test 1: No scheduler (current production)
print('=== No scheduler (lr=0.04) ===')
times1 = []
psnrs1 = []
for i in range(N):
    img = torch.from_numpy(images[i]).float() / 255.0
    t0 = time.time()
    W, loss = encode_image(img, K=500, epochs=3000, lr=0.04, kernel_size=32,
                           early_stop_threshold=1e-5, device='cuda',
                           soft_clamp=True, use_scheduler=False)
    elapsed = time.time() - t0
    psnr = 10*math.log10(1.0/loss) if loss > 1e-10 else 100.0
    times1.append(elapsed); psnrs1.append(psnr)
    print(f'  [{i+1}/{N}] {elapsed:.2f}s  PSNR={psnr:.1f}dB')
print(f'  Mean: {np.mean(times1):.2f}s  PSNR={np.mean(psnrs1):.1f}dB\n')

# Test 2: With scheduler 
print('=== With scheduler (lr=0.04) ===')
times2 = []
psnrs2 = []
for i in range(N):
    img = torch.from_numpy(images[i]).float() / 255.0
    t0 = time.time()
    W, loss = encode_image(img, K=500, epochs=3000, lr=0.04, kernel_size=32,
                           early_stop_threshold=1e-5, device='cuda',
                           soft_clamp=True, use_scheduler=True)
    elapsed = time.time() - t0
    psnr = 10*math.log10(1.0/loss) if loss > 1e-10 else 100.0
    times2.append(elapsed); psnrs2.append(psnr)
    print(f'  [{i+1}/{N}] {elapsed:.2f}s  PSNR={psnr:.1f}dB')
print(f'  Mean: {np.mean(times2):.2f}s  PSNR={np.mean(psnrs2):.1f}dB\n')

# Test 3: Higher lr for faster convergence
print('=== No scheduler (lr=0.06) ===')
times3 = []
psnrs3 = []
for i in range(N):
    img = torch.from_numpy(images[i]).float() / 255.0
    t0 = time.time()
    W, loss = encode_image(img, K=500, epochs=3000, lr=0.06, kernel_size=32,
                           early_stop_threshold=1e-5, device='cuda',
                           soft_clamp=True, use_scheduler=False)
    elapsed = time.time() - t0
    psnr = 10*math.log10(1.0/loss) if loss > 1e-10 else 100.0
    times3.append(elapsed); psnrs3.append(psnr)
    print(f'  [{i+1}/{N}] {elapsed:.2f}s  PSNR={psnr:.1f}dB')
print(f'  Mean: {np.mean(times3):.2f}s  PSNR={np.mean(psnrs3):.1f}dB\n')

# Test 4: Higher lr + scheduler
print('=== With scheduler (lr=0.06) ===')
times4 = []
psnrs4 = []
for i in range(N):
    img = torch.from_numpy(images[i]).float() / 255.0
    t0 = time.time()
    W, loss = encode_image(img, K=500, epochs=3000, lr=0.06, kernel_size=32,
                           early_stop_threshold=1e-5, device='cuda',
                           soft_clamp=True, use_scheduler=True)
    elapsed = time.time() - t0
    psnr = 10*math.log10(1.0/loss) if loss > 1e-10 else 100.0
    times4.append(elapsed); psnrs4.append(psnr)
    print(f'  [{i+1}/{N}] {elapsed:.2f}s  PSNR={psnr:.1f}dB')
print(f'  Mean: {np.mean(times4):.2f}s  PSNR={np.mean(psnrs4):.1f}dB\n')

print('SUMMARY:')
print(f'  No sched lr=0.04:   {np.mean(times1):.2f}s  {np.mean(psnrs1):.1f}dB  est60k={np.mean(times1)*5000/3600:.1f}h/shard')
print(f'  Sched    lr=0.04:   {np.mean(times2):.2f}s  {np.mean(psnrs2):.1f}dB  est60k={np.mean(times2)*5000/3600:.1f}h/shard')
print(f'  No sched lr=0.06:   {np.mean(times3):.2f}s  {np.mean(psnrs3):.1f}dB  est60k={np.mean(times3)*5000/3600:.1f}h/shard')
print(f'  Sched    lr=0.06:   {np.mean(times4):.2f}s  {np.mean(psnrs4):.1f}dB  est60k={np.mean(times4)*5000/3600:.1f}h/shard')
"
