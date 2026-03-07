#!/bin/bash
#SBATCH --job-name=bench_v100
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_renderer_v100_%j.out
#SBATCH --error=logs/bench_renderer_v100_%j.err

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion
mkdir -p logs

echo "=== Renderer microbenchmark: direct eval vs original ==="
$PYTHON -c "
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path('.')))
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting, _generate_direct

device = 'cuda'
K = 500
torch.manual_seed(42)

# Create test inputs
sigma_x = torch.sigmoid(torch.randn(K, device=device))
sigma_y = torch.sigmoid(torch.randn(K, device=device))
rho = torch.tanh(torch.randn(K, device=device))
coords = torch.tanh(torch.randn(K, 2, device=device))
colours = torch.sigmoid(torch.randn(K, 3, device=device))

# Warmup
for _ in range(10):
    _generate_direct(sigma_x, sigma_y, rho, coords, colours,
                     (32, 32), 3, device, soft_clamp=True)
torch.cuda.synchronize()

# Benchmark direct eval
N = 500
t0 = time.time()
for _ in range(N):
    _generate_direct(sigma_x, sigma_y, rho, coords, colours,
                     (32, 32), 3, device, soft_clamp=True)
torch.cuda.synchronize()
t_direct = (time.time() - t0) / N * 1000

print(f'Direct eval: {t_direct:.3f} ms/call ({N} calls)')

# For comparison, test the original path by calling with ks=11 (uses old path)
# Create test with ks < image_size to trigger old path
for _ in range(10):
    generate_2D_gaussian_splatting(11, sigma_x, sigma_y, rho, coords, colours,
                                   (28, 28), 1, device, soft_clamp=True)
torch.cuda.synchronize()

colours_gray = colours[:, :1]
t0 = time.time()
for _ in range(N):
    generate_2D_gaussian_splatting(11, sigma_x, sigma_y, rho, coords, colours_gray,
                                   (28, 28), 1, device, soft_clamp=True)
torch.cuda.synchronize()
t_old = (time.time() - t0) / N * 1000
print(f'Old path (ks=11, 28x28, gray): {t_old:.3f} ms/call ({N} calls)')

# Compare same-size: ks=32 via direct path vs ks=31 via old path
# ks=31 < 32 so it triggers old path
for _ in range(10):
    generate_2D_gaussian_splatting(31, sigma_x, sigma_y, rho, coords, colours,
                                   (32, 32), 3, device, soft_clamp=True)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N):
    generate_2D_gaussian_splatting(31, sigma_x, sigma_y, rho, coords, colours,
                                   (32, 32), 3, device, soft_clamp=True)
torch.cuda.synchronize()
t_old32 = (time.time() - t0) / N * 1000
print(f'Old path (ks=31, 32x32, RGB):  {t_old32:.3f} ms/call ({N} calls)')
print(f'Direct (ks=32, 32x32, RGB):    {t_direct:.3f} ms/call')
print(f'Speedup: {t_old32/t_direct:.2f}x')
"

echo ""
echo "=== Full encoding benchmark ==="
$PYTHON scripts/cifar10/bench_renderer.py cuda
