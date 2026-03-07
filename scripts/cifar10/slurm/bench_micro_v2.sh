#!/bin/bash
#SBATCH --job-name=bench_micro
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/bench_micro_v2_%j.out
#SBATCH --error=logs/bench_micro_v2_%j.err

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion
mkdir -p logs

$PYTHON -c "
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path('.')))
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting, _generate_direct

device = 'cuda'
K = 500
torch.manual_seed(42)
sigma_x = torch.sigmoid(torch.randn(K, device=device))
sigma_y = torch.sigmoid(torch.randn(K, device=device))
rho = torch.tanh(torch.randn(K, device=device))
coords = torch.tanh(torch.randn(K, 2, device=device))
colours = torch.sigmoid(torch.randn(K, 3, device=device))

# Warmup
for _ in range(20):
    _generate_direct(sigma_x, sigma_y, rho, coords, colours,
                     (32, 32), 3, device, soft_clamp=True)
torch.cuda.synchronize()

N = 1000
t0 = time.time()
for _ in range(N):
    _generate_direct(sigma_x, sigma_y, rho, coords, colours,
                     (32, 32), 3, device, soft_clamp=True)
torch.cuda.synchronize()
t_direct = (time.time() - t0) / N * 1000
print(f'Direct eval (optimized): {t_direct:.3f} ms/call ({N} calls)')

# Old path via ks=31
for _ in range(20):
    generate_2D_gaussian_splatting(31, sigma_x, sigma_y, rho, coords, colours,
                                   (32, 32), 3, device, soft_clamp=True)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N):
    generate_2D_gaussian_splatting(31, sigma_x, sigma_y, rho, coords, colours,
                                   (32, 32), 3, device, soft_clamp=True)
torch.cuda.synchronize()
t_old = (time.time() - t0) / N * 1000
print(f'Old path (ks=31):       {t_old:.3f} ms/call ({N} calls)')
print(f'Speedup: {t_old/t_direct:.2f}x')

# Forward + backward benchmark
sigma_x_r = sigma_x.clone().requires_grad_(True)
sigma_y_r = sigma_y.clone().requires_grad_(True)
rho_r = rho.clone().requires_grad_(True)
coords_r = coords.clone().requires_grad_(True)
colours_r = colours.clone().requires_grad_(True)

# Warmup fwd+bwd
for _ in range(20):
    out = _generate_direct(sigma_x_r, sigma_y_r, rho_r, coords_r, colours_r,
                          (32, 32), 3, device, soft_clamp=True)
    out.sum().backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N):
    out = _generate_direct(sigma_x_r, sigma_y_r, rho_r, coords_r, colours_r,
                          (32, 32), 3, device, soft_clamp=True)
    out.sum().backward()
torch.cuda.synchronize()
t_direct_bwd = (time.time() - t0) / N * 1000

# Old path fwd+bwd
sigma_x_r2 = sigma_x.clone().requires_grad_(True)
sigma_y_r2 = sigma_y.clone().requires_grad_(True)
rho_r2 = rho.clone().requires_grad_(True)
coords_r2 = coords.clone().requires_grad_(True)
colours_gray = colours[:, :1].clone().requires_grad_(True)

for _ in range(20):
    out = generate_2D_gaussian_splatting(31, sigma_x_r2, sigma_y_r2, rho_r2, coords_r2, colours_gray,
                                         (32, 32), 1, device, soft_clamp=True)
    out.sum().backward()
torch.cuda.synchronize()

t0 = time.time()
for _ in range(N):
    out = generate_2D_gaussian_splatting(31, sigma_x_r2, sigma_y_r2, rho_r2, coords_r2, colours_gray,
                                         (32, 32), 1, device, soft_clamp=True)
    out.sum().backward()
torch.cuda.synchronize()
t_old_bwd = (time.time() - t0) / N * 1000

print(f'')
print(f'Forward + Backward:')
print(f'  Direct (ks=32, RGB): {t_direct_bwd:.3f} ms/call')
print(f'  Old (ks=31, gray):   {t_old_bwd:.3f} ms/call')
print(f'  Speedup: {t_old_bwd/t_direct_bwd:.2f}x')
"
