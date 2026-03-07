#!/bin/bash
#SBATCH --job-name=bench_render
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_renderer_%j.out
#SBATCH --error=logs/bench_renderer_%j.err

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/12.2.1/gcc-11.2.0

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion
mkdir -p logs

echo "=== Direct eval benchmark (new code) ==="
$PYTHON scripts/cifar10/bench_renderer.py cuda

echo ""
echo "=== torch.compile + direct eval ==="
$PYTHON -c "
import sys, time, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path('.')))
from src.encode import encode_image
import src.encode as _enc_mod
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

# Compile the renderer
_compiled_fn = torch.compile(generate_2D_gaussian_splatting, mode='reduce-overhead')
_original_render = _enc_mod._render

def _render_compiled(p, kernel_size, image_size, device, channels=1, soft_clamp=False):
    coords = torch.stack([p['x'], p['y']], dim=1)
    return _compiled_fn(
        kernel_size=kernel_size, sigma_x=p['sigma_x'], sigma_y=p['sigma_y'],
        rho=p['rho'], coords=coords, colours=p['colours'],
        image_size=image_size, channels=channels, device=device,
        soft_clamp=soft_clamp)

_enc_mod._render = _render_compiled

torch.manual_seed(42)
times = []
for i in range(7):
    img = torch.rand(3, 32, 32)
    t0 = time.time()
    W, loss = encode_image(img, K=500, epochs=500, lr=0.04, kernel_size=32,
                           early_stop_threshold=1e-5, device='cuda',
                           soft_clamp=True, use_scheduler=False)
    elapsed = time.time() - t0
    psnr = 10*math.log10(1.0/loss) if loss > 1e-10 else 100.0
    print(f'  [{i+1}/7] {elapsed:.2f}s  PSNR={psnr:.1f}dB')
    if i >= 2:  # skip warmup
        times.append(elapsed)
print(f'Mean (post-warmup): {sum(times)/len(times):.2f}s/img')
"
