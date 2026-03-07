#!/bin/bash
#SBATCH --job-name=bench_b3
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_batch3_%j.out
#SBATCH --error=logs/bench_batch3_%j.err

module load gcc/11.2.0/gcc-4.8.5
module load anaconda3/2022.10/gcc-11.2.0
module --ignore_cache load cuda/12.2.1/gcc-11.2.0

PYTHON=/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python
cd /gpfs/workdir/coessenss/GaussianDiffusion
mkdir -p logs

$PYTHON -c "
import sys, time, math, pickle
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path('.')))
from src.encode import encode_image, encode_batch

cifar_dir = Path('data/cifar10/raw/cifar-10-batches-py')
with open(cifar_dir / 'data_batch_1', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
raw_images = batch[b'data'].reshape(-1, 3, 32, 32)

device = 'cuda'
N = 32
K = 500; ep = 3000; lr = 0.04; es = 1e-5

def psnr(mse):
    return 10*math.log10(1.0/mse) if mse > 1e-10 else 100.0

# Sequential baseline
print('=== Sequential (baseline) ===')
t0 = time.time()
psnrs_seq = []
for i in range(N):
    img = torch.from_numpy(raw_images[i]).float() / 255.0
    W, loss = encode_image(img, K=K, epochs=ep, lr=lr, kernel_size=32,
                           early_stop_threshold=es, device=device,
                           soft_clamp=True, use_scheduler=False)
    psnrs_seq.append(psnr(loss))
t_seq = time.time() - t0
print(f'  {t_seq:.1f}s ({t_seq/N:.2f}s/img) PSNR={np.mean(psnrs_seq):.1f}dB min={np.min(psnrs_seq):.1f}')
print()

# Batch with best-param snapshot
for BS in [1, 4, 8, 16, 32]:
    print(f'=== Batch size = {BS} (best-param snapshot) ===')
    all_imgs = torch.from_numpy(raw_images[:N]).float() / 255.0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    all_losses = []
    for start in range(0, N, BS):
        end = min(start + BS, N)
        W_batch, losses = encode_batch(all_imgs[start:end], K=K, epochs=ep, lr=lr,
                                        kernel_size=32, early_stop_threshold=es,
                                        device=device, soft_clamp=True, use_scheduler=False)
        all_losses.append(losses)
    t_b = time.time() - t0
    psnrs = [psnr(l) for l in torch.cat(all_losses).tolist()]
    mem = torch.cuda.max_memory_allocated()/1e9
    print(f'  {t_b:.1f}s ({t_b/N:.2f}s/img) PSNR={np.mean(psnrs):.1f}dB min={np.min(psnrs):.1f} speedup={t_seq/t_b:.2f}x mem={mem:.1f}GB')
    print()
"
