#!/bin/bash
#SBATCH --job-name=bench_batch
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_batch_%j.out
#SBATCH --error=logs/bench_batch_%j.err

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

# Load real CIFAR-10 images
cifar_dir = Path('data/cifar10/raw/cifar-10-batches-py')
with open(cifar_dir / 'data_batch_1', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
raw_images = batch[b'data'].reshape(-1, 3, 32, 32)

device = 'cuda'
N_TOTAL = 64  # Total images to encode
epochs = 3000
lr = 0.04
K = 500
es = 1e-5

def psnr(mse):
    return 10*math.log10(1.0/mse) if mse > 1e-10 else 100.0

# === Test 1: Sequential (baseline) ===
print('=== Sequential (1 image at a time) ===')
imgs_seq = [torch.from_numpy(raw_images[i]).float() / 255.0 for i in range(N_TOTAL)]
t0 = time.time()
psnrs_seq = []
for i, img in enumerate(imgs_seq):
    W, loss = encode_image(img, K=K, epochs=epochs, lr=lr, kernel_size=32,
                           early_stop_threshold=es, device=device,
                           soft_clamp=True, use_scheduler=False)
    psnrs_seq.append(psnr(loss))
    if (i+1) % 16 == 0:
        print(f'  [{i+1}/{N_TOTAL}] mean_psnr={np.mean(psnrs_seq):.1f}dB')
t_seq = time.time() - t0
print(f'  Total: {t_seq:.1f}s ({t_seq/N_TOTAL:.2f}s/img)  PSNR={np.mean(psnrs_seq):.1f}dB')
print()

# === Test batch sizes ===
for BS in [4, 8, 16, 32, 64]:
    if BS > N_TOTAL:
        break
    print(f'=== Batch size = {BS} ===')
    all_imgs = torch.from_numpy(raw_images[:N_TOTAL]).float() / 255.0  # [N, 3, 32, 32]
    t0 = time.time()
    all_W = []
    all_losses = []
    for start in range(0, N_TOTAL, BS):
        end = min(start + BS, N_TOTAL)
        batch_imgs = all_imgs[start:end]
        W_batch, losses = encode_batch(batch_imgs, K=K, epochs=epochs, lr=lr,
                                        kernel_size=32, early_stop_threshold=es,
                                        device=device, soft_clamp=True, use_scheduler=False)
        all_W.append(W_batch)
        all_losses.append(losses)
        print(f'  batch [{start}:{end}] psnr={np.mean([psnr(l) for l in losses.tolist()]):.1f}dB')
    t_batch = time.time() - t0
    all_losses_flat = torch.cat(all_losses)
    psnrs_batch = [psnr(l) for l in all_losses_flat.tolist()]
    speedup = t_seq / t_batch
    print(f'  Total: {t_batch:.1f}s ({t_batch/N_TOTAL:.2f}s/img)  PSNR={np.mean(psnrs_batch):.1f}dB  speedup={speedup:.2f}x')
    print(f'  GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
    torch.cuda.reset_peak_memory_stats()
    print()

print('SUMMARY:')
print(f'  Sequential: {t_seq/N_TOTAL:.2f}s/img')
"
