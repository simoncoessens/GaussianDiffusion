#!/bin/bash
#SBATCH --job-name=bench_bmax
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_batch_max_%j.out
#SBATCH --error=logs/bench_batch_max_%j.err

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
from src.encode import encode_batch

cifar_dir = Path('data/cifar10/raw/cifar-10-batches-py')
with open(cifar_dir / 'data_batch_1', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
raw_images = batch[b'data'].reshape(-1, 3, 32, 32)

device = 'cuda'
K = 500; ep = 3000; lr = 0.04; es = 1e-5
gpu_name = torch.cuda.get_device_name(0)
gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu_name} ({gpu_mem_total:.1f} GB)')

def psnr(mse):
    return 10*math.log10(1.0/mse) if mse > 1e-10 else 100.0

for BS in [32, 64, 128, 256, 512]:
    print(f'=== Batch size = {BS} ===')
    N = min(BS, len(raw_images))
    all_imgs = torch.from_numpy(raw_images[:N]).float() / 255.0
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        W_batch, losses = encode_batch(all_imgs, K=K, epochs=ep, lr=lr,
                                        kernel_size=32, early_stop_threshold=es,
                                        device=device, soft_clamp=True, use_scheduler=False)
        elapsed = time.time() - t0
        psnrs = [psnr(l) for l in losses.tolist()]
        mem = torch.cuda.max_memory_allocated()/1e9
        print(f'  {elapsed:.1f}s ({elapsed/N:.2f}s/img) PSNR={np.mean(psnrs):.1f}dB min={np.min(psnrs):.1f} mem={mem:.1f}GB ({100*mem/gpu_mem_total:.0f}%)')
    except torch.cuda.OutOfMemoryError:
        print(f'  OOM! Max batch for this GPU is < {BS}')
        break
    except Exception as e:
        print(f'  Error: {e}')
        break
    print()
"
