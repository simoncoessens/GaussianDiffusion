#!/bin/bash
#SBATCH --job-name=bench_vmx
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_batch_v100_max_%j.out
#SBATCH --error=logs/bench_batch_v100_max_%j.err

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
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')

def psnr(mse):
    return 10*math.log10(1.0/mse) if mse > 1e-10 else 100.0

for BS in [768, 1024, 1280, 1536]:
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
        psnrs_b = [psnr(l) for l in losses.tolist()]
        mem = torch.cuda.max_memory_allocated()/1e9
        print(f'  {elapsed:.1f}s ({elapsed/N:.3f}s/img) PSNR={np.mean(psnrs_b):.1f}dB min={np.min(psnrs_b):.1f} mem={mem:.1f}GB ({100*mem/gpu_mem:.0f}%)')
    except torch.cuda.OutOfMemoryError:
        print(f'  OOM! Max batch < {BS}')
        torch.cuda.empty_cache()
        break
    print()

# Production estimate
print('Production estimate:')
print(f'  5000 imgs/shard at 0.62s/img = {5000*0.62/3600:.1f}h/shard')
print(f'  60k imgs on 12 GPUs = {60000*0.62/3600/12:.1f}h total')
"
