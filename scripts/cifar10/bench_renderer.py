"""Benchmark old vs new renderer path for CIFAR-10 encoding speed."""
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encode import encode_image

def bench_encode(n_images=5, device="cuda"):
    """Encode random CIFAR-10-like images and measure time."""
    torch.manual_seed(42)
    times = []
    psnrs = []

    for i in range(n_images):
        img = torch.rand(3, 32, 32)  # Random RGB image
        t0 = time.time()
        W, loss = encode_image(
            img, K=500, epochs=500, lr=0.04,
            kernel_size=32, early_stop_threshold=1e-5,
            device=device, soft_clamp=True, use_scheduler=False,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        import math
        psnr = 10 * math.log10(1.0 / loss) if loss > 1e-10 else 100.0
        psnrs.append(psnr)
        print(f"  [{i+1}/{n_images}] {elapsed:.2f}s  PSNR={psnr:.1f}dB")

    mean_t = sum(times) / len(times)
    mean_p = sum(psnrs) / len(psnrs)
    print(f"\nMean: {mean_t:.2f}s/img  PSNR={mean_p:.1f}dB")
    return mean_t, mean_p


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    print(f"Device: {device}")
    print(f"Benchmarking direct evaluation renderer...")
    bench_encode(n_images=5, device=device)
