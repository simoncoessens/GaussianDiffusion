"""Benchmark CIFAR-10 training throughput with different batch sizes."""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.transformer_model import GaussianTransformer
from src.ddpm import DDPM


def benchmark(hidden_dim, num_blocks, num_heads, batch_sizes, device="cuda", n_iters=50):
    model = GaussianTransformer(
        input_dim=500, time_emb_dim=hidden_dim, feature_dim=8,
        num_timestamps=200, num_transformer_blocks=num_blocks,
        num_heads=num_heads, num_classes=10, class_dropout_prob=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_blocks}B/{num_heads}H/{hidden_dim}D = {n_params/1e6:.2f}M params")

    ddpm = DDPM(n_T=200, schedule_type="cosine", s=0.008)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    n_train = 54000  # 90% of 60k

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        steps_per_epoch = n_train // bs
        x = torch.randn(bs, 500, 8, device=device)
        t = torch.randint(1, 201, (bs,), device=device)
        y = torch.randint(0, 10, (bs,), device=device)
        noise = torch.randn_like(x)

        # Warmup
        try:
            for _ in range(3):
                with torch.amp.autocast("cuda"):
                    pred = model(x, t.float(), y=y)
                    loss = criterion(pred, noise)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  BS={bs:>5}: OOM during warmup")
                torch.cuda.empty_cache()
                continue
            raise

        # Benchmark
        t0 = time.time()
        for _ in range(n_iters):
            with torch.amp.autocast("cuda"):
                pred = model(x, t.float(), y=y)
                loss = criterion(pred, noise)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        it_per_s = n_iters / elapsed
        img_per_s = it_per_s * bs
        epoch_time = steps_per_epoch / it_per_s
        epochs_24h = 24 * 3600 / epoch_time

        print(f"  BS={bs:>5}: {it_per_s:.1f} it/s, {img_per_s:.0f} img/s, "
              f"{epoch_time:.0f}s/epoch, ~{epochs_24h:.0f} ep/24h, mem={mem_gb:.1f}GB")

        del x, t, y, noise
    del model, optimizer


if __name__ == "__main__":
    device = "cuda"
    gpu_name = torch.cuda.get_device_name()
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({total_mem:.0f} GB)")

    # 7.4M model
    benchmark(256, 6, 16, [64, 128, 256, 512, 768, 1024], device)

    # 4.2M model
    benchmark(192, 6, 8, [128, 256, 512, 768, 1024], device)

    # 2.5M model
    benchmark(128, 8, 8, [128, 256, 512, 1024], device)

    # 1.06M model
    benchmark(96, 6, 4, [256, 512, 1024, 2048], device)
