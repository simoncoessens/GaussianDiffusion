# FID Improvement Report — Gaussian Diffusion on CIFAR-10

**Date:** 2026-03-07
**Setup:** DiT variants, CIFAR-10 (60k images), K=500 Gaussians, 8-dim (alpha dropped), DDPM/DDIM cosine schedule T=200

---

## Summary

**Best FID: TBD** — experiments in progress.

CIFAR-10 Gaussian diffusion training campaign. Building on MNIST findings (best FID=6.66 with 7.4M model).

Key differences from MNIST:
- K=500 tokens (vs 70) — 51× more attention computation per layer
- 8-dim features: sigma_x, sigma_y, rho, r, g, b, x, y (vs 6-dim gray)
- 32×32 RGB images rendered with soft_clamp=True
- Encoder quality: 49.47 dB PSNR (vs 39.4 dB for MNIST)

---

## Encoding Summary

- 60,000 CIFAR-10 images encoded to K=500 Gaussians each
- Mean PSNR: 49.47 dB (std=1.31, min=35.15, max=51.97)
- Only 19 images below 40 dB (0.03%)
- Config: lr=0.04, epochs=3000, ks=32, soft_clamp=True, early_stop=1e-5
- Data: `data/cifar10/cifar10_gaussians_K500.h5` (1.08 GB)

---

## Training Configuration

Base config (from MNIST learnings):
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- LR schedule: 500-step warmup → cosine decay to 1e-6
- EMA: decay=0.9999 with power ramp
- AMP (fp16), torch.compile
- CFG: 10 classes, dropout=0.1, sample w=1.5
- Noise schedule: cosine, T=200, s=0.008

---

## Phase 1: Model Size Sweep (500 epochs)

Initial sweep to find the right model size for K=500 CIFAR-10.

| Job | Config | Params | GPU | BS | Status | FID | IS | Notes |
|-----|--------|--------|-----|----|----|-----|-----|-------|
| TBD | 6B/16H/256 | 7.4M | A100 | 64 | pending | — | — | MNIST best config |
| TBD | 12B/16H/256 | 14.5M | A100 | 32 | pending | — | — | Larger model |
| TBD | 6B/8H/192 | 4.2M | A100 | 128 | pending | — | — | |
| TBD | 8B/8H/128 | 2.5M | A100 | 128 | pending | — | — | |
| TBD | 6B/8H/128 | 1.9M | V100 | 128 | pending | — | — | |
| TBD | 6B/4H/96 | 1.06M | V100 | 256 | pending | — | — | MNIST smallest good |
| TBD | 4B/4H/96 | 0.72M | V100 | 256 | pending | — | — | |

---

## Experiment Log

### Phase 1 submissions

*Experiments will be logged here as they complete.*
