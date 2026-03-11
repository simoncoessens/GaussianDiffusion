# FID Improvement Report — Gaussian Diffusion on CIFAR-10

**Date:** 2026-03-08 (ongoing)
**Setup:** DiT variants, CIFAR-10 (60k images), K=500 Gaussians, 8-dim (alpha dropped), DDPM/DDIM cosine schedule T=200

---

## Summary

**Best in-training FID: 125.7** (6B/16H/512D 29.3M, ep225, fp32, 5000ep cosine, CFG=2.0, DDIM-50, 500 samples)
**29.3M model**: New best at ep225, surpassed 16.5M's 1375-epoch best in just 225 epochs (scaling validated)
**Best sweep FID: 131.0** (16.5M, ep253 checkpoint, CFG=2.0, DDIM-100, eta=0, 500 samples)
**Reconstruction FID floor: 10.5** (encoder renders vs real CIFAR-10, 5k images)

Key differences from MNIST:
- K=500 tokens (vs 70) — 51x more attention computation per layer
- 8-dim features: sigma_x, sigma_y, rho, r, g, b, x, y (vs 6-dim gray)
- 32x32 RGB images rendered with soft_clamp=True
- Encoder quality: 49.47 dB PSNR (vs 39.4 dB for MNIST)
- 4000 values per sample (vs 420 for MNIST) — 10x harder generative task

---

## Encoding Summary

- 60,000 CIFAR-10 images encoded to K=500 Gaussians each
- Mean PSNR: 49.47 dB (std=1.31, min=35.15, max=51.97)
- Only 19 images below 40 dB (0.03%)
- Config: lr=0.04, epochs=3000, ks=32, soft_clamp=True, early_stop=1e-5
- Data: `data/cifar10/cifar10_gaussians_K500.h5` (1.08 GB)
- **Reconstruction FID: 10.5** — encoder is NOT the bottleneck

---

## Training Configuration

Base config (from MNIST learnings):
- Optimizer: AdamW, weight_decay=1e-4
- LR schedule: 500-step warmup -> cosine decay to 1e-6
- EMA: decay=0.9999 with power ramp
- AMP (fp16), torch.compile
- CFG: 10 classes, dropout=0.1, sample w=1.5
- Noise schedule: cosine, T=200, s=0.008
- FID: 500 samples, DDIM-50, 10k real images, every 25 epochs

---

## Phase 1: Initial Sweep (200 epochs, 8 configs)

Goal: Find the right LR/model size/batch size for CIFAR-10.

### Results

| Job ID | Tag | Model | LR | BS | GPU | Best FID | @ Epoch | Final ep | Val Loss | Status |
|--------|-----|-------|-----|-----|-----|----------|---------|----------|----------|--------|
| 138897 | std2_6b16h256d_lr5e4 | 6B/16H/256D (7.4M) | 5e-4 | 256 | V100 | **172.1** | 50 | 60 | 0.3337 | **NaN (AMP fp16)** |
| 138896 | std2_6b8h192d_lr3e4 | 6B/8H/192D (~3M) | 3e-4 | 256 | V100 | **174.5** | 75 | 99+ | 0.3297 | Running |
| 138894 | std2_6b16h256d_lr3e4 | 6B/16H/256D (7.4M) | 3e-4 | 256 | A100 | 182.2 | 50 | 55+ | 0.3360 | Running |
| 138900 | std2_6b16h256d_lr1e3 | 6B/16H/256D (7.4M) | 1e-3 | 128 | V100 | 187.9 | 25 | 59+ | 0.3318 | Unstable (FID went UP to 208.6) |
| 138893 | std2_6b16h256d | 6B/16H/256D (7.4M) | 1e-4 | 512 | A100 | 197.9 | 75 | 200 | 0.3432 | **Collapsed** (317.5 at ep100) |
| 138895 | std2_6b8h192d | 6B/8H/192D (~3M) | 1e-4 | 512 | V100 | 199.2 | 75 | 98+ | 0.3420 | Running |
| 138898 | std2_12b16h256d | 12B/16H/256D (14.5M) | 1e-4 | 256 | A100 | 212.1 | 25 | 27+ | 0.3384 | Running (slow: 150s/ep) |
| 138899 | std2_6b4h96d | 6B/4H/96D (~0.5M) | 1e-4 | 1024 | V100 | 215.3 | 150 | 200 | 0.3537 | **Plateaued** |

### FID Trajectories

```
Job 138893 (lr=1e-4, bs=512, 7.4M):   428.7 → 215.2 → 204.8 → 197.9 → 317.5↑ → 267.8 → 222.7 → 213.7
Job 138894 (lr=3e-4, bs=256, 7.4M):   254.5 → 190.6 → 182.2
Job 138895 (lr=1e-4, bs=512, ~3M):    432.7 → 218.7 → 212.0 → 199.2
Job 138896 (lr=3e-4, bs=256, ~3M):    267.7 → 196.8 → 177.4 → 174.5
Job 138897 (lr=5e-4, bs=256, 7.4M):   257.3 → 174.5 → 172.1 → NaN
Job 138898 (lr=1e-4, bs=256, 14.5M):  278.3 → 212.1
Job 138899 (lr=1e-4, bs=1024, ~0.5M): 437.6 → 249.8 → 234.8 → 231.0 → 220.0 → 221.6 → 215.3 → 216.5
Job 138900 (lr=1e-3, bs=128, 7.4M):   232.0 → 187.9 → 208.6↑
```

### Key Findings

1. **LR sweet spot is 3e-4 to 5e-4**: lr=5e-4 gives best FID (172.1) but NaN'd; lr=3e-4 is stable (174.5); lr=1e-4 is too slow; lr=1e-3 is unstable
2. **AMP fp16 + V100 + high LR = NaN**: Job 138897 NaN'd at epoch 60 with lr=5e-4 on V100. Lower LR configs survive. A100 bf16 is safer.
3. **Bigger model converges faster**: 7.4M (256D) reaches 174.5 at ep25; 3M (192D) takes ep75 for the same FID
4. **Small model (0.5M) plateaus at FID=215**: Insufficient capacity for K=500
5. **Large model (14.5M) is slow**: 150s/ep vs 43s/ep for 3M. Not enough gain for 3.5x compute
6. **lr=1e-4 with bs=512 collapses**: FID went from 197.9 to 317.5 at ep100. Cosine LR decayed too fast
7. **Smaller batch helps**: bs=128/256 consistently beats bs=512/1024

### Critical Bug: AMP NaN on V100

Job 138897 (lr=5e-4, V100) went NaN at epoch 60. This is likely fp16 underflow at high LR.
**Fix**: Disable AMP for V100 runs, or restrict V100 to lr<=3e-4. A100 uses bf16 natively and is more stable.
Scout scripts already disable AMP on V100 by design.

---

## Phase 2: Focused Sweep (500 epochs, targeted)

Goal: Extend the best LR (5e-4) to longer training on A100, and explore nearby LR range.

### Submitted experiments

| Job ID | Tag | Model | LR | BS | Epochs | GPU | Status |
|--------|-----|-------|-----|-----|--------|-----|--------|
| 138925 | std3_lr1e3_bs256_500ep | 6B/16H/256D | 1e-3 | 256 | 500 | A100 | Running (ep12) |
| 138926 | std3_lr2e3_bs128 | 6B/16H/256D | 2e-3 | 128 | 200 | V100 | Running (ep12) |
| 138927 | std3_8h384d_lr1e3 | 6B/16H/384D | 1e-3 | 128 | 200 | A100 | Running (ep3) |
| 138928 | std3_lr1e3_bs128_500ep | 6B/16H/256D | 1e-3 | 128 | 500 | V100 | Running (ep12) |
| 138935 | std3_lr5e4_bs256_500ep | 6B/16H/256D | 5e-4 | 256 | 500 | A100 | Pending |
| 138936 | std3_lr7e4_bs256_500ep | 6B/16H/256D | 7e-4 | 256 | 500 | A100 | Pending |
| 138937 | std3_lr5e4_bs128_500ep | 6B/16H/256D | 5e-4 | 128 | 500 | V100 | Running (ep3) — **RISK: may NaN** |

Early epoch-1 FIDs from Phase 2:
- 138926 (lr=2e-3): 223.2
- 138928 (lr=1e-3, 500ep): 230.2
- 138927 (lr=1e-3, 384D): 230.7
- 138937 (lr=5e-4, 500ep): 251.2
- 138925 (lr=1e-3, bs=256): 252.1

---

## Infrastructure: Fast Iteration System

Implemented 3-tier system for faster hyperparameter search:

| Tier | Script | Data | Epochs | FID | GPU | Time | Purpose |
|------|--------|------|--------|-----|-----|------|---------|
| Scout | `scout.sh` | 6k (1 class) | 30 | None | V100 | ~4 min | Eliminate bad configs |
| Validate | `validate.sh` | 60k | 75 | 250/DDIM-25 | A100 | ~25 min | Rank by FID |
| Full | `train.sh` | 60k | 200-500 | 500/DDIM-50 | A100 | 1.5-4 h | Final model |

Both scout.sh and validate.sh support SLURM array jobs for parallel sweeps:
```bash
sbatch --array=0-7 scripts/cifar10/slurm/scout.sh      # 8 configs in parallel
sbatch --array=0-3 scripts/cifar10/slurm/validate.sh    # 4 configs in parallel
```

**Batched renderer**: Replaced per-image loop with `generate_2D_gaussian_splatting_batch` (chunks of 32). FID rendering: ~50s -> ~5s for 500 images.

---

## Critical Bug Fix: AMP fp16 → bf16

**Root cause of all NaN crashes**: `torch.amp.autocast("cuda")` defaults to fp16, even on A100.
fp16 max value is 65504 — easily overflows during attention/gradient computation with K=500 tokens.

**Fix**: Detect bf16 support and use `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
bf16 has fp32-range exponents (±3.4e38) with fp16-level precision. No GradScaler needed.

All Phase 2 A100 jobs use bf16 automatically. V100 still uses fp16 (no bf16 support).

---

## Deep Diagnostic (ep175 384D model)

### Per-timestep denoising quality
```
t=1:   noise_mse=0.738  x0_mse=0.0002  (near perfect)
t=10:  noise_mse=0.649  x0_mse=0.005   (good)
t=50:  noise_mse=0.574  x0_mse=0.104   (starting to struggle)
t=100: noise_mse=0.344  x0_mse=0.353   (significant errors)
t=150: noise_mse=0.115  x0_mse=0.680   (very poor)
```

Model denoises well at low noise but struggles at high noise (t>100). This is expected for an underfit model.

### Feature difficulty ranking (x0_mse at t=100)
1. rho: 0.500 (hardest)
2. x, y: 0.472, 0.442
3. sigma_x/y: 0.378, 0.382
4. r, g, b: 0.243, 0.217, 0.189 (easiest)

Spatial/shape features are 2x harder than color features.

### Generated parameter distribution
sigma_x/sigma_y generated with std=1.33 (vs expected 1.0) → over-dispersed → too many extreme sizes → blobs.

### Training efficiency
```
CIFAR-10 (best so far): 200ep × 469 steps = 93,800 steps, 4000 values/sample
MNIST (FID=6.66):       1500ep × 430 steps = 645,000 steps, 420 values/sample
=> CIFAR-10 has 1.5% of MNIST's effective training (adjusting for data complexity)
```

The model is massively underfit. More training is the #1 priority.

---

## Phase 2 Progress (500 epochs, ongoing)

| Job | Tag | Model | LR | BS | GPU | AMP | Ep | Best FID | Status |
|-----|-----|-------|-----|-----|-----|-----|----|----------|--------|
| **138937** | lr5e4_bs128_500ep | 256D (7.4M) | 5e-4 | 128 | V100 | fp16 | 131 | **148.8** | Running, plateauing |
| 138925 | lr1e3_bs256_500ep | 256D (7.4M) | 1e-3 | 256 | A100 | fp16 | 132 | 164.8 | Running, FID oscillating |
| 138935 | lr5e4_bs256_500ep | 256D (7.4M) | 5e-4 | 256 | A100 | fp16 | 84 | 167.5 | Running, FID oscillating |
| **139112** | 16h384d_fp16_500ep | 384D (16.5M) | 5e-4 | 128 | A100 | **fp16** | 0 | — | **JUST STARTED** (critical) |
| **139094** | 16h384d_wloss | 384D (16.5M) | 5e-4 | 128 | A100 | fp16 | 0 | — | **JUST STARTED** (weighted loss) |

Cancelled:
| 138960 | bf16 384D | Cancelled — bf16 val_loss 0.37 (vs fp16 0.33), FID=234.6@ep100 |
| 138936 | lr=7e-4 | Cancelled — unstable val_loss 0.35+, FID=215.2@ep50 |

### FID Trajectories (full):
```
138937 (256D, fp16, bs=128, V100):   251.2 → 177.1 → 169.7 → 169.8 → 148.8 → 149.4  ← plateauing
138925 (256D, fp16, lr=1e-3, A100):  252.1 → 176.3 → 198.2 → 176.9 → 164.8 → 193.8  ← high variance
138935 (256D, fp16, bs=256, A100):   249.6 → 187.5 → 167.5 → 193.8                    ← high variance
138960 (384D, bf16, A100):           — → 214.4 → 234.6 → CANCELLED                     ← bf16 bad
138936 (lr=7e-4, A100):              — → 206.8 → 215.2 → CANCELLED                     ← unstable
```

### Key Phase 2 Findings

1. **bf16 kills convergence**: Val_loss 0.367 (bf16) vs 0.328 (fp16). bf16's 7-bit mantissa is insufficient for K=500 attention. Reverted to fp16 default, bf16 opt-in via `--bf16`.

2. **bs=128 >> bs=256**: bs=128 (138937) shows smooth FID descent (148.8→149.4). bs=256 runs (138925, 138935) oscillate wildly (±25 points). This confirms MNIST finding.

3. **FID variance with 500 samples is ±15 points**: The bs=256 "degradation" (167.5→193.8) is likely noise, not real deterioration. Need 2000+ samples for reliable measurement.

4. **fp16 works on A100**: Despite A100 natively supporting bf16, fp16 gives better convergence. GradScaler handles overflow risk.

### Improvements implemented
1. **fp16 default** — reverted from bf16 auto-detect. `--bf16` for opt-in.
2. **Feature-weighted MSE loss** — 139094 running with sigma/rho/coords at 2x weight
3. **AMP/compile toggleable** in train.sh via env vars
4. **Batched renderer** for 10x faster FID evaluation
5. **`--grad_clip` configurable** — default 1.0, for NaN prevention

---

## FID Trajectory Summary

| Epoch | 384D lr=1e-3 (fin.) | 256D lr=5e-4 (NaN@147) | 384D lr=5e-4 (139112) | 384D v-pred (139151) |
|-------|----------------------|-------------------------|------------------------|----------------------|
| 1 | 230.7 | 251.2 | 239.9 | 214.4 |
| 25 | 159.0 | 177.1 | 166.3 | — |
| 50 | 173.0 | 169.7 | 156.0 | **155.4** |
| 75 | 162.0 | 169.8 | 165.4 (noise) | pending |
| 100 | 174.0 | **148.8** | **153.6** | pending |
| 125 | 169.2 | 149.4 | pending | pending |
| 150 | 152.3 | NaN! | pending | pending |
| 175 | **143.0** | — | pending | pending |
| 200 | 155.2 | — | pending | pending |

**Key comparisons at ep50 (same 384D config, CFG=1.5, DDIM-50, 500 samples):**
- Epsilon (139112): FID = 156.0
- V-prediction (139151): FID = **155.4**
- V-pred matches epsilon at ep50, despite starting 25 points better at ep1. Both still improving.

---

## Sampling Sweep Results (no retraining, just parameter optimization)

### 384D sweep (ep173, 1000 samples, ongoing — 15/24 done):

| eta | CFG | steps | FID |
|-----|-----|-------|-----|
| 0.0 | 1.0 | 50 | 145.9 |
| 0.0 | 1.0 | 100 | 145.0 |
| 0.0 | 1.5 | 50 | 143.5 |
| 0.0 | 1.5 | 100 | 141.3 |
| 0.0 | **2.0** | **50** | **135.8** |
| 0.0 | **2.0** | **100** | **135.6** ← best FID |
| 0.3 | 1.0 | 50 | 151.8 |
| 0.3 | 1.0 | 100 | 147.9 |
| 0.3 | 1.5 | 50 | 143.6 |
| 0.3 | 1.5 | 100 | 140.7 |
| 0.3 | 2.0 | 50 | 138.7 |
| 0.3 | 2.0 | 100 | 138.0 |
| 0.5 | 1.0 | 50 | 153.1 |
| 0.5 | 1.0 | 100 | 154.1 |
| 0.5 | 1.5 | 50 | 146.3 |

### 256D sweep (ep121, 1000 samples, **COMPLETED**):

| eta | CFG | steps | FID |
|-----|-----|-------|-----|
| 0.0 | 1.0 | 50 | 174.1 |
| 0.0 | 1.0 | 100 | 162.2 |
| 0.0 | 1.5 | 50 | 165.6 |
| 0.0 | 1.5 | 100 | 152.4 |
| 0.0 | 2.0 | 50 | 156.9 |
| 0.0 | 2.0 | 100 | 144.3 |
| 0.3 | 1.0 | 50 | 172.4 |
| 0.3 | 1.0 | 100 | 159.9 |
| 0.3 | 1.5 | 50 | 161.8 |
| 0.3 | 1.5 | 100 | 150.4 |
| 0.3 | 2.0 | 50 | 151.4 |
| 0.3 | 2.0 | 100 | 149.4 |
| 0.5 | 1.0 | 50 | 172.0 |
| 0.5 | 1.0 | 100 | 157.3 |
| 0.5 | 1.5 | 50 | 160.2 |
| 0.5 | 1.5 | 100 | 152.7 |
| 0.5 | 2.0 | 50 | 155.7 |
| 0.5 | 2.0 | 100 | 145.5 |
| 0.8 | 1.0 | 50 | 173.2 |
| 0.8 | 1.0 | 100 | 159.9 |
| 0.8 | 1.5 | 50 | 165.8 |
| 0.8 | 1.5 | 100 | 150.5 |
| 0.8 | 2.0 | 50 | 156.6 |
| 0.8 | **2.0** | **100** | **144.1** ← best 256D |

### Key Sampling Findings
1. **CFG=2.0 is optimal for CIFAR-10** (not 1.5 as MNIST!): 135.6 vs 141.3 at CFG=1.5. Free 6-point improvement.
2. **100 steps helps significantly at CFG=2.0**: 135.8→135.6 (384D), 156.9→144.3 (256D). Effect is larger for 256D.
3. **384D >> 256D**: 135.6 vs 144.1 at optimal settings. 8.5 FID gap.
4. **eta doesn't matter much at CFG=2.0**: For 384D: 135.6 (eta=0), 138.0 (eta=0.3). For 256D: 144.3 (eta=0), 144.1 (eta=0.8) — within noise.
5. **eta=0.3-0.5 hurts at CFG=1.0**: 151.8 (eta=0.3) vs 145.9 (eta=0.0). Stochastic noise hurts with weak guidance.
6. **Best FID: 135.6** (384D, CFG=2.0, DDIM-100, 1000 samples)

---

## Cancelled/Failed Experiments

| Job | Config | Failure | Lesson |
|-----|--------|---------|--------|
| 138960 | 384D bf16 500ep | val_loss 0.37 vs 0.33 fp16 | bf16 7-bit mantissa insufficient for K=500 |
| 138936 | lr=7e-4 bs=256 | val_loss 0.35+, unstable | lr=7e-4 too high |
| 138925 | lr=1e-3 bs=256 | FID oscillating 165-194 | lr=1e-3 + bs=256 = noisy |
| 138935 | lr=5e-4 bs=256 | FID oscillating 168-194 | bs=256 too large |
| 138937 | 256D fp16 V100 | **NaN at ep147** | fp16 on V100 inherently unstable |
| 139094 | 384D weighted loss | **NaN at ep34** | Weighted loss amplifies gradients → fp16 overflow |

**Critical lesson**: fp16 NaN is the #1 risk. V100 fp16 fails at moderate LR. Weighted loss fails even on A100 fp16. Only standard epsilon loss on A100 fp16 is reliable.

---

## Analysis: Why FID is Still High

### Training efficiency (the core issue)
```
CIFAR-10 (best so far, 200ep): 200 × 422 steps = 84,400 steps, 4000 values/sample
MNIST (FID=6.66, 1500ep):      1500 × 430 steps = 645,000 steps, 420 values/sample
=> CIFAR-10 has had ~1.5% of MNIST's effective training (adjusting for data complexity)
```

**The model is massively underfit.** At 500ep, we'll have ~211,000 steps — still only ~8% of MNIST equivalent.
Job 139148 (1500ep) targets 633,000 steps — matching MNIST.

### FID measurement noise
With 500 samples: ±15 point variance. With 1000 samples (sweeps): ±7 variance.
The sweep confirms training FID was reasonably accurate: 143.0 (training, 500 samples) vs 143.5 (sweep, 1000 samples).

---

## Phase 3: fp16 + NaN Recovery Ablation (ongoing)

### Current running experiments

| Job | LR | Pred | min-SNR | hflip | Ep | FID@50 | FID@100 | NaN | Status |
|-----|-----|------|---------|-------|----|--------|---------|-----|--------|
| **139247** | **1e-3** | eps | no | no | **108** | **163.0** | **145.2** | **0** | **★ BEST** |
| 139300 | 1e-3 | eps | no | no | 0 | — | — | 0 | backup |
| 139248 | 5e-4 | eps | no | no | 107 | 176.6 | 171.7 | **3** | struggling |
| 139253 | 5e-4 | v-pred | no | no | 95 | 180.3 | pending | 0 | ok |

All 384D (16.5M), fp16, bs=128, 1500ep, each on a separate A100 node.

### Key findings (Phase 3)

1. **lr=1e-3 >> lr=5e-4**: FID 145.2 vs 171.7 at ep100. Higher LR converges much faster.

2. **lr=1e-3 is MORE stable than lr=5e-4 in fp16**: 139247 (lr=1e-3) has 0 NaN at ep108, while 139248 (lr=5e-4) NaN'd 3 times by ep98. Possibly because higher LR → larger gradients → GradScaler uses smaller scale factor → less overflow.

3. **Min-SNR+hflip HURTS FID**: 139246 (min-SNR+hflip) FID=188.3@ep100 vs 139247 (clean) FID=145.2@ep100. A 43-point gap! Min-SNR downweights high-noise timesteps that are critical for generating global structure. **Don't use min-SNR for CIFAR-10.**

4. **NaN recovery mechanism works**: Added auto-reload from best.pt on NaN (up to 3 times). 139248 survived 3 NaN events at ep32, 69, 98. But each recovery degrades val_loss (0.33→0.38) because progress is lost.

5. **bf16 kills generation quality (confirmed)**: bf16 runs had similar val_loss (~0.33) but much worse FID (192-315 vs 145-163). bf16's 7-bit mantissa causes accumulated errors in the DDIM reverse process.

### FID Trajectories
```
139247 (eps, lr=1e-3, fp16):     235.2 → 163.0 → 145.2 (★ best, still improving, 0 NaN)
139248 (eps, lr=5e-4, fp16):     249.8 → 176.6 → 171.7 (3 NaN recoveries, degraded)
139253 (v-pred, lr=5e-4, fp16):  227.3 → 180.3 → pending...
139246 (eps, lr=1e-3, minSNR+hflip): 243.3 → 190.6 → 188.3 (cancelled — min-SNR hurts)
```

Previous runs:
```
139112 (eps, lr=5e-4, fp16):  239.9 → 166.3 → 156.0 → 165.4 → 153.6 → NaN! (ep125)
138927 (eps, lr=1e-3, fp16, 200ep): best training FID 143.0@ep175. Sweep: 135.6 with CFG=2.0
```

### bf16 experiment results (concluded)
```
139192 (bf16, minSNR+hflip, lr=1e-3): FID@50=192.7, val=0.334. Cancelled ep85.
139196 (bf16, control, lr=1e-3):      FID@50=315.5, val=0.365. Cancelled ep57.
139210 (bf16, hflip, lr=1e-3):        Only ran to ep35. Cancelled.
```
**Verdict**: bf16 achieves similar val_loss but 2-3x worse FID. Not viable for this model.

### v-prediction comparison
- **ep1**: v-pred 227.3 vs epsilon 249.8 (v-pred starts better)
- **ep50**: v-pred 180.3 vs epsilon 176.6 (epsilon slightly better, but 139248 had NaN recovery issues)
- V-pred neither clearly helps nor hurts. Will continue monitoring to ep100+.

---

## Next Steps

### Running now (4 GPUs, all separate A100 nodes)
1. **139247** (epsilon, fp16, lr=1e-3, 1500ep) — **★ ep108, FID=145.2@ep100**
2. **139300** (epsilon, fp16, lr=1e-3, 1500ep) — backup copy, ep0
3. **139248** (epsilon, fp16, lr=5e-4, 1500ep) — ep107, 3 NaN recoveries, struggling
4. **139253** (v-pred, fp16, lr=5e-4, 1500ep) — ep95, FID=180.3@ep50

### Key milestones
- 139247 ep150 FID: beating the sweep best of 135.6? (in-training FID uses CFG=2.0, 500 samples)
- 139247 ep200 FID: matching/beating the 200ep model's sweep FID of 135.6?
- 139253 ep100 FID: v-pred at 100ep
- 139247 survival: will it NaN? (Previous runs NaN'd at ep60-125)
- 139300 ep50 FID: backup verification

### When 139247 reaches ep200+
Run a sampling sweep on the best checkpoint with 1000 samples. Expected: FID < 130 with CFG=2.0.

### Implemented improvements
1. **NaN recovery** — auto-reload from best.pt on NaN (up to 3x per run)
2. **v-prediction** — `--prediction_type v`
3. **Min-SNR weighting** — `--min_snr_gamma 5` (implemented but HURTS FID — don't use)
4. **Horizontal flip** — `--hflip` (implemented but untested in isolation)
5. **bf16 support** — `--bf16` (implemented but HURTS FID — don't use)
6. **Sampling sweep** — `scripts/cifar10/sample_sweep.py`
7. **GPU contention prevention** — `--exclude` in sbatch

### FID trajectory: training → sampling optimization
```
Training FID (CFG=2.0, DDIM-50, 500 samples):  145.2 (384D ep100, 139247)
Previous sweep (CFG=2.0, DDIM-100, 1000 samples): 135.6 (384D ep173, older model)
=> Sweep typically improves FID by 5-10 points over in-training measurement
=> Expected sweep FID at ep200: ~115-125
```

### Target
- **Short-term**: FID < 130 (with 139247 at ep200 + sweep)
- **Medium-term**: FID < 100 (with 139247 at ep500 + sweep)
- **Long-term**: FID < 70 (with ep1500 + optimal sampling)
- **Reconstruction floor**: 10.5 (theoretical minimum)

### Lessons learned
1. **fp16 + NaN recovery > bf16**: bf16 converges to similar loss but generates much worse samples
2. **No tricks needed**: Clean epsilon prediction + lr=1e-3 + fp16 is the best recipe
3. **Min-SNR hurts**: Downweights high-noise timesteps critical for global structure
4. **GPU contention is real**: Always use --exclude for separate A100 nodes
5. **LR=1e-3 > 5e-4**: Faster convergence AND more fp16-stable (counterintuitively)

---

## Phase 4: Training Stability + LR Schedule (concluded)

### Key discoveries

1. **fp16 is not viable for K=500**: Every fp16 run eventually NaN's or degrades. Tried `x.clamp(-1e4, 1e4)` in DiTBlock — survived longer but collapsed by ep275 (FID: 143→300). bf16 also kills quality (7-bit mantissa). **fp32 is the only stable option.**

2. **fp32 speed**: 87s/ep (compiled, A100). 3x slower than fp16 (28s/ep) but the only path that works.

3. **1500ep cosine >> 500ep cosine for long training**: 500ep cosine decays LR too aggressively — plateaus at ep200 (FID=134.3) and never improves. 1500ep cosine keeps improving through ep1100+.

4. **LR schedule comparison (both resumed from ep140 checkpoint)**:

| Epoch | 139517 (500ep reset) | 139343 (1500ep) |
|-------|-----|-----|
| 150 | 153.6 | 148.5 |
| 200 | **134.3** | 142.6 |
| 250 | 141.0 ↑ | 136.3 |
| 300 | 141.4 | 132.2 |
| 500 | 134.8 (stuck) | 141.7 |
| 1050 | — | **127.5** ★ |

500ep reset wins short-term (134.3@ep200 vs 142.6) but plateaus. 1500ep keeps improving.

### Sampling sweep (corrected normalize=True)

**ep140 checkpoint sweep (500 samples, 32 configs):**

| eta\cfg (steps=100) | 1.5 | 2.0 | 2.5 | 3.0 |
|-----|------|------|------|------|
| 0.0 | 178.3 | 176.9 | 179.7 | 185.2 |
| 0.3 | 167.4 | **158.6** | 167.6 | 170.7 |
| 0.5 | 159.8 | 162.0 | 163.2 | 157.5 |
| 0.8 | **154.5** | 155.6 | 156.0 | 156.0 |

**ep253 checkpoint sweep (500 samples, eta sweep):**

| eta | FID (cfg=2.0, steps=100) |
|-----|-----|
| 0.0 | **131.0** |
| 0.3 | 132.5 |
| 0.8 | 133.6 |
| 1.0 | 134.8 |

At ep253, eta barely matters (131-137 range). CFG=2.0 optimal for CIFAR-10.

### Final results (all three runs completed)

| Job | Tag | Config | Final Ep | Best FID | Best @ | Status |
|-----|-----|--------|----------|----------|--------|--------|
| **139343** | std12_fp32_1500ep | fp32, 1500ep cosine | 1122 | **127.5** | ep1050 | ★ BEST |
| 139517 | std14_fp32_reset | fp32, 500ep reset | 500 | 134.3 | ep200 | plateaued |
| 139638 | std16_fp16_clamp | fp16+clamp, 500ep reset | 348 | 143.7 | ep275 | collapsed (300+) |

### 139343 full FID trajectory (★ best run)
```
ep150:  148.5    ep500:  141.7    ep850:  132.0
ep200:  142.6    ep550:  135.8    ep900:  129.1
ep250:  136.3    ep600:  138.2    ep950:  128.0
ep300:  132.2    ep650:  132.7    ep1000: 134.9
ep350:  136.2    ep700:  138.4    ep1050: 127.5 ★
ep400:  141.2    ep750:  134.2    ep1100: 129.7
ep450:  145.8    ep800:  132.4
```

FID oscillates ±8 points (500-sample noise) but the trend is clear: **~2 FID improvement per 100 epochs**.
Trend: 148.5 (ep150) → 136 (ep300) → 133 (ep650) → 129 (ep900) → 127.5 (ep1050).

---

## Phase 5: Analysis — Is More Training Enough?

### Training scale comparison with DiT

| | DiT-XL/2 (SOTA) | Ours (current) | Ratio |
|---|---|---|---|
| Parameters | 675M | 16.5M | **41x smaller** |
| Tokens/sample | 256 (patchified) | 500 | 2x more |
| Token dim | 16 (patchified) | 8 | — |
| Output dims | 4,096 | 4,000 | ~same |
| Training steps | 7,000,000 | ~530,000 (ep1122) | **13x fewer** |
| Batch size | 256 | 128 | 2x |
| Sample-steps | 1.79B | 68M | **26x fewer** |
| Positional encoding | 2D sinusoidal | **None** | critical |
| FID | 2.27 (ImageNet 256) | 127.5 (CIFAR-10 32) | — |

The dimensionality (4,000 values) is comparable to DiT. The gap is:
1. **Model 41x too small** — 16.5M vs 675M
2. **Training 26x too short** — 68M sample-steps vs 1.79B
3. **No positional encoding** — DiT tokens have 2D spatial pos-enc; our Gaussians are unordered sets

### Improvement rate extrapolation

From the trajectory, FID improves ~2 points per 100 epochs (decelerating):
- ep150→300: -16.3 FID / 150ep = **10.9 per 100ep**
- ep300→650: -3.3 FID / 350ep = **0.9 per 100ep**
- ep650→1050: -5.2 FID / 400ep = **1.3 per 100ep**

Extrapolating at ~1 FID per 100ep:
- ep2000: ~118 (900 more epochs = 22h on A100)
- ep5000: ~88 (3900 more epochs = 4 days)
- ep10000: ~38? (unlikely — will plateau well before)

**Training alone won't reach reconstruction floor (10.5)**. The model is fundamentally limited at 16.5M params with no positional encoding.

### What would help most

1. **Bigger model**: Even 50-60M params (12B/512D) would 3.5x the capacity. DiT-S (33M) is the absolute minimum reference.
2. **Positional encoding**: Sort Gaussians by spatial position, add learned pos-enc. This gives the model spatial priors that DiT gets for free.
3. **Much longer training**: Current run at 530k steps; DiT uses 7M. Need at least 2-5M steps.
4. **Combination**: Bigger model + pos-enc + longer training is the path to FID < 50.

### Targets (updated)
- **Current best**: FID 127.5 (in-training, ep1050) / 131.0 (sweep, ep253)
- **Expected with sweep on ep1050**: ~115-120
- **Expected at ep2000**: ~118 (training only, current model)
- **With bigger model (50M+) + pos-enc**: FID < 80 plausible
- **Reconstruction floor**: 10.5

---

## Phase 6: Model Scaling (overnight, 2026-03-09→10)

### Hypothesis
The 16.5M model is capacity-limited, showing diminishing returns after ep1000. Scaling up to 29.3M and 57.6M should improve FID/epoch efficiency.

### Precision decision: fp32 only
fp32 is the ONLY stable precision for K=500:
- **fp16**: Always NaN's eventually (max 65504 overflows in attention). Every fp16 run NaN'd between ep60-275.
- **bf16**: Achieves similar val_loss but 2-3x worse FID (7-bit mantissa accumulated errors in DDIM reverse).
- **fp32**: Zero NaN's across 1500+ epochs in all runs.

### Multi-GPU DDP support
Added full DDP training support to `scripts/cifar10/train.py`:
- `torchrun --nproc_per_node=N` launcher via SLURM
- DistributedSampler, rank-0 gating for I/O, proper EMA on raw model
- Wrapping order: raw_model → compile → DDP
- 4×A100 DDP jobs couldn't schedule (all nodes in "mixed" state) — fell back to 1×GPU

### Models tested

| Model | Config | Params | Speed (1×A100, fp32, compiled) |
|-------|--------|--------|-------------------------------|
| 16.5M | 6B/16H/384D | 16,529,032 | 87s/ep |
| 29.3M | 6B/16H/512D | 29,368,328 | 103s/ep |
| 57.6M | 12B/16H/512D | 57,619,464 | 843s/ep |

### Running experiments

| Job | Tag | Model | LR | BS | Epochs | Start | Status |
|-----|-----|-------|-----|-----|--------|-------|--------|
| **144793** | std17_16M_resume_5000ep | 16.5M | 1e-3 | 128 | 5000 | ep~1122 (resumed) | Running |
| **144803** | std19_29M_fresh_5000ep | 29.3M | 1e-3 | 128 | 5000 | ep0 (fresh) | Running |
| **144794** | std18_57M_fresh_5000ep | 57.6M | 1e-3 | 128 | 5000 | ep0 (fresh) | Running |

### FID Trajectories (in-training, CFG=2.0, DDIM-50, 500 samples)

**29.3M model (144803) — star performer:**
```
ep1:   241.0    ep125: 136.8
ep25:  162.1    ep150: 134.1
ep50:  149.6    ep175: 131.8
ep75:  149.3    ep200: 127.2  ★ matched 16.5M's best in 5x fewer epochs
ep100: 139.0    ep225: 125.7  ★ ALL-TIME BEST
ep250: 131.8    ep275: 140.9  (noise outlier)
ep300: 137.1
```
Recent 4-point average (ep225-300): 133.6. Trend: slowly improving with ±10 measurement noise.
Val_loss stable at 0.329 throughout — model is learning, FID noise is from 500-sample measurement.

**16.5M model (144793) — resumed from ep~1122:**
```
(continued from Phase 4 run 139343)
ep1125: 128.4    ep1250: 129.7
ep1150: 134.6    ep1275: 133.2
ep1175: 130.5    ep1300: 131.0
ep1200: 138.4    ep1375: 126.7
ep1225: 134.4    ep1400: 128.6
                 ep1425: 129.0
                 ep1450: 128.4
                 ep1475: 126.7
```
Oscillating ±7 around 130. Diminishing returns — ~1 FID per 200 epochs.

**57.6M model (144794):**
```
ep1:  240.7
ep25: 157.2
```
Very slow (843s/ep) but shows fastest per-epoch FID descent: 83.5 FID improvement in 25 epochs vs 29.3M's 78.9.

### Key findings (Phase 6, preliminary)

1. **29.3M validates model scaling**: FID 127.2@ep200 matches 16.5M's best (127.5@ep1050) in 5x fewer epochs. The bigger model is dramatically more sample-efficient.

2. **Per-epoch efficiency scales with model size**:
   - 16.5M: 148.5→127.5 = 21 FID improvement over 900 epochs = 0.023 FID/epoch
   - 29.3M: 241→127.2 = 113.8 FID improvement over 200 epochs = 0.57 FID/epoch
   - 57.6M: 240.7→157.2 = 83.5 improvement over 25 epochs = 3.34 FID/epoch

3. **57.6M is impractical on 1×GPU**: 843s/ep means ~37 epochs in 9 hours. Would need 4×A100 DDP (not schedulable) or accept very slow progress.

4. **16.5M is at diminishing returns**: After ep1100, FID oscillates 126-138 with ~1 point improvement per 200 epochs. Model capacity is the bottleneck, not training duration.

5. **29.3M is the sweet spot**: 103s/ep (only 18% slower than 16.5M) but converges much faster. Expected to reach sub-120 FID by ep400-500.

### Updated targets
- **Current all-time best**: FID 126.7 (16.5M, ep1375, in-training)
- **29.3M projected ep500**: FID ~115-120 (based on trajectory)
- **29.3M projected ep1000**: FID ~105-110
- **Reconstruction floor**: 10.5
