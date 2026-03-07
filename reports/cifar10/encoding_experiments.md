# CIFAR-10 Encoding Experiments

## Goal
Find optimal hyperparameters for encoding CIFAR-10 (32x32 RGB) as 2D Gaussian splats.
Target: PSNR >= 40 dB (MSE <= 1e-4), with reasonable encoding speed for 60k images.

## Key Differences from MNIST
| Property | MNIST | CIFAR-10 |
|----------|-------|----------|
| Size | 28x28 | 32x32 |
| Channels | 1 (grayscale) | 3 (RGB) |
| Complexity | Simple digits | Natural scenes |
| Param dim | 7 (raw) / 6 (diffusion) | 9 (raw) / 8 (diffusion) |
| MNIST best K | 70 | TBD |

## Experiment Plan

### Phase 1: K Sweep (number of Gaussians)
- K = {50, 100, 150, 200, 300, 500}
- Fixed: epochs=3000, lr=5e-3, kernel_size=11, 20 sample images
- Goal: Find minimum K for >= 40 dB mean PSNR

### Phase 2: Epoch Sweep
- epochs = {1000, 2000, 3000, 5000}
- Fixed: best K from Phase 1, lr=5e-3
- Goal: Find minimum epochs for convergence

### Phase 3: LR Sweep
- lr = {1e-3, 3e-3, 5e-3, 1e-2}
- Fixed: best K and epochs from above
- Goal: Optimize convergence speed

### Phase 4: Additional tuning
- kernel_size = {7, 11, 15}
- recycle_every = {100, 200, 300, 500}
- early_stop threshold

## Results

### Phase 1: K Sweep (lr=5e-3, epochs=3000, ks=11)
SLURM job 137489 (A100, completed)

| K | PSNR mean | PSNR std | PSNR min | PSNR max | >=40dB | >=35dB | >=30dB | Time/img | Est 60k |
|-----|-----------|----------|----------|----------|--------|--------|--------|----------|---------|
| 50 | 11.30 | 2.64 | 7.23 | 18.92 | 0% | 0% | 0% | 10.88s | 181h |
| 100 | 17.43 | 2.60 | 13.60 | 25.40 | 0% | 0% | 0% | 11.15s | 186h |
| 150 | 22.81 | 2.10 | 19.32 | 29.19 | 0% | 0% | 0% | 10.92s | 182h |
| 200 | 26.45 | 1.68 | 23.38 | 31.97 | 0% | 0% | 5% | 11.01s | 184h |
| 300 | 31.09 | 2.23 | 25.93 | 35.80 | 0% | 5% | 70% | 10.83s | 181h |
| **500** | **34.78** | **4.25** | **27.59** | **40.01** | **25%** | **50%** | **85%** | **10.31s** | **172h** |

Key findings:
- K=500 clearly best: mean 34.78 dB, 25% of images converge to 40 dB
- Time/image is ~11s REGARDLESS of K (GPU-parallelized renderer)
- No dead Gaussians detected in ANY experiment (recycling broken for RGB)
- High variance (std=4.25 dB): simple images reach 40+ dB, complex textures ~27 dB

### Quick single-image diagnostics (same CIFAR-10 frog image)
| K | LR | Epochs | KS | PSNR | Time | Note |
|------|--------|--------|-----|-------|------|------|
| 100 | 5e-3 | 500 | 11 | 10.56 | 6.2s | baseline |
| 200 | 5e-3 | 3000 | 11 | 25.20 | 15.0s | |
| 200 | 1e-2 | 3000 | 11 | 27.28 | 16.5s | lr=0.01 helps +2 dB |
| 200 | 5e-3 | 5000 | 11 | 27.47 | 22.2s | +2 dB but 50% slower |
| 200 | 5e-3 | 3000 | 7 | 10.20 | 13.4s | ks=7 BROKEN |
| 500 | 5e-3 | 3000 | 11 | 33.15 | 14.0s | big jump |
| 500 | 1e-2 | 3000 | 11 | 33.39 | 13.1s | |
| 500 | 1e-2 | 5000 | 11 | 34.36 | 20.5s | diminishing returns |
| 500 | 1e-2 | 8000 | 11 | 33.96 | 33.2s | worse! warm restarts hurt |
| 750 | 5e-3 | 3000 | 11 | 28.69 | 14.1s | WORSE than K=500! |
| 750 | 1e-2 | 3000 | 11 | 30.73 | 13.1s | higher lr helps |
| 1000 | 1e-2 | 3000 | 11 | 26.58 | 11.8s | WORSE, optimizer struggles |
| 1000 | 2e-2 | 3000 | 11 | 30.35 | 12.0s | lr scaling partially fixes |
| 1500 | 5e-3 | 3000 | 11 | 21.61 | 11.9s | worst: too many params |

Key: K>500 DEGRADES because optimizer can't handle the landscape. Higher lr partially compensates.

### Phase 2+3+4 Combined Results (all completed)
Ran 32 experiments across 4 phases, ~50+ SLURM jobs.

**Definitive results (A100, N=50, most reliable):**

| K | LR | Epochs | N | PSNR mean | PSNR std | PSNR min | >=40dB | >=30dB | Time/img | Est 60k |
|-----|--------|--------|-----|-----------|----------|----------|--------|--------|----------|---------|
| **450** | **1e-2** | **2500** | **50** | **35.02** | **3.56** | **28.08** | **12%** | **94%** | **8.6s** | **143h** |
| 500 | 1e-2 | 3000 | 50 | 34.88 | 3.76 | 25.75 | 18% | 92% | 10.0s | 167h |
| 450 | 1e-2 | 3000 | 50 | 34.77 | 3.21 | 29.08 | 14% | 94% | 10.4s | 174h |
| 500 | 1e-2 | 2500 | 50 | 34.66 | 3.26 | 26.98 | 8% | 90% | 8.7s | 146h |
| 500 | 5e-3 | 3000 | 50 | 34.17 | 3.89 | 25.78 | 12% | 88% | 16.6s | 277h |

**Additional results (A100/V100, N=10-30):**

| K | LR | Epochs | N | PSNR mean | >=40dB | >=30dB | Time/img |
|-----|--------|--------|-----|-----------|--------|--------|----------|
| 500 | 1e-2 | 4000 | 10 | 36.63 | 40% | 90% | 18.3s |
| 450 | 7e-3 | 3000 | 10 | 36.63 | 30% | 90% | 15.6s |
| 450 | 1e-2 | 4000 | 10 | 36.50 | 40% | 90% | 19.0s |
| 400 | 1e-2 | 3000 | 20 | 35.70 | 15% | 95% | 10.4s |
| 350 | 1e-2 | 3000 | 10 | 35.14 | 10% | 90% | 10.7s |
| 300 | 1e-2 | 3000 | 20 | 33.20 | 0% | 95% | 17.7s |
| 550 | 1e-2 | 3000 | 10 | 35.18 | 30% | 90% | 9.2s |
| 600 | 1e-2 | 3000 | 10 | 34.25 | 20% | 70% | 9.8s |
| 750 | 2e-2 | 3000 | 20 | 31.82 | 20% | 55% | 15.3s |
| 1000 | 2e-2 | 3000 | 20 | 28.65 | 10% | 30% | 16.6s |

**Optimizer/scheduler experiments (K=500, lr=0.01, ep=3000, 10 images):**

| Variant | PSNR mean | Note |
|---------|-----------|------|
| CosineWarmRestarts (default) | 33.67 | Best mean |
| CosineAnnealing (no restarts) | 33.00 | Better >=35dB rate |
| OneCycleLR | 33.23 | Best >=30dB rate |
| GradClip=1.0 | 30.90 | No improvement |
| MSE+SSIM | 30.85 | No improvement |

Scheduler/optimizer changes give <1 dB difference. Not worth changing.

**Error-based recycling test**: Tried moving bottom-10% Gaussians (by error at center) to
high-error regions. Result: -2 to -7 dB WORSE. Gaussians at low-error locations ARE the
reason error is low. Removing them increases error.

---

## Phase 5: Renderer Improvements (2026-03-07)

### Hypothesis
Three bottlenecks identified through deep analysis of the renderer:

1. **Kernel resolution**: The Gaussian kernel is evaluated on only 11 points spanning [-5, 5].
   Larger kernels provide more sample points for better gradients.
2. **Hard clipping kills gradients**: `clamp(0, 1)` creates a flat ceiling when overlapping
   Gaussians sum > 1. The optimizer gets zero gradient in saturated regions.
3. **Sigma parameterization**: `sigmoid(raw)` has vanishing gradients for small sigmas.
   `softplus(raw)` gives constant gradient flow.

### Code Changes
All changes are backward-compatible with optional parameters (defaults match old behavior):

1. **Soft saturation** (`gaussian_to_image.py`): `soft_clamp=False` parameter.
   When True: `1.0 - exp(-x.clamp(min=0))` instead of `clamp(0, 1)`.
2. **Softplus sigma** (`encode.py`): `sigma_activation="sigmoid"` parameter.
   When "softplus": `F.softplus(raw)` instead of `sigmoid(raw)`.
3. **Gradient-weighted init** (`encode.py`): `init_mode="brightness"` parameter.
   When "gradient": 70% gradient magnitude + 30% brightness for position sampling.

### Phase 5a: Kernel Size Sweep (N=10, K=500, lr=0.01, ep=3000)
Jobs 137635-137641 (A100) + 137651-137653 (V100)

| KS | PSNR mean | >=40dB | Min | Time/img | Note |
|----|-----------|--------|-----|----------|------|
| 11 | 35.59 | 20% | 28.4 | 10.2s | baseline |
| 13 | 36.22 | 30% | 31.0 | 9.3s | +0.6 dB |
| 15 | 35.70 | 40% | 29.3 | 8.5s | more convergences |
| 17 | 36.21 | 30% | 30.3 | 9.0s | +0.6 dB |
| 21 | 34.70 | 20% | 29.6 | 15.2s | WORSE (less padding room) |
| 25 | 32.54-33.96 | 10-20% | 27.4 | 9.5-16.3s | WORSE |
| 32 | 19.70 | 0% | 16.8 | 11.1s | **CATASTROPHIC** (no padding) |

Key insight: Without soft_clamp, larger kernels HURT because:
- The kernel spans the full image (pad=0 for ks=32), leaving no room for positional translation
- Overlapping Gaussians saturate at clamp(1.0), killing gradients

### Phase 5b: Soft Clamp + Kernel Size (N=10)
Jobs 137642-137644, 137654-137656

| Config | PSNR mean | >=40dB | Min | Time/img |
|--------|-----------|--------|-----|----------|
| ks=17 +soft_clamp | 39.18 | 90% | 31.7 | 8.4s |
| ks=25 +soft_clamp | 39.21-39.22 | 90% | 32.1-32.2 | 4.4-7.0s |
| **ks=32 +soft_clamp** | **40.01-40.06** | **100%** | **40.0** | **2.5-3.9s** |

**BREAKTHROUGH**: Soft clamp is the single biggest improvement ever found.
- ks=32 goes from 19.70 → 40.06 dB (+20.4 dB!) with soft_clamp
- ks=25 goes from 33.96 → 39.22 dB (+5.3 dB)
- 100% of images converge to 40 dB target
- 2.5-4x FASTER than baseline (early convergence)

Why it works: `1 - exp(-x)` preserves gradient flow when overlapping Gaussians sum > 1.
With hard clamp, any region where sum > 1 gets zero gradient — the optimizer can't
reduce individual Gaussians because d/dx clamp(sum, 0, 1) = 0 when sum > 1.
With soft clamp, the gradient is always exp(-sum) > 0, so the optimizer can always improve.

### Phase 5c: Softplus Sigma (N=10)
Jobs 137645-137647, 137657-137658

| Config | PSNR mean | >=40dB | Min | Time/img |
|--------|-----------|--------|-----|----------|
| ks=17 +sc +softplus | 39.43 | 90% | 34.3 | 4.3s |
| ks=25 +sc +softplus | 39.52 | 90% | 35.2 | 3.7-5.7s |
| ks=32 +sc +softplus | 40.03-40.05 | 100% | 40.0 | 2.8-4.6s |

Softplus adds +0.02-0.31 dB on top of soft_clamp. Small but consistent.

### Phase 5d: Gradient-Weighted Init (N=10)
Jobs 137648-137650

| Config | PSNR mean | >=40dB | Min | Time/img |
|--------|-----------|--------|-----|----------|
| ks=17 +sc +softplus +gradient | 39.51 | 90% | 35.1 | 4.8s |
| ks=25 +sc +softplus +gradient | 39.59 | 90% | 35.8 | 3.8s |
| ks=32 +sc +softplus +gradient | 40.02 | 100% | 40.0 | 2.7s |

Gradient init adds ~0.07 dB. Negligible when ks=32+sc already converges everything.

### Phase 5e: 50-Image Validation (early_stop=1e-5 to see true PSNR)
Jobs 137674-137685 (4 A100 + 8 V100).

**Core results (A100, N=50):**

| Config | PSNR mean | std | >=40dB | min | Time/img |
|--------|-----------|-----|--------|-----|----------|
| **ks=32 +sc (lr=0.01)** | **46.70** | 3.36 | **94%** | 36.18 | 10.3s |
| ks=32 +sc +softplus (lr=0.01) | 46.65 | 3.40 | 94% | 36.54 | 10.4s |
| ks=11 baseline (lr=0.01) | 35.54 | 4.79 | 14% | 26.55 | 11.0s |
| ks=32 no sc (lr=0.01) | 18.87 | 1.63 | 0% | 15.76 | 11.0s |

**Soft clamp gives +11.16 dB** (35.54 → 46.70) on 50 images.
Without soft_clamp, ks=32 is catastrophic (18.87 dB).

### Phase 5f: LR Optimization (N=50, ks=32 +sc, early_stop=1e-5)
Jobs 137679, 137710-137713

| LR | PSNR mean | std | >=40dB | min | Time/img |
|------|-----------|-----|--------|------|----------|
| 0.005 | 43.08 | 4.50 | 82% | 33.08 | 17.3s |
| 0.01 | 46.70 | 3.36 | 94% | 36.18 | 10.3s |
| 0.015 | 47.89 | 2.95 | 100% | 40.14 | 9.9s |
| 0.02 | 48.39 | 2.43 | 100% | 40.65 | 14.5s |
| 0.025 | 48.68 | 2.32 | 100% | 41.29 | 8.7s |
| **0.03** | **48.95** | **2.00** | **100%** | **41.91** | **8.7s** |
| 0.04 | 48.64 | 2.01 | 100% | 42.22 | 9.4s |

**lr=0.03 is optimal**: 48.95 dB mean, 100% >=40dB, min=41.91 dB.
- lr=0.04 slightly worse mean but better min (42.22) — U-shaped curve peaks at 0.03
- lr=0.005 dramatically worse (43.08) — too slow to converge in 3000 epochs

### Phase 5g: Epoch Optimization (N=50, ks=32 +sc, lr=0.02)
Jobs 137714-137716, 137678

| Epochs | PSNR mean | >=40dB | min | Time/img |
|--------|-----------|--------|------|----------|
| 1000 | 44.10 | 84% | 34.96 | 5.9s |
| 1500 | 46.54 | 96% | 38.70 | 8.6s |
| 2000 | 47.46 | 96% | 39.68 | 11.2s |
| 3000 | 48.39 | 100% | 40.65 | 14.5s |
| 5000 | 48.45 | 96% | 38.79 | 23.2s |

**3000 epochs is optimal**. 5000 gives no improvement (+0.06 dB) at 60% more time.
1500ep still good (96% >=40dB) but loses 1.85 dB mean.

### Phase 5h: K Optimization (N=50, ks=32 +sc)
Jobs 137682-137683, 137717-137719 (partial results)

| K | LR | PSNR mean | >=40dB | min | Time/img |
|-----|------|-----------|--------|------|----------|
| 200 | 0.01 | 39.11 | 38% | 33.79 | 18.0s |
| 200 | 0.02 | 40.60 | 54% | 34.07 | 17.6s |
| 300 | 0.01 | 43.61 | 88% | 36.12 | 17.7s |
| 300 | 0.02 | 45.15 | 94% | 38.87 | 17.7s |
| 400 | 0.02 | 47.49 | 98% | 37.78 | 16.3s |
| 400 | 0.035 -sched | 47.63 | 98% | 38.85 | 13.7s |
| **500** | **0.035** | **49.05** | **100%** | **42.12** | **8.5s** |

K=500 is clearly best. K=400 is 1.4 dB worse, K=300 is 3.9 dB worse.

### Phase 5i: Other experiments (N=50)

| Config | PSNR mean | >=40dB | min | Time/img |
|--------|-----------|--------|------|----------|
| ks=25 +sc (lr=0.01) | 44.95 | 84% | 31.94 | 16.9s |
| ks=32 +sc +gradient (lr=0.01) | 47.22 | 96% | 38.33 | 16.7s |
| ks=32 +sc +sp +gradient (lr=0.01) | 46.87 | 94% | 35.99 | 17.0s |
| ks=32 +sc +sp (lr=0.02) | 48.64 | 100% | 42.23 | 14.3s |

### Phase 5j: Scheduler Ablation (N=50, ks=32 +sc, es=1e-5)

| LR | +sched | -sched | Delta | Note |
|------|--------|--------|-------|------|
| 0.02 | 48.39 | 48.62 | +0.23 | Nosched helps |
| 0.025 | 48.68 | 49.10 | **+0.42** | **Nosched best overall** |
| 0.03 | 48.95 | 48.94 | -0.01 | Tie |
| 0.035 | **49.05** | 48.32 | **-0.73** | Sched helps at high lr |
| 0.04 | 48.64 | 49.09 | +0.45 | Nosched helps |
| 0.05 | 47.97 | 48.73 | +0.76 | Nosched helps |

**Key insight**: No-scheduler helps at lr<=0.03 and lr>=0.04, but HURTS at lr=0.035.
The scheduler periodically resets lr which helps escape local minima at lr=0.035,
but at other lr values the warm restarts undo progress.

Best overall: **lr=0.025 no-scheduler: 49.10 dB** (min=41.55, 100% >=40dB)
Best with sched: **lr=0.035: 49.05 dB** (min=42.12, 100% >=40dB, **best min**)

### Phase 5k: LR Sweep at es=1e-6 (full 3000 epochs, N=20)

With es=1e-6, no images converge — all run full 3000 epochs. This reveals the
"ceiling" PSNR for each LR+scheduler combination.

| LR | +sched (dB) | N | GPU |
|------|-------------|-----|------|
| 0.03 | 50.97 | 20 | V100 |
| 0.035 | 50.24 | 50 | A100 |
| **0.04** | **51.15** | 20 | V100 |
| 0.045 | 50.45 | 20 | V100 |
| 0.05 | 49.73 | 20 | V100 |

50-image validations (A100 unless noted):

| LR | es=1e-6 PSNR (N=50) | es=1e-5 PSNR | es=1e-5 Time/img |
|------|----------------------|--------------|------------------|
| 0.03 | **50.50** (A100) | 48.95 (N=50) | 8.69s |
| 0.035 | 50.24 (A100) | 49.07 (N=20) | 8.33s |
| 0.04 | 50.18 (V100) | 48.96 (N=20) | 7.79s |

At es=1e-6, lr=0.03 is marginally best (50.50 dB), but all three are within noise.
At es=1e-5, all LRs in 0.03-0.04 give ~49 dB.
lr=0.04 is ~6% faster at es=1e-5 (7.79 vs 8.33 s/img).

**Epoch scaling at lr=0.04 (V100, es=1e-6):**
| Epochs | PSNR | Time/img | Delta from 3000ep |
|--------|------|----------|-------------------|
| 2000 | 49.37 | 11.8s | -1.8 dB |
| 3000 | 51.15 | 17.8s | baseline |

**K scaling at lr=0.035 (V100, es=1e-6):**
| K | PSNR | Note |
|-----|------|------|
| 300 | 45.96 | -4.3 dB from K=500 |
| 400 | 49.21 | -1.0 dB from K=500 |
| 500 | 50.24 | baseline (N=50) |

**AMP benchmark (A100, K=500, ks=32):**
| Precision | ms/step | Speedup |
|-----------|---------|---------|
| FP32 | 3.38 | 1.00x |
| FP16 (AMP) | 3.74 | 0.91x (SLOWER) |
| BF16 | 3.57 | 0.95x (SLOWER) |

AMP adds overhead (autocast, scaler) that exceeds savings on small tensors (K=500).
Not worth pursuing.

### Phase 5l: Speed Optimization

**Profiling results (A100, K=500, ks=32, soft_clamp):**
- Per-step: 3.83ms (forward 1.88ms, backward 2.50ms, optimizer 0.33ms)
- Per-image overhead: init 1.5ms, optimizer creation 0.5ms (negligible)

**Optimization 1: torch.compile**
- Per-step: 1.80ms (2.0x faster per step)
- First-call compile overhead: ~33s (amortized over 5000+ images = negligible)
- **Production test (encode.py, 20 images, A100, lr=0.04, es=1e-5):**
  - Without compile: 168s total (8.4s/img)
  - With compile: 127s total (4.14s/img steady-state after 48s first image)
  - **Steady-state speedup: 2.03x** (saves 50% encoding time)
  - For 5000 images/shard: 11.7h → 5.76h (**2.03x**)

**Optimization 2: Direct Gaussian Evaluation (renderer rewrite)**

Replaced `F.affine_grid + F.grid_sample` pipeline with direct analytical Gaussian
evaluation at pixel positions. When `kernel_size >= image_size` (CIFAR-10: ks=32=32):

1. Evaluate Gaussians directly at each pixel position (no padding, no grid creation)
2. Analytical 2x2 covariance inverse (no `torch.linalg.inv` LAPACK calls)
3. Simplified quadratic form: `z = -0.5/(1-r²) * (ux² - 2r·ux·uy + uy²)`
4. Efficient matmul for colour combination: `colours.T @ kernel` instead of broadcast multiply
5. Log-space normalization: `exp(z - z_max)` for peak = 1.0

File: `src/utils/gaussian_to_image.py` — new `_generate_direct()` function, auto-dispatched
when `kernel_size >= min(image_size)`. Old path preserved for MNIST (ks=11 < 28).

**Renderer microbenchmark (V100, K=500):**

| Metric | Old (affine_grid, gray) | New (direct, RGB) | Speedup |
|--------|-------------------------|--------------------|---------|
| Forward only | 1.793 ms | 0.603 ms | **2.98x** |
| Forward + Backward | 4.626 ms | 2.574 ms | **1.80x** |

Note: new path does 3-channel RGB; old does 1-channel gray. The new RGB path is still
faster than the old gray path.

**Full encoding comparison (V100, N=5, K=500, lr=0.04, ep=3000, es=1e-5, no compile):**

| Path | Time/img | PSNR |
|------|----------|------|
| Old (ks=31, affine_grid) | 14.51s | 46.5 dB |
| New (ks=32, direct eval) | 10.77s | 45.9 dB |
| **Speedup** | **1.35x** | |

**Production benchmark (A100, N=10, compile + direct eval, es=1e-5):**

| Config | Time/img | PSNR | Est/shard (5000 imgs) |
|--------|----------|------|-----------------------|
| No sched lr=0.04 | **3.91s** | 48.6 dB | 5.4h |
| Sched lr=0.04 | 4.34s | **49.3 dB** | 6.0h |
| No sched lr=0.06 | 3.99s | 46.6 dB | 5.5h |
| Sched lr=0.06 | 4.38s | 46.9 dB | 6.1h |

Previous result with old renderer + compile: ~4.14s/img. New: 3.91s/img (5% faster
on top of compile). The main gain is in the non-compiled path (1.35x) since torch.compile
already fuses many of the same operations.

**Optimization 3: Early stop (the main speed lever)**
With ks=32 + soft_clamp + lr=0.035 + scheduler, images converge at varying speeds
depending on the early_stop threshold (A100, N=20):

| Early Stop | PSNR Target | PSNR mean | PSNR std | min | Time/img | Est 60k (12 GPU) |
|------------|-------------|-----------|----------|-----|----------|-------------------|
| **1e-4** | 40 dB | 40.02 | 0.03 | 40.00 | **0.93s** | **1.3h** |
| 5e-5 | 43 dB | 43.05 | 0.10 | 42.91 | 1.91s | 2.7h |
| 3e-5 | 45 dB | 45.16 | 0.33 | 43.74 | 3.06s | 4.3h |
| 1e-5 | 50 dB | ~49 | ~2 | ~42 | ~8.5s | ~12h |
| 1e-6 | 60 dB | **50.24** | 3.65 | 42.26 | 17.6s | 24.4h |

Key observations:
- Every 3 dB of quality roughly doubles encoding time
- 100% of images converge to 40 dB within 0.93s (avg ~150 epochs)
- The hardest images (cls=9, complex textures) dominate at higher thresholds
- es=1e-6 gives no convergence — all images run full 3000 epochs (17.6s each)

**Previous es=1e-4 benchmarks (without scheduler):**

| Config (A100, N=50) | es | PSNR mean | min | Time/img |
|---------------------|------|-----------|-----|----------|
| lr=0.035 -sched | 1e-4 | 40.04 | 40.00 | **0.76s** |
| lr=0.03 -sched | 1e-4 | 40.04 | 40.00 | **0.83s** |

**Measured speed estimates for full 60k encoding (updated with direct eval):**

| Scenario | Time/img (A100) | Total GPU-h | With 12 GPUs |
|----------|----------------|-------------|--------------|
| Old config (ks=11, lr=0.01, es=1e-4) | ~10s | 167h | 14h |
| New: es=1e-5 (no compile, old renderer) | ~8.4s | ~140h | ~12h |
| New: es=1e-5 +compile (old renderer) | ~4.1s | ~69h | ~6h |
| **New: es=1e-5 +compile +direct eval** | **~3.9s** | **~65h** | **~5.4h** |
| New: es=1e-4 +compile | ~0.9s | ~15h | **~1.3h** |

**AMP benchmark: no benefit (A100, K=500, ks=32)**

| Precision | ms/step | Speedup |
|-----------|---------|---------|
| FP32 | 3.38 | 1.00x |
| FP16 (AMP) | 3.74 | 0.91x (SLOWER) |
| BF16 | 3.57 | 0.95x (SLOWER) |

AMP adds overhead (autocast, scaler) that exceeds savings on small tensors (K=500).

### Phase 5m: Batched Encoding (multiple images per GPU)

Instead of encoding images sequentially (one encode_image call per image), process B images
simultaneously with a single Adam optimizer over [B, K, C+6] parameters.

**Implementation** (`src/encode.py`):
- `encode_batch()`: processes [B, C, H, W] image tensor in parallel
- `generate_2D_gaussian_splatting_batch()`: batched renderer [B, K, H, W] kernels
- Per-image early stopping: zero gradients for converged images
- Best-param snapshot: save W_raw at each image's best loss (avoids Adam momentum drift)
- Sum (not mean) loss across images: preserves per-image gradient magnitude

**Key design decisions**:
1. **Loss = sum, not mean**: Each image's K params are independent. Averaging divides
   gradients by B, reducing effective lr. Sum gives identical gradients to sequential encoding.
2. **Best-param snapshot**: Adam momentum continues pushing params after convergence,
   degrading quality. Snapshotting W_raw at best_loss preserves peak PSNR. Without this,
   PSNR drops from 49.5 → 40 dB.

**Benchmark results (K=500, lr=0.04, ep=3000, ks=32, es=1e-5, soft_clamp, no scheduler):**

V100 (32 GB):
| BS | s/img | Total time | PSNR | Min PSNR | Memory | GPU Util |
|----|-------|------------|------|----------|--------|----------|
| 32 | 0.81 | 25.9s | 49.6 | 46.6 | 0.7 GB | 2% |
| 64 | 0.70 | 45.0s | 49.5 | 45.0 | 1.3 GB | 4% |
| 128 | 0.65 | 83.7s | 49.4 | 41.1 | 2.7 GB | 8% |
| 256 | 0.63 | 162.1s | 49.4 | 40.9 | 5.3 GB | 16% |
| 512 | 0.62 | 319.0s | 49.5 | 42.8 | 10.6 GB | 31% |
| Sequential | 7.57 | - | 49.4 | - | - | - |

A100 (40 GB):
| BS | s/img | Total time | PSNR | Min PSNR | Memory | GPU Util |
|----|-------|------------|------|----------|--------|----------|
| 64 | 0.58 | 36.8s | 49.5 | 46.8 | 1.3 GB | 3% |
| 128 | 0.50 | 64.3s | 49.4 | 41.0 | 2.7 GB | 6% |
| 256 | 0.48 | 123.6s | 49.5 | 42.2 | 5.3 GB | 12% |
| 512 | 0.47 | 242.5s | 49.5 | 42.8 | 10.6 GB | 25% |
| 1024 | 0.47 | 481.0s | 49.4 | 41.6 | 21.1 GB | 50% |
| 2048 | OOM | - | - | - | >40 GB | - |
| Sequential | 4.76 | - | ~49 | - | - | - |

**Speedup**: 12x on V100 (7.57→0.62 s/img), 10x on A100 (4.76→0.47 s/img).

**Why per-image time plateaus at large BS**: GPU compute units saturate around BS=128-256.
Beyond that, doubling BS doubles both work and time. The 12x speedup comes from eliminating
per-image Python overhead — all B images share one set of 3000 optimizer steps.

**Updated production encoding estimates (batched, no compile needed):**

| GPU | BS | s/img | 5000 imgs/shard | 60k on 12 GPUs |
|-----|-----|-------|-----------------|----------------|
| V100 | 512 | 0.62 | 51 min | **51 min** |
| A100 | 1024 | 0.47 | 39 min | **39 min** |
| Mixed (4 A100 + 8 V100) | - | - | - | **~51 min** |

**vs old sequential encoding**: 5.4h → 51 min = **6.4x faster** end-to-end.

---

## Phase 5 Summary

### Improvement Trajectory
| Step | Change | PSNR | Delta | Note |
|------|--------|------|-------|------|
| Old baseline | ks=11, lr=0.01, hard clamp | 35.54 | - | 14% >=40dB, N=50 |
| +soft_clamp | `1-exp(-x)` saturation | 46.70 | **+11.16** | 94% >=40dB, N=50 |
| +lr=0.035 | Higher learning rate | 49.05 | +2.35 | **100% >=40dB**, N=50 |
| +es=1e-6 | Deeper convergence | **50.24** | +1.19 | 100% >=40dB, N=50 |
| | | | **+14.70 total** | |

Note: lr=0.04 showed 51.15 dB on N=20 but regressed to 50.18 dB on N=50 validation
(within noise of lr=0.035 at 50.24 dB). Both LRs are equally good.

### Key Findings

1. **Soft clamp is the single biggest improvement ever found** (+11.16 dB): Changes
   `clamp(0,1)` to `1-exp(-x)` in the renderer. Enables large kernels by preserving
   gradient flow when overlapping Gaussians sum > 1.
2. **ks=32 (full-image kernel) requires soft_clamp**: Without it, 18.87 dB (catastrophic).
   With it, 46.70 dB. The kernel fills the entire 32x32 image.
3. **lr=0.025-0.04 is optimal range**: Best results at lr=0.025 nosched (49.10 dB) and
   lr=0.035 sched (50.24 dB at es=1e-6). U-shaped curve from lr=0.005 to lr=0.05.
4. **Scheduler vs no-scheduler depends on lr**: No-sched helps at lr<=0.03 and lr>=0.04,
   but hurts at lr=0.035. Best to test both.
5. **Softplus and gradient init are negligible**: <0.3 dB with soft_clamp.
6. **K=500 is optimal**: K=400 loses 1.4 dB, K=300 loses 3.9 dB. K>500 still degrades.
7. **3000 epochs sufficient**: 5000 gives +0.06 dB at +60% cost.
8. **Early stop is the main speed lever**: es=1e-4 → 0.93s/img, es=3e-5 → 3.1s/img,
   es=1e-5 → 8.5s/img, es=1e-6 → 17.6s/img (no convergence, full epochs).
9. **torch.compile gives 2x production speedup**: 8.4→4.1 s/img on encode.py (A100).
   33s warmup amortized over 5000+ images. Enabled by default in encode.sh.
10. **Direct Gaussian evaluation**: 3x faster renderer forward pass (2.98x), 1.8x faster
    forward+backward. Eliminates affine_grid+grid_sample bottleneck with analytical
    2x2 covariance inverse and matmul colour mixing.
11. **Every 3 dB of quality roughly doubles encoding time**: Clean exponential tradeoff.
12. **AMP (FP16/BF16) gives no speedup**: Autocast overhead exceeds savings on small tensors.

### Final Encoding Configs

**Option A: Maximum Quality** — maximize PSNR (50+ dB)
```
K=500, lr=0.04, epochs=3000, kernel_size=32, soft_clamp=True, early_stop=1e-6
```
- PSNR: 50.18 ± 3.34 dB (min=41.72, 100% >=40dB, N=50 validated)
- Time: ~18s/img on V100 → ~300 GPU-h → **~25h with 12 GPUs**

**Option B: Best Quality** — high PSNR, practical time (RECOMMENDED)
```
K=500, lr=0.04, epochs=3000, kernel_size=32, soft_clamp=True, early_stop=1e-5, batch_size=512
```
- PSNR: ~49 ± 2 dB (min=~42, 100% >=40dB)
- Time: 0.47-0.62 s/img (batched) → **~51 min with 12 GPUs**

**Option C: High Quality** — 45 dB with fast turnaround
```
K=500, lr=0.04, epochs=3000, kernel_size=32, soft_clamp=True, early_stop=3e-5, compile=True
```
- PSNR: ~45 ± 0.5 dB (min=~43, 100% >=40dB)
- Time: ~1.5s/img on A100 with compile → ~25 GPU-h → **~2h with 12 GPUs**

**Option D: Fast** — 40 dB target, minimum encoding time
```
K=500, lr=0.04, epochs=3000, kernel_size=32, soft_clamp=True, early_stop=1e-4
```
- PSNR: 40.02 ± 0.03 dB (min=40.00, 100% >=40dB)
- Time: ~0.9s/img on A100 → 15 GPU-h → **~1.3h with 12 GPUs**

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| kernel_size | 11 | **32** | Full-image kernel with soft saturation |
| soft_clamp | False | **True** | `1-exp(-x)` preserves gradient flow |
| lr | 0.01 | **0.04** | +2.35 dB over lr=0.01, 6% faster than 0.035 |
| early_stop | 1e-4 | **1e-5** | Allow convergence to 49+ dB (adjustable) |
| recycle_every | 300 | **0** | Disabled; doesn't work for RGB images |

**Encoding command (best quality, batched):**
```bash
# 4 A100 shards (BS=1024) + 8 V100 shards (BS=512) = 12 GPUs
BATCH_SIZE=1024 sbatch --array=0-3 -p gpua100 scripts/cifar10/slurm/encode.sh
sbatch --array=4-11 -p gpu scripts/cifar10/slurm/encode.sh
# Total: ~51 min for all 60k images
```

---

## Known Issues
1. **Dead Gaussian recycling broken for RGB**: `_dead_mask` checks target brightness < 0.05,
   but CIFAR-10 images have color everywhere. 0 dead detected = no recycling.
   Error-based recycling tested but HURTS quality. Best to disable (recycle_every=0).
2. **K>500 degradation**: Optimizer can't handle the landscape. Higher lr partially
   compensates but doesn't close the gap.
3. **Soft clamp changes renderer output**: The diffusion model and sample.py must
   also use soft_clamp=True at inference time for consistency. Need to update
   sample.py/train.py when building the CIFAR-10 diffusion pipeline.

## SLURM Job IDs
| Experiment | Job ID | Partition | Status | Notes |
|------------|--------|-----------|--------|-------|
| Phase 1: K sweep | 137489 | gpua100 | DONE | K=50-500, N=20 |
| Phase 2: comprehensive | 137503-137514 | mixed | DONE | 12 experiments |
| Phase 3: fine K/ep | 137541-137552 | mixed | DONE | 12 experiments |
| Phase 4: validation | 137564-137577 | mixed | DONE | 12 experiments |
| Phase 5a: KS sweep | 137635-137641 | gpua100 | DONE | ks=11-32, N=10 |
| Phase 5a: KS sweep (V100) | 137651-137653 | gpu | DONE | ks=21-32, N=10 |
| Phase 5b: soft_clamp | 137642-137644, 137654-137656 | mixed | DONE | +sc, N=10 |
| Phase 5c: softplus | 137645-137647, 137657-137658 | mixed | DONE | +sc+sp, N=10 |
| Phase 5d: gradient init | 137648-137650 | gpua100 | DONE | +sc+sp+grad, N=10 |
| Phase 5e: 50-img validation | 137674-137685 | mixed | DONE | best configs, N=50 |
| Phase 5f: lr sweep | 137710-137713 | gpua100 | DONE | lr=0.015-0.04, N=50 |
| Phase 5g: epoch sweep | 137714-137716, 137678 | gpu | DONE | ep=1000-5000, N=50 |
| Phase 5h: K sweep | 137717-137719 | gpu | DONE | K=200-400, N=50 |
| Phase 5j: no-scheduler | 137730-137732 | gpu | DONE | lr=0.02-0.04, N=50 |
| Phase 5f: lr fine-tune | 137725-137728 | gpua100 | DONE | lr=0.035-0.05, ep=1500-2000 |
| Phase 5k/l: es sweep | 137758-137759, 137767-137768 | gpua100 | DONE | es=1e-4 to 1e-5, N=20 |
| Phase 5k/l: lr/K/ep at es=1e-6 | 137760-137766, 137769 | gpu | DONE | lr=0.03-0.05, K=300-400 |
| AMP benchmark | 137775, 137782 | gpua100 | DONE | FP16/BF16 vs FP32: no speedup |
| Phase 5k: lr=0.04 sweep | 137784-137789 | mixed | DONE | es sweep + 50-img val: 50.18 dB |
| torch.compile prod test | 137813, 137827 | gpua100 | DONE | 2.0x speedup at es=1e-5 and 1e-6 |
| lr=0.03 50-img val | 137821 | gpua100 | DONE | 50.50 dB at es=1e-6 (best N=50) |
| Renderer benchmark | 137945-137946 | mixed | DONE | Direct eval: 3x fwd, 1.8x fwd+bwd |
| Direct eval microbench | 137962, 137980 | gpu | DONE | 0.603ms vs 1.793ms per call |
| Production bench | 137952, 137979 | gpua100 | DONE | 3.91s/img (compile+direct) |
| Batched encoding bench | 138102-138108 | mixed | DONE | 10-12x throughput improvement |
| V100 max batch bench | 138123 | gpu | RUNNING | BS=768-1536 |
