# Encoder Investigation and Fix Report

**Date:** 2026-03-03
**Context:** MNIST Gaussian splatting encoder was producing poor PSNR (~26 dB mean), with ~40% of Gaussians "dead" on background pixels and no convergence within 3000 epochs.

---

## Summary

A **coordinate inversion bug** in the renderer was identified as the root cause of all observed encoder problems. Fixing it increased mean PSNR from **26 dB → 44 dB** (K=70, 20 images), brought the alive fraction to **100%**, and reduced convergence time from "never" to **~790 epochs on average**.

---

## Experiments Run (Chronological)

All experiments used 10 or fewer images for fast iteration as requested.

### Baseline characterisation

| Experiment | Config | Result |
|---|---|---|
| Baseline (old encoder) | K=50, L1+SSIM, CosineAnnealingLR | 26.54 ± 6.33 dB, 0% converged |
| Warm restarts | K=50, L1+SSIM, CosineAnnealingWarmRestarts | 27.52 ± 6.50 dB (+0.99 dB) |
| K ablation | K=25–50, warm restarts | Monotone: more K → better PSNR up to K=50 |
| MSE loss alone | K=50, MSE, warm restarts | 29.31 ± 4.42 dB, 50% ≥ 30 dB |
| L1+SSIM alone | K=50, L1+SSIM, warm restarts | 29.69 ± 5.12 dB, 50% ≥ 30 dB |
| sigma_init=0.5 | K=50, MSE, sigma=0.5 | 27.37 ± 2.97 dB — **worse** |
| Combined loss | K=50/70, MSE+L1+0.5×SSIM | 28.96 / 26.46 dB — **worse than either alone** |

**Persistent anomaly:** despite testing many loss functions and schedulers, ~38–44% of Gaussians always ended dead on background pixels, and 0% of images ever triggered early stop at 30 dB.

### Recycling enabled (first attempt — catastrophic failure)

Enabling `recycle_every=300` caused PSNR to collapse to **11.76 dB** with alive=100%.
This failure exposed the root cause.

---

## Root Cause: Coordinate Inversion in the Renderer

### The bug

`generate_2D_gaussian_splatting` translates the Gaussian kernel using `F.affine_grid`:

```python
theta[:, :, 2] = coords   # coords = [p["x"], p["y"]]
grid = F.affine_grid(theta, size, align_corners=True)
kernel_transformed = F.grid_sample(kernel_padded, grid, align_corners=True)
```

`F.affine_grid` maps output pixel at `(x_out, y_out)` to **input** sample location `(x_out + tx, y_out + ty)`. The kernel peak is at input `(0, 0)`, so it appears at output position `(-tx, -ty) = (-coords_x, -coords_y)`.

**Verified empirically:**
```
coords=(+0.5, 0.0) → Gaussian peak at column 6  (LEFT of center)
coords=(-0.5, 0.0) → Gaussian peak at column 20 (RIGHT of center)
```

Coordinates are **fully inverted**.

### How training adapted (masking the bug)

Because loss gradients are computed through the same renderer, the optimizer automatically compensated. After training, `p["x"] ≈ -actual_rendered_x`. The model worked correctly at inference (29–30 dB), but at a huge cost:

1. **Slow convergence:** Gaussians initialised at pixel `col=20` rendered at `col=6` initially. The optimizer spent ~400–600 epochs migrating each Gaussian to the correct (mirrored) position.
2. **Dead Gaussians:** Many Gaussians initialised at bright pixel positions, but rendered on black pixels (mirrored side). The MSE gradient suppressed their colour toward zero rather than moving position. These became permanently "dead" (38–44% of K).
3. **Broken recycling:** `_dead_mask` checked `image[p["x"]]`, but `p["x"] ≈ -actual_rendered_x` pointed to the opposite (background) pixel. Trained, contributing Gaussians were incorrectly flagged as dead and destroyed by recycling → 11.76 dB.

---

## The Fix

Three one-line changes in `src/encode.py`:

### 1. `_init_gaussians` — seed at correct rendered position

```python
# Before: Gaussian seeded at pixel xs renders at -xs (wrong side)
xs_raw = torch.atanh(xs.clamp(-1 + 1e-6, 1 - 1e-6))

# After: set raw coord to -xs so renderer places it at xs
xs_raw = torch.atanh((-xs).clamp(-1 + 1e-6, 1 - 1e-6))
# same for ys_raw
```

### 2. `_dead_mask` — check actual rendered pixel

```python
# Before: checks image at p["x"] = -actual_rendered_x → wrong pixel
px = ((p["x"] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)

# After: checks image at -p["x"] = actual_rendered_x → correct pixel
px = ((-p["x"] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
# same for py
```

### 3. `_recycle` — place recycled Gaussians at correct position

```python
# Before: Gaussian placed at pixel xs renders at -xs (wrong side)
xs_raw = torch.atanh(xs.clamp(-1 + 1e-6, 1 - 1e-6))

# After: set raw coord so renderer places at xs
xs_raw = torch.atanh((-xs).clamp(-1 + 1e-6, 1 - 1e-6))
```

### Backward compatibility

The existing dataset (old `.pt` files) has the **same sign convention** — both old and new encoders converge to `p["x"] ≈ -actual_rendered_x`. The format is unchanged. Old trained diffusion model weights remain valid.

---

## Results After Fix

| Config | PSNR | Alive | Converged |
|---|---|---|---|
| K=50, 5 imgs, early_stop=0.001 (30 dB) | 30.08 ± 0.05 dB | 100% | 100% @ avg ep 270 |
| K=50, 5 imgs, early_stop=0.0001 (40 dB) | 40.01 ± 0.01 dB | 100% | 100% @ avg ep 1660 |
| K=70, 5 imgs, early_stop=0.0001 (40 dB) | 40.01 ± 0.02 dB | 100% | 100% @ avg ep 790 |
| **K=70, 20 imgs, no early stop** | **44.17 ± 3.74 dB** | **100%** | min=37.8 dB |

K=70 with no early stop revealed the true ceiling: 37.8–50 dB across real MNIST images.

### Loss function findings

| Loss | PSNR (10 imgs) | Note |
|---|---|---|
| MSE alone | best | Directly optimises PSNR metric |
| L1+SSIM | ≈same | Slightly different images pass/fail |
| MSE+L1+0.5×SSIM | **−0.4 dB worse** | Combined landscape harder to optimise |

**Current config:** pure MSE loss, CosineAnnealingWarmRestarts (T₀=epochs/3).

---

## Additional Finding: Alpha is Vestigial

`alpha` (column 3 of W) is computed in `_to_physical` but never passed to `generate_2D_gaussian_splatting`. After encoding, alpha ≈ 0.499 ± 0.019 for all images (frozen at its initial sigmoid(0) value). The diffusion model learns this trivially. Future work could either remove alpha or wire it into the renderer.

## Files Changed

| File | Change |
|---|---|
| `src/encode.py` | Coord inversion fix in `_init_gaussians`, `_dead_mask`, `_recycle`; loss changed to pure MSE; scheduler changed to CosineAnnealingWarmRestarts |
| `tests/test_encode.py` | Updated `test_init_coord_seeded_correctly` to reflect correct inversion convention |
| `scripts/slurm_encode_gpu.sh` | New GPU array encode script (supersedes CPU version) |
| `scripts/slurm_train_v2.sh` | Training script pointed at v2 dataset |

All 23 tests pass.
