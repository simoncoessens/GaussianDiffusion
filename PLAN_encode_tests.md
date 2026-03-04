# Plan: Rigorous Testing of the Gaussian Encoding Pipeline

## What we're testing

Five components, each gets its own test file:

```
tests/
  test_renderer.py       ← generate_2D_gaussian_splatting
  test_ssim.py           ← _gaussian_kernel + ssim_loss
  test_encode.py         ← _init_gaussians, _to_physical, _render, encode_image (replaces current)
  test_dataset.py        ← GaussianDataset
  test_normalize.py      ← normalize_parameters, denormalize_parameters
```

---

## Bugs found during analysis (before writing a single test)

### Bug 1 — Init colour/coord mismatch [HIGH]
In `_init_gaussians`, colour and x/y are seeded with *physical* values (pixel intensities
in [0,1], coordinates in [-1,1]), but they're stored as *raw* values and later passed through
`sigmoid()` / `tanh()` in `_to_physical()`. This means:

- A pixel with intensity 0.8 → stored raw = 0.8 → physical colour = sigmoid(0.8) = 0.69 ≠ 0.8
- A pixel at coordinate +0.9 → stored raw = 0.9 → physical x = tanh(0.9) = 0.716 ≠ 0.9

Fix: initialize raw colour = logit(pixel), raw x/y = atanh(coord).

### Bug 2 — Kernel div-by-zero on near-zero sigma [MEDIUM]
In `generate_2D_gaussian_splatting`, line 87-88:
```python
kernel_max = kernel.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
kernel_normalized = kernel / kernel_max   # NaN if kernel_max == 0
```
If sigma is very small (not achievable via sigmoid, but could appear with extreme negative raw),
the kernel can be numerically zero, giving NaN. No guard exists.

### Bug 3 — `torch.inverse` is deprecated [LOW]
`torch.inverse(covariance)` is deprecated for batched tensors in PyTorch ≥ 1.9.
Should be `torch.linalg.inv(covariance)`.

### Bug 4 — Dead determinant guard [informational]
The `if (determinant <= 0).any()` branch can never trigger given our parameterisation:
det = σx² σy² (1 - ρ²) > 0 always, since sigmoid outputs ∈ (0,1) and |tanh| < 1.
The epsilon nudge is dead code. Worth a test to document this.

### Alpha is vestigial [design issue, not a bug]
`_render()` never passes `alpha` to the renderer. By design (additive compositing),
but it wastes one dimension of the latent space trained by the diffusion model.
We document this with a test but do not fix it here.

---

## Test file specifications

### 1. `tests/test_renderer.py` — 12 tests

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_output_shape` | basic shape contract | shape == (H, W, 1) |
| `test_output_range` | values clamped correctly | all values ∈ [0, 1] |
| `test_centered_gaussian_peaks_at_center` | coord=(0,0) → peak near image centre | argmax within ±2 px of centre |
| `test_position_effect` | coord shift moves peak | argmax shifts ≥ 3 px when coord changes by 0.3 |
| `test_sigma_effect` | larger σ → more diffuse | image entropy increases with σ |
| `test_colour_proportional` | colour scales output linearly | render(colour=0.5) ≈ 0.5 * render(colour=1.0) before clamp |
| `test_additive_compositing` | two non-overlapping Gaussians sum | sum of two single-G renders ≈ joint render (before clamp) |
| `test_gradient_flows` | backprop succeeds | loss.backward() executes, no RuntimeError, sigma.grad is not None |
| `test_near_singular_rho` | rho = ±0.99 → no NaN/Inf | isfinite(output).all() |
| `test_small_sigma` | sigma = 0.01 → no crash | completes without error; **may expose Bug 2** |
| `test_kernel_size_equals_image_size` | kernel_size == image_size is valid | no error, shape correct |
| `test_kernel_larger_than_image_raises` | kernel > image → ValueError | pytest.raises(ValueError) |

### 2. `tests/test_ssim.py` — 8 tests

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_identical_images_zero_loss` | ssim_loss(x, x) = 0 | loss < 1e-5 |
| `test_loss_in_range` | loss bounded | loss ∈ [0, 1] for any input |
| `test_black_vs_white_high_loss` | maximally different → loss ≈ 1 | loss > 0.9 |
| `test_noisy_image_positive_loss` | noise increases loss | ssim_loss(x+ε, x) > 0 |
| `test_symmetry` | ssim_loss(a,b) == ssim_loss(b,a) | |abs difference| < 1e-6 |
| `test_no_nan_uniform_image` | all-zeros / all-ones → no NaN | isfinite(loss) |
| `test_gradient_flows` | backprop through SSIM | loss.backward() ok, grad not None |
| `test_shape_agnostic` | [H,W] and [1,H,W] give same result | values match |

### 3. `tests/test_encode.py` — 15 tests (replaces current file)

**Unit tests for sub-functions:**

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_init_output_shape` | _init_gaussians returns [K, 7] | shape == (K, 7) |
| `test_init_sigma_raw` | raw sigma columns == -1.5 | all values == -1.5 |
| `test_init_rho_raw_zero` | raw rho column == 0 | all values == 0.0 |
| `test_init_colour_bug` | **exposes Bug 1**: physical colour ≠ pixel value | `sigmoid(W[:,4])` ≠ `pixel_values` → **documents the bug** |
| `test_init_coord_bug` | **exposes Bug 1**: physical coords ≠ seeded coords | `tanh(W[:,5:7])` ≠ `seeded_xy` → **documents the bug** |
| `test_init_black_image_safe` | all-zeros image → no error, uniform sampling | completes without error |
| `test_to_physical_ranges` | all outputs in correct physical ranges | sigma∈(0,1), rho∈(-1,1), colour∈(0,1), x,y∈(-1,1) |
| `test_to_physical_gradient` | backprop through _to_physical | .backward() ok, grad not None |
| `test_alpha_is_vestigial` | alpha column has no effect on rendered output | render(W with alpha=0) == render(W with alpha=1) |

**Integration tests for encode_image:**

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_encode_output_shape` | [K, 7] physical params | shape == (K, 7) |
| `test_encode_params_valid` | all params in-range after encode | sigma>0, |rho|<1, etc. |
| `test_encode_no_nan` | no NaN/Inf in output | isfinite(W).all() |
| `test_encode_loss_decreases` | optimization makes progress | loss at epoch 50 < loss at epoch 1 |
| `test_encode_early_stop` | stops before max_epochs when threshold met | run 1000 epochs, easy image → N_actual < 1000 |
| `test_encode_round_trip_psnr` | rendered reconstruction quality | PSNR > 15 dB (short run, K=20, 200 epochs) |

### 4. `tests/test_dataset.py` — 9 tests

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_loads_all_valid_files` | N files with correct K → N samples | len(dataset) == N |
| `test_sample_shape` | single item shape | shape == (K, 7) |
| `test_full_data_shape` | stacked tensor shape | dataset.data.shape == (N, K, 7) |
| `test_filters_wrong_K` | files with K'≠K are silently skipped | len == only valid files |
| `test_raises_on_empty_dir` | no .pt files → ValueError | pytest.raises(ValueError) |
| `test_missing_W_key_raises_or_skips` | .pt without 'W' key | doesn't silently corrupt data |
| `test_getitem_returns_tensor` | dataset[i] is a FloatTensor | isinstance check |
| `test_values_preserved` | loaded values match saved values | max(abs(loaded - saved)) < 1e-7 |
| `test_sorted_order_deterministic` | same dir → same sample ordering every run | two loads give identical data |

### 5. `tests/test_normalize.py` — 7 tests

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| `test_normalize_to_minus1_plus1` | in-range params map to [-1, 1] | output ∈ [-1, 1] |
| `test_round_trip_identity` | normalize → denormalize = identity | max abs error < 1e-6 |
| `test_boundary_values` | min/max of each param → exactly -1/+1 | output == ±1.0 |
| `test_degenerate_range` | min==max → normalize outputs 0, denormalize outputs min | values as expected |
| `test_gradient_flows_normalize` | backprop through normalize | .backward() ok |
| `test_shape_preserved` | works on [K, 7] and [N, K, 7] | output.shape == input.shape |
| `test_denormalize_reverses_normalize` | on real PARAM_RANGES with random valid data | round-trip error < 1e-6 |

---

## After tests are written: bugs to fix

Once we have green tests (except the intentional `[EXPOSES BUG]` ones):

1. **Fix Bug 1** — change `_init_gaussians` to use `torch.logit(colours)` and `torch.atanh(xs.clamp(-1+1e-6, 1-1e-6))` for proper raw initialization. Re-run `test_init_colour_bug` and `test_init_coord_bug`, they should now *pass* (meaning the bug is gone).

2. **Fix Bug 3** — replace `torch.inverse(covariance)` with `torch.linalg.inv(covariance)`.

3. **Fix Bug 2 guard** — add `kernel_max = kernel_max.clamp(min=1e-8)` before division. Test `test_small_sigma` should pass.

---

## Verification sequence

```bash
# 1. Run all tests (expect Bug 1 tests to fail = expected)
pytest tests/ -v

# 2. After fixes, all tests should pass
pytest tests/ -v --tb=short

# 3. Smoke-test encode CLI on a real MNIST image
python -m src.encode \
    --data_dir /gpfs/workdir/coessenss/gsplat/data/datasets/MNIST/raw_images/ \
    --out_dir data/mnist_gaussian_representations/ \
    --num_gaussians 70 --epochs 10

# 4. Smoke-test dataset loading
python -c "from src.dataset import GaussianDataset; d=GaussianDataset('data/mnist_gaussian_representations/'); print(d.data.shape)"
```

---

## Runtime estimate (CPU)

| File | Tests | Estimated total |
|------|-------|----------------|
| test_renderer.py | 12 | ~20s |
| test_ssim.py | 8 | ~2s |
| test_encode.py | 15 | ~60s (encode tests are slow) |
| test_dataset.py | 9 | ~2s |
| test_normalize.py | 7 | ~1s |
| **Total** | **51** | **~85s** |
