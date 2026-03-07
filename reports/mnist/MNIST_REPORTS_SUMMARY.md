# MNIST Reports Summary

This file summarizes the two MNIST reports:

- `ENCODER_FIX_REPORT.md`
- `FID_IMPROVEMENT_REPORT.md`

## High-Level Outcome

The MNIST pipeline improved in two major steps:

1. The encoder was fixed by correcting a coordinate inversion bug in the renderer. This raised encoding quality from roughly unusable levels to strong reconstructions.
2. The diffusion model and sampler were then tuned extensively, reducing generation FID from 43.8 to 6.66.

Together, these reports establish a stable MNIST setup with high-quality Gaussian encodings and a much stronger diffusion model.

## 1. Encoder Fix Report: What Changed and Why It Mattered

### Root cause

The encoder problems came from a coordinate inversion bug in the Gaussian renderer. A Gaussian initialized at a target pixel was actually rendered at the mirrored position. Training partially compensated for this, but it made optimization inefficient and misleading.

### Main symptoms before the fix

- Mean PSNR was only about 26 dB
- Around 38-44% of Gaussians became dead on background pixels
- Images did not converge within 3000 epochs
- Recycling made things much worse because it misidentified useful Gaussians as dead

### Fixes applied

Three one-line corrections were made in `src/encode.py`:

- `_init_gaussians`: seed coordinates with the correct sign so the renderer places Gaussians where intended
- `_dead_mask`: check the actual rendered pixel instead of the mirrored one
- `_recycle`: respawn recycled Gaussians at the correct rendered location

The encoder was also simplified to use:

- pure MSE loss
- `CosineAnnealingWarmRestarts`

### Results after the fix

- Mean PSNR improved from about 26 dB to 44.17 dB on 20 MNIST images with `K=70`
- Alive fraction went to 100%
- Convergence became fast and reliable
- `K=70` reached 40 dB on 5-image tests with full convergence and about 790 epochs on average

### Important practical notes

- Existing encoded `.pt` datasets remain compatible
- Existing diffusion model weights remain valid
- `alpha` is currently vestigial in the encoder pipeline and could be removed or properly wired into rendering later
- The report states that all 23 tests passed after the fix

## 2. FID Improvement Report: What Drove the Best Generation Quality

### Final result

FID improved from 43.8 to 6.66 over more than 110 experiments.

### Best overall configuration

- Model: 7.4M parameter DiT (`6B/16H/256`)
- Training: 1500 epochs
- Data: full MNIST
- Sampling: DDIM with 200 steps
- DDIM `eta=0.4`
- CFG weight `w=1.5`

### Biggest findings

- DDIM was the single largest no-retraining improvement, reducing FID from 8.09 to 6.66 on the best model
- CFG helped dramatically, with a clear optimum around `w=1.5`
- The 7.4M model was the sweet spot; larger was not better
- Training beyond about 1500 epochs hurt the 7.4M model due to overfitting
- Batch size 256 worked better than 512
- Default diffusion hyperparameters were validated as best or near-best:
  - `T=200`
  - cosine schedule with `s=0.008`
  - CFG dropout `p=0.1`

### Architecture and training takeaways

- Width mattered more than simply adding depth at larger model sizes
- Small models were surprisingly competitive; the 1.06M model came within about 7% of the best FID
- Full-dataset training consistently improved over partial-data training

### Implementation changes that mattered

- EMA with a power-ramp warmup
- Per-step learning-rate schedule with warmup then cosine decay
- A100-oriented performance improvements including TF32 and `torch.compile`
- Classifier-free guidance support in the transformer
- DDIM sampling with a bug fix to start from `T-1`

## 3. Current Recommended MNIST Setup

For encoding:

- Use the fixed encoder
- Use `K=70`
- Use pure MSE loss
- Use `CosineAnnealingWarmRestarts`

For diffusion training and sampling:

- Prefer the 7.4M DiT (`6B/16H/256`) as the main quality model
- Train on full MNIST
- Use around 1500 epochs for the best 7.4M checkpoint
- Use CFG with `w=1.5`
- Sample with DDIM-200 and `eta=0.4`

## Bottom Line

The encoder report solved the foundational reconstruction issue by fixing mirrored coordinates in the renderer. The FID report then showed that the strongest gains in generation quality came from the right model scale, correct training duration, classifier-free guidance, and especially DDIM sampling. The resulting MNIST pipeline now has strong encodings and a best reported FID of 6.66.
