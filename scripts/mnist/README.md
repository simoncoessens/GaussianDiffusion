# Full MNIST Encoding Pipeline

Complete workflow to encode all 70,000 MNIST images into a single HDF5 file with comprehensive metadata and validation.

## Overview

This pipeline encodes the full MNIST dataset (60k train + 10k test) into Gaussian splatting representations using 4 parallel A100 GPUs via SLURM array jobs. Each image is fitted with K=70 Gaussians using gradient descent with dead-Gaussian recycling.

**Output**: `data/mnist_gaussians_K70.h5` (single HDF5 file, ~35 MB)

**Time estimate**: ~24-30 hours total (4 GPUs in parallel, ~6-8 hours per shard)

## Architecture

```
                ┌─────────────────────────────────────┐
                │  70,000 MNIST images                │
                │  (60k train + 10k test)             │
                └────────────┬────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │  Interleaved sharding   │
                │  (shard s → i % 4 == s) │
                └─────┬───────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
    ┌───▼───┐    ┌───▼───┐    ┌───▼───┐    ┌───▼───┐
    │Shard 0│    │Shard 1│    │Shard 2│    │Shard 3│
    │17.5k  │    │17.5k  │    │17.5k  │    │17.5k  │
    │A100 #0│    │A100 #1│    │A100 #2│    │A100 #3│
    └───┬───┘    └───┬───┘    └───┬───┘    └───┬───┘
        │             │             │             │
        └─────────────┼─────────────┴─────────────┘
                      │
               ┌──────▼──────┐
               │ merge_mnist │
               │ shards.py   │
               └──────┬──────┘
                      │
            ┌─────────▼─────────────┐
            │  mnist_gaussians_K70  │
            │  .h5 (merged + sorted)│
            └───────────────────────┘
```

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `encode_full_mnist.py` | Per-shard encoder with HDF5 incremental writing | ~500 |
| `slurm_encode_full_mnist.sh` | SLURM array job launcher (4 GPUs) | ~100 |
| `merge_mnist_shards.py` | Merge + validate 4 shards into one file | ~400 |
| `slurm_merge_mnist.sh` | SLURM merge job (CPU only) | ~50 |
| `dataset_v2.py` | PyTorch Dataset for HDF5 backend | ~200 |

## HDF5 Schema

**Per-shard files** (`data/mnist_gaussians_shards/shard_{0..3}.h5`):
```
/W            [N_shard, 70, 7]  float32  Gaussian parameters (chunked, lz4)
/labels       [N_shard]         uint8    Digit class 0–9
/psnr         [N_shard]         float32  Reconstruction quality (dB)
/n_epochs     [N_shard]         int32    Epochs taken
/converged    [N_shard]         bool     Reached early_stop threshold?
/n_dead       [N_shard]         int8     Dead Gaussians at end
/orig_split   [N_shard]         uint8    0=train, 1=test
/orig_index   [N_shard]         int32    Index within MNIST split
/encode_time  [N_shard]         float32  Wall-clock seconds per image
/global_idx   [N_shard]         int32    Global index (0–69999)
/done         [N_shard]         bool     Row written? (for resume)
```

**Merged file** (`data/mnist_gaussians_K70.h5`): Same datasets, sorted by global_idx, plus root-level attrs:
```python
attrs = {
    "K": 70,
    "lr": 5e-3,
    "epochs_max": 3000,
    "early_stop": 1e-4,  # ~40 dB PSNR
    "encoder_version": "2.0",
    "coord_fix_applied": True,
    "mean_psnr": 42.3,  # example
    "pct_converged": 95.2,
    ...
}
```

## Workflow

### Step 1: Submit encoding jobs

Fresh encoding (4 shards in parallel):
```bash
sbatch --array=0-3 scripts/slurm_encode_full_mnist.sh
```

After submission, note the job ID (e.g., `129400`):
```
Submitted batch job 129400
```

### Step 2: Monitor progress

**Live tail of a single shard:**
```bash
tail -f logs/encode_full_mnist_shard0.log
```

**Check all shard logs:**
```bash
ls -lh logs/encode_full_*.out logs/encode_full_mnist_shard*.log
```

**Decode log output:**
```
[2026-03-04 20:15:42] [S0 | 1000/17500 | 5.7%]  idx=4000  digit=7  PSNR=42.1dB  ep=789  dead=0/70  t=3.2s
[2026-03-04 20:16:10] [S0 | PROGRESS | 1100/17500 | 6.3%]  window: mean=41.8dB  conv=94%  0.312img/s  ETA=14h23m  cumul_mean=41.6dB
[2026-03-04 20:30:00] [S0 | CKPT] 1500/17500 written.  Elapsed=1h23m.  ETA=13h45m
```

**SLURM queue status:**
```bash
squeue -u $USER
```

**Check GPU utilization:**
```bash
sstat -j 129400.0 --format=AveCPU,AveRSS,MaxRSS
```

### Step 3: Resume after crash (optional)

If a shard fails (e.g., timeout, OOM, node failure), resume from checkpoint:
```bash
# Resume shard 2 only
RESUME=1 sbatch --array=2 scripts/slurm_encode_full_mnist.sh
```

The `done` boolean dataset tracks per-image completion, so the script will skip already-encoded rows.

### Step 4: Merge shards

After all 4 encode jobs finish, merge into one file:
```bash
# Manual submission (after checking logs)
sbatch scripts/slurm_merge_mnist.sh

# OR: Automatic dependency (submit at same time as encode jobs)
JID=$(sbatch --array=0-3 scripts/slurm_encode_full_mnist.sh | awk '{print $NF}')
sbatch --dependency=afterok:${JID} scripts/slurm_merge_mnist.sh
```

Output will be written to `data/mnist_gaussians_K70.h5` with full validation report.

### Step 5: Validate output

The merge script prints a comprehensive report:
```
PSNR Summary
────────────────────────────────────────────────────────────────────
  P10  : 38.12 dB
  P25  : 40.45 dB
  P50  : 42.18 dB
  P75  : 43.89 dB
  P90  : 45.23 dB
  Min  : 32.89 dB
  Mean : 42.04 dB  (±2.34)
  Max  : 48.56 dB

  below 30 dB  : 12 (0.02%)
  below 35 dB  : 234 (0.33%)
  converged    : 66789 (95.4%)

Per-digit PSNR
────────────────────────────────────────────────────────────────────
  Digit       N     Mean     Std      Min      Max    <30dB
  ──────  ──────  ───────  ──────  ───────  ───────  ──────
      0    6903    42.12    2.45    35.23    47.89       2
      1    7877    43.01    2.11    36.78    48.34       1
      ...

  File size    : 34.7 MB
  Est. load    : 0.1s  (preload=True, ~500 MB/s)
  Output       : /gpfs/workdir/coessenss/GaussianDiffusion/data/mnist_gaussians_K70.h5
```

A JSON summary is also saved to `data/mnist_gaussians_K70_summary.json`.

## Using the Encoded Dataset

### Load with GaussianDatasetV2

```python
from src.dataset_v2 import GaussianDatasetV2
from torch.utils.data import DataLoader

# Load full dataset (preloaded into RAM)
ds = GaussianDatasetV2("data/mnist_gaussians_K70.h5")
print(ds)  # GaussianDatasetV2(n=70000, K=70, preload=True, ...)

# Filter by split
ds_train = GaussianDatasetV2("data/mnist_gaussians_K70.h5", split="train")   # 60k
ds_test  = GaussianDatasetV2("data/mnist_gaussians_K70.h5", split="test")    # 10k

# Filter by PSNR
ds_clean = GaussianDatasetV2("data/mnist_gaussians_K70.h5", min_psnr=35.0)

# Filter by digits
ds_digits = GaussianDatasetV2("data/mnist_gaussians_K70.h5", digits=[0, 1, 2])

# Lazy mode (low memory, slower __getitem__)
ds_lazy = GaussianDatasetV2("data/mnist_gaussians_K70.h5", preload=False)

# DataLoader (standard PyTorch)
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
for W_batch, labels_batch in loader:
    # W_batch: [64, 70, 7]  Gaussian params
    # labels_batch: [64]    digit classes
    ...
```

### PSNR statistics

```python
ds = GaussianDatasetV2("data/mnist_gaussians_K70.h5", split="train")
stats = ds.get_psnr_stats()
print(stats)
# {'n': 60000, 'mean': 42.1, 'std': 2.3, 'min': 33.2, 'max': 48.5,
#  'pct_below_30': 0.01, 'pct_below_35': 0.29}
```

### Direct HDF5 access (advanced)

```python
import h5py

with h5py.File("data/mnist_gaussians_K70.h5", "r") as hf:
    # Load specific indices
    W = hf["W"][[0, 1, 2]]  # shape [3, 70, 7]
    labels = hf["labels"][[0, 1, 2]]

    # Metadata
    print(hf.attrs["mean_psnr"])
    print(hf.attrs["pct_converged"])

    # Filter by PSNR in HDF5
    psnr = hf["psnr"][:]
    high_quality = psnr >= 40.0
    W_hq = hf["W"][high_quality]
```

## Encoder Hyperparameters

Default settings (can override via environment variables in SLURM script):

| Param | Value | Description |
|-------|-------|-------------|
| `K` | 70 | Number of Gaussians per image |
| `lr` | 5e-3 | Adam learning rate |
| `epochs` | 3000 | Max epochs per image |
| `early_stop` | 1e-4 | MSE threshold (~40 dB PSNR) |
| `kernel_size` | 11 | Gaussian kernel size for rendering |
| `recycle_every` | 300 | Dead-Gaussian recycling interval |

Override example:
```bash
K=100 EPOCHS=5000 sbatch --array=0-3 scripts/slurm_encode_full_mnist.sh
```

## Checkpointing & Resume

The encoder writes to HDF5 incrementally (every 10 images) and tracks completion via a `done` boolean dataset. If a job crashes:

1. The shard file remains valid (all written rows are intact)
2. Re-run the same shard with `--resume`:
   ```bash
   RESUME=1 sbatch --array=2 scripts/slurm_encode_full_mnist.sh
   ```
3. The script will read the `done` array and skip already-encoded images
4. Encoding continues from the first incomplete row

**No data loss** — the HDF5 file is flushed every 10 images, so even hard crashes lose at most 10 images worth of work.

## Interleaved Sharding Rationale

MNIST is sorted by digit class (0–9). Contiguous sharding (shard 0 = indices 0–17499, etc.) would give each shard an unbalanced class distribution. **Interleaved sharding** ensures each shard processes a representative sample:

- Shard 0: indices 0, 4, 8, 12, … (every 4th index)
- Shard 1: indices 1, 5, 9, 13, …
- Shard 2: indices 2, 6, 10, 14, …
- Shard 3: indices 3, 7, 11, 15, …

After merging, the final HDF5 is sorted by `global_idx` (0 to 69999), restoring the original MNIST order.

## Troubleshooting

### Job killed with exit code 137 (OOM)
- Increase `--mem` in SLURM script (default: 60G)
- Check if another process is using GPU memory: `nvidia-smi`

### Job timeout (48h limit)
- Resume with `RESUME=1 sbatch --array=X`
- Or increase `--time` (max: depends on partition)

### Low PSNR warnings
- Check logs for images with PSNR < 30 dB (marked with `← below 30dB floor`)
- Inspect with: `grep "WARN" logs/encode_full_mnist_shard*.log`
- This is expected for ~0.1% of images (complex textures, near-blank digits)

### h5py import error
- Ensure NLS env is activated: `/gpfs/workdir/coessenss/NLS/nls_env_new/bin/python`
- h5py 3.15.1 is installed in this env

### Merge fails with "shard incomplete"
- One or more encode jobs didn't finish
- Check SLURM logs: `tail logs/encode_full_129400_*.out`
- Re-run incomplete shards before merging

## Performance Benchmarks

Based on SLURM job 129389 (encoder research report):
- **Mean encode time**: 3.2 s/image (K=70, early_stop=1e-4)
- **Per-shard estimate**: 17500 images × 3.2 s = 15.6 hours
- **4 shards in parallel**: ~16 hours total (assuming balanced convergence)
- **Merge time**: ~2 minutes (70k images, 35 MB file)

**Total pipeline**: ~16-20 hours wall-clock (encode) + 2 min (merge)

## Next Steps After Encoding

1. **Train diffusion model** on the encoded representations:
   ```python
   from src.dataset_v2 import GaussianDatasetV2
   ds = GaussianDatasetV2("data/mnist_gaussians_K70.h5", split="train")
   # Train GaussianTransformer with DDPM on ds
   ```

2. **Evaluate reconstruction quality**:
   - Compare against pixel-space baseline (FID, IS, KID)
   - Compute per-digit PSNR distributions
   - Visualize latent space with PCA/t-SNE

3. **Ablation studies**:
   - Re-encode with K=50 or K=100 (set `K=` in SLURM script)
   - Test different `early_stop` thresholds (trade quality vs speed)
   - Compare MSE vs L1+SSIM loss (requires encoder code change)

4. **Scale to larger datasets** (CIFAR-10, Sprites):
   - Update `encode_full_mnist.py` to load CIFAR-10 via torchvision
   - Adjust K (recommend K=200 for 32×32 CIFAR)
   - Same workflow: 4 shards → merge

## Contact

For issues or questions about this pipeline:
- Check logs: `logs/encode_full_mnist_shard*.log`
- Review encoder report: `reports/encoder_research/main.pdf` (compile LaTeX locally)
- See memory notes: `~/.claude/projects/.../MEMORY.md`
