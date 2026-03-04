# Quick Start: Encode Full MNIST Dataset

Three commands to encode all 70,000 MNIST images into a single HDF5 file.

## Prerequisites

- [x] h5py 3.15.1 installed in `/gpfs/workdir/coessenss/NLS/nls_env_new`
- [x] Access to 4 A100 GPUs via SLURM (partition: `gpua100`)
- [x] ~35 MB disk space for output file

## Commands

### 1. Submit encoding jobs (4 GPUs in parallel)

```bash
cd /gpfs/workdir/coessenss/GaussianDiffusion
sbatch --array=0-3 scripts/slurm_encode_full_mnist.sh
```

**Output:**
```
Submitted batch job 129400
```

Note the job ID (e.g., `129400`). This will be used for the merge step.

### 2. Monitor progress

**Watch live log for shard 0:**
```bash
tail -f logs/encode_full_mnist_shard0.log
```

**Check all shards:**
```bash
watch -n 10 'tail -1 logs/encode_full_mnist_shard*.log'
```

**SLURM queue:**
```bash
squeue -u $USER
```

**Expected progress markers:**
- `[S0 | 1000/17500 | 5.7%]` — Per-image logs (every 25 images)
- `[S0 | PROGRESS | ...]` — Window statistics (every 100 images)
- `[S0 | CKPT]` — Checkpoint marker (every 500 images)

**ETA:** ~16-20 hours per shard (will complete in parallel)

### 3. Merge shards into one file

After all 4 jobs finish (check with `squeue` or logs), merge:

```bash
sbatch scripts/slurm_merge_mnist.sh
```

**OR** submit merge with dependency at the same time as encode jobs:
```bash
JID=$(sbatch --array=0-3 scripts/slurm_encode_full_mnist.sh | awk '{print $NF}')
sbatch --dependency=afterok:${JID} scripts/slurm_merge_mnist.sh
```

**Output:** `data/mnist_gaussians_K70.h5` (~35 MB)

## Verify

Check the merged file exists and has correct shape:

```bash
python -c "
import h5py
with h5py.File('data/mnist_gaussians_K70.h5', 'r') as f:
    print(f'Shape: {f[\"W\"].shape}')
    print(f'Mean PSNR: {f.attrs[\"mean_psnr\"]:.2f} dB')
    print(f'Converged: {f.attrs[\"pct_converged\"]:.1f}%')
"
```

**Expected output:**
```
Shape: (70000, 70, 7)
Mean PSNR: 42.04 dB
Converged: 95.4%
```

## Use in Training

```python
from src.dataset_v2 import GaussianDatasetV2
from torch.utils.data import DataLoader

# Load train split
ds = GaussianDatasetV2("data/mnist_gaussians_K70.h5", split="train")
loader = DataLoader(ds, batch_size=64, shuffle=True)

for W_batch, labels in loader:
    # W_batch: [64, 70, 7] Gaussian parameters
    # Train diffusion model here
    ...
```

## Troubleshooting

**Job killed (exit code 137):**
- OOM — increase `--mem=60G` to `--mem=100G` in `slurm_encode_full_mnist.sh`

**Job timed out (exit code 140):**
- Resume: `RESUME=1 sbatch --array=X scripts/slurm_encode_full_mnist.sh` (replace X with shard ID)

**Merge fails:**
- One or more shards incomplete — check logs and re-run incomplete shards

**Full docs:** `scripts/README_FULL_MNIST_ENCODING.md`
