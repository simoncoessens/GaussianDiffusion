"""
encode_celeba64.py -- Encode CelebA images into Gaussian splatting representations.

Each shard processes a subset of images determined by interleaved sharding:
  shard s -> global indices i where i % n_shards == s

Global index layout (deterministic, reproducible):
  0      - 162769 : CelebA train set
  162770 - 182636 : CelebA val set
  182637 - 202598 : CelebA test set

Preprocessing: center crop 140x140 -> resize 64x64 (standard CelebA-64)

Output: data/celeba64/shards/shard_{s}.h5
  /W            float32  [N_shard, K, 9]  Gaussian parameters (RGB)
  /labels       uint8    [N_shard]        dummy label (0 for all, unconditional)
  /psnr         float32  [N_shard]        final PSNR (dB)
  /n_epochs     int32    [N_shard]        max epochs
  /converged    bool     [N_shard]        reached early_stop_threshold?
  /orig_split   uint8    [N_shard]        0=train, 1=val, 2=test
  /orig_index   int32    [N_shard]        index within CelebA split
  /encode_time  float32  [N_shard]        wall-clock seconds per image
  /global_idx   int32    [N_shard]        global index (0-202598)
  /done         bool     [N_shard]        True = row is written (for resume)

Submit: sbatch --array=0-19 scripts/celeba64/slurm/encode.sh
Resume: add RESUME=1 env var
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encode import encode_batch  # noqa: E402


def _psnr(mse: float) -> float:
    if mse < 1e-10:
        return 100.0
    return min(100.0, 10.0 * math.log10(1.0 / mse))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _eta_str(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "unknown"
    rate = done / elapsed
    remaining = (total - done) / rate
    return _fmt_duration(remaining)


def load_celeba64(data_dir: Path, crop_size: int = 140, image_size: int = 64):
    """Load all CelebA images as preprocessed tensors, with split info.

    Returns:
        images: np.ndarray [N, 3, 64, 64] uint8->float will be done per batch
        splits: np.ndarray [N] uint8 (0=train, 1=val, 2=test)
    """
    img_dir = data_dir / "celeba" / "img_align_celeba"
    partition_file = data_dir / "celeba" / "list_eval_partition.txt"

    if not img_dir.exists():
        sys.exit(f"[FATAL] Image directory not found: {img_dir}\n"
                 f"Run: python scripts/celeba64/download.py")

    # Load partition info
    split_map = {}
    if partition_file.exists():
        with open(partition_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    split_map[parts[0]] = int(parts[1])

    # Build sorted list of all image filenames
    filenames = sorted(f.name for f in img_dir.iterdir() if f.suffix == ".jpg")
    n_total = len(filenames)
    print(f"  Found {n_total} images in {img_dir}")

    if n_total < 202599:
        print(f"  [WARN] Expected 202599 images, found {n_total}")

    # Preprocessing transform: center crop -> resize -> tensor
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),  # [0, 1] float32
    ])

    # Load all images into memory as uint8 [N, 3, H, W]
    from PIL import Image
    images_list = []
    splits = np.zeros(n_total, dtype=np.uint8)

    for i, fname in enumerate(filenames):
        img = Image.open(img_dir / fname).convert("RGB")
        img_tensor = transform(img)  # [3, 64, 64] float32 [0, 1]
        images_list.append((img_tensor.numpy() * 255).astype(np.uint8))
        splits[i] = split_map.get(fname, 0)

        if (i + 1) % 50000 == 0:
            print(f"    Loaded {i+1}/{n_total} images ...", flush=True)

    images = np.stack(images_list)  # [N, 3, 64, 64] uint8
    n_train = int((splits == 0).sum())
    n_val = int((splits == 1).sum())
    n_test = int((splits == 2).sum())
    print(f"  Split: train={n_train}, val={n_val}, test={n_test}")
    return images, splits


def main():
    parser = argparse.ArgumentParser(
        description="Encode CelebA-64 images to Gaussian representations (one HDF5 shard)."
    )
    parser.add_argument("--shard_id", type=int, required=True, help="Shard index (0 to n_shards-1)")
    parser.add_argument("--n_shards", type=int, default=20, help="Total number of shards")
    parser.add_argument("--data_dir", type=str, default="data/celeba64/raw",
                        help="Root dir with celeba/ subdirectory")
    parser.add_argument("--out_dir", type=str, default="data/celeba64/shards",
                        help="Directory for shard HDF5 files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from existing shard file")
    # Encoder hyperparameters
    parser.add_argument("--K", type=int, default=1000, help="Number of Gaussians")
    parser.add_argument("--epochs", type=int, default=3000, help="Max epochs per image")
    parser.add_argument("--lr", type=float, default=4e-2, help="Learning rate")
    parser.add_argument("--early_stop", type=float, default=1e-5, help="MSE early-stop (~50dB)")
    parser.add_argument("--kernel_size", type=int, default=32, help="Gaussian kernel size")
    parser.add_argument("--sigma_activation", type=str, default="sigmoid",
                        choices=["sigmoid", "softplus"], help="Sigma activation function")
    parser.add_argument("--init_mode", type=str, default="brightness",
                        choices=["brightness", "gradient"], help="Init position sampling mode")
    parser.add_argument("--soft_clamp", action="store_true",
                        help="Use soft saturation (1-exp(-x)) instead of hard clamp")
    parser.add_argument("--no_scheduler", action="store_true",
                        help="Disable CosineAnnealingWarmRestarts (constant lr)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Images per batch (V100: 256, A100: 512)")
    parser.add_argument("--crop_size", type=int, default=140,
                        help="Center crop size before resize to 64x64")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        sys.exit("[FATAL] h5py not installed.")

    assert 0 <= args.shard_id < args.n_shards

    s = args.shard_id
    out_dir = Path(PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{s}.h5"
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"encode_celeba64_shard{s}.log"

    # Load CelebA
    print(f"[{_now()}] Loading CelebA-64...", flush=True)
    data_dir = PROJECT_ROOT / args.data_dir
    images, splits = load_celeba64(data_dir, crop_size=args.crop_size, image_size=64)
    n_total = len(splits)

    my_global_indices = [i for i in range(n_total) if i % args.n_shards == s]
    n_shard = len(my_global_indices)

    # Param dim for RGB: 9 (sigma_x, sigma_y, rho, alpha, r, g, b, x, y)
    param_dim = 9

    header = (
        f"\n{'='*70}\n"
        f"  encode_celeba64.py  --  shard {s}/{args.n_shards}\n"
        f"  Started : {_now()}\n"
        f"  Device  : {args.device}\n"
        f"  K={args.K}  lr={args.lr}  epochs={args.epochs}  ks={args.kernel_size}"
        f"  batch_size={args.batch_size}  crop={args.crop_size}\n"
        f"  Shard   : {n_shard} images ({n_shard/n_total*100:.1f}%)\n"
        f"  Output  : {shard_path}\n"
        f"{'='*70}\n"
    )
    print(header, flush=True)

    done_set: set = set()

    def _create_shard(f):
        f.create_dataset("W", shape=(n_shard, args.K, param_dim), dtype="float32",
                         chunks=(min(256, n_shard), args.K, param_dim), compression="gzip")
        f.create_dataset("labels", shape=(n_shard,), dtype="uint8")
        f.create_dataset("psnr", shape=(n_shard,), dtype="float32")
        f.create_dataset("n_epochs", shape=(n_shard,), dtype="int32")
        f.create_dataset("converged", shape=(n_shard,), dtype="bool")
        f.create_dataset("orig_split", shape=(n_shard,), dtype="uint8")
        f.create_dataset("orig_index", shape=(n_shard,), dtype="int32")
        f.create_dataset("encode_time", shape=(n_shard,), dtype="float32")
        f.create_dataset("global_idx", shape=(n_shard,), dtype="int32")
        f.create_dataset("done", shape=(n_shard,), dtype="bool",
                         data=np.zeros(n_shard, dtype=bool))
        f["global_idx"][:] = np.array(my_global_indices, dtype=np.int32)
        f.attrs["shard_id"] = s
        f.attrs["n_shards"] = args.n_shards
        f.attrs["K"] = args.K
        f.attrs["lr"] = args.lr
        f.attrs["epochs_max"] = args.epochs
        f.attrs["early_stop"] = args.early_stop
        f.attrs["kernel_size"] = args.kernel_size
        f.attrs["param_dim"] = param_dim
        f.attrs["channels"] = 3
        f.attrs["image_size"] = 64
        f.attrs["crop_size"] = args.crop_size
        f.attrs["dataset"] = "celeba64"
        f.attrs["created_at"] = _now()

    if args.resume and shard_path.exists():
        print(f"Resume mode: opening {shard_path}", flush=True)
        mode = "r+"
    elif shard_path.exists():
        print(f"[WARN] {shard_path} exists but --resume not set. Overwriting.", flush=True)
        shard_path.unlink()
        mode = "w"
    else:
        mode = "w"

    import h5py

    with open(log_path, "a") as log_fh:
        log_fh.write(header)

        with h5py.File(shard_path, mode) as hf:
            if mode == "w":
                _create_shard(hf)
            elif mode == "r+":
                done_arr = hf["done"][:]
                done_set = set(int(r) for r in np.where(done_arr)[0])
                print(f"  Resuming: {len(done_set)}/{n_shard} rows done.", flush=True)

            t_start = time.time()
            cumul_psnr = []
            BS = args.batch_size

            # Build list of (row, global_idx) pairs that still need encoding
            todo = [(row, gidx) for row, gidx in enumerate(my_global_indices)
                    if row not in done_set]

            n_todo = len(todo)
            n_encoded = 0

            for batch_start in range(0, n_todo, BS):
                batch_items = todo[batch_start:batch_start + BS]
                batch_rows = [r for r, _ in batch_items]
                batch_gidxs = [g for _, g in batch_items]
                B_cur = len(batch_items)

                # Build batch tensor [B, 3, 64, 64]
                batch_imgs = torch.from_numpy(
                    np.stack([images[g] for g in batch_gidxs])
                ).float() / 255.0

                t0 = time.time()
                W_batch, losses = encode_batch(
                    batch_imgs,
                    K=args.K,
                    epochs=args.epochs,
                    lr=args.lr,
                    kernel_size=args.kernel_size,
                    early_stop_threshold=args.early_stop,
                    device=args.device,
                    sigma_activation=args.sigma_activation,
                    init_mode=args.init_mode,
                    soft_clamp=args.soft_clamp,
                    use_scheduler=not args.no_scheduler,
                )
                elapsed_batch = time.time() - t0

                # Write results to HDF5
                W_np = W_batch.numpy()   # [B, K, 9]
                losses_np = losses.numpy()

                for i, (row, global_idx) in enumerate(batch_items):
                    mse = float(losses_np[i])
                    psnr = _psnr(mse)
                    split_val = int(splits[global_idx])

                    # Compute orig_index: index within its split
                    if split_val == 0:
                        orig_index = global_idx
                    elif split_val == 1:
                        orig_index = global_idx - 162770
                    else:
                        orig_index = global_idx - 182637

                    hf["W"][row] = W_np[i]
                    hf["labels"][row] = 0  # unconditional
                    hf["psnr"][row] = psnr
                    hf["n_epochs"][row] = args.epochs
                    hf["converged"][row] = bool(mse < args.early_stop)
                    hf["orig_split"][row] = split_val
                    hf["orig_index"][row] = orig_index
                    hf["encode_time"][row] = elapsed_batch / B_cur
                    hf["done"][row] = True
                    cumul_psnr.append(psnr)

                hf.flush()
                n_encoded += B_cur
                elapsed_total = time.time() - t_start
                cum_mean = float(np.mean(cumul_psnr))
                rate = n_encoded / elapsed_total if elapsed_total > 0 else 0
                eta = _eta_str(elapsed_total, n_encoded, n_todo)

                batch_psnrs = [_psnr(float(losses_np[i])) for i in range(B_cur)]
                batch_min = min(batch_psnrs)
                batch_mean = sum(batch_psnrs) / len(batch_psnrs)

                line = (f"[{_now()}] [S{s} | {n_encoded:>5}/{n_todo} | "
                        f"{100*n_encoded/n_todo:5.1f}%]  "
                        f"batch={B_cur}  {elapsed_batch:.1f}s ({elapsed_batch/B_cur:.2f}s/img)  "
                        f"PSNR={batch_mean:.1f}/{batch_min:.1f}dB(mean/min)  "
                        f"cumul={cum_mean:.1f}dB  {rate:.2f}img/s  ETA={eta}")
                print(line, flush=True)
                log_fh.write(line + "\n")
                log_fh.flush()

        # Summary
        elapsed_total = time.time() - t_start
        with h5py.File(shard_path, "r") as hf:
            done_arr = hf["done"][:]
            psnr_arr = hf["psnr"][:][done_arr]

        summary = {
            "shard_id": s,
            "n_shard": n_shard,
            "n_written": int(done_arr.sum()),
            "psnr_mean": float(np.mean(psnr_arr)),
            "psnr_std": float(np.std(psnr_arr)),
            "psnr_min": float(np.min(psnr_arr)),
            "psnr_max": float(np.max(psnr_arr)),
            "total_wall_h": elapsed_total / 3600.0,
            "finished_at": _now(),
        }
        with open(out_dir / f"shard_{s}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        summary_msg = (
            f"\n{'='*70}\n"
            f"[S{s} | DONE]  {summary['n_written']}/{n_shard} images\n"
            f"  PSNR: mean={summary['psnr_mean']:.2f}  std={summary['psnr_std']:.2f}"
            f"  min={summary['psnr_min']:.2f}  max={summary['psnr_max']:.2f} dB\n"
            f"  Time: {_fmt_duration(elapsed_total)}\n"
            f"{'='*70}\n"
        )
        print(summary_msg, flush=True)
        log_fh.write(summary_msg)


if __name__ == "__main__":
    main()
