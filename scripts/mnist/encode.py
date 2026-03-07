"""
encode_full_mnist.py — Encode all 70,000 MNIST images into a single shard HDF5 file.

Each shard processes a subset of images determined by interleaved sharding:
  shard s → global indices i where i % n_shards == s

Global index layout (deterministic, reproducible):
  0  – 59999 : MNIST train set, original order
  60000 – 69999 : MNIST test  set, original order

Output: data/mnist_gaussians_shards/shard_{s}.h5
  /W            float32  [N_shard, 70, 7]  Gaussian parameters
  /labels       uint8    [N_shard]          digit class 0–9
  /psnr         float32  [N_shard]          final PSNR (dB)
  /n_epochs     int32    [N_shard]          epoch at convergence / max epochs
  /converged    bool     [N_shard]          reached early_stop_threshold?
  /n_dead       int8     [N_shard]          dead Gaussians at end
  /orig_split   uint8    [N_shard]          0=train, 1=test
  /orig_index   int32    [N_shard]          index within MNIST split
  /encode_time  float32  [N_shard]          wall-clock seconds per image
  /global_idx   int32    [N_shard]          global index (0–69999)
  /done         bool     [N_shard]          True = row is written (for resume)

Submit: sbatch --array=0-3 scripts/slurm_encode_full_mnist.sh
Resume: add --resume flag to sbatch command or re-run with same --shard_id
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Resolve project root and set up imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encode import encode_image  # noqa: E402


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
    rate = done / elapsed  # images/s
    remaining = (total - done) / rate
    return _fmt_duration(remaining)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_image(log_fh, shard_id: int, row: int, total: int,
              idx: int, digit: int, psnr: float, n_ep: int,
              n_dead: int, K: int, t: float, is_warn: bool):
    pct = 100.0 * (row + 1) / total
    prefix = f"[{_now()}] [S{shard_id} | {row+1:>5}/{total} | {pct:5.1f}%]"
    msg = (f"  idx={idx:<6}  digit={digit}  PSNR={psnr:.1f}dB"
           f"  ep={n_ep:<5}  dead={n_dead}/{K}  t={t:.1f}s")
    line = prefix + msg
    if is_warn:
        line += "  ← below 30dB floor"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def log_progress(log_fh, shard_id: int, row: int, total: int,
                 window_psnr: list, window_conv: list,
                 elapsed: float, cumul_psnr: list):
    pct = 100.0 * (row + 1) / total
    mean_psnr = float(np.mean(window_psnr)) if window_psnr else 0.0
    conv_rate = float(np.mean(window_conv)) * 100 if window_conv else 0.0
    rate = (row + 1) / elapsed if elapsed > 0 else 0.0
    eta = _eta_str(elapsed, row + 1, total)
    cum_mean = float(np.mean(cumul_psnr)) if cumul_psnr else 0.0
    line = (
        f"[{_now()}] [S{shard_id} | PROGRESS | {row+1:>5}/{total} | {pct:5.1f}%]"
        f"  window: mean={mean_psnr:.2f}dB  conv={conv_rate:.0f}%"
        f"  {rate:.3f}img/s  ETA={eta}  cumul_mean={cum_mean:.2f}dB"
    )
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def log_checkpoint(log_fh, shard_id: int, row: int, total: int, elapsed: float):
    line = (
        f"[{_now()}] [S{shard_id} | CKPT] {row+1}/{total} written."
        f"  Elapsed={_fmt_duration(elapsed)}."
        f"  ETA={_eta_str(elapsed, row+1, total)}"
    )
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def log_warn(log_fh, shard_id: int, idx: int, digit: int, psnr: float,
             n_ep: int, n_dead: int):
    line = (
        f"[{_now()}] [S{shard_id} | WARN] idx={idx}  digit={digit}"
        f"  PSNR={psnr:.1f}dB  ep={n_ep}  dead={n_dead}"
    )
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Encode all MNIST images to Gaussian representations (one HDF5 shard)."
    )
    parser.add_argument("--shard_id",  type=int, required=True, help="Shard index (0 to n_shards-1)")
    parser.add_argument("--n_shards",  type=int, default=4,     help="Total number of shards")
    parser.add_argument("--out_dir",   type=str, default="data/mnist_gaussians_shards",
                        help="Directory for shard HDF5 files")
    parser.add_argument("--device",    type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume from existing shard file (skip done rows)")
    # Encoder hyperparameters
    parser.add_argument("--K",             type=int,   default=70,    help="Number of Gaussians")
    parser.add_argument("--epochs",        type=int,   default=3000,  help="Max epochs per image")
    parser.add_argument("--lr",            type=float, default=5e-3,  help="Learning rate")
    parser.add_argument("--early_stop",    type=float, default=1e-4,  help="MSE early-stop threshold (~40dB)")
    parser.add_argument("--kernel_size",   type=int,   default=11,    help="Gaussian kernel size")
    parser.add_argument("--recycle_every", type=int,   default=300,   help="Dead Gaussian recycle interval")
    # Logging intervals
    parser.add_argument("--log_every",      type=int, default=25,  help="Console log every N images")
    parser.add_argument("--progress_every", type=int, default=100, help="PROGRESS line every N images")
    parser.add_argument("--ckpt_log_every", type=int, default=500, help="CHECKPOINT log marker every N images")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        sys.exit("[FATAL] h5py not installed. Activate the gaussiandiffusion conda env.")

    try:
        from torchvision.datasets import MNIST
        from torchvision import transforms as T
    except ImportError:
        sys.exit("[FATAL] torchvision not installed.")

    # ------------------------------------------------------------------
    # Validate args
    # ------------------------------------------------------------------
    assert 0 <= args.shard_id < args.n_shards, \
        f"shard_id must be in [0, n_shards). Got {args.shard_id}/{args.n_shards}"

    s = args.shard_id
    out_dir = Path(PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{s}.h5"

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"encode_full_mnist_shard{s}.log"

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    header = (
        f"\n{'='*70}\n"
        f"  encode_full_mnist.py  —  shard {s}/{args.n_shards}\n"
        f"  Started : {_now()}\n"
        f"  Device  : {args.device}\n"
        f"  K       : {args.K}  lr={args.lr}  epochs={args.epochs}\n"
        f"  Output  : {shard_path}\n"
        f"  Log     : {log_path}\n"
        f"{'='*70}\n"
    )
    print(header, flush=True)

    with open(log_path, "a") as log_fh:
        log_fh.write(header)

    # ------------------------------------------------------------------
    # Load MNIST (download=False; data must exist in standard torchvision location)
    # ------------------------------------------------------------------
    mnist_root = str(PROJECT_ROOT / "data" / "mnist_raw")
    tf = T.ToTensor()

    try:
        train_ds = MNIST(root=mnist_root, train=True,  download=True, transform=tf)
        test_ds  = MNIST(root=mnist_root, train=False, download=True, transform=tf)
    except Exception as e:
        sys.exit(f"[FATAL] Could not load MNIST: {e}")

    n_train = len(train_ds)   # 60000
    n_test  = len(test_ds)    # 10000
    n_total_global = n_train + n_test  # 70000

    # Build list of global indices assigned to this shard
    my_global_indices = [i for i in range(n_total_global) if i % args.n_shards == s]
    n_shard = len(my_global_indices)

    msg = f"Shard {s}: {n_shard} images ({n_shard/n_total_global*100:.1f}% of dataset)"
    print(msg, flush=True)

    # ------------------------------------------------------------------
    # Open / create HDF5 shard file
    # ------------------------------------------------------------------
    done_set: set[int] = set()  # local row indices already written

    def _create_shard(f: "h5py.File"):
        """Pre-allocate all datasets in a new shard file."""
        f.create_dataset("W",           shape=(n_shard, args.K, 7), dtype="float32",
                         chunks=(min(256, n_shard), args.K, 7), compression="gzip")
        f.create_dataset("labels",      shape=(n_shard,), dtype="uint8")
        f.create_dataset("psnr",        shape=(n_shard,), dtype="float32")
        f.create_dataset("n_epochs",    shape=(n_shard,), dtype="int32")
        f.create_dataset("converged",   shape=(n_shard,), dtype="bool")
        f.create_dataset("n_dead",      shape=(n_shard,), dtype="int8")
        f.create_dataset("orig_split",  shape=(n_shard,), dtype="uint8")
        f.create_dataset("orig_index",  shape=(n_shard,), dtype="int32")
        f.create_dataset("encode_time", shape=(n_shard,), dtype="float32")
        f.create_dataset("global_idx",  shape=(n_shard,), dtype="int32")
        f.create_dataset("done",        shape=(n_shard,), dtype="bool",
                         data=np.zeros(n_shard, dtype=bool))
        # Write global indices map upfront
        f["global_idx"][:] = np.array(my_global_indices, dtype=np.int32)
        # Attrs
        f.attrs["shard_id"]    = s
        f.attrs["n_shards"]    = args.n_shards
        f.attrs["K"]           = args.K
        f.attrs["lr"]          = args.lr
        f.attrs["epochs_max"]  = args.epochs
        f.attrs["early_stop"]  = args.early_stop
        f.attrs["kernel_size"] = args.kernel_size
        f.attrs["recycle_every"] = args.recycle_every
        f.attrs["encoder_version"] = "2.0"
        f.attrs["coord_fix_applied"] = True
        f.attrs["n_shard"] = n_shard
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

    with open(log_path, "a") as log_fh:
        import h5py

        with h5py.File(shard_path, mode) as hf:
            if mode == "w":
                _create_shard(hf)
            elif mode == "r+":
                done_arr = hf["done"][:]
                done_set = set(int(r) for r in np.where(done_arr)[0])
                print(f"  Resuming: {len(done_set)}/{n_shard} rows already done.", flush=True)
                # Resize check: shard file must match current n_shard
                assert hf["W"].shape[0] == n_shard, \
                    f"Shard file has {hf['W'].shape[0]} rows but expected {n_shard}"

            # --------------------------------------------------------------
            # Encoding loop
            # --------------------------------------------------------------
            t_start = time.time()
            window_psnr:  list = []
            window_conv:  list = []
            cumul_psnr:   list = []
            flush_counter = 0

            for row, global_idx in enumerate(my_global_indices):
                if row in done_set:
                    continue  # already encoded (resume mode)

                # Retrieve image
                if global_idx < n_train:
                    img_tensor, label = train_ds[global_idx]
                    orig_split = 0
                    orig_index = global_idx
                else:
                    local_idx = global_idx - n_train
                    img_tensor, label = test_ds[local_idx]
                    orig_split = 1
                    orig_index = local_idx

                # img_tensor: [1, H, W] → [H, W]
                img = img_tensor.squeeze(0)
                digit = int(label)

                # Encode
                t0 = time.time()
                W_phys, final_loss = encode_image(
                    img,
                    K=args.K,
                    epochs=args.epochs,
                    lr=args.lr,
                    kernel_size=args.kernel_size,
                    early_stop_threshold=args.early_stop,
                    device=args.device,
                    recycle_every=args.recycle_every,
                )
                elapsed_img = time.time() - t0

                # Compute stats
                psnr = _psnr(final_loss)
                converged = bool(final_loss < args.early_stop)

                # Count dead Gaussians (rough: colour < 0.05 heuristic)
                with torch.no_grad():
                    colour_vals = W_phys[:, 4]  # colour channel
                    n_dead = int((colour_vals < 0.05).sum().item())

                # Estimate n_epochs (we don't track this in encode_image return,
                # so use a proxy: if converged, converged early; else max epochs)
                n_ep = args.epochs  # worst case; actual epoch not returned by default

                # Write to HDF5
                hf["W"][row]           = W_phys.numpy()
                hf["labels"][row]      = digit
                hf["psnr"][row]        = psnr
                hf["n_epochs"][row]    = n_ep
                hf["converged"][row]   = converged
                hf["n_dead"][row]      = n_dead
                hf["orig_split"][row]  = orig_split
                hf["orig_index"][row]  = orig_index
                hf["encode_time"][row] = elapsed_img
                hf["done"][row]        = True

                flush_counter += 1
                if flush_counter >= 10:
                    hf.flush()
                    flush_counter = 0

                # Logging
                is_warn = psnr < 30.0
                window_psnr.append(psnr)
                window_conv.append(float(converged))
                cumul_psnr.append(psnr)

                elapsed_total = time.time() - t_start
                n_done_so_far = row + 1 - len(done_set)  # rows written this run

                # Per-image (throttled to log_every on console but always to file)
                if is_warn:
                    log_warn(log_fh, s, global_idx, digit, psnr, n_ep, n_dead)
                if n_done_so_far % args.log_every == 1 or is_warn:
                    log_image(log_fh, s, row, n_shard, global_idx, digit,
                              psnr, n_ep, n_dead, args.K, elapsed_img, is_warn)

                # PROGRESS every progress_every images
                if n_done_so_far % args.progress_every == 0:
                    log_progress(log_fh, s, row, n_shard,
                                 window_psnr[-args.progress_every:],
                                 window_conv[-args.progress_every:],
                                 elapsed_total, cumul_psnr)

                # CHECKPOINT log marker every ckpt_log_every images
                if n_done_so_far % args.ckpt_log_every == 0:
                    log_checkpoint(log_fh, s, row, n_shard, elapsed_total)

            # Final flush
            hf.flush()

        # --------------------------------------------------------------
        # End-of-shard summary
        # --------------------------------------------------------------
        elapsed_total = time.time() - t_start

        # Reload for summary stats
        with h5py.File(shard_path, "r") as hf:
            done_arr  = hf["done"][:]
            n_written = int(done_arr.sum())
            psnr_arr  = hf["psnr"][:][done_arr]
            conv_arr  = hf["converged"][:][done_arr]
            time_arr  = hf["encode_time"][:][done_arr]

        summary = {
            "shard_id": s,
            "n_shard":  n_shard,
            "n_written": n_written,
            "psnr_mean":  float(np.mean(psnr_arr))  if len(psnr_arr) else 0.0,
            "psnr_std":   float(np.std(psnr_arr))   if len(psnr_arr) else 0.0,
            "psnr_min":   float(np.min(psnr_arr))   if len(psnr_arr) else 0.0,
            "psnr_max":   float(np.max(psnr_arr))   if len(psnr_arr) else 0.0,
            "pct_converged":   float(np.mean(conv_arr)) * 100 if len(conv_arr) else 0.0,
            "pct_below_30dB":  float(np.mean(psnr_arr < 30.0)) * 100 if len(psnr_arr) else 0.0,
            "mean_encode_time_s": float(np.mean(time_arr)) if len(time_arr) else 0.0,
            "total_wall_h": elapsed_total / 3600.0,
            "finished_at": _now(),
        }

        summary_path = out_dir / f"shard_{s}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        summary_msg = (
            f"\n{'='*70}\n"
            f"[S{s} | DONE]  {n_written}/{n_shard} images written\n"
            f"  PSNR  : mean={summary['psnr_mean']:.2f}  std={summary['psnr_std']:.2f}"
            f"  min={summary['psnr_min']:.2f}  max={summary['psnr_max']:.2f} dB\n"
            f"  Conv  : {summary['pct_converged']:.1f}%"
            f"  below30dB={summary['pct_below_30dB']:.2f}%\n"
            f"  Time  : {summary['mean_encode_time_s']:.2f} s/img"
            f"  total={_fmt_duration(elapsed_total)}\n"
            f"  Summary JSON : {summary_path}\n"
            f"{'='*70}\n"
        )
        print(summary_msg, flush=True)
        log_fh.write(summary_msg)


if __name__ == "__main__":
    main()
