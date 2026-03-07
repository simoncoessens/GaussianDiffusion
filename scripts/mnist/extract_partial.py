"""
extract_partial_data.py — Extract completed rows from in-progress HDF5 shards.

Reads each shard, selects only rows where done=True, runs sanity checks,
and writes a standalone HDF5 file compatible with GaussianDatasetV2.

Usage:
    python scripts/extract_partial_data.py
    python scripts/extract_partial_data.py --shard_dir data/mnist_gaussians_shards \
        --out_file data/mnist_gaussians_partial.h5 --min_psnr 25.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Extract completed rows from in-progress shards")
    parser.add_argument("--shard_dir", default="data/mnist_gaussians_shards",
                        help="Directory containing shard_*.h5 files")
    parser.add_argument("--out_file", default="data/mnist_gaussians_partial.h5",
                        help="Output HDF5 file")
    parser.add_argument("--min_psnr", type=float, default=None,
                        help="Exclude rows with PSNR below this threshold (dB)")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        sys.exit("[FATAL] h5py not installed.")

    shard_dir = PROJECT_ROOT / args.shard_dir
    out_file = PROJECT_ROOT / args.out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Discover shards
    shard_paths = sorted(shard_dir.glob("shard_*.h5"))
    if not shard_paths:
        sys.exit(f"[FATAL] No shard_*.h5 files found in {shard_dir}")

    print(f"Found {len(shard_paths)} shards in {shard_dir}")

    # Collect completed rows from all shards
    all_W, all_labels, all_psnr, all_orig_split = [], [], [], []
    K = None
    encoder_version = "unknown"

    for path in shard_paths:
        with h5py.File(path, "r") as hf:
            done = hf["done"][:]
            n_done = int(done.sum())
            n_total = len(done)
            print(f"  {path.name}: {n_done}/{n_total} done")

            if n_done == 0:
                continue

            if K is None:
                K = int(hf.attrs.get("K", hf["W"].shape[1]))
                encoder_version = str(hf.attrs.get("encoder_version", "unknown"))

            # Shards may be actively written; read in aligned chunks,
            # skipping any corrupted gzip blocks.
            chunk_size = 256  # matches HDF5 chunk layout
            shard_W, shard_labels, shard_psnr, shard_split = [], [], [], []
            labels_full = hf["labels"][:]
            psnr_full = hf["psnr"][:]
            split_full = hf["orig_split"][:]

            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                chunk_done = done[start:end]
                if not chunk_done.any():
                    continue
                try:
                    W_chunk = hf["W"][start:end]
                except OSError:
                    print(f"    chunk [{start}:{end}] unreadable — skipping")
                    continue
                shard_W.append(W_chunk[chunk_done])
                shard_labels.append(labels_full[start:end][chunk_done])
                shard_psnr.append(psnr_full[start:end][chunk_done])
                shard_split.append(split_full[start:end][chunk_done])

            if shard_W:
                all_W.append(np.concatenate(shard_W))
                all_labels.append(np.concatenate(shard_labels))
                all_psnr.append(np.concatenate(shard_psnr))
                all_orig_split.append(np.concatenate(shard_split))

    if not all_W:
        sys.exit("[FATAL] No completed rows found across shards.")

    W = np.concatenate(all_W, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    psnr = np.concatenate(all_psnr, axis=0)
    orig_split = np.concatenate(all_orig_split, axis=0)

    print(f"\nTotal completed: {len(W)}")

    # --- Sanity checks ---
    n_nan = int(np.isnan(W).any(axis=(1, 2)).sum())
    n_inf = int(np.isinf(W).any(axis=(1, 2)).sum())
    print(f"NaN rows: {n_nan}, Inf rows: {n_inf}")
    if n_nan > 0 or n_inf > 0:
        good = ~(np.isnan(W).any(axis=(1, 2)) | np.isinf(W).any(axis=(1, 2)))
        W, labels, psnr, orig_split = W[good], labels[good], psnr[good], orig_split[good]
        print(f"  Removed {n_nan + n_inf} bad rows → {len(W)} remaining")

    # PSNR filter
    if args.min_psnr is not None:
        before = len(W)
        keep = psnr >= args.min_psnr
        W, labels, psnr, orig_split = W[keep], labels[keep], psnr[keep], orig_split[keep]
        print(f"PSNR filter (>= {args.min_psnr} dB): {before} → {len(W)}")

    # Stats
    print(f"\n--- PSNR stats ---")
    print(f"  Mean: {psnr.mean():.2f} ± {psnr.std():.2f} dB")
    print(f"  Min:  {psnr.min():.2f} dB")
    print(f"  Max:  {psnr.max():.2f} dB")
    print(f"  < 30 dB: {(psnr < 30).sum()} ({100*(psnr < 30).mean():.1f}%)")
    print(f"  < 35 dB: {(psnr < 35).sum()} ({100*(psnr < 35).mean():.1f}%)")

    print(f"\n--- Label distribution ---")
    for d in range(10):
        n = int((labels == d).sum())
        print(f"  {d}: {n}")

    print(f"\n--- Value ranges (7-dim) ---")
    col_names = ["sigma_x", "sigma_y", "rho", "alpha", "colour", "x", "y"]
    for c in range(W.shape[2]):
        vals = W[:, :, c].ravel()
        print(f"  {col_names[c]:>8}: [{vals.min():.4f}, {vals.max():.4f}]")

    # --- Write output ---
    print(f"\nWriting {out_file} …")
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("W", data=W, chunks=(min(256, len(W)), K, 7), compression="gzip")
        hf.create_dataset("labels", data=labels, compression="gzip")
        hf.create_dataset("psnr", data=psnr, compression="gzip")
        hf.create_dataset("orig_split", data=orig_split, compression="gzip")

        hf.attrs["K"] = K
        hf.attrs["encoder_version"] = encoder_version
        hf.attrs["n_total"] = len(W)
        hf.attrs["source"] = "extract_partial_data.py"
        hf.attrs["mean_psnr"] = float(psnr.mean())

    file_mb = out_file.stat().st_size / 1e6
    print(f"Done. {len(W)} rows, {file_mb:.1f} MB → {out_file}")


if __name__ == "__main__":
    main()
