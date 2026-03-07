"""
merge_mnist_shards.py — Merge 4 shard HDF5 files into one master HDF5 file.

Usage:
    python scripts/merge_mnist_shards.py
    python scripts/merge_mnist_shards.py --shard_dir data/mnist_gaussians_shards \
        --out_file data/mnist_gaussians_K70.h5

The script:
  1. Reads each shard file (shard_0.h5 … shard_3.h5)
  2. Validates completeness and integrity (no NaN, all done=True, etc.)
  3. Writes merged file sorted by global_idx (0 to 69999)
  4. Writes root-level statistics attributes
  5. Prints a validation report and per-digit PSNR table

Submit after encode jobs finish:
    sbatch --dependency=afterok:JID0:JID1:JID2:JID3 scripts/slurm_merge_mnist.sh
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _hr() -> str:
    return "─" * 68


def validate_shard(hf, shard_id: int, expected_n: int) -> list[str]:
    """Return list of error strings (empty = OK)."""
    errors = []
    done = hf["done"][:]
    n_done = int(done.sum())
    if n_done != expected_n:
        errors.append(f"  Shard {shard_id}: {n_done}/{expected_n} done rows (incomplete)")
    if not done.all():
        errors.append(f"  Shard {shard_id}: {(~done).sum()} rows not done")
    W = hf["W"][:]
    if np.isnan(W).any():
        errors.append(f"  Shard {shard_id}: NaN found in W")
    if np.isinf(W).any():
        errors.append(f"  Shard {shard_id}: Inf found in W")
    psnr = hf["psnr"][:]
    if (psnr <= 0).any():
        errors.append(f"  Shard {shard_id}: {(psnr<=0).sum()} PSNR <= 0")
    labels = hf["labels"][:]
    if ((labels < 0) | (labels > 9)).any():
        errors.append(f"  Shard {shard_id}: labels out of [0,9] range")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Merge MNIST Gaussian shard files into one HDF5.")
    parser.add_argument("--shard_dir",  default="data/mnist_gaussians_shards",
                        help="Directory containing shard_0.h5 … shard_N.h5")
    parser.add_argument("--out_file",   default="data/mnist_gaussians_K70.h5",
                        help="Output merged HDF5 file")
    parser.add_argument("--n_shards",   type=int, default=4, help="Number of shards")
    parser.add_argument("--n_total",    type=int, default=70000, help="Total images (70k MNIST)")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        sys.exit("[FATAL] h5py not installed. Activate gaussiandiffusion conda env.")

    shard_dir = PROJECT_ROOT / args.shard_dir
    out_file  = PROJECT_ROOT / args.out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  merge_mnist_shards.py")
    print(f"  Started : {_now()}")
    print(f"  Input   : {shard_dir}")
    print(f"  Output  : {out_file}")
    print(f"{'='*68}\n")

    # ------------------------------------------------------------------
    # Step 1: Open and validate all shards
    # ------------------------------------------------------------------
    print(_hr())
    print("  Step 1: Validating shard files")
    print(_hr())

    shard_files = []
    all_errors  = []
    n_per_shard = []

    for s in range(args.n_shards):
        path = shard_dir / f"shard_{s}.h5"
        if not path.exists():
            all_errors.append(f"  [MISSING] {path}")
            continue

        with h5py.File(path, "r") as hf:
            n_shard = int(hf.attrs.get("n_shard", hf["W"].shape[0]))
            n_per_shard.append(n_shard)
            errors = validate_shard(hf, s, n_shard)
            all_errors.extend(errors)
            status = "OK " if not errors else "ERR"
            print(f"  [{status}] shard_{s}.h5  rows={n_shard}  "
                  f"done={int(hf['done'][:].sum())}  "
                  f"mean_psnr={float(hf['psnr'][:].mean()):.2f} dB")

        shard_files.append(path)

    if all_errors:
        print("\n[ERRORS FOUND — aborting merge]")
        for e in all_errors:
            print(e)
        sys.exit(1)

    n_expected = sum(n_per_shard)
    if n_expected != args.n_total:
        print(f"\n[WARN] Total rows across shards = {n_expected}, expected {args.n_total}")

    print(f"\n  All {len(shard_files)} shards validated OK. Total rows: {n_expected}\n")

    # ------------------------------------------------------------------
    # Step 2: Load all shard data into memory
    # ------------------------------------------------------------------
    print(_hr())
    print("  Step 2: Loading shard data")
    print(_hr())

    t0 = time.time()
    # Preallocate arrays indexed by global_idx
    W_all           = np.zeros((args.n_total, 70, 7), dtype=np.float32)
    labels_all      = np.zeros(args.n_total, dtype=np.uint8)
    psnr_all        = np.zeros(args.n_total, dtype=np.float32)
    n_epochs_all    = np.zeros(args.n_total, dtype=np.int32)
    converged_all   = np.zeros(args.n_total, dtype=bool)
    n_dead_all      = np.zeros(args.n_total, dtype=np.int8)
    orig_split_all  = np.zeros(args.n_total, dtype=np.uint8)
    orig_index_all  = np.zeros(args.n_total, dtype=np.int32)
    encode_time_all = np.zeros(args.n_total, dtype=np.float32)
    present         = np.zeros(args.n_total, dtype=bool)

    for s, path in enumerate(shard_files):
        print(f"  Loading shard {s} from {path.name} …", flush=True)
        with h5py.File(path, "r") as hf:
            g_idx = hf["global_idx"][:]
            W_all[g_idx]           = hf["W"][:]
            labels_all[g_idx]      = hf["labels"][:]
            psnr_all[g_idx]        = hf["psnr"][:]
            n_epochs_all[g_idx]    = hf["n_epochs"][:]
            converged_all[g_idx]   = hf["converged"][:]
            n_dead_all[g_idx]      = hf["n_dead"][:]
            orig_split_all[g_idx]  = hf["orig_split"][:]
            orig_index_all[g_idx]  = hf["orig_index"][:]
            encode_time_all[g_idx] = hf["encode_time"][:]
            present[g_idx]         = True

    print(f"  Load time: {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 3: Final validation on merged arrays
    # ------------------------------------------------------------------
    print(_hr())
    print("  Step 3: Merged validation")
    print(_hr())

    checks_passed = True

    # All indices present
    missing = int((~present).sum())
    ok = missing == 0
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} All {args.n_total} indices present (missing={missing})")

    # Shape
    ok = W_all.shape == (args.n_total, 70, 7)
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} W.shape == ({args.n_total}, 70, 7)  got {W_all.shape}")

    # NaN / Inf
    ok_nan = not np.isnan(W_all).any()
    ok_inf = not np.isinf(W_all).any()
    checks_passed &= (ok_nan and ok_inf)
    print(f"  {'[OK]' if ok_nan else '[FAIL]'} No NaN in W")
    print(f"  {'[OK]' if ok_inf else '[FAIL]'} No Inf in W")

    ok_nan_p = not np.isnan(psnr_all).any()
    checks_passed &= ok_nan_p
    print(f"  {'[OK]' if ok_nan_p else '[FAIL]'} No NaN in psnr")

    # PSNR > 0
    ok = (psnr_all > 0).all()
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} All PSNR > 0  (min={psnr_all.min():.2f} dB)")

    # Labels in range
    ok = ((labels_all >= 0) & (labels_all <= 9)).all()
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} All labels in [0, 9]")

    # Split counts
    n_tr = int((orig_split_all == 0).sum())
    n_te = int((orig_split_all == 1).sum())
    ok = (n_tr == 60000 and n_te == 10000)
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} Train={n_tr} Test={n_te}  (expected 60000/10000)")

    if not checks_passed:
        print("\n[ABORT] Validation failures detected — not writing output file.")
        sys.exit(1)

    print("\n  All checks passed.\n")

    # ------------------------------------------------------------------
    # Step 4: Write merged HDF5
    # ------------------------------------------------------------------
    print(_hr())
    print(f"  Step 4: Writing {out_file}")
    print(_hr())

    K = W_all.shape[1]
    t0 = time.time()
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("W", data=W_all,
                          chunks=(256, K, 7), compression="gzip")
        hf.create_dataset("labels",      data=labels_all,      compression="gzip")
        hf.create_dataset("psnr",        data=psnr_all,        compression="gzip")
        hf.create_dataset("n_epochs",    data=n_epochs_all,    compression="gzip")
        hf.create_dataset("converged",   data=converged_all,   compression="gzip")
        hf.create_dataset("n_dead",      data=n_dead_all,      compression="gzip")
        hf.create_dataset("orig_split",  data=orig_split_all,  compression="gzip")
        hf.create_dataset("orig_index",  data=orig_index_all,  compression="gzip")
        hf.create_dataset("encode_time", data=encode_time_all, compression="gzip")

        # Root attrs
        hf.attrs["K"]               = K
        hf.attrs["lr"]              = 5e-3
        hf.attrs["epochs_max"]      = 3000
        hf.attrs["early_stop"]      = 1e-4
        hf.attrs["kernel_size"]     = 11
        hf.attrs["recycle_every"]   = 300
        hf.attrs["encoder_version"] = "2.0"
        hf.attrs["coord_fix_applied"] = True
        hf.attrs["n_total"]         = args.n_total
        hf.attrs["n_train"]         = 60000
        hf.attrs["n_test"]          = 10000
        hf.attrs["mean_psnr"]       = float(psnr_all.mean())
        hf.attrs["std_psnr"]        = float(psnr_all.std())
        hf.attrs["min_psnr"]        = float(psnr_all.min())
        hf.attrs["max_psnr"]        = float(psnr_all.max())
        hf.attrs["pct_converged"]   = float(converged_all.mean()) * 100
        hf.attrs["mean_encode_time_s"] = float(encode_time_all.mean())
        hf.attrs["total_encode_time_h"] = float(encode_time_all.sum()) / 3600.0
        hf.attrs["created_at"]      = _now()

    write_time = time.time() - t0
    file_mb = out_file.stat().st_size / 1e6
    print(f"  Written in {write_time:.1f}s  —  {file_mb:.1f} MB\n")

    # ------------------------------------------------------------------
    # Step 5: Print summary report
    # ------------------------------------------------------------------
    print(_hr())
    print("  PSNR Summary")
    print(_hr())
    pcts = [10, 25, 50, 75, 90]
    deciles = np.percentile(psnr_all, pcts)
    for p, v in zip(pcts, deciles):
        print(f"  P{p:2d}  : {v:.2f} dB")
    print(f"  Min  : {psnr_all.min():.2f} dB")
    print(f"  Mean : {psnr_all.mean():.2f} dB  (±{psnr_all.std():.2f})")
    print(f"  Max  : {psnr_all.max():.2f} dB")

    n_below_30 = int((psnr_all < 30.0).sum())
    n_below_35 = int((psnr_all < 35.0).sum())
    print(f"\n  below 30 dB  : {n_below_30} ({100*n_below_30/args.n_total:.2f}%)")
    print(f"  below 35 dB  : {n_below_35} ({100*n_below_35/args.n_total:.2f}%)")
    print(f"  converged    : {converged_all.sum()} ({100*converged_all.mean():.1f}%)")

    print(f"\n{_hr()}")
    print("  Per-digit PSNR")
    print(_hr())
    print(f"  {'Digit':>6}  {'N':>6}  {'Mean':>7}  {'Std':>6}  {'Min':>7}  {'Max':>7}  {'<30dB':>6}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}")
    for d in range(10):
        mask = labels_all == d
        psnr_d = psnr_all[mask]
        n_d = len(psnr_d)
        n_low = int((psnr_d < 30.0).sum())
        print(
            f"  {d:>6}  {n_d:>6}  {psnr_d.mean():>7.2f}  {psnr_d.std():>6.2f}"
            f"  {psnr_d.min():>7.2f}  {psnr_d.max():>7.2f}  {n_low:>6}"
        )

    print(f"\n{_hr()}")
    print(f"  File size    : {file_mb:.1f} MB")
    est_load_s = file_mb / 1000 * 2  # rough: ~500 MB/s disk
    print(f"  Est. load    : {est_load_s:.1f}s  (preload=True, ~500 MB/s)")
    print(f"  Output       : {out_file}")
    print(f"  Finished     : {_now()}")
    print(f"{'='*68}\n")

    # Write summary JSON
    summary = {
        "n_total": args.n_total,
        "mean_psnr": float(psnr_all.mean()),
        "std_psnr":  float(psnr_all.std()),
        "min_psnr":  float(psnr_all.min()),
        "max_psnr":  float(psnr_all.max()),
        "pct_converged": float(converged_all.mean()) * 100,
        "n_below_30dB": n_below_30,
        "n_below_35dB": n_below_35,
        "file_mb": file_mb,
        "finished_at": _now(),
        "per_digit": {
            str(d): {
                "n": int((labels_all == d).sum()),
                "mean": float(psnr_all[labels_all == d].mean()),
                "min":  float(psnr_all[labels_all == d].min()),
            }
            for d in range(10)
        }
    }
    summary_path = out_file.parent / "mnist_gaussians_K70_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON : {summary_path}\n")


if __name__ == "__main__":
    main()
