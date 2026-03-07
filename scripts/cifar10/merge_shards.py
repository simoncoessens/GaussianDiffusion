"""
merge_cifar10_shards.py — Merge 12 shard HDF5 files into one master HDF5 file.

Usage:
    python scripts/cifar10/merge_shards.py
    python scripts/cifar10/merge_shards.py --shard_dir data/cifar10/shards \
        --out_file data/cifar10/cifar10_gaussians_K500.h5
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


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
    parser = argparse.ArgumentParser(description="Merge CIFAR-10 Gaussian shard files into one HDF5.")
    parser.add_argument("--shard_dir", default="data/cifar10/shards",
                        help="Directory containing shard_0.h5 … shard_N.h5")
    parser.add_argument("--out_file", default="data/cifar10/cifar10_gaussians_K500.h5",
                        help="Output merged HDF5 file")
    parser.add_argument("--n_shards", type=int, default=12, help="Number of shards")
    parser.add_argument("--n_total", type=int, default=60000, help="Total images (60k CIFAR-10)")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        sys.exit("[FATAL] h5py not installed.")

    shard_dir = PROJECT_ROOT / args.shard_dir
    out_file = PROJECT_ROOT / args.out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  merge_cifar10_shards.py")
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
    all_errors = []
    n_per_shard = []
    K = None
    param_dim = None

    for s in range(args.n_shards):
        path = shard_dir / f"shard_{s}.h5"
        if not path.exists():
            all_errors.append(f"  [MISSING] {path}")
            continue

        with h5py.File(path, "r") as hf:
            n_shard = hf["W"].shape[0]
            if K is None:
                K = hf["W"].shape[1]
                param_dim = hf["W"].shape[2]
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

    print(f"\n  All {len(shard_files)} shards validated OK. Total rows: {n_expected}")
    print(f"  K={K}, param_dim={param_dim}\n")

    # ------------------------------------------------------------------
    # Step 2: Load all shard data into memory
    # ------------------------------------------------------------------
    print(_hr())
    print("  Step 2: Loading shard data")
    print(_hr())

    t0 = time.time()
    W_all = np.zeros((args.n_total, K, param_dim), dtype=np.float32)
    labels_all = np.zeros(args.n_total, dtype=np.uint8)
    psnr_all = np.zeros(args.n_total, dtype=np.float32)
    n_epochs_all = np.zeros(args.n_total, dtype=np.int32)
    converged_all = np.zeros(args.n_total, dtype=bool)
    orig_split_all = np.zeros(args.n_total, dtype=np.uint8)
    orig_index_all = np.zeros(args.n_total, dtype=np.int32)
    encode_time_all = np.zeros(args.n_total, dtype=np.float32)
    present = np.zeros(args.n_total, dtype=bool)

    for s, path in enumerate(shard_files):
        print(f"  Loading shard {s} from {path.name} …", flush=True)
        with h5py.File(path, "r") as hf:
            g_idx = hf["global_idx"][:]
            W_all[g_idx] = hf["W"][:]
            labels_all[g_idx] = hf["labels"][:]
            psnr_all[g_idx] = hf["psnr"][:]
            n_epochs_all[g_idx] = hf["n_epochs"][:]
            converged_all[g_idx] = hf["converged"][:]
            orig_split_all[g_idx] = hf["orig_split"][:]
            orig_index_all[g_idx] = hf["orig_index"][:]
            encode_time_all[g_idx] = hf["encode_time"][:]
            present[g_idx] = True

    print(f"  Load time: {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Step 3: Final validation on merged arrays
    # ------------------------------------------------------------------
    print(_hr())
    print("  Step 3: Merged validation")
    print(_hr())

    checks_passed = True

    missing = int((~present).sum())
    ok = missing == 0
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} All {args.n_total} indices present (missing={missing})")

    ok = W_all.shape == (args.n_total, K, param_dim)
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} W.shape == ({args.n_total}, {K}, {param_dim})  got {W_all.shape}")

    ok_nan = not np.isnan(W_all).any()
    ok_inf = not np.isinf(W_all).any()
    checks_passed &= (ok_nan and ok_inf)
    print(f"  {'[OK]' if ok_nan else '[FAIL]'} No NaN in W")
    print(f"  {'[OK]' if ok_inf else '[FAIL]'} No Inf in W")

    ok_nan_p = not np.isnan(psnr_all).any()
    checks_passed &= ok_nan_p
    print(f"  {'[OK]' if ok_nan_p else '[FAIL]'} No NaN in psnr")

    ok = (psnr_all > 0).all()
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} All PSNR > 0  (min={psnr_all.min():.2f} dB)")

    ok = ((labels_all >= 0) & (labels_all <= 9)).all()
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} All labels in [0, 9]")

    n_tr = int((orig_split_all == 0).sum())
    n_te = int((orig_split_all == 1).sum())
    ok = (n_tr == 50000 and n_te == 10000)
    checks_passed &= ok
    print(f"  {'[OK]' if ok else '[FAIL]'} Train={n_tr} Test={n_te}  (expected 50000/10000)")

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

    t0 = time.time()
    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("W", data=W_all,
                          chunks=(min(256, args.n_total), K, param_dim))
        hf.create_dataset("labels", data=labels_all)
        hf.create_dataset("psnr", data=psnr_all)
        hf.create_dataset("n_epochs", data=n_epochs_all)
        hf.create_dataset("converged", data=converged_all)
        hf.create_dataset("orig_split", data=orig_split_all)
        hf.create_dataset("orig_index", data=orig_index_all)
        hf.create_dataset("encode_time", data=encode_time_all)

        hf.attrs["K"] = K
        hf.attrs["param_dim"] = param_dim
        hf.attrs["channels"] = 3
        hf.attrs["image_size"] = 32
        hf.attrs["dataset"] = "cifar10"
        hf.attrs["lr"] = 0.04
        hf.attrs["epochs_max"] = 3000
        hf.attrs["early_stop"] = 1e-5
        hf.attrs["kernel_size"] = 32
        hf.attrs["soft_clamp"] = True
        hf.attrs["n_total"] = args.n_total
        hf.attrs["n_train"] = 50000
        hf.attrs["n_test"] = 10000
        hf.attrs["mean_psnr"] = float(psnr_all.mean())
        hf.attrs["std_psnr"] = float(psnr_all.std())
        hf.attrs["min_psnr"] = float(psnr_all.min())
        hf.attrs["max_psnr"] = float(psnr_all.max())
        hf.attrs["pct_converged"] = float(converged_all.mean()) * 100
        hf.attrs["mean_encode_time_s"] = float(encode_time_all.mean())
        hf.attrs["total_encode_time_h"] = float(encode_time_all.sum()) / 3600.0
        hf.attrs["created_at"] = _now()

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

    n_below_35 = int((psnr_all < 35.0).sum())
    n_below_40 = int((psnr_all < 40.0).sum())
    n_below_45 = int((psnr_all < 45.0).sum())
    print(f"\n  below 35 dB  : {n_below_35} ({100*n_below_35/args.n_total:.3f}%)")
    print(f"  below 40 dB  : {n_below_40} ({100*n_below_40/args.n_total:.3f}%)")
    print(f"  below 45 dB  : {n_below_45} ({100*n_below_45/args.n_total:.2f}%)")
    print(f"  converged    : {converged_all.sum()} ({100*converged_all.mean():.1f}%)")

    print(f"\n{_hr()}")
    print("  Per-class PSNR")
    print(_hr())
    print(f"  {'Class':>12}  {'N':>6}  {'Mean':>7}  {'Std':>6}  {'Min':>7}  {'Max':>7}  {'<40dB':>6}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}")
    for d in range(10):
        mask = labels_all == d
        psnr_d = psnr_all[mask]
        n_d = len(psnr_d)
        n_low = int((psnr_d < 40.0).sum())
        print(
            f"  {CIFAR10_CLASSES[d]:>12}  {n_d:>6}  {psnr_d.mean():>7.2f}  {psnr_d.std():>6.2f}"
            f"  {psnr_d.min():>7.2f}  {psnr_d.max():>7.2f}  {n_low:>6}"
        )

    print(f"\n{_hr()}")
    print(f"  File size    : {file_mb:.1f} MB")
    print(f"  Output       : {out_file}")
    print(f"  Finished     : {_now()}")
    print(f"{'='*68}\n")

    # Write summary JSON
    summary = {
        "n_total": args.n_total,
        "mean_psnr": float(psnr_all.mean()),
        "std_psnr": float(psnr_all.std()),
        "min_psnr": float(psnr_all.min()),
        "max_psnr": float(psnr_all.max()),
        "pct_converged": float(converged_all.mean()) * 100,
        "n_below_35dB": n_below_35,
        "n_below_40dB": n_below_40,
        "n_below_45dB": n_below_45,
        "file_mb": file_mb,
        "finished_at": _now(),
        "per_class": {
            CIFAR10_CLASSES[d]: {
                "n": int((labels_all == d).sum()),
                "mean": float(psnr_all[labels_all == d].mean()),
                "min": float(psnr_all[labels_all == d].min()),
            }
            for d in range(10)
        }
    }
    summary_path = out_file.parent / "cifar10_gaussians_K500_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON : {summary_path}\n")


if __name__ == "__main__":
    main()
