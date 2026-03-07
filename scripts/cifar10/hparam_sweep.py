"""
CIFAR-10 Gaussian Encoding Hyperparameter Sweep

Encodes a fixed sample of CIFAR-10 images with configurable hyperparameters
and reports PSNR, encoding time, and dead Gaussian statistics.

Usage:
    python scripts/cifar10/hparam_sweep.py --K 200 --epochs 3000 --lr 5e-3
    python scripts/cifar10/hparam_sweep.py --K 200 --epochs 3000 --lr 5e-3 --n_images 50

Results are saved to: reports/cifar10/sweep_K{K}_ep{epochs}_lr{lr}_ks{kernel_size}.json
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encode import encode_image, _to_physical, _dead_mask  # noqa: E402


def _psnr(mse: float) -> float:
    if mse < 1e-10:
        return 100.0
    return min(100.0, 10.0 * math.log10(1.0 / mse))


def load_cifar10_sample(n_images: int = 20, seed: int = 42) -> list:
    """Load a fixed sample of CIFAR-10 images (2 per class by default)."""
    cifar_dir = PROJECT_ROOT / "data" / "cifar10" / "raw" / "cifar-10-batches-py"

    # Load all training batches
    all_images = []
    all_labels = []
    for i in range(1, 6):
        batch_path = cifar_dir / f"data_batch_{i}"
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        all_images.append(batch[b"data"])
        all_labels.extend(batch[b"labels"])

    # Also load test batch
    with open(cifar_dir / "test_batch", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    all_images.append(batch[b"data"])
    all_labels.extend(batch[b"labels"])

    images = np.concatenate(all_images, axis=0)  # [60000, 3072]
    labels = np.array(all_labels)

    # Reshape to [N, 3, 32, 32]
    images = images.reshape(-1, 3, 32, 32)

    # Select n_images: stratified by class
    rng = np.random.RandomState(seed)
    per_class = max(1, n_images // 10)
    selected_indices = []
    for cls in range(10):
        cls_indices = np.where(labels == cls)[0]
        chosen = rng.choice(cls_indices, size=per_class, replace=False)
        selected_indices.extend(chosen.tolist())

    # Trim to exact n_images
    selected_indices = selected_indices[:n_images]

    samples = []
    for idx in selected_indices:
        img = torch.from_numpy(images[idx]).float() / 255.0  # [3, 32, 32]
        samples.append({"image": img, "label": int(labels[idx]), "index": idx})

    return samples


def run_sweep(args):
    device = args.device
    samples = load_cifar10_sample(n_images=args.n_images, seed=args.seed)
    print(f"Loaded {len(samples)} CIFAR-10 images")
    print(f"Config: K={args.K}, epochs={args.epochs}, lr={args.lr}, "
          f"kernel_size={args.kernel_size}, recycle_every={args.recycle_every}, "
          f"sigma_act={args.sigma_activation}, init={args.init_mode}, "
          f"soft_clamp={args.soft_clamp}, no_sched={args.no_scheduler}")
    print(f"Device: {device}")
    print()

    results = []
    total_start = time.time()

    for i, sample in enumerate(samples):
        img = sample["image"]  # [3, 32, 32]
        label = sample["label"]

        t0 = time.time()
        W_phys, final_loss = encode_image(
            img,
            K=args.K,
            epochs=args.epochs,
            lr=args.lr,
            kernel_size=args.kernel_size,
            early_stop_threshold=args.early_stop,
            device=device,
            recycle_every=args.recycle_every,
            recycle_threshold=args.recycle_threshold,
            sigma_activation=args.sigma_activation,
            init_mode=args.init_mode,
            soft_clamp=args.soft_clamp,
            use_scheduler=not args.no_scheduler,
        )
        elapsed = time.time() - t0

        psnr = _psnr(final_loss)
        converged = final_loss < args.early_stop

        # Count dead Gaussians: check alpha < 0.05
        with torch.no_grad():
            alpha_vals = W_phys[:, 3]  # alpha column
            n_dead = int((alpha_vals < 0.05).sum().item())

        results.append({
            "index": sample["index"],
            "label": label,
            "psnr": psnr,
            "mse": final_loss,
            "converged": converged,
            "n_dead": n_dead,
            "time_s": elapsed,
        })

        status = "CONV" if converged else "    "
        print(f"  [{i+1:3d}/{len(samples)}] class={label}  PSNR={psnr:6.2f}dB  "
              f"dead={n_dead:3d}/{args.K}  t={elapsed:.1f}s  {status}")

    total_elapsed = time.time() - total_start

    # Compute summary statistics
    psnrs = [r["psnr"] for r in results]
    times = [r["time_s"] for r in results]
    deads = [r["n_dead"] for r in results]

    summary = {
        "config": {
            "K": args.K,
            "epochs": args.epochs,
            "lr": args.lr,
            "kernel_size": args.kernel_size,
            "early_stop": args.early_stop,
            "recycle_every": args.recycle_every,
            "recycle_threshold": args.recycle_threshold,
            "n_images": len(samples),
            "seed": args.seed,
            "device": device,
            "sigma_activation": args.sigma_activation,
            "init_mode": args.init_mode,
            "soft_clamp": args.soft_clamp,
            "no_scheduler": args.no_scheduler,
        },
        "summary": {
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
            "psnr_min": float(np.min(psnrs)),
            "psnr_max": float(np.max(psnrs)),
            "psnr_median": float(np.median(psnrs)),
            "pct_above_40dB": float(np.mean(np.array(psnrs) >= 40.0)) * 100,
            "pct_above_35dB": float(np.mean(np.array(psnrs) >= 35.0)) * 100,
            "pct_above_30dB": float(np.mean(np.array(psnrs) >= 30.0)) * 100,
            "pct_converged": float(np.mean([r["converged"] for r in results])) * 100,
            "dead_mean": float(np.mean(deads)),
            "dead_max": int(np.max(deads)),
            "time_mean_s": float(np.mean(times)),
            "time_total_s": total_elapsed,
            "time_per_60k_h": float(np.mean(times)) * 60000 / 3600,
        },
        "per_image": results,
    }

    # Save results
    out_dir = PROJECT_ROOT / "reports" / "cifar10"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build filename with optional suffixes for non-default settings
    fname = f"sweep_K{args.K}_ep{args.epochs}_lr{args.lr}_ks{args.kernel_size}"
    if args.soft_clamp:
        fname += "_softclamp"
    if args.sigma_activation != "sigmoid":
        fname += f"_{args.sigma_activation}"
    if args.init_mode != "brightness":
        fname += f"_{args.init_mode}"
    if args.no_scheduler:
        fname += "_nosched"
    if args.early_stop != 1e-4:
        fname += f"_es{args.early_stop}"
    fname += ".json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: K={args.K}  epochs={args.epochs}  lr={args.lr}  ks={args.kernel_size}")
    print(f"  PSNR : mean={summary['summary']['psnr_mean']:.2f}  "
          f"std={summary['summary']['psnr_std']:.2f}  "
          f"min={summary['summary']['psnr_min']:.2f}  "
          f"max={summary['summary']['psnr_max']:.2f} dB")
    print(f"  >=40dB: {summary['summary']['pct_above_40dB']:.0f}%  "
          f">=35dB: {summary['summary']['pct_above_35dB']:.0f}%  "
          f">=30dB: {summary['summary']['pct_above_30dB']:.0f}%")
    print(f"  Dead : mean={summary['summary']['dead_mean']:.1f}  "
          f"max={summary['summary']['dead_max']}")
    print(f"  Time : {summary['summary']['time_mean_s']:.2f} s/img  "
          f"total={total_elapsed:.0f}s  "
          f"est 60k: {summary['summary']['time_per_60k_h']:.1f}h")
    print(f"  Saved: {out_path}")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 encoding hyperparameter sweep")
    parser.add_argument("--K", type=int, default=200, help="Number of Gaussians")
    parser.add_argument("--epochs", type=int, default=3000, help="Max epochs per image")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--kernel_size", type=int, default=11, help="Gaussian kernel size")
    parser.add_argument("--early_stop", type=float, default=1e-4,
                        help="MSE early-stop threshold (~40dB)")
    parser.add_argument("--recycle_every", type=int, default=300,
                        help="Dead Gaussian recycle interval")
    parser.add_argument("--recycle_threshold", type=float, default=0.05,
                        help="Brightness threshold for dead detection")
    parser.add_argument("--n_images", type=int, default=20,
                        help="Number of sample images (2 per class)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sigma_activation", type=str, default="sigmoid",
                        choices=["sigmoid", "softplus"], help="Sigma activation function")
    parser.add_argument("--init_mode", type=str, default="brightness",
                        choices=["brightness", "gradient"], help="Init position sampling mode")
    parser.add_argument("--soft_clamp", action="store_true",
                        help="Use soft saturation (1-exp(-x)) instead of hard clamp")
    parser.add_argument("--no_scheduler", action="store_true",
                        help="Disable CosineAnnealingWarmRestarts (constant lr)")
    args = parser.parse_args()

    run_sweep(args)


if __name__ == "__main__":
    main()
