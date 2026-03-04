"""
Hyperparameter search for Gaussian encoding on MNIST.

Scientific design:
  Phase 1 — K sweep:  K ∈ [10, 20, 35, 50, 70, 100], fixed lr/epochs/kernel_size
  Phase 2 — Optimizer sweep: lr × epochs × kernel_size (27 configs), given best K

Usage:
    python scripts/hparam_search_encode.py --phase 1 [--n_eval 200] [--device cpu]
    python scripts/hparam_search_encode.py --phase 2 --best_k 70 [--n_eval 200]
"""

import argparse
import csv
import json
import math
import os
import time
from itertools import product

import torch
import torchvision

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.encode import encode_image
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting


# ---------------------------------------------------------------------------
# Evaluation set
# ---------------------------------------------------------------------------

def load_eval_images(n: int = 200, seed: int = 42,
                     mnist_root: str = "/tmp/mnist") -> list:
    """Return n fixed MNIST test images as (28,28) float tensors in [0,1]."""
    dataset = torchvision.datasets.MNIST(
        root=mnist_root, train=False, download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    torch.manual_seed(seed)
    idx = torch.randperm(len(dataset))[:n]
    images = [dataset[i.item()][0].squeeze(0) for i in idx]  # [28,28]
    return images


# ---------------------------------------------------------------------------
# Single-image encode + measure
# ---------------------------------------------------------------------------

def compute_psnr(rendered: torch.Tensor, original: torch.Tensor) -> float:
    mse = torch.mean((rendered - original) ** 2).item()
    return 100.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)


def encode_and_measure(image: torch.Tensor, config: dict, device: str) -> dict:
    """Encode one image, render back, return {psnr, ssim_loss, time_s}."""
    t0 = time.time()
    W_phys, _ = encode_image(
        image,
        K=config["K"],
        epochs=config["epochs"],
        lr=config["lr"],
        kernel_size=config["kernel_size"],
        early_stop_threshold=config.get("early_stop", 0.005),
        device=device,
    )
    t1 = time.time()

    # Render back
    sigma_x = W_phys[:, 0].clamp(1e-4, 1.0)
    sigma_y = W_phys[:, 1].clamp(1e-4, 1.0)
    rho = W_phys[:, 2].clamp(-0.999, 0.999)
    colour = W_phys[:, 4].clamp(0, 1).unsqueeze(1)
    coords = torch.stack([W_phys[:, 5], W_phys[:, 6]], dim=1)

    rendered = generate_2D_gaussian_splatting(
        kernel_size=config["kernel_size"],
        sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
        coords=coords, colours=colour,
        image_size=(28, 28), channels=1, device=device,
    )[:, :, 0]

    psnr = compute_psnr(rendered, image.to(device))
    return {"psnr": psnr, "time_s": t1 - t0}


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

FIELDNAMES = ["K", "epochs", "lr", "kernel_size",
              "mean_psnr", "std_psnr", "mean_ssim", "std_ssim", "mean_time_s"]


def _config_key(config: dict) -> str:
    return f"{config['K']}_{config['epochs']}_{config['lr']}_{config['kernel_size']}"


def _load_completed(csv_path: str) -> set:
    """Return set of already-completed config keys from an existing CSV."""
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['K']}_{row['epochs']}_{row['lr']}_{row['kernel_size']}"
            done.add(key)
    return done


def run_phase(phase_id: int, configs: list, images: list,
              out_csv: str, device: str):
    """Iterate configs × images, write each row to CSV immediately (fault-tolerant)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    completed = _load_completed(out_csv)

    file_exists = os.path.exists(out_csv)
    csv_file = open(out_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    best_row = None

    for config in configs:
        key = _config_key(config)
        if key in completed:
            print(f"  [skip] {config}")
            continue

        print(f"  [run ] {config}")
        psnrs, times = [], []
        for i, img in enumerate(images):
            metrics = encode_and_measure(img, config, device)
            psnrs.append(metrics["psnr"])
            times.append(metrics["time_s"])
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(images)} — mean PSNR so far: {sum(psnrs)/len(psnrs):.2f}")

        psnr_t = torch.tensor(psnrs)
        row = {
            "K": config["K"],
            "epochs": config["epochs"],
            "lr": config["lr"],
            "kernel_size": config["kernel_size"],
            "mean_psnr": f"{psnr_t.mean().item():.4f}",
            "std_psnr": f"{psnr_t.std().item():.4f}",
            "mean_ssim": "N/A",   # SSIM-loss not separately tracked to keep it fast
            "std_ssim": "N/A",
            "mean_time_s": f"{sum(times)/len(times):.4f}",
        }
        writer.writerow(row)
        csv_file.flush()

        if best_row is None or float(row["mean_psnr"]) > float(best_row["mean_psnr"]):
            best_row = row

    csv_file.close()
    print(f"\nPhase {phase_id} done → {out_csv}")
    return best_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for Gaussian encoding")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--n_eval", type=int, default=200, help="Number of eval images")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--best_k", type=int, default=70,
                        help="Best K to fix for phase 2 (required for --phase 2)")
    parser.add_argument("--out_dir", default="results/", help="Directory for CSV outputs")
    parser.add_argument("--mnist_root", default="/tmp/mnist")
    args = parser.parse_args()

    print(f"Loading {args.n_eval} eval images…")
    images = load_eval_images(n=args.n_eval, mnist_root=args.mnist_root)
    print(f"Loaded {len(images)} images.")

    if args.phase == 1:
        configs = [
            {"K": K, "epochs": 1000, "lr": 5e-3, "kernel_size": 11, "early_stop": 0.005}
            for K in [10, 20, 35, 50, 70, 100]
        ]
        out_csv = os.path.join(args.out_dir, "hparam_phase1.csv")
        print(f"\n=== Phase 1: K sweep ({len(configs)} configs) ===")
        best = run_phase(1, configs, images, out_csv, args.device)
        if best:
            print(f"Best K by PSNR: K={best['K']} — mean PSNR={best['mean_psnr']} dB")

    elif args.phase == 2:
        lrs = [1e-3, 5e-3, 1e-2]
        epochs_list = [500, 1000, 2000]
        kernel_sizes = [9, 11, 15]
        configs = [
            {"K": args.best_k, "epochs": e, "lr": lr, "kernel_size": ks, "early_stop": 0.005}
            for lr, e, ks in product(lrs, epochs_list, kernel_sizes)
        ]
        out_csv = os.path.join(args.out_dir, "hparam_phase2.csv")
        print(f"\n=== Phase 2: Optimizer sweep ({len(configs)} configs, K={args.best_k}) ===")
        best = run_phase(2, configs, images, out_csv, args.device)
        if best:
            best_json = os.path.join(args.out_dir, "best_config.json")
            with open(best_json, "w") as f:
                json.dump(best, f, indent=2)
            print(f"Best config saved to {best_json}")
            print(f"Best: {best}")


if __name__ == "__main__":
    main()
