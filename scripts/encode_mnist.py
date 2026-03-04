"""
Encode MNIST images into 2D Gaussian splatting representations.

SLURM-parallelisable via --chunk_id / --num_chunks (interleaved assignment).
Resumable: skips already-written .pt files.

Output .pt format:
    {"W": W_phys,    # [K, 7] float32
     "label": int,   # MNIST digit class
     "psnr": float}  # reconstruction PSNR at save time

Usage:
    python scripts/encode_mnist.py \\
        --split test --chunk_id 0 --num_chunks 10 \\
        --out_dir data/mnist_gaussian_representations/test/ \\
        --K 70 --epochs 1000 --lr 5e-3 --kernel_size 11 --device cpu
"""

import argparse
import json
import math
import os
import time

import torch
import torchvision

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.encode import encode_image
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_mnist_images(split: str, data_root: str = "/tmp/mnist") -> list:
    """Return list of (image [28,28] float, label) pairs for train or test split."""
    is_train = split == "train"
    dataset = torchvision.datasets.MNIST(
        root=data_root, train=is_train, download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return [(dataset[i][0].squeeze(0), dataset[i][1]) for i in range(len(dataset))]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(rendered: torch.Tensor, original: torch.Tensor) -> float:
    """10 * log10(1/MSE), capped at 100 dB."""
    mse = torch.mean((rendered - original) ** 2).item()
    return 100.0 if mse < 1e-10 else min(100.0, 10 * math.log10(1.0 / mse))


def _render_w(W_phys: torch.Tensor, kernel_size: int, device: str) -> torch.Tensor:
    """Render [K, 7] physical params → [28,28] image tensor."""
    sigma_x = W_phys[:, 0].clamp(1e-4, 1.0)
    sigma_y = W_phys[:, 1].clamp(1e-4, 1.0)
    rho = W_phys[:, 2].clamp(-0.999, 0.999)
    colour = W_phys[:, 4].clamp(0, 1).unsqueeze(1)
    coords = torch.stack([W_phys[:, 5], W_phys[:, 6]], dim=1)
    img = generate_2D_gaussian_splatting(
        kernel_size=kernel_size,
        sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
        coords=coords, colours=colour,
        image_size=(28, 28), channels=1, device=device,
    )
    return img[:, :, 0]


# ---------------------------------------------------------------------------
# Chunk encoding
# ---------------------------------------------------------------------------

def encode_chunk(items: list, out_dir: str, config: dict, device: str) -> dict:
    """
    Encode a list of (tensor, label, global_idx) items.
    Skips already-written .pt files. Returns summary stats.
    """
    n_encoded = n_skipped = n_failed = 0
    psnrs = []
    t_start = time.time()

    for image, label, global_idx in items:
        out_path = os.path.join(out_dir, f"{global_idx:06d}.pt")

        if os.path.exists(out_path):
            n_skipped += 1
            continue

        try:
            W_phys, _ = encode_image(
                image,
                K=config["K"],
                epochs=config["epochs"],
                lr=config["lr"],
                kernel_size=config["kernel_size"],
                early_stop_threshold=config.get("early_stop", 0.005),
                device=device,
            )
            rendered = _render_w(W_phys.to(device), config["kernel_size"], device)
            psnr = compute_psnr(rendered, image.to(device))
            torch.save(
                {"W": W_phys.cpu(), "label": int(label), "psnr": float(psnr)},
                out_path,
            )
            psnrs.append(psnr)
            n_encoded += 1
        except Exception as e:
            print(f"  [WARN] Failed on index {global_idx}: {e}")
            n_failed += 1

    total_time = time.time() - t_start
    n_total = n_encoded + n_skipped + n_failed
    ips = n_encoded / total_time if total_time > 0 and n_encoded > 0 else 0.0
    psnr_t = torch.tensor(psnrs) if psnrs else torch.zeros(1)

    return {
        "n_encoded": n_encoded,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "median_psnr": float(psnr_t.median()),
        "mean_psnr": float(psnr_t.mean()),
        "std_psnr": float(psnr_t.std()) if len(psnrs) > 1 else 0.0,
        "total_time_s": round(total_time, 1),
        "images_per_second": round(ips, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Encode MNIST into Gaussian representations")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--chunk_id", type=int, required=True,
                        help="Index of this chunk (0-based)")
    parser.add_argument("--num_chunks", type=int, required=True,
                        help="Total number of parallel chunks")
    parser.add_argument("--out_dir", required=True, help="Output directory for .pt files")
    parser.add_argument("--K", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--early_stop", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data_root", default="/tmp/mnist",
                        help="Root dir for torchvision MNIST download")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = {
        "K": args.K,
        "epochs": args.epochs,
        "lr": args.lr,
        "kernel_size": args.kernel_size,
        "early_stop": args.early_stop,
    }

    print(f"Loading MNIST {args.split} split…")
    all_images = get_mnist_images(args.split, data_root=args.data_root)
    print(f"Total images: {len(all_images)}")

    # Interleaved chunk assignment: chunk c processes global indices c, c+n, c+2n, ...
    items = [
        (all_images[i][0], all_images[i][1], i)
        for i in range(args.chunk_id, len(all_images), args.num_chunks)
    ]
    print(f"Chunk {args.chunk_id}/{args.num_chunks}: {len(items)} images to process")
    print(f"Config: {config}")

    summary = encode_chunk(items, args.out_dir, config, args.device)
    summary["chunk_id"] = args.chunk_id

    summary_path = os.path.join(args.out_dir, f"chunk_{args.chunk_id:04d}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nChunk {args.chunk_id} summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
