"""
Find the minimum number of Gaussians (K) needed to encode MNIST at ≥ 30 dB PSNR.

Strategy:
  - Evaluate K ∈ [50, 75, 100, 125, 150, 200, 300, 500] in ascending order
  - For each K, encode N_EVAL fixed test images (seed=42) and measure PSNR
  - Stop and report as soon as mean PSNR ≥ TARGET_PSNR dB
  - Results written to 3 CSVs row-by-row (fault-tolerant, resumable)

Output CSVs:
  results/k_sweep_psnr.csv        — one row per K (summary statistics)
  results/k_sweep_convergence.csv — one row per (K, image, log_every epoch)
  results/k_sweep_per_image.csv   — one row per (K, image) with Gaussian health

Usage:
    python scripts/k_sweep_mnist.py [--device cuda] [--n_eval 200]
                                    [--target_psnr 30] [--epochs 3000]
                                    [--out_dir results/]
                                    [--recycle_every 300] [--recycle_thresh 0.05]
"""

import argparse
import csv
import datetime
import json
import math
import os
import time

import torch
import torchvision

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.encode import encode_image, _dead_mask, _to_physical
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_VALUES = [50, 75, 100, 125, 150, 200, 300, 500]

PSNR_FIELDNAMES = [
    "K", "epochs", "lr", "kernel_size", "recycle_every",
    "n_images",
    "mean_psnr", "std_psnr", "min_psnr", "max_psnr", "pct_above_target",
    "mean_alive_frac", "std_alive_frac",
    "mean_recycling_events",
    "pct_early_stop",
    "mean_conv_epoch",
    "mean_time_s", "total_time_s",
]

CONV_FIELDNAMES = [
    "K", "image_idx", "epoch", "loss", "psnr_db", "n_dead", "recycling_event",
]

PER_IMAGE_FIELDNAMES = [
    "K", "image_idx", "final_psnr", "convergence_epoch", "n_recycling_events",
    "n_dead_final", "mean_colour", "std_colour", "mean_sigma_x", "mean_sigma_y",
    "pos_x_std", "pos_y_std",
]

LOG_EVERY = 50  # history entry every N epochs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_images(n: int = 200, seed: int = 42,
                     mnist_root: str = "data") -> list:
    """Return n fixed MNIST test images as [28,28] float tensors in [0,1]."""
    dataset = torchvision.datasets.MNIST(
        root=mnist_root, train=False, download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    torch.manual_seed(seed)
    idx = torch.randperm(len(dataset))[:n]
    return [dataset[i.item()][0].squeeze(0) for i in idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(rendered: torch.Tensor, original: torch.Tensor) -> float:
    """10 * log10(1/MSE), capped at 100 dB."""
    mse = torch.mean((rendered - original) ** 2).item()
    return 100.0 if mse < 1e-10 else min(100.0, 10 * math.log10(1.0 / mse))


def render_w(W_phys: torch.Tensor, kernel_size: int, device: str) -> torch.Tensor:
    """Render [K, 7] physical params → [28, 28] image tensor on device."""
    sigma_x = W_phys[:, 0].clamp(1e-4, 1.0)
    sigma_y = W_phys[:, 1].clamp(1e-4, 1.0)
    rho     = W_phys[:, 2].clamp(-0.999, 0.999)
    colour  = W_phys[:, 4].clamp(0, 1).unsqueeze(1)
    coords  = torch.stack([W_phys[:, 5], W_phys[:, 6]], dim=1)
    img = generate_2D_gaussian_splatting(
        kernel_size=kernel_size,
        sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
        coords=coords, colours=colour,
        image_size=(28, 28), channels=1, device=device,
    )
    return img[:, :, 0]  # [28, 28]


# ---------------------------------------------------------------------------
# Single-K evaluation
# ---------------------------------------------------------------------------

def evaluate_k(
    K: int,
    images: list,
    config: dict,
    device: str,
    target_psnr: float,
    conv_writer: csv.DictWriter,
    per_image_writer: csv.DictWriter,
    conv_file,
    per_image_file,
    verbose: bool = True,
) -> dict:
    """Encode all images with K Gaussians, return PSNR statistics."""
    psnrs, times = [], []
    alive_fracs = []
    recycling_events_per_image = []
    convergence_epochs = []
    ks = config["kernel_size"]
    n = len(images)
    k_start = time.time()

    for i, img in enumerate(images):
        t0 = time.time()
        W_phys, _, history = encode_image(
            img,
            K=K,
            epochs=config["epochs"],
            lr=config["lr"],
            kernel_size=ks,
            early_stop_threshold=config["early_stop"],
            device=device,
            return_history=True,
            recycle_every=config["recycle_every"],
            recycle_threshold=config["recycle_thresh"],
            log_every=LOG_EVERY,
        )
        rendered = render_w(W_phys.to(device), ks, device)
        psnr = compute_psnr(rendered, img.to(device))
        elapsed = time.time() - t0

        # Extract convergence info from history
        n_recycling_events = sum(1 for h in history if h["recycling_event"])
        last_epoch = history[-1]["epoch"] if history else config["epochs"] - 1
        # Convergence = early stop triggered (last logged epoch << max epochs)
        max_epoch = config["epochs"] - 1
        convergence_epoch = last_epoch if last_epoch < max_epoch - LOG_EVERY else -1

        # Alive fraction: Gaussians not dead at end
        n_dead_final = history[-1]["n_dead"] if history else 0
        alive_frac = (K - n_dead_final) / K

        # Per-Gaussian statistics from physical params
        mean_colour = float(W_phys[:, 4].mean())
        std_colour = float(W_phys[:, 4].std())
        mean_sigma_x = float(W_phys[:, 0].mean())
        mean_sigma_y = float(W_phys[:, 1].mean())
        pos_x_std = float(W_phys[:, 5].std())
        pos_y_std = float(W_phys[:, 6].std())

        psnrs.append(psnr)
        times.append(elapsed)
        alive_fracs.append(alive_frac)
        recycling_events_per_image.append(n_recycling_events)
        convergence_epochs.append(convergence_epoch)

        # Write convergence rows (fault-tolerant: one per history entry)
        for h in history:
            conv_writer.writerow({
                "K": K,
                "image_idx": i,
                "epoch": h["epoch"],
                "loss": round(h["loss"], 6),
                "psnr_db": round(h["psnr_db"], 3),
                "n_dead": h["n_dead"],
                "recycling_event": int(h["recycling_event"]),
            })
        conv_file.flush()

        # Write per-image row
        per_image_writer.writerow({
            "K": K,
            "image_idx": i,
            "final_psnr": round(psnr, 3),
            "convergence_epoch": convergence_epoch,
            "n_recycling_events": n_recycling_events,
            "n_dead_final": n_dead_final,
            "mean_colour": round(mean_colour, 4),
            "std_colour": round(std_colour, 4),
            "mean_sigma_x": round(mean_sigma_x, 4),
            "mean_sigma_y": round(mean_sigma_y, 4),
            "pos_x_std": round(pos_x_std, 4),
            "pos_y_std": round(pos_y_std, 4),
        })
        per_image_file.flush()

        if verbose:
            running_mean = sum(psnrs) / len(psnrs)
            elapsed_total = time.time() - k_start
            eta_s = elapsed_total / (i + 1) * (n - i - 1)
            eta_str = f"{eta_s/60:.1f}min" if eta_s >= 60 else f"{eta_s:.0f}s"
            alive_pct = alive_frac * 100
            conv_str = f"YES@{convergence_epoch}" if convergence_epoch >= 0 else "NO"
            print(
                f"  [{i+1:3d}/{n}] PSNR={psnr:5.2f}dB | "
                f"alive={K-n_dead_final}/{K}({alive_pct:.0f}%) | "
                f"recycled={n_recycling_events}x | "
                f"conv={conv_str} | "
                f"{elapsed:.1f}s | ETA={eta_str}",
                flush=True,
            )

    psnr_t = torch.tensor(psnrs)
    alive_t = torch.tensor(alive_fracs)
    conv_epochs_valid = [e for e in convergence_epochs if e >= 0]

    return {
        "K": K,
        "epochs": config["epochs"],
        "lr": config["lr"],
        "kernel_size": ks,
        "recycle_every": config["recycle_every"],
        "n_images": len(images),
        "mean_psnr": float(psnr_t.mean()),
        "std_psnr": float(psnr_t.std()),
        "min_psnr": float(psnr_t.min()),
        "max_psnr": float(psnr_t.max()),
        "pct_above_target": float(100 * (psnr_t >= target_psnr).float().mean()),
        "mean_alive_frac": float(alive_t.mean()),
        "std_alive_frac": float(alive_t.std()),
        "mean_recycling_events": float(sum(recycling_events_per_image) / len(recycling_events_per_image)),
        "pct_early_stop": float(100 * sum(1 for e in convergence_epochs if e >= 0) / len(convergence_epochs)),
        "mean_conv_epoch": float(sum(conv_epochs_valid) / len(conv_epochs_valid)) if conv_epochs_valid else -1.0,
        "mean_time_s": float(sum(times) / len(times)),
        "total_time_s": float(sum(times)),
        "all_psnrs": psnrs,
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_ks(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(int(row["K"]))
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="K-sweep to find minimum Gaussians for PSNR ≥ target"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_eval", type=int, default=200, help="Number of eval images")
    parser.add_argument("--target_psnr", type=float, default=30.0,
                        help="Target mean PSNR in dB (default: 30)")
    parser.add_argument("--epochs", type=int, default=3000,
                        help="Max encoding epochs per image")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--early_stop", type=float, default=1e-4,
                        help="Stop encoding early if loss < threshold")
    parser.add_argument("--k_values", nargs="+", type=int, default=K_VALUES,
                        help="K values to sweep (ascending)")
    parser.add_argument("--out_dir", default="results/")
    parser.add_argument("--mnist_root", default="data")
    parser.add_argument("--recycle_every", type=int, default=300,
                        help="Recycle dead Gaussians every N epochs (0 = disabled)")
    parser.add_argument("--recycle_thresh", type=float, default=0.05,
                        help="Dead-Gaussian detection threshold")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv      = os.path.join(args.out_dir, "k_sweep_psnr.csv")
    out_conv_csv = os.path.join(args.out_dir, "k_sweep_convergence.csv")
    out_pi_csv   = os.path.join(args.out_dir, "k_sweep_per_image.csv")
    out_json     = os.path.join(args.out_dir, "k_sweep_result.json")

    def log(msg=""):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    log(f"Device : {args.device}")
    if args.device == "cuda":
        log(f"GPU    : {torch.cuda.get_device_name(0)}")
    log(f"Target : mean PSNR ≥ {args.target_psnr} dB")
    log(f"Config : epochs={args.epochs}, lr={args.lr}, "
        f"kernel_size={args.kernel_size}, early_stop={args.early_stop}, "
        f"recycle_every={args.recycle_every}, recycle_thresh={args.recycle_thresh}")
    log(f"K sweep: {args.k_values}")
    log()

    log(f"Loading {args.n_eval} eval images from {args.mnist_root} …")
    images = load_eval_images(n=args.n_eval, mnist_root=args.mnist_root)
    log(f"Loaded {len(images)} images.")

    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "kernel_size": args.kernel_size,
        "early_stop": args.early_stop,
        "recycle_every": args.recycle_every,
        "recycle_thresh": args.recycle_thresh,
    }

    completed = load_completed_ks(out_csv)

    # Open all 3 CSV files
    psnr_file_exists = os.path.exists(out_csv)
    conv_file_exists = os.path.exists(out_conv_csv)
    pi_file_exists   = os.path.exists(out_pi_csv)

    psnr_file = open(out_csv, "a", newline="")
    conv_file = open(out_conv_csv, "a", newline="")
    pi_file   = open(out_pi_csv, "a", newline="")

    psnr_writer = csv.DictWriter(psnr_file, fieldnames=PSNR_FIELDNAMES)
    conv_writer = csv.DictWriter(conv_file, fieldnames=CONV_FIELDNAMES)
    pi_writer   = csv.DictWriter(pi_file, fieldnames=PER_IMAGE_FIELDNAMES)

    if not psnr_file_exists:
        psnr_writer.writeheader()
        psnr_file.flush()
    if not conv_file_exists:
        conv_writer.writeheader()
        conv_file.flush()
    if not pi_file_exists:
        pi_writer.writeheader()
        pi_file.flush()

    if completed:
        log(f"Resuming — already done: K={sorted(completed)}")
    log()

    optimal_K = None
    all_results = []

    for K in sorted(args.k_values):
        if K in completed:
            with open(out_csv, newline="") as f:
                for row in csv.DictReader(f):
                    if int(row["K"]) == K:
                        mean_psnr = float(row["mean_psnr"])
                        log(f"K={K:3d} — skip (already done, mean PSNR={mean_psnr:.2f} dB)")
                        if mean_psnr >= args.target_psnr and optimal_K is None:
                            optimal_K = K
            continue

        log(f"{'='*60}")
        log(f"K={K}  ({len(images)} images, up to {config['epochs']} epochs each)")
        log(f"{'='*60}")
        t_k_start = time.time()
        result = evaluate_k(
            K, images, config, args.device, args.target_psnr,
            conv_writer, pi_writer, conv_file, pi_file,
        )
        t_k_total = time.time() - t_k_start

        # K-level summary block
        alive_pct = result["mean_alive_frac"] * 100
        min_alive_pct = (min(result.get("all_psnrs", [0])) / 100) * 100  # placeholder
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] {'═'*54}", flush=True)
        print(f"[{ts}] K={K} DONE", flush=True)
        print(f"[{ts}] PSNR    : {result['mean_psnr']:.2f} ± {result['std_psnr']:.2f} dB  "
              f"[{result['min_psnr']:.1f} – {result['max_psnr']:.1f}]  "
              f"{result['pct_above_target']:.0f}% ≥ {args.target_psnr} dB", flush=True)
        print(f"[{ts}] Alive   : {alive_pct:.0f}% avg  — want >85% for good diffusion data", flush=True)
        print(f"[{ts}] Recycled: {result['mean_recycling_events']:.1f} events/img avg", flush=True)
        print(f"[{ts}] Converge: {result['pct_early_stop']:.0f}% early-stopped  (want >50% for diffusion quality)", flush=True)
        mean_ce = result['mean_conv_epoch']
        ce_str = f"avg epoch {mean_ce:.0f}" if mean_ce >= 0 else "none"
        print(f"[{ts}] Conv epoch: {ce_str}", flush=True)
        print(f"[{ts}] CSV     : {out_csv}  (row appended & flushed)", flush=True)
        print(f"[{ts}] {'═'*54}\n", flush=True)

        row = {k: v for k, v in result.items() if k in PSNR_FIELDNAMES}
        psnr_writer.writerow(row)
        psnr_file.flush()

        all_results.append(result)

        if result["mean_psnr"] >= args.target_psnr and optimal_K is None:
            optimal_K = K
            log(f"*** TARGET REACHED: K={K} → mean PSNR {result['mean_psnr']:.2f} dB ≥ {args.target_psnr} dB ***")
        log()

    psnr_file.close()
    conv_file.close()
    pi_file.close()

    # Summary JSON
    summary = {
        "target_psnr": args.target_psnr,
        "optimal_K": optimal_K,
        "config": config,
        "n_eval": args.n_eval,
        "results": [
            {k: v for k, v in r.items() if k != "all_psnrs"}
            for r in all_results
        ],
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"{'='*60}")
    log(f"K sweep complete.")
    log(f"Results      → {out_csv}")
    log(f"Convergence  → {out_conv_csv}")
    log(f"Per-image    → {out_pi_csv}")
    log(f"Summary JSON → {out_json}")
    if optimal_K is not None:
        log(f"Optimal K = {optimal_K}  (first K achieving mean PSNR ≥ {args.target_psnr} dB)")
    else:
        log(f"Target PSNR {args.target_psnr} dB NOT reached for any K in {args.k_values}.")
        log("Consider: larger K, more epochs, or a higher lr.")


if __name__ == "__main__":
    main()
