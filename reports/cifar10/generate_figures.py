#!/usr/bin/env python
"""
Generate figures and convergence animations for CIFAR-10 encoding report.

Produces:
  figures/fig_01_reconstruction_grid.png   — 5×3 grid (original / encoded / residual)
  figures/fig_02_convergence_curves.png    — PSNR vs epoch for the 5 examples
  figures/fig_03_psnr_histogram.png        — PSNR distribution over 50 images
  videos/<class>_convergence.gif           — encoding animation for 5 examples

Usage (from project root):
    python reports/cifar10/generate_figures.py [--device cpu|cuda]
        [--n_hist 50] [--frame_every 20]
"""

import sys
import math
import time
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encode import (
    encode_image, encode_batch,
    _init_gaussians, _to_physical, _render,
)

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
VID_DIR = REPORT_DIR / "videos"

# Encoding config — matches production settings
ENC = dict(K=500, epochs=3000, lr=0.04, kernel_size=32,
           early_stop_threshold=1e-5, soft_clamp=True,
           use_scheduler=False, sigma_activation="sigmoid",
           init_mode="brightness")


def psnr_db(mse):
    if mse < 1e-10:
        return 100.0
    return min(100.0, 10.0 * math.log10(1.0 / mse))


def load_cifar10():
    """Load CIFAR-10 images and labels."""
    cifar_dir = PROJECT_ROOT / "data" / "cifar10" / "raw" / "cifar-10-batches-py"
    all_images, all_labels = [], []
    for i in range(1, 6):
        with open(cifar_dir / f"data_batch_{i}", "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        all_images.append(batch[b"data"])
        all_labels.extend(batch[b"labels"])
    with open(cifar_dir / "test_batch", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    all_images.append(batch[b"data"])
    all_labels.extend(batch[b"labels"])
    images = np.concatenate(all_images).reshape(-1, 3, 32, 32)
    labels = np.array(all_labels, dtype=np.uint8)
    return images, labels


# Class names for CIFAR-10
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# ─── Encode with frame capture ───────────────────────────────────────────────

def encode_with_frames(image_chw, device, frame_every=20):
    """
    Encode a single CIFAR-10 image, capturing intermediate frames.

    Args:
        image_chw: [3, 32, 32] float tensor in [0, 1]
        device: "cpu" or "cuda"
        frame_every: capture a frame every N epochs

    Returns:
        frames: list of (epoch, rendered_hwc_np, mse, psnr)
        W_phys: final physical parameters
    """
    # Convert to [H, W, C]
    image_hwc = image_chw.permute(1, 2, 0).to(device)
    H, W_img, C = image_hwc.shape
    image_size = (H, W_img)
    K = ENC["K"]

    W_raw = _init_gaussians(image_hwc, K, device,
                            ENC["init_mode"], ENC["sigma_activation"])
    W_raw = W_raw.requires_grad_(True)
    opt = torch.optim.Adam([W_raw], lr=ENC["lr"])

    frames = []

    for epoch in range(ENC["epochs"]):
        opt.zero_grad()
        p = _to_physical(W_raw, C, ENC["sigma_activation"])
        rendered = _render(p, ENC["kernel_size"], image_size, device, C,
                           ENC["soft_clamp"])
        loss = F.mse_loss(rendered, image_hwc)
        loss.backward()
        opt.step()

        mse = loss.item()
        if epoch % frame_every == 0 or epoch == ENC["epochs"] - 1 or mse < ENC["early_stop_threshold"]:
            psnr = psnr_db(mse)
            frames.append((epoch, rendered.detach().cpu().numpy(), mse, psnr))

        if mse < ENC["early_stop_threshold"]:
            break

    # Final physical params
    with torch.no_grad():
        p = _to_physical(W_raw, C, ENC["sigma_activation"])
        W_phys = torch.cat([
            torch.stack([p["sigma_x"], p["sigma_y"], p["rho"], p["alpha"]], dim=1),
            p["colours"],
            torch.stack([p["x"], p["y"]], dim=1),
        ], dim=1)

    return frames, W_phys.cpu()


# ─── Figure 1: Reconstruction grid ───────────────────────────────────────────

def fig_01_reconstruction_grid(examples):
    """
    5×3 grid: Original | Reconstructed | Residual (×5)
    examples: list of (class_name, original_hwc, reconstructed_hwc, psnr)
    """
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(8, 2.4 * n))
    fig.patch.set_facecolor("white")

    for row, (cls, orig, recon, psnr) in enumerate(examples):
        residual = np.clip(np.abs(orig - recon) * 5, 0, 1)

        for col, (data, title) in enumerate([
            (orig, f"{cls}" if row == 0 else cls),
            (recon, f"PSNR = {psnr:.1f} dB" if row == 0 else f"{psnr:.1f} dB"),
            (residual, "Residual ×5" if row == 0 else ""),
        ]):
            ax = axes[row, col]
            ax.imshow(np.clip(data, 0, 1))
            ax.set_axis_off()
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            elif col <= 1:
                ax.set_title(title, fontsize=10)

    # Column headers
    cols = ["Original", "Reconstructed (K=500)", "Residual ×5"]
    for col, label in enumerate(cols):
        axes[0, col].set_title(label, fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout(h_pad=0.3, w_pad=0.3)
    path = FIG_DIR / "fig_01_reconstruction_grid.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ─── Figure 2: Convergence curves ────────────────────────────────────────────

def fig_02_convergence_curves(all_frames, class_names):
    """PSNR vs epoch for each example."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_frames)))

    for i, (frames, cls) in enumerate(zip(all_frames, class_names)):
        epochs = [f[0] for f in frames]
        psnrs = [f[3] for f in frames]
        ax.plot(epochs, psnrs, color=colors[i], label=cls, linewidth=1.5)

    ax.axhline(y=40, color="red", linestyle="--", alpha=0.5, label="40 dB target")
    ax.axhline(y=50, color="green", linestyle="--", alpha=0.3, label="50 dB")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Convergence: PSNR vs Epoch (K=500, lr=0.04, ks=32, soft clamp)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "fig_02_convergence_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ─── Figure 3: PSNR histogram ────────────────────────────────────────────────

def fig_03_psnr_histogram(psnrs, labels):
    """PSNR distribution over N images, colored by class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    # Left: overall histogram
    ax1.hist(psnrs, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.axvline(np.mean(psnrs), color="red", linestyle="--",
                label=f"Mean = {np.mean(psnrs):.1f} dB")
    ax1.axvline(np.min(psnrs), color="orange", linestyle="--",
                label=f"Min = {np.min(psnrs):.1f} dB")
    ax1.set_xlabel("PSNR (dB)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"PSNR Distribution (N={len(psnrs)})", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)

    # Right: per-class box plot
    class_psnrs = {i: [] for i in range(10)}
    for p, l in zip(psnrs, labels):
        class_psnrs[l].append(p)

    box_data = [class_psnrs[i] for i in range(10) if class_psnrs[i]]
    box_labels = [CLASS_NAMES[i] for i in range(10) if class_psnrs[i]]
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("PSNR (dB)", fontsize=12)
    ax2.set_title("PSNR by Class", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = FIG_DIR / "fig_03_psnr_histogram.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ─── Convergence GIF ─────────────────────────────────────────────────────────

def _render_gif_frame(target_hwc, rendered_hwc, epoch, mse, psnr, cls_name):
    """Build one animation frame as numpy RGB array."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor("white")

    residual = np.clip(np.abs(target_hwc - rendered_hwc) * 5, 0, 1)

    titles = [
        f"Target ({cls_name})",
        f"Epoch {epoch}  |  PSNR = {psnr:.1f} dB",
        f"Residual ×5  |  MSE = {mse:.6f}",
    ]

    for ax, data, title in zip(axes, [target_hwc, rendered_hwc, residual], titles):
        ax.imshow(np.clip(data, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()

    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


def generate_gif(frames, target_hwc, cls_name, out_path):
    """Generate a convergence GIF from captured frames."""
    from PIL import Image as PILImage

    frame_imgs = []
    for (epoch, rendered_hwc, mse, psnr) in frames:
        buf = _render_gif_frame(target_hwc, rendered_hwc, epoch, mse, psnr, cls_name)
        frame_imgs.append(buf)

    # Hold the last frame longer
    for _ in range(15):
        frame_imgs.append(frame_imgs[-1])

    pil_frames = [PILImage.fromarray(f) for f in frame_imgs]
    pil_frames[0].save(
        str(out_path), save_all=True,
        append_images=pil_frames[1:],
        duration=100, loop=0,
    )
    print(f"  Saved {out_path.name} ({len(frames)} frames)")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_hist", type=int, default=50,
                        help="Number of images for PSNR histogram")
    parser.add_argument("--frame_every", type=int, default=20,
                        help="Capture frame every N epochs for GIF")
    parser.add_argument("--skip_hist", action="store_true",
                        help="Skip histogram (slow, encodes n_hist images)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VID_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Encoding config: {ENC}")
    print()

    # Load data
    print("[1/4] Loading CIFAR-10 ...")
    images_np, labels = load_cifar10()

    # Pick 5 diverse examples (one per class pair: easy/medium/hard)
    # Use fixed indices for reproducibility
    example_indices = [
        49,    # airplane (simple, sky background)
        1001,  # automobile
        7777,  # bird
        3333,  # cat (harder, textured)
        8888,  # ship
    ]
    example_imgs = []
    for idx in example_indices:
        img = torch.from_numpy(images_np[idx]).float() / 255.0  # [3, 32, 32]
        lbl = int(labels[idx])
        example_imgs.append((img, lbl, CLASS_NAMES[lbl], idx))

    # [2/4] Encode examples with frame capture
    print("\n[2/4] Encoding 5 examples with frame capture ...")
    all_frames = []
    examples_for_grid = []

    for img_chw, lbl, cls, idx in example_imgs:
        print(f"  Encoding idx={idx} ({cls}) ...")
        t0 = time.time()
        frames, W_phys = encode_with_frames(img_chw, args.device,
                                            frame_every=args.frame_every)
        elapsed = time.time() - t0
        final_psnr = frames[-1][3]
        print(f"    {len(frames)} frames, {elapsed:.1f}s, "
              f"final PSNR={final_psnr:.1f} dB")

        all_frames.append(frames)
        orig_hwc = img_chw.permute(1, 2, 0).numpy()
        recon_hwc = frames[-1][1]
        examples_for_grid.append((cls, orig_hwc, recon_hwc, final_psnr))

    # [3/4] Generate figures
    print("\n[3/4] Generating figures ...")
    fig_01_reconstruction_grid(examples_for_grid)
    fig_02_convergence_curves(all_frames,
                              [cls for _, _, cls, _ in example_imgs])

    # [3b] PSNR histogram (encode more images)
    if not args.skip_hist:
        print(f"\n[3b] Encoding {args.n_hist} images for PSNR histogram ...")
        torch.manual_seed(42)
        hist_indices = torch.randperm(len(labels))[:args.n_hist].tolist()

        # Use batched encoding for speed
        hist_imgs = torch.from_numpy(
            np.stack([images_np[i] for i in hist_indices])
        ).float() / 255.0

        t0 = time.time()
        _, losses = encode_batch(
            hist_imgs, device=args.device, **ENC,
        )
        elapsed = time.time() - t0
        hist_psnrs = [psnr_db(l) for l in losses.tolist()]
        hist_labels = [int(labels[i]) for i in hist_indices]
        print(f"  {args.n_hist} images in {elapsed:.1f}s "
              f"({elapsed/args.n_hist:.2f}s/img)")
        print(f"  PSNR: mean={np.mean(hist_psnrs):.1f}, "
              f"std={np.std(hist_psnrs):.1f}, "
              f"min={np.min(hist_psnrs):.1f}, "
              f"max={np.max(hist_psnrs):.1f}")

        fig_03_psnr_histogram(hist_psnrs, hist_labels)
    else:
        print("\n[3b] Skipping histogram (--skip_hist)")

    # [4/4] Generate convergence GIFs
    print("\n[4/4] Generating convergence GIFs ...")
    for (img_chw, lbl, cls, idx), frames in zip(example_imgs, all_frames):
        target_hwc = img_chw.permute(1, 2, 0).numpy()
        gif_path = VID_DIR / f"{cls}_convergence.gif"
        generate_gif(frames, target_hwc, cls, gif_path)

    print("\nDone! All outputs in:")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Videos:  {VID_DIR}/")


if __name__ == "__main__":
    main()
