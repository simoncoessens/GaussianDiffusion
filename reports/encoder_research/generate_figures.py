#!/usr/bin/env python
"""
Generate all figures for the Gaussian encoder research report.

Runtime: ~30-45 minutes on CPU with default settings (--n_per_class 5).
Use --n_per_class 2 and --skip_experiments for a quick smoke test (~5 min).

Usage (from project root):
    python reports/encoder_research/generate_figures.py [--device cpu|cuda]
        [--n_per_class N] [--no_cache] [--skip_experiments] [--skip_videos]
        [--data_root PATH] [--frame_every N]
"""

import sys
import os
import math
import time
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.encode import (
    encode_image, _init_gaussians, _to_physical, _render,
    _dead_mask, _recycle, ssim_loss,
)
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

# ── output directories ────────────────────────────────────────────────────────
REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR    = REPORT_DIR / "figures"
VID_DIR    = REPORT_DIR / "videos"
CACHE_FILE = FIG_DIR / "cache.pt"

# ── parameter metadata ────────────────────────────────────────────────────────
PARAM_NAMES  = ["sigma_x", "sigma_y", "rho", "alpha", "colour", "x", "y"]
PARAM_LABELS = [r"$\sigma_x$", r"$\sigma_y$", r"$\rho$", r"$\alpha$",
                "colour", r"$x$", r"$y$"]
PARAM_RANGES = [(0, 1), (0, 1), (-1, 1), (0, 1), (0, 1), (-1, 1), (-1, 1)]

DIGIT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})

# Default encoding settings (matching best known config from MEMORY.md)
ENC_KWARGS = dict(
    K=70, epochs=3000, lr=5e-3, kernel_size=11,
    early_stop_threshold=1e-4, recycle_every=300,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def psnr_db(mse: float) -> float:
    return min(100.0, 10.0 * math.log10(1.0 / mse)) if mse > 1e-10 else 100.0


def save_fig(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"    Saved {path.name}")


def render_W(W: torch.Tensor, kernel_size: int = 11,
             image_size: tuple = (28, 28)) -> np.ndarray:
    """Render a physical W tensor [K,7] to a numpy [H,W] image."""
    p = {
        "sigma_x": W[:, 0], "sigma_y": W[:, 1], "rho": W[:, 2],
        "colour": W[:, 4], "x": W[:, 5], "y": W[:, 6],
    }
    coords = torch.stack([p["x"], p["y"]], dim=1)
    with torch.no_grad():
        img_t = generate_2D_gaussian_splatting(
            kernel_size=kernel_size,
            sigma_x=p["sigma_x"], sigma_y=p["sigma_y"], rho=p["rho"],
            coords=coords, colours=p["colour"].unsqueeze(1),
            image_size=image_size, channels=1, device="cpu",
        )[:, :, 0]
    return img_t.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# MNIST loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_samples(n_per_class: int = 5, seed: int = 42,
                       data_root=None):
    """Return list of (image_tensor [H,W], digit_label) pairs."""
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    if data_root is None:
        data_root = PROJECT_ROOT / "data"
    ds = MNIST(root=str(data_root), train=True, download=True,
               transform=ToTensor())
    rng = np.random.default_rng(seed)
    samples = []
    for digit in range(10):
        indices = [i for i, (_, lbl) in enumerate(ds) if lbl == digit]
        chosen = rng.choice(indices,
                            size=min(n_per_class, len(indices)),
                            replace=False)
        for idx in chosen:
            img, _ = ds[int(idx)]
            samples.append((img.squeeze(0), digit))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Encoding with caching
# ─────────────────────────────────────────────────────────────────────────────

def encode_batch(samples, device: str, cache_path: Path, force: bool = False):
    """Encode all (image, digit) pairs; load from cache if available."""
    if not force and cache_path.exists():
        print("  Loading main encodings from cache …")
        return torch.load(cache_path, weights_only=False)

    results = []
    for i, (img, digit) in enumerate(samples):
        t0 = time.time()
        W, loss, hist = encode_image(img, device=device,
                                     return_history=True, **ENC_KWARGS)
        elapsed = time.time() - t0
        psnr = psnr_db(loss)
        print(f"  [{i+1:2d}/{len(samples)}] digit={digit}  "
              f"PSNR={psnr:.1f} dB  "
              f"epochs={hist[-1]['epoch']+1}  t={elapsed:.1f}s")
        results.append({
            "W": W, "loss": loss, "psnr": psnr,
            "hist": hist, "digit": digit, "image": img,
        })

    torch.save(results, cache_path)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Buggy encoder (pre-fix, for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def encode_image_buggy(image: torch.Tensor, K: int = 70, epochs: int = 500,
                       lr: float = 5e-3, kernel_size: int = 11,
                       device: str = "cpu") -> float:
    """
    Encoder with the coordinate-inversion bug present (atanh(xs) not atanh(-xs)).
    Returns final PSNR in dB after `epochs` iterations (no early stop).
    """
    H, W_img = image.shape
    image = image.to(device)
    image_size = (H, W_img)
    flat = image.view(-1)
    probs = (flat / (flat.sum() + 1e-8)
             if flat.sum() > 1e-7
             else torch.ones_like(flat) / flat.numel())
    indices = torch.multinomial(probs, num_samples=K, replacement=True)
    ys = (indices // W_img).float() / (H - 1) * 2 - 1
    xs = (indices % W_img).float() / (W_img - 1) * 2 - 1
    colours_raw = torch.logit(flat[indices].clamp(0.05, 0.95))
    # BUG: no negation — xs_raw = atanh(xs) instead of atanh(-xs)
    xs_raw = torch.atanh(xs.clamp(-1 + 1e-6, 1 - 1e-6))
    ys_raw = torch.atanh(ys.clamp(-1 + 1e-6, 1 - 1e-6))
    log_sig = torch.full((K,), -1.5, device=device)
    W_raw = torch.stack([
        log_sig, log_sig, torch.zeros(K, device=device),
        torch.zeros(K, device=device),
        colours_raw.to(device), xs_raw.to(device), ys_raw.to(device),
    ], dim=1).requires_grad_(True)
    opt = torch.optim.Adam([W_raw], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, epochs // 3), eta_min=1e-5,
    )
    for _ in range(epochs):
        opt.zero_grad()
        p = _to_physical(W_raw)
        ren = _render(p, kernel_size, image_size, device)
        loss = F.mse_loss(ren, image)
        loss.backward()
        opt.step()
        sch.step()
    return psnr_db(loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# Alternative-loss encoder for ablation (fig 12)
# ─────────────────────────────────────────────────────────────────────────────

def encode_with_loss_fn(image: torch.Tensor, loss_fn_name: str,
                        K: int = 70, epochs: int = 1000,
                        lr: float = 5e-3, kernel_size: int = 11,
                        early_stop: float = 1e-4,
                        device: str = "cpu") -> list:
    """Encode with 'mse' or 'l1ssim'; return history list."""
    H, W_img = image.shape
    image = image.to(device)
    image_size = (H, W_img)
    W_raw = _init_gaussians(image, K, device).requires_grad_(True)
    opt = torch.optim.Adam([W_raw], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, epochs // 3), eta_min=1e-5,
    )
    hist = []
    for epoch in range(epochs):
        opt.zero_grad()
        p = _to_physical(W_raw)
        ren = _render(p, kernel_size, image_size, device)
        if loss_fn_name == "mse":
            loss = F.mse_loss(ren, image)
        else:
            loss = 0.5 * F.l1_loss(ren, image) + 0.5 * ssim_loss(ren, image)
        loss.backward()
        opt.step()
        sch.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                mse_val = F.mse_loss(ren.detach(), image).item()
            hist.append({
                "epoch": epoch,
                "loss": loss.item(),
                "psnr_db": psnr_db(mse_val),
            })
        if loss_fn_name == "mse" and loss.item() < early_stop:
            break
    return hist


# ─────────────────────────────────────────────────────────────────────────────
# Figure 01 — Reconstruction grid
# ─────────────────────────────────────────────────────────────────────────────

def fig_01(results):
    """10 × 3 grid: original | reconstruction | ×5 residual, best per digit."""
    best = {}
    for r in results:
        d = r["digit"]
        if d not in best or r["psnr"] > best[d]["psnr"]:
            best[d] = r

    fig, axes = plt.subplots(10, 3, figsize=(6, 22))
    for col, title in enumerate(["Original", "Reconstructed", "|Residual| ×5"]):
        axes[0, col].set_title(title, fontsize=12, pad=6)

    for row, digit in enumerate(range(10)):
        r = best[digit]
        img = r["image"].numpy()
        recon = render_W(r["W"])
        error = np.clip(np.abs(img - recon) * 5, 0, 1)

        for col, (data, cmap) in enumerate(
            [(img, "gray"), (recon, "gray"), (error, "hot")]
        ):
            ax = axes[row, col]
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            ax.set_axis_off()

        axes[row, 0].set_ylabel(
            f"Digit {digit}\n{r['psnr']:.1f} dB",
            fontsize=9, rotation=0, labelpad=50, va="center",
        )

    plt.suptitle("Encoder Reconstruction Quality — Best Encoding per Digit",
                 fontsize=13, y=1.001)
    plt.tight_layout()
    save_fig("fig_01_reconstruction_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 02 — PSNR before vs after fix
# ─────────────────────────────────────────────────────────────────────────────

def fig_02(results, buggy_by_digit: dict):
    after = {d: [] for d in range(10)}
    for r in results:
        after[r["digit"]].append(r["psnr"])

    after_mean = [np.mean(after[d]) if after[d] else 0 for d in range(10)]
    after_std  = [np.std(after[d])  if after[d] else 0 for d in range(10)]
    before_mean = [np.mean(buggy_by_digit.get(d, [26.0])) for d in range(10)]
    before_std  = [np.std(buggy_by_digit.get(d, [0.5]))   for d in range(10)]

    x, w = np.arange(10), 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, before_mean, w, yerr=before_std, capsize=4,
           color="#d62728", alpha=0.85, label="Before fix (buggy init, 500 ep.)")
    ax.bar(x + w/2, after_mean, w, yerr=after_std, capsize=4,
           color="#2ca02c", alpha=0.85, label="After fix (corrected init, ≤3000 ep.)")

    ax.axhline(np.mean(before_mean), color="#d62728", ls="--", lw=1.5,
               label=f"Mean before: {np.mean(before_mean):.1f} dB")
    ax.axhline(np.mean(after_mean), color="#2ca02c", ls="--", lw=1.5,
               label=f"Mean after:  {np.mean(after_mean):.1f} dB")
    ax.axhline(40, color="gray", ls=":", lw=1.2,
               label="40 dB early-stop target")

    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_xlabel("Digit class"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR Before vs. After Coordinate-Inversion Fix\n"
                 "(K = 70 Gaussians, MNIST test images)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_ylim(0, 58)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("fig_02_psnr_before_after.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 03 — Convergence curves
# ─────────────────────────────────────────────────────────────────────────────

def fig_03(results):
    """Loss and PSNR vs. epoch for 5 representative images."""
    chosen = []
    for d in [0, 2, 4, 6, 8]:
        cands = [r for r in results if r["digit"] == d and r.get("hist")]
        if cands:
            chosen.append(max(cands, key=lambda r: r["psnr"]))
    if not chosen:
        print("    Skipping fig_03: no history available.")
        return

    fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 3.2 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(chosen):
        hist = r["hist"]
        epochs  = [h["epoch"]   for h in hist]
        losses  = [h["loss"]    for h in hist]
        psnrs   = [h["psnr_db"] for h in hist]
        recycle_epochs = [h["epoch"] for h in hist if h.get("recycling_event")]

        ax_l, ax_p = axes[row, 0], axes[row, 1]

        ax_l.semilogy(epochs, losses, color="steelblue", lw=1.8)
        for ep in recycle_epochs:
            ax_l.axvline(ep, color="orange", ls=":", lw=1.0, alpha=0.8)
        ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("MSE loss (log)")
        ax_l.set_title(f"Digit {r['digit']} — Loss")
        ax_l.grid(alpha=0.3)

        ax_p.plot(epochs, psnrs, color="darkorange", lw=1.8)
        ax_p.axhline(40, color="gray", ls="--", lw=1.2, label="40 dB target")
        for i, ep in enumerate(recycle_epochs):
            ax_p.axvline(ep, color="orange", ls=":", lw=1.0, alpha=0.8,
                         label="Recycle event" if i == 0 else "")
        ax_p.set_xlabel("Epoch"); ax_p.set_ylabel("PSNR (dB)")
        ax_p.set_title(f"Digit {r['digit']} — PSNR (final: {r['psnr']:.1f} dB)")
        ax_p.legend(fontsize=8); ax_p.grid(alpha=0.3)

    plt.suptitle("Encoder Convergence Curves (5 Representative Digits)",
                 fontsize=12, y=1.002)
    plt.tight_layout()
    save_fig("fig_03_convergence_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 04 — PSNR distribution
# ─────────────────────────────────────────────────────────────────────────────

def fig_04(results):
    all_psnr = [r["psnr"] for r in results]
    psnr_by_digit = {d: [] for d in range(10)}
    for r in results:
        psnr_by_digit[r["digit"]].append(r["psnr"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    bins = np.linspace(min(all_psnr) - 1, max(all_psnr) + 1, 30)
    for d in range(10):
        if psnr_by_digit[d]:
            ax1.hist(psnr_by_digit[d], bins=bins, color=DIGIT_COLORS[d],
                     alpha=0.7, label=str(d))
    ax1.axvline(np.mean(all_psnr), color="black", ls="--", lw=2.0,
                label=f"Mean: {np.mean(all_psnr):.1f} dB")
    ax1.axvline(min(all_psnr), color="red", ls=":", lw=1.5,
                label=f"Min: {min(all_psnr):.1f} dB")
    ax1.set_xlabel("PSNR (dB)"); ax1.set_ylabel("Count")
    ax1.set_title("PSNR Distribution (all images, colored by digit)")
    ax1.legend(fontsize=7.5, ncol=2, title="Digit")
    ax1.grid(alpha=0.3)

    data_by_digit = [psnr_by_digit[d] for d in range(10)]
    bp = ax2.boxplot(data_by_digit, patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"], DIGIT_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax2.axhline(40, color="gray", ls="--", lw=1.2, label="40 dB target")
    ax2.set_xticks(range(1, 11))
    ax2.set_xticklabels([str(d) for d in range(10)])
    ax2.set_xlabel("Digit class"); ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("PSNR per Digit (Box Plot)")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Reconstruction Quality Summary  "
        f"(N={len(results)},  "
        f"mean={np.mean(all_psnr):.1f}±{np.std(all_psnr):.1f} dB,  "
        f"min={min(all_psnr):.1f} dB)",
        fontsize=12,
    )
    plt.tight_layout()
    save_fig("fig_04_psnr_histogram.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 05 — Parameter distributions
# ─────────────────────────────────────────────────────────────────────────────

def fig_05(results):
    all_W = torch.cat([r["W"].unsqueeze(0) for r in results], dim=0)
    flat  = all_W.view(-1, 7).numpy()  # [N*K, 7]
    N_gaussians = flat.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes_flat = axes.flatten()

    for i, (name, label, (lo, hi)) in enumerate(
        zip(PARAM_NAMES, PARAM_LABELS, PARAM_RANGES)
    ):
        ax = axes_flat[i]
        vals = flat[:, i]
        ax.hist(vals, bins=70, color="steelblue", alpha=0.78, density=True)
        ax.axvline(lo, color="red",  ls="--", lw=1.5, label=f"lo={lo}")
        ax.axvline(hi, color="red",  ls="--", lw=1.5, label=f"hi={hi}")
        ax.axvline(np.mean(vals), color="darkorange", ls="-", lw=1.8,
                   label=f"μ={np.mean(vals):.3f}")
        ax.set_title(f"{label}   σ={np.std(vals):.3f}")
        ax.set_xlabel("Value (physical)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    axes_flat[-1].set_visible(False)

    plt.suptitle(
        f"Gaussian Parameter Distributions\n"
        f"(N={len(results)} images × K=70 = {N_gaussians:,} Gaussians total)",
        fontsize=12,
    )
    plt.tight_layout()
    save_fig("fig_05_param_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 06 — Parameter correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_06(results):
    flat = torch.cat([r["W"].unsqueeze(0) for r in results], dim=0).view(-1, 7).numpy()

    corr = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            rho_val, _ = spearmanr(flat[:, i], flat[:, j])
            corr[i, j] = rho_val

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(PARAM_LABELS, fontsize=11)
    ax.set_yticklabels(PARAM_LABELS, fontsize=11)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if abs(corr[i, j]) > 0.55 else "black")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(
        f"Spearman Rank Correlation of Gaussian Parameters\n"
        f"({flat.shape[0]:,} Gaussians from {len(results)} encoded images)",
        fontsize=12,
    )
    plt.tight_layout()
    save_fig("fig_06_param_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 07 — Spatial distribution per digit
# ─────────────────────────────────────────────────────────────────────────────

def fig_07(results):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()

    for d in range(10):
        ax = axes[d]
        digit_results = [r for r in results if r["digit"] == d]

        # Average digit image as faded background
        if digit_results:
            avg_img = torch.stack([r["image"] for r in digit_results]).mean(0).numpy()
            ax.imshow(avg_img, extent=[-1, 1, 1, -1], cmap="Blues",
                      alpha=0.35, aspect="auto", origin="upper")

        for r in digit_results:
            W = r["W"].numpy()
            # Renderer inverts: actual rendered position = (-p["x"], -p["y"])
            x_render = -W[:, 5]
            y_render = -W[:, 6]
            colour = W[:, 4]
            sc = ax.scatter(x_render, y_render, c=colour, cmap="inferno",
                            vmin=0, vmax=1, s=6, alpha=0.55)

        ax.set_title(f"Digit {d}", fontsize=10)
        ax.set_xlim(-1, 1); ax.set_ylim(1, -1)  # image coords: y increases down
        ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    plt.colorbar(sc, cax=cbar_ax, label="colour")

    plt.suptitle(
        "Spatial Distribution of Gaussians per Digit\n"
        r"(positions show actual rendered location: $x_\mathrm{render}=-p_x$, "
        r"$y_\mathrm{render}=-p_y$; colour = colour param)",
        fontsize=12,
    )
    save_fig("fig_07_spatial_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 08 — K-sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def fig_08(k_sens_data: dict):
    Ks    = sorted(k_sens_data.keys())
    means = [np.mean(k_sens_data[k]) for k in Ks]
    stds  = [np.std( k_sens_data[k]) for k in Ks]
    mins  = [np.min( k_sens_data[k]) for k in Ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.errorbar(Ks, means, yerr=stds, marker="o", color="steelblue",
                 capsize=5, lw=2.0, label="Mean PSNR ± std")
    ax1.plot(Ks, mins, marker="^", color="red", ls="--", lw=1.3,
             label="Minimum PSNR")
    ax1.axhline(40, color="gray", ls=":", lw=1.2, label="40 dB target")
    ax1.set_xlabel("Number of Gaussians K")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Reconstruction Quality vs. K")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.set_xticks(Ks)

    # Marginal gain per additional Gaussian
    gains = [0.0]
    for i in range(1, len(Ks)):
        dK = Ks[i] - Ks[i - 1]
        gains.append((means[i] - means[i - 1]) / dK if dK > 0 else 0.0)
    ax2.bar([str(k) for k in Ks], gains, color="darkorange", alpha=0.8,
            edgecolor="black", lw=0.6)
    ax2.set_xlabel("K (number of Gaussians)")
    ax2.set_ylabel("ΔdB / ΔK (marginal gain)")
    ax2.set_title("Marginal PSNR Gain per Additional Gaussian")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "K-Sensitivity Analysis: Effect of Number of Gaussians on PSNR",
        fontsize=12,
    )
    plt.tight_layout()
    save_fig("fig_08_k_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 09 — Coordinate inversion diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig_09():
    """Schematic illustration of the affine_grid coordinate inversion bug."""
    H, W = 28, 28
    target_col = 20       # bright pixel column in image coords
    # In normalised coords [-1,1]: x_norm = 2*col/(W-1) - 1
    x_norm = 2 * target_col / (W - 1) - 1   # ≈ +0.48

    # Affine_grid translate: theta[:,2]=+c  →  sample from position (1+c)/2*(W-1) = col~23
    # Wait: affine_grid uses: x_sample = x_grid * 0 + x_translation in the NORMALISED grid.
    # Actually the translation in theta shifts the *sample position* in normalised coords.
    # With theta[:,0,2] = c, output pixel at normalised grid pos g samples input at g+c.
    # So center kernel at grid pos 0 with theta[0,2]=+c → samples input at +c → col=(1+c)/2*(W-1).
    # But for a translation (not sampling FROM but PLACING TO), the formula inverts.
    # In the renderer: setting coords[i] = c places the Gaussian peak at output x = -c
    # (because affine_grid(theta, ...) with theta[:,2]=c maps output position x to input x+c,
    # so the kernel maximum at input x=0 appears at OUTPUT x=-c).
    # Therefore: to render at output col = target_col, we need coords = -x_norm.

    # Buggy: coords set to +x_norm → renders at output col corresponding to -x_norm
    buggy_render_col = int((1 - x_norm) / 2 * (W - 1))   # ≈ 6 (left side)

    def gaussian_blob(center_col, sigma=3.0):
        cols = np.arange(W)
        return np.exp(-0.5 * ((cols - center_col) / sigma) ** 2)

    target_row = gaussian_blob(target_col)
    buggy_row  = gaussian_blob(buggy_render_col)
    fixed_row  = gaussian_blob(target_col)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    col_x = np.arange(W)

    # Panel 1: Target
    axes[0].fill_between(col_x, target_row, alpha=0.6, color="steelblue")
    axes[0].axvline(target_col, color="green", lw=2, label=f"Target col={target_col}")
    axes[0].set_title(
        f"Target image\nBright region at col={target_col} (x={x_norm:.2f})",
        fontsize=10,
    )
    axes[0].set_xlim(0, W - 1); axes[0].set_ylim(0, 1.25)
    axes[0].legend(fontsize=8.5); axes[0].set_xlabel("pixel column")
    axes[0].set_ylabel("intensity")

    # Panel 2: Buggy init
    axes[1].fill_between(col_x, buggy_row, alpha=0.6, color="#d62728")
    axes[1].axvline(target_col, color="green", lw=1.5, ls="--",
                    label=f"Target col={target_col}")
    axes[1].axvline(buggy_render_col, color="#d62728", lw=2,
                    label=f"Buggy render col={buggy_render_col}")
    axes[1].set_title(
        f"Buggy init: raw_x = atanh(+{x_norm:.2f})\n"
        f"→ renders at col={buggy_render_col}  (wrong side!)",
        fontsize=10, color="#d62728",
    )
    axes[1].set_xlim(0, W - 1); axes[1].set_ylim(0, 1.25)
    axes[1].legend(fontsize=8.5); axes[1].set_xlabel("pixel column")

    # Panel 3: Fixed init
    axes[2].fill_between(col_x, fixed_row, alpha=0.6, color="#2ca02c")
    axes[2].axvline(target_col, color="green", lw=2,
                    label=f"Correct render col={target_col}")
    axes[2].set_title(
        f"Fixed init: raw_x = atanh(−{x_norm:.2f})\n"
        f"→ renders at col={target_col}  (correct!)",
        fontsize=10, color="#2ca02c",
    )
    axes[2].set_xlim(0, W - 1); axes[2].set_ylim(0, 1.25)
    axes[2].legend(fontsize=8.5); axes[2].set_xlabel("pixel column")

    fig.text(
        0.5, -0.04,
        r"$F.\mathrm{affine\_grid}$: setting $\theta_{:,2} = c$ maps output position $x$ "
        r"to input $x + c$, so a kernel centered at input $x=0$ appears at output $x = -c$."
        "\n"
        r"Fix: $W_\mathrm{raw}[i,5] = \mathrm{atanh}(-x_\mathrm{pixel})$ "
        r"so $\tanh(W_\mathrm{raw}[i,5]) = -x_\mathrm{pixel}$ "
        r"and the renderer places the Gaussian at $-(-x_\mathrm{pixel}) = x_\mathrm{pixel}$.",
        ha="center", fontsize=9, style="italic",
    )
    plt.suptitle(
        r"Root Cause: $F.\mathrm{affine\_grid}$ Inverts Translation Coordinates",
        fontsize=12,
    )
    plt.tight_layout()
    save_fig("fig_09_coord_inversion_diagram.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — Normalization coverage
# ─────────────────────────────────────────────────────────────────────────────

def fig_10(results):
    all_W = torch.cat([r["W"].unsqueeze(0) for r in results], dim=0).view(-1, 7).numpy()

    coverage   = []
    actual_min = []
    actual_max = []
    for i, (lo, hi) in enumerate(PARAM_RANGES):
        v = all_W[:, i]
        vmin, vmax = v.min(), v.max()
        actual_min.append(vmin)
        actual_max.append(vmax)
        span = hi - lo
        coverage.append((vmax - vmin) / span if span > 0 else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: coverage fraction bar
    colors_bar = [
        "green" if c > 0.75 else "orange" if c > 0.45 else "red"
        for c in coverage
    ]
    ax1.barh(range(7), coverage, color=colors_bar, alpha=0.85, edgecolor="black", lw=0.5)
    ax1.axvline(1.0, color="black", ls="--", lw=1.5, label="Full theoretical range")
    ax1.set_yticks(range(7)); ax1.set_yticklabels(PARAM_LABELS, fontsize=12)
    ax1.set_xlabel("Coverage fraction (observed span / theoretical span)")
    ax1.set_title("Parameter Range Coverage\n(how much of the valid range is explored)")
    ax1.set_xlim(0, 1.15)
    ax1.legend(fontsize=9); ax1.grid(axis="x", alpha=0.3)
    for i, (cov, lo, hi, amin, amax) in enumerate(
        zip(coverage, *zip(*PARAM_RANGES), actual_min, actual_max)
    ):
        ax1.text(cov + 0.01, i, f" {cov:.0%}   [{amin:.2f}, {amax:.2f}]",
                 va="center", fontsize=8.5)

    # Right: normalized distributions
    all_normed = np.zeros_like(all_W)
    for i, (lo, hi) in enumerate(PARAM_RANGES):
        all_normed[:, i] = 2.0 * (all_W[:, i] - lo) / max(hi - lo, 1e-8) - 1.0
    bins = np.linspace(-1.15, 1.15, 45)
    pal  = plt.cm.Set2(np.linspace(0, 1, 7))
    for i in range(7):
        ax2.hist(all_normed[:, i], bins=bins, alpha=0.55, density=True,
                 color=pal[i], label=PARAM_LABELS[i])
    ax2.axvline(-1, color="black", ls="--", lw=1.2, label="Bounds [−1, 1]")
    ax2.axvline( 1, color="black", ls="--", lw=1.2)
    ax2.set_xlabel("Normalized value (linear map to [−1, 1])")
    ax2.set_ylabel("Density")
    ax2.set_title("Parameter Distributions After Normalization\n"
                  "(required for diffusion model training)")
    ax2.legend(fontsize=8, ncol=2); ax2.grid(alpha=0.3)

    plt.suptitle("Normalization Readiness for Diffusion Model Input", fontsize=12)
    plt.tight_layout()
    save_fig("fig_10_normalization_coverage.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — PCA of latent representations
# ─────────────────────────────────────────────────────────────────────────────

def fig_11(results):
    try:
        from sklearn.decomposition import PCA
        use_sklearn = True
    except ImportError:
        print("    sklearn not available; using numpy SVD for PCA")
        use_sklearn = False

    # Each image → flatten W to [K*7] and normalize
    X = np.stack([r["W"].numpy().flatten() for r in results])
    labels = np.array([r["digit"] for r in results])

    # Normalize each parameter slice
    for i, (lo, hi) in enumerate(PARAM_RANGES):
        idx = np.arange(i, X.shape[1], 7)
        X[:, idx] = 2.0 * (X[:, idx] - lo) / max(hi - lo, 1e-8) - 1.0

    n_comp = min(10, X.shape[0] - 1)

    if use_sklearn:
        pca = PCA(n_components=n_comp)
        Z = pca.fit_transform(X)
        var_ratio = pca.explained_variance_ratio_
    else:
        # Numpy SVD fallback
        Xc = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :n_comp] * S[:n_comp]
        total_var = (Xc ** 2).sum()
        var_ratio = (S[:n_comp] ** 2) / total_var

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: 2D scatter
    ax = axes[0]
    for d in range(10):
        mask = labels == d
        if mask.any():
            ax.scatter(Z[mask, 0], Z[mask, 1], c=[DIGIT_COLORS[d]], s=100,
                       label=str(d), alpha=0.85, edgecolors="none")
    ax.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% var.)")
    ax.set_title("PCA of Gaussian Latent Representations\n"
                 "(one point per encoded image)")
    ax.legend(title="Digit", fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3)

    # Right: scree plot
    ax2 = axes[1]
    n_show = min(n_comp, len(var_ratio))
    ax2.bar(range(1, n_show + 1), var_ratio[:n_show] * 100,
            color="steelblue", alpha=0.85)
    ax2.plot(range(1, n_show + 1), np.cumsum(var_ratio[:n_show]) * 100,
             color="darkorange", marker="o", lw=2.0, label="Cumulative")
    ax2.axhline(90, color="gray", ls="--", lw=1.2, label="90% threshold")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title("PCA Explained Variance")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.suptitle("Latent Space Structure of Gaussian Representations\n"
                 r"(input: flattened, normalized $W \in \mathbb{R}^{K \times 7}$)",
                 fontsize=12)
    plt.tight_layout()
    save_fig("fig_11_pca_latent.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 — Loss function ablation
# ─────────────────────────────────────────────────────────────────────────────

def fig_12(loss_comp_data: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"mse": "steelblue", "l1ssim": "darkorange"}
    labels = {"mse": "MSE only", "l1ssim": "0.5·L1 + 0.5·SSIM"}

    all_final = {"mse": [], "l1ssim": []}

    for loss_name, all_hists in loss_comp_data.items():
        # Individual curves (light)
        for hist in all_hists:
            ax1.plot([h["epoch"] for h in hist],
                     [h["psnr_db"] for h in hist],
                     color=colors[loss_name], alpha=0.3, lw=1.0)
            all_final[loss_name].append(hist[-1]["psnr_db"])

        # Average curve (bold)
        all_epochs = sorted({h["epoch"] for hist in all_hists for h in hist})
        avg_psnr = []
        for ep in all_epochs:
            vals = [h["psnr_db"] for hist in all_hists for h in hist
                    if h["epoch"] == ep]
            avg_psnr.append(np.mean(vals))
        ax1.plot(all_epochs, avg_psnr, color=colors[loss_name], lw=2.5,
                 label=f"{labels[loss_name]} (avg)")

    ax1.axhline(40, color="gray", ls="--", lw=1.2, label="40 dB target")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Convergence: MSE vs L1+SSIM Loss Function")
    ax1.legend(fontsize=8.5); ax1.grid(alpha=0.3)

    # Final PSNR grouped bar
    n = max(len(v) for v in all_final.values())
    x = np.arange(n)
    w = 0.38
    for j, (loss_name, final_list) in enumerate(all_final.items()):
        ax2.bar(x[:len(final_list)] + j * w, final_list, w,
                color=colors[loss_name], alpha=0.85, label=labels[loss_name])
    ax2.axhline(40, color="gray", ls="--", lw=1.2)
    ax2.set_xlabel("Image index"); ax2.set_ylabel("Final PSNR (dB)")
    ax2.set_title("Final PSNR Comparison")
    ax2.legend(fontsize=8.5); ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Loss Function Ablation Study", fontsize=12)
    plt.tight_layout()
    save_fig("fig_12_loss_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 — Alpha is vestigial
# ─────────────────────────────────────────────────────────────────────────────

def fig_13(results):
    r = max(results, key=lambda r: r["psnr"])
    W_base = r["W"].clone()
    img_orig = r["image"]

    alpha_values = [0.01, 0.50, 0.99]
    rendered_imgs = []

    for alpha_val in alpha_values:
        W_mod = W_base.clone()
        W_mod[:, 3] = alpha_val       # change physical alpha — not passed to renderer
        rendered_imgs.append(render_W(W_mod))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    axes[0].imshow(img_orig.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original\n(digit {r['digit']},  {r['psnr']:.1f} dB)")
    axes[0].set_axis_off()

    ref = rendered_imgs[0]
    for i, (alpha_val, rend) in enumerate(zip(alpha_values, rendered_imgs)):
        diff = np.abs(rend - ref).max()
        axes[i + 1].imshow(rend, cmap="gray", vmin=0, vmax=1)
        axes[i + 1].set_title(f"α = {alpha_val:.2f}\nmax|diff| = {diff:.2e}")
        axes[i + 1].set_axis_off()

    plt.suptitle(
        r"$\alpha$ Parameter is Vestigial — Changing $\alpha$ Has Zero Effect on Rendering"
        "\n(α is decoded by the encoder but never passed to "
        r"\texttt{generate\_2D\_gaussian\_splatting})",
        fontsize=11,
    )
    plt.tight_layout()
    save_fig("fig_13_alpha_vestigial.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 14 — Encoding consistency
# ─────────────────────────────────────────────────────────────────────────────

def fig_14(consistency_data):
    Ws     = [r["W"]    for r in consistency_data]
    psnrs  = [r["psnr"] for r in consistency_data]
    img    = consistency_data[0]["image"]

    renders = [render_W(W) for W in Ws]

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.45)

    # Row 0: original + 3 reconstructions + std map
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img.numpy(), cmap="gray"); ax0.set_title("Original"); ax0.set_axis_off()

    for i in range(min(3, len(renders))):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(renders[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Run {i+1}\n{psnrs[i]:.1f} dB"); ax.set_axis_off()

    # Std across runs
    W_stack = torch.stack(Ws).numpy()  # [runs, K, 7]
    param_stds = W_stack.std(axis=0)   # [K, 7]
    ax_std = fig.add_subplot(gs[0, 4])
    im_std = ax_std.imshow(param_stds.T, cmap="viridis", aspect="auto")
    ax_std.set_yticks(range(7)); ax_std.set_yticklabels(PARAM_LABELS, fontsize=8)
    ax_std.set_xlabel("Gaussian index (0…K-1)"); ax_std.set_ylabel("")
    ax_std.set_title("Param std across runs\n(per Gaussian, per dim)")
    plt.colorbar(im_std, ax=ax_std, fraction=0.04)

    # Row 1 left: position scatter overlay
    ax_pos = fig.add_subplot(gs[1, :2])
    markers = ["o", "s", "^"]
    for i, W in enumerate(Ws):
        x_r = -W[:, 5].numpy()
        y_r = -W[:, 6].numpy()
        ax_pos.scatter(x_r, y_r, s=25, alpha=0.6, marker=markers[i % 3],
                       label=f"Run {i+1}")
    ax_pos.set_xlim(-1, 1); ax_pos.set_ylim(1, -1)
    ax_pos.set_xlabel("Rendered x"); ax_pos.set_ylabel("Rendered y")
    ax_pos.set_title("Gaussian Position Overlap Across Runs")
    ax_pos.legend(fontsize=9); ax_pos.set_aspect("equal"); ax_pos.grid(alpha=0.3)

    # Row 1 right: per-parameter std violin plot
    ax_vio = fig.add_subplot(gs[1, 2:])
    vp = ax_vio.violinplot(
        [param_stds[:, i] for i in range(7)],
        positions=range(7), showmedians=True,
    )
    for body in vp["bodies"]:
        body.set_alpha(0.6)
    ax_vio.set_xticks(range(7)); ax_vio.set_xticklabels(PARAM_LABELS, fontsize=10)
    ax_vio.set_ylabel("Std across runs")
    ax_vio.set_title("Per-Parameter Consistency (lower = more deterministic)")
    ax_vio.grid(axis="y", alpha=0.3)

    psnr_str = ", ".join(f"{p:.1f}" for p in psnrs)
    plt.suptitle(
        f"Encoder Consistency: {len(consistency_data)} Independent Runs on Same Image\n"
        f"PSNR: {psnr_str} dB",
        fontsize=12,
    )
    save_fig("fig_14_consistency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Video generation — custom training loop with frame capture
# ─────────────────────────────────────────────────────────────────────────────

def encode_with_frames(image: torch.Tensor, K: int = 70, epochs: int = 3000,
                       lr: float = 5e-3, kernel_size: int = 11,
                       early_stop: float = 1e-4, device: str = "cpu",
                       frame_every: int = 10, recycle_every: int = 300,
                       recycle_threshold: float = 0.05):
    """Training loop that saves rendered frames every `frame_every` epochs."""
    H, W_img = image.shape
    image_size = (H, W_img)
    image = image.to(device)

    W_raw = _init_gaussians(image, K, device).requires_grad_(True)
    opt   = torch.optim.Adam([W_raw], lr=lr)
    sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, epochs // 3), T_mult=1, eta_min=1e-5,
    )
    frames = []  # (epoch, rendered_np, loss, psnr, n_dead)

    for epoch in range(epochs):
        opt.zero_grad()
        p       = _to_physical(W_raw)
        rendered = _render(p, kernel_size, image_size, device)
        loss    = F.mse_loss(rendered, image)
        loss.backward()
        opt.step()
        sch.step()

        if recycle_every > 0 and (epoch + 1) % recycle_every == 0:
            _recycle(W_raw, image, opt, kernel_size, image_size,
                     device, recycle_threshold)

        if epoch % frame_every == 0 or epoch == epochs - 1:
            loss_val = loss.item()
            psnr     = psnr_db(loss_val)
            n_dead   = int(_dead_mask(W_raw, image, recycle_threshold).sum())
            frames.append((epoch, rendered.detach().cpu().numpy(),
                           loss_val, psnr, n_dead))

        if loss.item() < early_stop:
            break

    return frames


def _render_video_frame(img_target: np.ndarray, img_render: np.ndarray,
                        epoch: int, loss: float, psnr: float,
                        n_dead: int, K: int) -> np.ndarray:
    """Build one video frame (numpy RGB array)."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    fig.patch.set_facecolor("white")

    residual = np.clip(np.abs(img_target - img_render) * 5, 0, 1)

    for ax, data, title, cmap in zip(
        axes,
        [img_target, img_render, residual],
        [
            "Target",
            f"Epoch {epoch}  |  PSNR = {psnr:.1f} dB",
            f"Residual ×5  |  MSE = {loss:.5f}  |  dead = {n_dead}/{K}",
        ],
        ["gray", "gray", "hot"],
    ):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9); ax.set_axis_off()

    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


def generate_videos(samples, device: str, frame_every: int = 10,
                    K: int = 70):
    """Generate convergence MP4/GIF for 3 representative digits."""
    VID_DIR.mkdir(exist_ok=True)

    # Pick one image for digits 1, 5, 8
    target_digits = [1, 5, 8]
    chosen = []
    for d in target_digits:
        cands = [(img, lbl) for img, lbl in samples if lbl == d]
        if cands:
            chosen.append(cands[0])

    for img, digit in chosen:
        print(f"  Encoding digit {digit} for video …")
        t0 = time.time()
        frames_data = encode_with_frames(
            img, K=K, device=device, frame_every=frame_every,
        )
        print(f"    {len(frames_data)} frames captured in {time.time()-t0:.0f}s")

        # Build frame images
        frame_imgs = []
        for (epoch, rendered, loss, psnr, n_dead) in frames_data:
            buf = _render_video_frame(
                img.numpy(), rendered, epoch, loss, psnr, n_dead, K,
            )
            frame_imgs.append(buf)

        # Try MP4 via imageio, fallback to GIF
        saved = False
        try:
            import imageio
            mp4_path = VID_DIR / f"digit_{digit}_convergence.mp4"
            imageio.mimwrite(str(mp4_path), frame_imgs, fps=15,
                             macro_block_size=None)
            print(f"    Saved {mp4_path.name}")
            saved = True
        except Exception as e:
            print(f"    MP4 failed ({type(e).__name__}); trying GIF …")

        if not saved:
            try:
                from PIL import Image as PILImage
                gif_path = VID_DIR / f"digit_{digit}_convergence.gif"
                pil_frames = [PILImage.fromarray(f) for f in frame_imgs]
                pil_frames[0].save(
                    str(gif_path), save_all=True,
                    append_images=pil_frames[1:],
                    duration=int(1000 / 10), loop=0,
                )
                print(f"    Saved {gif_path.name}")
            except Exception as e2:
                print(f"    GIF also failed ({e2}); saving individual frames …")
                frames_dir = VID_DIR / f"digit_{digit}_frames"
                frames_dir.mkdir(exist_ok=True)
                for idx, buf in enumerate(frame_imgs):
                    PILImage.fromarray(buf).save(
                        frames_dir / f"frame_{idx:04d}.png"
                    )
                print(f"    Frames saved to {frames_dir}/")
                print(f"    (assemble with: ffmpeg -r 15 -i frame_%04d.png -c:v libx264 out.mp4)")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device",           default="cpu",
                        help="Compute device (cpu or cuda)")
    parser.add_argument("--n_per_class",      type=int, default=5,
                        help="MNIST images per digit class (default 5)")
    parser.add_argument("--no_cache",         action="store_true",
                        help="Ignore cached encodings and re-encode")
    parser.add_argument("--skip_experiments", action="store_true",
                        help="Skip K-sensitivity, loss ablation, consistency tests")
    parser.add_argument("--skip_videos",      action="store_true",
                        help="Skip video generation")
    parser.add_argument("--data_root",        default=None,
                        help="MNIST data directory (default: data/)")
    parser.add_argument("--frame_every",      type=int, default=10,
                        help="Capture video frame every N epochs (default 10)")
    args = parser.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    VID_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  GAUSSIAN ENCODER RESEARCH — FIGURE GENERATOR")
    print("=" * 60)

    # ── 1. Load MNIST ─────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading MNIST ({args.n_per_class} images per digit) …")
    samples = load_mnist_samples(n_per_class=args.n_per_class,
                                 data_root=args.data_root)
    print(f"  Loaded {len(samples)} images total")

    # ── 2. Encode main batch ──────────────────────────────────────────────────
    print(f"\n[2/7] Encoding {len(samples)} images (K=70, ≤3000 epochs) …")
    results = encode_batch(samples, device=args.device,
                           cache_path=CACHE_FILE, force=args.no_cache)
    all_psnr = [r["psnr"] for r in results]
    n_alive  = [r["hist"][-1]["n_dead"] for r in results if r.get("hist")]
    print(f"  PSNR: mean={np.mean(all_psnr):.2f}  "
          f"std={np.std(all_psnr):.2f}  "
          f"min={min(all_psnr):.2f}  max={max(all_psnr):.2f} dB")
    if n_alive:
        print(f"  Dead Gaussians at end: mean={np.mean(n_alive):.1f}/70")

    # ── 3. Additional experiments ─────────────────────────────────────────────
    buggy_by_digit   = {}
    k_sens_data      = {}
    loss_comp_data   = {}
    consistency_data = []

    if not args.skip_experiments:
        print("\n[3/7] Running additional experiments …")

        # 3a. Buggy encoder (before-fix comparison)
        print("  3a. Before-fix encoder (500 epochs, buggy init) …")
        for d in range(10):
            d_imgs = [r["image"] for r in results if r["digit"] == d][:2]
            buggy_by_digit[d] = [
                encode_image_buggy(img, K=70, epochs=500, device=args.device)
                for img in d_imgs
            ]
        mean_buggy = np.mean([v for vs in buggy_by_digit.values() for v in vs])
        print(f"    Buggy mean PSNR: {mean_buggy:.1f} dB")

        # 3b. K-sensitivity
        print("  3b. K-sensitivity sweep [10, 20, 35, 50, 70, 100] …")
        test_imgs = [r["image"] for r in results if r["digit"] in [2, 5, 8]][:3]
        for K in [10, 20, 35, 50, 70, 100]:
            k_psnrs = []
            for img in test_imgs:
                W, loss = encode_image(
                    img, K=K, epochs=2000, lr=5e-3, kernel_size=11,
                    early_stop_threshold=1e-4, device=args.device,
                )
                k_psnrs.append(psnr_db(loss))
            k_sens_data[K] = k_psnrs
            print(f"    K={K:3d}: {np.mean(k_psnrs):.1f} ± {np.std(k_psnrs):.1f} dB")

        # 3c. Loss function comparison
        print("  3c. Loss ablation: MSE vs L1+SSIM (5 images × 1000 epochs) …")
        comp_imgs = []
        for d in [0, 2, 5, 7, 9]:
            cands = [r["image"] for r in results if r["digit"] == d]
            if cands:
                comp_imgs.append(cands[0])
        comp_imgs = comp_imgs[:5]
        loss_comp_data = {"mse": [], "l1ssim": []}
        for img in comp_imgs:
            for fn in ["mse", "l1ssim"]:
                hist = encode_with_loss_fn(img, fn, K=70, epochs=1000,
                                           device=args.device)
                loss_comp_data[fn].append(hist)

        # 3d. Consistency (3 runs same image)
        print("  3d. Consistency test: 3 independent runs on same image …")
        ref_img = next(r["image"] for r in results if r["digit"] == 5)
        for run_i in range(3):
            W, loss, hist = encode_image(ref_img, device=args.device,
                                         return_history=True, **ENC_KWARGS)
            consistency_data.append({
                "W": W, "loss": loss, "psnr": psnr_db(loss),
                "image": ref_img, "hist": hist,
            })
            print(f"    Run {run_i+1}: PSNR = {psnr_db(loss):.1f} dB")

    else:
        print("\n[3/7] Skipping additional experiments (--skip_experiments)")
        # Provide stub data so figure functions still run
        buggy_by_digit   = {d: [26.0] for d in range(10)}
        k_sens_data      = {k: [v] for k, v in
                            [(10, 20.0), (20, 27.0), (35, 33.0),
                             (50, 38.5), (70, 44.0), (100, 46.5)]}
        loss_comp_data   = {
            "mse":    [[{"epoch": 0, "psnr_db": 10.0},
                        {"epoch": 1000, "psnr_db": 42.0}]],
            "l1ssim": [[{"epoch": 0, "psnr_db": 10.0},
                        {"epoch": 1000, "psnr_db": 36.0}]],
        }
        consistency_data = [
            {"W": results[0]["W"], "loss": results[0]["loss"],
             "psnr": results[0]["psnr"], "image": results[0]["image"]}
            for _ in range(3)
        ]

    # ── 4. Generate all 14 figures ────────────────────────────────────────────
    print("\n[4/7] Generating figures …")
    fig_funcs = [
        ("01: reconstruction grid",        lambda: fig_01(results)),
        ("02: PSNR before/after fix",      lambda: fig_02(results, buggy_by_digit)),
        ("03: convergence curves",         lambda: fig_03(results)),
        ("04: PSNR histogram",             lambda: fig_04(results)),
        ("05: parameter distributions",    lambda: fig_05(results)),
        ("06: parameter correlation",      lambda: fig_06(results)),
        ("07: spatial distribution",       lambda: fig_07(results)),
        ("08: K-sensitivity",              lambda: fig_08(k_sens_data)),
        ("09: coordinate inversion diag",  fig_09),
        ("10: normalization coverage",     lambda: fig_10(results)),
        ("11: PCA latent space",           lambda: fig_11(results)),
        ("12: loss function ablation",     lambda: fig_12(loss_comp_data)),
        ("13: alpha vestigial",            lambda: fig_13(results)),
        ("14: consistency",                lambda: fig_14(consistency_data)),
    ]
    for name, fn in fig_funcs:
        print(f"  Figure {name}")
        try:
            fn()
        except Exception as e:
            print(f"    WARNING: {e}")
            import traceback; traceback.print_exc()

    print(f"\n[5/7] All figures saved to {FIG_DIR}")

    # ── 5. Videos ──────────────────────────────────────────────────────────────
    if not args.skip_videos:
        print(f"\n[6/7] Generating convergence videos "
              f"(frame_every={args.frame_every}) …")
        generate_videos(samples, device=args.device,
                        frame_every=args.frame_every)
    else:
        print("\n[6/7] Skipping video generation (--skip_videos)")

    # ── 6. Final summary ───────────────────────────────────────────────────────
    print("\n[7/7] Summary:")
    print(f"  N images encoded : {len(results)}")
    print(f"  PSNR mean        : {np.mean(all_psnr):.2f} dB")
    print(f"  PSNR std         : {np.std(all_psnr):.2f} dB")
    print(f"  PSNR min / max   : {min(all_psnr):.2f} / {max(all_psnr):.2f} dB")
    pct_above_40 = 100 * sum(p >= 40 for p in all_psnr) / len(all_psnr)
    print(f"  Fraction ≥ 40 dB : {pct_above_40:.1f}%")
    print(f"\nFigures: {FIG_DIR}")
    print(f"Videos:  {VID_DIR}")
    print(f"\nTo compile the report:")
    print(f"  cd {REPORT_DIR}")
    print(f"  pdflatex main.tex && pdflatex main.tex")


if __name__ == "__main__":
    main()
