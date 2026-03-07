"""
Encode raw images into 2D Gaussian splatting representations.

Each image is fitted independently via gradient descent. The result is a
tensor W of shape [K, C+6] with columns:
  Grayscale (C=1): [sigma_x, sigma_y, rho, alpha, colour, x, y]        → 7 cols
  RGB       (C=3): [sigma_x, sigma_y, rho, alpha, r, g, b, x, y]      → 9 cols

General layout: [sigma_x, sigma_y, rho, alpha, colour_0..colour_{C-1}, x, y]

Usage:
    python -m src.encode \\
        --data_dir <path_to_raw_images> \\
        --out_dir data/mnist_gaussian_representations/ \\
        --num_gaussians 70 \\
        --epochs 1000
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.utils.gaussian_to_image import (
    generate_2D_gaussian_splatting,
    generate_2D_gaussian_splatting_batch,
)


# ---------------------------------------------------------------------------
# SSIM helper (single-scale, grayscale)
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device="cpu") -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return torch.outer(g, g)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    1 - SSIM so that minimising it maximises structural similarity.
    pred/target: [H, W] or [1, H, W] float tensors in [0, 1].
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pred = pred.view(1, 1, *pred.shape[-2:])
    target = target.view(1, 1, *target.shape[-2:])
    kernel = _gaussian_kernel(window_size, device=pred.device)
    kernel = kernel.view(1, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad)
    mu2 = F.conv2d(target, kernel, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Initialisation from pixel statistics
# ---------------------------------------------------------------------------

def _init_gaussians(
    image: torch.Tensor,  # [H, W, C] in [0, 1]
    K: int,
    device: str,
    init_mode: str = "brightness",
    sigma_activation: str = "sigmoid",
) -> torch.Tensor:
    """
    Seed Gaussian positions weighted by pixel statistics.

    Args:
        init_mode: "brightness" (default) or "gradient" (weight by edge magnitude).
        sigma_activation: "sigmoid" (default) or "softplus" (adjusts init value).

    Returns raw (unconstrained) parameters of shape [K, C+6]:
    [log_sigma_x, log_sigma_y, atanh_rho, alpha_logit, colour_logits..., x_raw, y_raw]
    """
    if image.dim() == 2:
        image = image.unsqueeze(-1)  # [H, W] → [H, W, 1]
    H, W, C = image.shape
    brightness = image.mean(dim=-1)  # [H, W]

    if init_mode == "gradient":
        # Compute gradient magnitude via finite differences (Sobel-like)
        # Pad to handle borders
        padded = brightness.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        padded = F.pad(padded, (1, 1, 1, 1), mode="reflect")
        dx = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]  # horizontal
        dy = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]  # vertical
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8).squeeze()  # [H, W]
        # Blend: 70% gradient, 30% brightness
        weights = 0.7 * grad_mag + 0.3 * brightness
        flat = weights.view(-1)
    else:
        flat = brightness.view(-1)

    if flat.sum() < 1e-7:
        probs = torch.ones_like(flat) / flat.numel()
    else:
        probs = flat / (flat.sum() + 1e-8)

    indices = torch.multinomial(probs, num_samples=K, replacement=True)
    ys = (indices // W).float() / (H - 1) * 2 - 1   # in [-1, 1]
    xs = (indices % W).float() / (W - 1) * 2 - 1

    # Get colour values at sampled positions
    flat_colours = image.reshape(-1, C)  # [H*W, C]
    colours_phys = flat_colours[indices].clamp(0.05, 0.95)  # [K, C]
    colours_raw  = torch.logit(colours_phys)                # [K, C]

    # Renderer inverts coordinates: Gaussian with W_raw[i, 4+C]=r renders at x = -tanh(r).
    # To render at actual pixel position xs, we need W_raw[i, 4+C] = atanh(-xs).
    xs_raw = torch.atanh((-xs).clamp(-1 + 1e-6, 1 - 1e-6))
    ys_raw = torch.atanh((-ys).clamp(-1 + 1e-6, 1 - 1e-6))

    # Sigma init: adjust for activation function
    if sigma_activation == "softplus":
        # softplus(-1.5) ≈ 0.167, close to sigmoid(-1.5) ≈ 0.182
        sigma_init = torch.full((K,), -1.5, device=device)
    else:
        sigma_init = torch.full((K,), -1.5, device=device)  # sigmoid(-1.5) ≈ 0.182

    atanh_rho = torch.zeros(K, device=device)
    alpha_logit = torch.zeros(K, device=device)

    W_init = torch.cat([
        torch.stack([sigma_init, sigma_init, atanh_rho, alpha_logit], dim=1),  # [K, 4]
        colours_raw.to(device),                                                # [K, C]
        torch.stack([xs_raw.to(device), ys_raw.to(device)], dim=1),           # [K, 2]
    ], dim=1)  # [K, C+6]
    return W_init


# ---------------------------------------------------------------------------
# Raw → physical parameters
# ---------------------------------------------------------------------------

def _to_physical(W_raw: torch.Tensor, channels: int = 1,
                  sigma_activation: str = "sigmoid") -> dict:
    """Convert unconstrained raw params to physically valid ranges.

    Column layout: [sigma_x, sigma_y, rho, alpha, colour_0..colour_{C-1}, x, y]
    """
    C = channels
    if sigma_activation == "softplus":
        sigma_fn = F.softplus
    else:
        sigma_fn = torch.sigmoid
    return {
        "sigma_x": sigma_fn(W_raw[:, 0]),                # (0, inf) or (0, 1)
        "sigma_y": sigma_fn(W_raw[:, 1]),                # (0, inf) or (0, 1)
        "rho":     torch.tanh(W_raw[:, 2]),              # (-1, 1)
        "alpha":   torch.sigmoid(W_raw[:, 3]),           # (0, 1)
        "colours": torch.sigmoid(W_raw[:, 4:4+C]),      # [K, C] in (0, 1)
        "x":       torch.tanh(W_raw[:, 4+C]),            # (-1, 1)
        "y":       torch.tanh(W_raw[:, 4+C+1]),          # (-1, 1)
    }


def _render(p: dict, kernel_size: int, image_size: tuple, device: str,
            channels: int = 1, soft_clamp: bool = False) -> torch.Tensor:
    """Render Gaussians to a [H, W, C] image tensor in [0, 1]."""
    coords = torch.stack([p["x"], p["y"]], dim=1)   # [K, 2]
    img = generate_2D_gaussian_splatting(
        kernel_size=kernel_size,
        sigma_x=p["sigma_x"],
        sigma_y=p["sigma_y"],
        rho=p["rho"],
        coords=coords,
        colours=p["colours"],                        # [K, C]
        image_size=image_size,
        channels=channels,
        device=device,
        soft_clamp=soft_clamp,
    )  # [H, W, C]
    return img


# ---------------------------------------------------------------------------
# Dead-Gaussian detection
# ---------------------------------------------------------------------------

def _dead_mask(
    W_raw: torch.Tensor,   # [K, C+6] raw params (no grad needed)
    image: torch.Tensor,   # [H, W, C] float in [0, 1]
    threshold: float = 0.05,
    channels: int = 1,
    sigma_activation: str = "sigmoid",
) -> torch.Tensor:
    """
    Return a BoolTensor[K] marking Gaussians contributing nothing:
      - center lies on a background pixel (brightness < threshold).
    """
    H, W_img = image.shape[0], image.shape[1]
    with torch.no_grad():
        p = _to_physical(W_raw, channels, sigma_activation)
        # Renderer inverts coordinates: actual rendered position is (-p["x"], -p["y"]).
        px = ((-p["x"] + 1) / 2 * (W_img - 1)).long().clamp(0, W_img - 1)
        py = ((-p["y"] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
        brightness = image.mean(dim=-1)              # [H, W]
        target_at_center = brightness[py, px]        # [K]
        dead = target_at_center < threshold
    return dead


# ---------------------------------------------------------------------------
# Recycle dead Gaussians into under-reconstructed regions
# ---------------------------------------------------------------------------

def _recycle(
    W_raw: torch.Tensor,       # [K, C+6] — modified in-place
    image: torch.Tensor,       # [H, W, C] float in [0, 1]
    optimizer: torch.optim.Optimizer,
    kernel_size: int,
    image_size: tuple,
    device: str,
    threshold: float = 0.05,
    channels: int = 1,
    sigma_activation: str = "sigmoid",
    soft_clamp: bool = False,
) -> int:
    """
    Detect dead Gaussians and teleport them to under-reconstructed image regions.
    Returns the number of recycled Gaussians.
    """
    dead = _dead_mask(W_raw, image, threshold, channels, sigma_activation)
    n_dead = int(dead.sum().item())
    if n_dead == 0:
        return 0

    C = channels
    H, W_img = image_size
    with torch.no_grad():
        # Render current approximation
        p = _to_physical(W_raw, C, sigma_activation)
        rendered = _render(p, kernel_size, image_size, device, C, soft_clamp)

        # Sample from residual (under-reconstructed regions)
        residual = (image - rendered).clamp(min=0)            # [H, W, C]
        residual_brightness = residual.mean(dim=-1)           # [H, W]
        residual_probs = residual_brightness.view(-1) / (residual_brightness.sum() + 1e-8)

        indices = torch.multinomial(residual_probs, num_samples=n_dead, replacement=True)
        ys = (indices // W_img).float() / (H - 1) * 2 - 1
        xs = (indices % W_img).float() / (W_img - 1) * 2 - 1

        # Build new raw params for recycled rows.
        xs_raw = torch.atanh((-xs).clamp(-1 + 1e-6, 1 - 1e-6))
        ys_raw = torch.atanh((-ys).clamp(-1 + 1e-6, 1 - 1e-6))

        flat_colours = image.reshape(-1, C)                    # [H*W, C]
        pixel_vals = flat_colours[indices].clamp(0.05, 0.95)   # [n_dead, C]
        colour_raw = torch.logit(pixel_vals)                   # [n_dead, C]

        log_sigma = torch.full((n_dead,), -1.5, device=device)
        new_rows = torch.cat([
            torch.stack([
                log_sigma, log_sigma,
                torch.zeros(n_dead, device=device),            # atanh_rho
                torch.full((n_dead,), -2.0, device=device),   # alpha_logit
            ], dim=1),                                         # [n_dead, 4]
            colour_raw,                                        # [n_dead, C]
            torch.stack([xs_raw, ys_raw], dim=1),              # [n_dead, 2]
        ], dim=1)                                              # [n_dead, C+6]

        W_raw.data[dead] = new_rows

        # Reset Adam moment tensors for recycled rows
        if W_raw in optimizer.state:
            state = optimizer.state[W_raw]
            if "exp_avg" in state:
                state["exp_avg"][dead] = 0.0
            if "exp_avg_sq" in state:
                state["exp_avg_sq"][dead] = 0.0

    return n_dead


# ---------------------------------------------------------------------------
# Single-image encode
# ---------------------------------------------------------------------------

def encode_image(
    image: torch.Tensor,          # [H, W] for grayscale or [C, H, W] for multi-channel
    K: int = 70,
    epochs: int = 1000,
    lr: float = 5e-3,
    kernel_size: int = 11,
    early_stop_threshold: float = 0.005,
    device: str = "cpu",
    return_history: bool = False,
    recycle_every: int = 300,
    recycle_threshold: float = 0.05,
    log_every: int = 50,
    sigma_activation: str = "sigmoid",
    init_mode: str = "brightness",
    soft_clamp: bool = False,
    use_scheduler: bool = True,
):
    """
    Fit K Gaussians to a single image (grayscale or RGB).

    Input formats:
        [H, W]    → grayscale, C=1
        [C, H, W] → multi-channel (e.g. RGB with C=3)

    Returns:
        W: tensor of shape [K, C+6] with physical parameters
           [sigma_x, sigma_y, rho, alpha, colour_0..colour_{C-1}, x, y]
        final_loss: float
        history (only if return_history=True): list[dict]
    """
    # Normalise input to [H, W, C]
    if image.dim() == 2:
        image = image.unsqueeze(-1)          # [H, W] → [H, W, 1]
    elif image.dim() == 3:
        image = image.permute(1, 2, 0)       # [C, H, W] → [H, W, C]
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.dim()}D")

    H, W_img, C = image.shape
    image = image.to(device)

    W_raw = _init_gaussians(image, K, device, init_mode, sigma_activation).requires_grad_(True)
    optimizer = torch.optim.Adam([W_raw], lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, epochs // 3), T_mult=1, eta_min=1e-5
        )

    history: list = [] if return_history else None
    final_loss = float("inf")
    image_size = (H, W_img)

    for epoch in range(epochs):
        optimizer.zero_grad()
        p = _to_physical(W_raw, C, sigma_activation)
        rendered = _render(p, kernel_size, image_size, device, C, soft_clamp)
        loss = F.mse_loss(rendered, image)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        final_loss = loss.item()

        # Recycle dead Gaussians
        if recycle_every > 0 and (epoch + 1) % recycle_every == 0:
            _recycle(
                W_raw, image, optimizer, kernel_size, image_size, device,
                recycle_threshold, C, sigma_activation, soft_clamp,
            )

        if return_history and (epoch % log_every == 0 or epoch == epochs - 1):
            with torch.no_grad():
                mse = F.mse_loss(rendered.detach(), image).item()
                psnr = min(100.0, 10 * math.log10(1.0 / mse)) if mse > 1e-10 else 100.0
                n_dead = int(_dead_mask(W_raw, image, recycle_threshold, C, sigma_activation).sum().item())
            history.append({
                "epoch": epoch,
                "loss": final_loss,
                "psnr_db": psnr,
                "n_dead": n_dead,
            })

        if final_loss < early_stop_threshold:
            break

    with torch.no_grad():
        p = _to_physical(W_raw, C, sigma_activation)
        W_phys = torch.cat([
            torch.stack([p["sigma_x"], p["sigma_y"], p["rho"], p["alpha"]], dim=1),  # [K, 4]
            p["colours"],                                                              # [K, C]
            torch.stack([p["x"], p["y"]], dim=1),                                     # [K, 2]
        ], dim=1)  # [K, C+6]

    if return_history:
        return W_phys.cpu(), final_loss, history
    return W_phys.cpu(), final_loss


# ---------------------------------------------------------------------------
# Batched encode (multiple images in parallel on one GPU)
# ---------------------------------------------------------------------------

def _to_physical_batch(W_raw: torch.Tensor, channels: int = 3,
                       sigma_activation: str = "sigmoid") -> dict:
    """Batched version of _to_physical. W_raw: [B, K, C+6]."""
    C = channels
    if sigma_activation == "softplus":
        sigma_fn = F.softplus
    else:
        sigma_fn = torch.sigmoid
    return {
        "sigma_x": sigma_fn(W_raw[:, :, 0]),           # [B, K]
        "sigma_y": sigma_fn(W_raw[:, :, 1]),
        "rho":     torch.tanh(W_raw[:, :, 2]),
        "alpha":   torch.sigmoid(W_raw[:, :, 3]),
        "colours": torch.sigmoid(W_raw[:, :, 4:4+C]),  # [B, K, C]
        "x":       torch.tanh(W_raw[:, :, 4+C]),
        "y":       torch.tanh(W_raw[:, :, 4+C+1]),
    }


def _render_batch(p: dict, kernel_size: int, image_size: tuple,
                  device: str, channels: int = 3,
                  soft_clamp: bool = False) -> torch.Tensor:
    """Render B images in parallel. Returns [B, H, W, C]."""
    coords = torch.stack([p["x"], p["y"]], dim=2)  # [B, K, 2]
    return generate_2D_gaussian_splatting_batch(
        kernel_size=kernel_size,
        sigma_x=p["sigma_x"],
        sigma_y=p["sigma_y"],
        rho=p["rho"],
        coords=coords,
        colours=p["colours"],
        image_size=image_size,
        channels=channels,
        device=device,
        soft_clamp=soft_clamp,
    )


def encode_batch(
    images: torch.Tensor,             # [B, C, H, W] float in [0, 1]
    K: int = 500,
    epochs: int = 3000,
    lr: float = 0.04,
    kernel_size: int = 32,
    early_stop_threshold: float = 1e-5,
    device: str = "cuda",
    sigma_activation: str = "sigmoid",
    init_mode: str = "brightness",
    soft_clamp: bool = True,
    use_scheduler: bool = False,
):
    """
    Fit K Gaussians to B images in parallel on one GPU.

    Args:
        images: [B, C, H, W] float tensor in [0, 1]

    Returns:
        W_phys: [B, K, C+6] physical parameters
        losses: [B] final per-image MSE losses
    """
    B, C, H, W_img = images.shape
    images_hwc = images.permute(0, 2, 3, 1).to(device)  # [B, H, W, C]
    image_size = (H, W_img)

    # Initialize each image's Gaussians
    W_list = []
    for i in range(B):
        W_list.append(_init_gaussians(images_hwc[i], K, device, init_mode, sigma_activation))
    W_raw = torch.stack(W_list)  # [B, K, C+6]
    W_raw = W_raw.requires_grad_(True)

    optimizer = torch.optim.Adam([W_raw], lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, epochs // 3), T_mult=1, eta_min=1e-5
        )

    # Track per-image convergence; snapshot params at convergence to avoid
    # Adam momentum drift degrading quality after early stop.
    active = torch.ones(B, dtype=torch.bool, device=device)
    best_W = W_raw.detach().clone()      # [B, K, C+6] — best params so far
    best_losses = torch.full((B,), float('inf'), device=device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        p = _to_physical_batch(W_raw, C, sigma_activation)
        rendered = _render_batch(p, kernel_size, image_size, device, C, soft_clamp)

        # Per-image MSE: [B]
        per_loss = ((rendered - images_hwc) ** 2).mean(dim=(1, 2, 3))

        # Sum (not mean!) across images — each image's params are independent,
        # so summing gives the same per-image gradient as sequential encoding.
        loss = (per_loss * active.float()).sum()
        loss.backward()

        # Zero gradients for converged images so Adam doesn't touch them
        with torch.no_grad():
            if not active.all():
                W_raw.grad[~active] = 0.0

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Snapshot best params and check convergence
        with torch.no_grad():
            cur_losses = per_loss.detach()
            improved = cur_losses < best_losses
            if improved.any():
                best_W[improved] = W_raw.detach()[improved]
                best_losses[improved] = cur_losses[improved]

            newly_done = active & (cur_losses < early_stop_threshold)
            active = active & ~newly_done
            if not active.any():
                break

    # Extract physical parameters from best snapshot
    with torch.no_grad():
        p = _to_physical_batch(best_W, C, sigma_activation)
        W_phys = torch.cat([
            torch.stack([p["sigma_x"], p["sigma_y"], p["rho"], p["alpha"]], dim=2),  # [B, K, 4]
            p["colours"],                                                              # [B, K, C]
            torch.stack([p["x"], p["y"]], dim=2),                                     # [B, K, 2]
        ], dim=2)  # [B, K, C+6]

    return W_phys.cpu(), best_losses.cpu()


# ---------------------------------------------------------------------------
# CLI (grayscale legacy)
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> Optional[torch.Tensor]:
    """Load an image file as a [H, W] float tensor in [0, 1], or None on failure."""
    try:
        img = Image.open(path).convert("L")
        return transforms.ToTensor()(img).squeeze(0)  # [H, W]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Encode images into Gaussian representations")
    parser.add_argument("--data_dir", required=True, help="Directory containing input images")
    parser.add_argument("--out_dir", default="data/mnist_gaussian_representations/",
                        help="Output directory for .pt files")
    parser.add_argument("--num_gaussians", type=int, default=70, help="Number of Gaussians K")
    parser.add_argument("--epochs", type=int, default=1000, help="Fitting epochs per image")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--kernel_size", type=int, default=11, help="Gaussian kernel size")
    parser.add_argument("--early_stop", type=float, default=0.005,
                        help="Stop early if loss < threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_paths = sorted(
        p for p in Path(args.data_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.data_dir}")

    print(f"Encoding {len(image_paths)} images → {args.out_dir}")
    for path in tqdm(image_paths):
        out_path = os.path.join(args.out_dir, path.stem + ".pt")
        if os.path.exists(out_path):
            continue  # skip already encoded
        img = _load_image(path)
        if img is None:
            print(f"  Skipping {path.name} (load error)")
            continue
        W, loss = encode_image(
            img,
            K=args.num_gaussians,
            epochs=args.epochs,
            lr=args.lr,
            kernel_size=args.kernel_size,
            early_stop_threshold=args.early_stop,
            device=args.device,
        )
        torch.save({"W": W}, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
