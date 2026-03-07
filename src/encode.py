"""
Encode a folder of raw grayscale images into 2D Gaussian splatting representations.

Each image is fitted independently via gradient descent. The result is a
tensor W of shape [K, 7] with columns:
  [sigma_x, sigma_y, rho, alpha, colour, x, y]
saved as a .pt file containing {"W": W}.

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

from src.utils.gaussian_to_image import generate_2D_gaussian_splatting


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
    image: torch.Tensor,  # [H, W] in [0, 1]
    K: int,
    device: str,
) -> torch.Tensor:
    """
    Seed Gaussian positions weighted by pixel brightness: Gaussians start inside
    bright regions (stroke interiors) where they need to contribute.

    Returns raw (unconstrained) parameters of shape [K, 7]:
    [log_sigma_x, log_sigma_y, atanh_rho, alpha_logit, colour_logit, x_raw, y_raw]
    """
    H, W = image.shape
    flat = image.view(-1)

    if flat.sum() < 1e-7:
        # All-black image: pure uniform sampling
        probs = torch.ones_like(flat) / flat.numel()
    else:
        # Sample positions weighted by brightness: Gaussians start inside bright
        # regions (e.g. digit strokes) where they actually need to fit.
        # NOTE: gradient-weighted init (Sobel) sounds appealing but is wrong for
        # flat images like MNIST — gradients are zero inside strokes and high only
        # at 1-px edges, so Gaussians initialise straddling background/stroke and
        # take much longer to converge inward.
        # NOTE: replacement=False was tested and hurts PSNR significantly (~6 dB at
        # epoch 1000). With replacement=True, multiple Gaussians piling onto the same
        # bright pixels creates strong competitive pressure that rapidly diversifies
        # positions through gradient-driven competition in epochs 0-500. Without
        # replacement, Gaussians spread to dimmer pixels and lose the strong initial
        # gradient signal that drives fast convergence.
        probs = flat / (flat.sum() + 1e-8)

    indices = torch.multinomial(probs, num_samples=K, replacement=True)
    ys = (indices // W).float() / (H - 1) * 2 - 1   # in [-1, 1]
    xs = (indices % W).float() / (W - 1) * 2 - 1

    colours_phys = flat[indices].clamp(0.05, 0.95)
    colours_raw  = torch.logit(colours_phys)                            # inverse of sigmoid
    # Renderer inverts coordinates: Gaussian with W_raw[i,5]=r renders at x = -tanh(r).
    # To render at actual pixel position xs, we need W_raw[i,5] = atanh(-xs).
    xs_raw = torch.atanh((-xs).clamp(-1 + 1e-6, 1 - 1e-6))
    ys_raw = torch.atanh((-ys).clamp(-1 + 1e-6, 1 - 1e-6))
    log_sigma = torch.full((K,), -1.5, device=device)  # sigma ~ 0.22
    atanh_rho = torch.zeros(K, device=device)
    alpha_logit = torch.zeros(K, device=device)

    W_init = torch.stack(
        [log_sigma, log_sigma, atanh_rho, alpha_logit,
         colours_raw.to(device), xs_raw.to(device), ys_raw.to(device)],
        dim=1,
    )  # [K, 7]
    return W_init


# ---------------------------------------------------------------------------
# Raw → physical parameters
# ---------------------------------------------------------------------------

def _to_physical(W_raw: torch.Tensor) -> dict:
    """Convert unconstrained raw params to physically valid ranges."""
    return {
        "sigma_x": torch.sigmoid(W_raw[:, 0]),          # (0, 1)
        "sigma_y": torch.sigmoid(W_raw[:, 1]),          # (0, 1)
        "rho":     torch.tanh(W_raw[:, 2]),              # (-1, 1)
        "alpha":   torch.sigmoid(W_raw[:, 3]),           # (0, 1)
        "colour":  torch.sigmoid(W_raw[:, 4]),           # (0, 1)
        "x":       torch.tanh(W_raw[:, 5]),              # (-1, 1)
        "y":       torch.tanh(W_raw[:, 6]),              # (-1, 1)
    }


def _render(p: dict, kernel_size: int, image_size: tuple, device: str) -> torch.Tensor:
    """Render Gaussians to a [H, W] image tensor in [0, 1]."""
    coords = torch.stack([p["x"], p["y"]], dim=1)   # [K, 2]
    colours = p["colour"].unsqueeze(1)               # [K, 1]
    img = generate_2D_gaussian_splatting(
        kernel_size=kernel_size,
        sigma_x=p["sigma_x"],
        sigma_y=p["sigma_y"],
        rho=p["rho"],
        coords=coords,
        colours=colours,
        image_size=image_size,
        channels=1,
        device=device,
    )  # [H, W, 1]
    return img[:, :, 0]  # [H, W]


# ---------------------------------------------------------------------------
# Dead-Gaussian detection
# ---------------------------------------------------------------------------

def _dead_mask(
    W_raw: torch.Tensor,   # [K, 7] raw params (no grad needed)
    image: torch.Tensor,   # [H, W] float in [0, 1]
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Return a BoolTensor[K] marking Gaussians that are contributing nothing:
      - center lies on a background pixel (target < threshold), OR
      - colour has been suppressed below threshold by the optimizer.
    """
    H, W = image.shape
    with torch.no_grad():
        p = _to_physical(W_raw)
        # The renderer inverts coordinates: a Gaussian with p["x"]=v renders at
        # output position x = -v (due to affine_grid translation convention).
        # After training the optimizer compensates by setting p["x"] ≈ -actual_x.
        # So the ACTUAL rendered pixel is at (-p["x"], -p["y"]).
        px = ((-p["x"] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
        py = ((-p["y"] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
        target_at_center = image[py, px]           # [K]
        # Only flag Gaussians whose center has drifted onto a background pixel.
        # Do NOT use colour < threshold: the optimizer legitimately suppresses
        # colour for overlapping Gaussians (trading off coverage), which causes
        # false positives that recycle converging Gaussians and destroy progress.
        dead = target_at_center < threshold
    return dead


# ---------------------------------------------------------------------------
# Recycle dead Gaussians into under-reconstructed regions
# ---------------------------------------------------------------------------

def _recycle(
    W_raw: torch.Tensor,       # [K, 7] — modified in-place
    image: torch.Tensor,       # [H, W] float in [0, 1]
    optimizer: torch.optim.Optimizer,
    kernel_size: int,
    image_size: tuple,
    device: str,
    threshold: float = 0.05,
) -> int:
    """
    Detect dead Gaussians and teleport them to under-reconstructed image regions.
    Returns the number of recycled Gaussians.
    """
    dead = _dead_mask(W_raw, image, threshold)
    n_dead = int(dead.sum().item())
    if n_dead == 0:
        return 0

    H, W_img = image_size
    with torch.no_grad():
        # Render current approximation
        p = _to_physical(W_raw)
        rendered = _render(p, kernel_size, image_size, device)

        # Sample from residual (under-reconstructed regions)
        residual = (image - rendered).clamp(min=0)
        residual_probs = residual.view(-1) / (residual.sum() + 1e-8)

        indices = torch.multinomial(residual_probs, num_samples=n_dead, replacement=True)
        ys = (indices // W_img).float() / (H - 1) * 2 - 1
        xs = (indices % W_img).float() / (W_img - 1) * 2 - 1

        # Build new raw params for recycled rows.
        # Renderer inverts: Gaussian with W_raw[i,5]=r renders at x = -tanh(r).
        # To render at actual pixel position xs, we need tanh(W_raw[i,5]) = -xs.
        xs_raw = torch.atanh((-xs).clamp(-1 + 1e-6, 1 - 1e-6))
        ys_raw = torch.atanh((-ys).clamp(-1 + 1e-6, 1 - 1e-6))
        pixel_vals = image.view(-1)[indices].clamp(0.05, 0.95)
        colour_raw = torch.logit(pixel_vals)

        log_sigma = torch.full((n_dead,), -1.5, device=device)  # sigma ~ 0.22, same as init
        new_rows = torch.stack([
            log_sigma, log_sigma,
            torch.zeros(n_dead, device=device),   # atanh_rho
            torch.full((n_dead,), -2.0, device=device),  # alpha_logit
            colour_raw,
            xs_raw,
            ys_raw,
        ], dim=1)

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
    image: torch.Tensor,          # [H, W] float in [0, 1]
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
):
    """
    Fit K Gaussians to a single grayscale image.

    Returns:
        W: tensor of shape [K, 7] with physical parameters
           [sigma_x, sigma_y, rho, alpha, colour, x, y]
        final_loss: float
        history (only if return_history=True): list[dict] with keys
            {epoch, loss, psnr_db, n_dead, recycling_event}
    """
    H, W_img = image.shape
    image = image.to(device)

    W_raw = _init_gaussians(image, K, device).requires_grad_(True)
    optimizer = torch.optim.Adam([W_raw], lr=lr)
    # CosineAnnealingWarmRestarts with 3 equal cycles gives the optimizer multiple
    # fresh high-LR starts to escape local minima.  Empirical observation: images
    # that plateau at ~26 dB can jump 6-9 dB when the optimizer escapes a local
    # minimum with LR ≈ 1e-3; restarting at full LR every epochs/3 iterations makes
    # this more reliable.  T_0=max(1, epochs//3) gives 3 cycles regardless of
    # total epoch count.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 3), T_mult=1, eta_min=1e-5
    )

    history: list = [] if return_history else None
    final_loss = float("inf")
    convergence_epoch = -1
    image_size = (H, W_img)

    for epoch in range(epochs):
        optimizer.zero_grad()
        p = _to_physical(W_raw)
        rendered = _render(p, kernel_size, image_size, device)
        loss = F.mse_loss(rendered, image)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = loss.item()

        # Recycle dead Gaussians
        recycling_this_window = False
        if recycle_every > 0 and (epoch + 1) % recycle_every == 0:
            n_recycled = _recycle(
                W_raw, image, optimizer, kernel_size, image_size, device, recycle_threshold
            )
            recycling_this_window = (n_recycled > 0)

        if return_history and (epoch % log_every == 0 or epoch == epochs - 1):
            with torch.no_grad():
                mse = F.mse_loss(rendered.detach(), image).item()
                psnr = min(100.0, 10 * math.log10(1.0 / mse)) if mse > 1e-10 else 100.0
                n_dead = int(_dead_mask(W_raw, image, recycle_threshold).sum().item())
            history.append({
                "epoch": epoch,
                "loss": final_loss,
                "psnr_db": psnr,
                "n_dead": n_dead,
                "recycling_event": recycling_this_window,
            })

        if final_loss < early_stop_threshold:
            convergence_epoch = epoch
            break

    with torch.no_grad():
        p = _to_physical(W_raw)
        W_phys = torch.stack(
            [p["sigma_x"], p["sigma_y"], p["rho"], p["alpha"], p["colour"], p["x"], p["y"]],
            dim=1,
        )  # [K, 7]

    if return_history:
        return W_phys.cpu(), final_loss, history
    return W_phys.cpu(), final_loss


# ---------------------------------------------------------------------------
# CLI
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
