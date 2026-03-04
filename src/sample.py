"""
Sample from a trained GaussianDiffusion checkpoint and compute metrics.

Usage:
    python -m src.sample \\
        --checkpoint checkpoints/best.pt \\
        --n_samples 2000
"""

import argparse
import json
import os

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.ddpm import DDPM
from src.models.transformer_model import GaussianTransformer
from src.utils.denormalize import denormalize_parameters
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

# Must match the ranges used during training
PARAM_RANGES = [
    (0.0, 1.0),    # sigma_x
    (0.0, 1.0),    # sigma_y
    (-1.0, 1.0),   # rho
    (-1.0, 1.0),   # alpha
    (0.0, 1.0),    # colour
    (-1.0, 1.0),   # x
    (-1.0, 1.0),   # y
]


# ---------------------------------------------------------------------------
# Reverse diffusion
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_gaussians(
    model: torch.nn.Module,
    ddpm: DDPM,
    n: int,
    K: int,
    feat_dim: int,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Generate n samples in normalised Gaussian space. Returns [n, K, feat_dim]."""
    model.eval()
    all_samples = []
    remaining = n
    while remaining > 0:
        bs = min(batch_size, remaining)
        x = torch.randn(bs, K, feat_dim, device=device)
        for t in range(ddpm.n_T, 0, -1):
            t_tensor = torch.full((bs,), t, dtype=torch.float32, device=device)
            eps_pred = model(x, t_tensor)
            alpha_t = ddpm.alpha_t[t].to(device)
            oneover_sqrta = ddpm.oneover_sqrta[t].to(device)
            mab_over_sqrtmab = ddpm.mab_over_sqrtmab[t].to(device)
            sqrt_beta_t = ddpm.sqrt_beta_t[t].to(device)
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            x = oneover_sqrta * (x - mab_over_sqrtmab * eps_pred) + sqrt_beta_t * z
        all_samples.append(x.cpu())
        remaining -= bs
    return torch.cat(all_samples, dim=0)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_batch(
    W_norm: torch.Tensor,  # [N, K, 7]
    param_ranges: list,
    kernel_size: int = 11,
    image_size: tuple = (28, 28),
    device: str = "cpu",
) -> torch.Tensor:
    """Render normalised Gaussians to uint8 [N, 3, H, W] for metric computation."""
    W_phys = denormalize_parameters(W_norm.to(device), param_ranges)
    images = []
    for i in range(W_phys.shape[0]):
        w = W_phys[i]
        sigma_x = w[:, 0].clamp(1e-4, 1.0)
        sigma_y = w[:, 1].clamp(1e-4, 1.0)
        rho = w[:, 2].clamp(-0.999, 0.999)
        colour = w[:, 4].clamp(0, 1).unsqueeze(1)
        coords = torch.stack([w[:, 5], w[:, 6]], dim=1)
        img = generate_2D_gaussian_splatting(
            kernel_size=kernel_size,
            sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
            coords=coords, colours=colour,
            image_size=image_size, channels=1, device=device,
        )  # [H, W, 1]
        # Expand to 3 channels, convert to uint8
        img_3c = img.permute(2, 0, 1).repeat(3, 1, 1)  # [3, H, W]
        images.append(img_3c)
    return torch.stack(images, dim=0)  # [N, 3, H, W] float in [0, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sample from GaussianDiffusion checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", default="samples/")
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--real_data_dir", default=None,
                        help="Directory of .pt files for real-data FID comparison (optional)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt.get("args", {})
    K = train_args.get("num_gaussians", 70)
    T = train_args.get("timesteps", 200)
    feat_dim = 7

    ddpm = DDPM(n_T=T, schedule_type="cosine")
    model = GaussianTransformer(
        input_dim=K,
        time_emb_dim=512,
        feature_dim=feat_dim,
        num_timestamps=T,
        num_transformer_blocks=32,
        num_heads=64,
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ---- Sample ----
    print(f"Sampling {args.n_samples} images …")
    W_norm = sample_gaussians(model, ddpm, n=args.n_samples, K=K, feat_dim=feat_dim,
                               device=args.device, batch_size=args.batch_size)

    # ---- Render all ----
    image_size = (args.image_size, args.image_size)
    print("Rendering …")
    rendered = render_batch(W_norm, PARAM_RANGES,
                            kernel_size=args.kernel_size,
                            image_size=image_size,
                            device=args.device)  # [N, 3, H, W] float

    # ---- Save grid of first 64 ----
    grid = make_grid(rendered[:64], nrow=8, normalize=False)
    grid_path = os.path.join(args.out_dir, "sample_grid.png")
    save_image(grid, grid_path)
    print(f"Saved sample grid → {grid_path}")

    # ---- Metrics ----
    rendered_u8 = (rendered * 255).clamp(0, 255).to(torch.uint8)

    metrics = {}

    # IS
    print("Computing IS …")
    is_metric = InceptionScore(normalize=True).to(args.device)
    for i in tqdm(range(0, len(rendered_u8), args.batch_size)):
        batch = rendered_u8[i: i + args.batch_size].to(args.device)
        is_metric.update(batch)
    is_mean, is_std = is_metric.compute()
    metrics["IS_mean"] = is_mean.item()
    metrics["IS_std"] = is_std.item()

    # FID (requires real images)
    if args.real_data_dir is not None:
        from src.dataset import GaussianDataset
        from src.utils.normalize import normalize_parameters

        print("Loading real data for FID/KID …")
        real_ds = GaussianDataset(args.real_data_dir, num_gaussians=K)
        real_W = real_ds.data  # [N, K, 7]
        real_rendered = render_batch(real_W, PARAM_RANGES,
                                     kernel_size=args.kernel_size,
                                     image_size=image_size,
                                     device=args.device)
        real_u8 = (real_rendered * 255).clamp(0, 255).to(torch.uint8)

        print("Computing FID …")
        fid_metric = FrechetInceptionDistance(normalize=True).to(args.device)
        for i in tqdm(range(0, len(real_u8), args.batch_size)):
            fid_metric.update(real_u8[i: i + args.batch_size].to(args.device), real=True)
        for i in tqdm(range(0, len(rendered_u8), args.batch_size)):
            fid_metric.update(rendered_u8[i: i + args.batch_size].to(args.device), real=False)
        metrics["FID"] = fid_metric.compute().item()

        print("Computing KID …")
        kid_metric = KernelInceptionDistance(subset_size=min(1000, len(real_u8)),
                                              normalize=True).to(args.device)
        for i in tqdm(range(0, len(real_u8), args.batch_size)):
            kid_metric.update(real_u8[i: i + args.batch_size].to(args.device), real=True)
        for i in tqdm(range(0, len(rendered_u8), args.batch_size)):
            kid_metric.update(rendered_u8[i: i + args.batch_size].to(args.device), real=False)
        kid_mean, kid_std = kid_metric.compute()
        metrics["KID_mean"] = kid_mean.item()
        metrics["KID_std"] = kid_std.item()

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Saved → {metrics_path}")


if __name__ == "__main__":
    main()
