"""
Sample from a trained CIFAR-10 GaussianDiffusion checkpoint and compute FID/IS/KID.

Usage:
    python scripts/cifar10/sample.py \
        --checkpoint checkpoints/cifar10/best.pt \
        --real_data_h5 data/cifar10/cifar10_gaussians_K500.h5 \
        --n_samples 10000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.cifar10 import CIFAR10_CONFIG  # noqa: E402
from src.ddpm import DDPM  # noqa: E402
from src.models.transformer_model import GaussianTransformer  # noqa: E402
from src.utils.denormalize import denormalize_parameters  # noqa: E402
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting  # noqa: E402
from src.utils.normalize import normalize_parameters  # noqa: E402

CFG = CIFAR10_CONFIG
PARAM_RANGES = CFG["param_ranges"]
FEAT_DIM = CFG["feature_dim"]  # 8
IMAGE_SIZE = (CFG["image_size"], CFG["image_size"])  # (32, 32)
KERNEL_SIZE = CFG["kernel_size"]  # 32
SOFT_CLAMP = CFG["soft_clamp"]  # True
NUM_CLASSES_DEFAULT = CFG["num_classes"]  # 10


# ---------------------------------------------------------------------------
# DDPM sampler
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_gaussians(model, ddpm, n, K, device, batch_size=64,
                     num_classes=0, cfg_scale=0.0):
    """DDPM reverse diffusion. Returns [n, K, FEAT_DIM]."""
    model.eval()
    use_cfg = num_classes > 0 and cfg_scale > 0
    all_samples = []
    remaining = n
    while remaining > 0:
        bs = min(batch_size, remaining)
        x = torch.randn(bs, K, FEAT_DIM, device=device)
        if use_cfg:
            labels = torch.arange(num_classes, device=device).repeat(
                (bs + num_classes - 1) // num_classes)[:bs]
            null_labels = torch.full((bs,), num_classes, dtype=torch.long, device=device)
        for t in range(ddpm.n_T, 0, -1):
            t_tensor = torch.full((bs,), t, dtype=torch.float32, device=device)
            if use_cfg:
                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t_tensor, t_tensor], dim=0)
                y_double = torch.cat([labels, null_labels], dim=0)
                eps_double = model(x_double, t_double, y=y_double)
                eps_cond, eps_uncond = eps_double.chunk(2, dim=0)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = model(x, t_tensor)
            oneover_sqrta = ddpm.oneover_sqrta[t].to(device)
            mab_over_sqrtmab = ddpm.mab_over_sqrtmab[t].to(device)
            sqrt_beta_t = ddpm.sqrt_beta_t[t].to(device)
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            x = oneover_sqrta * (x - mab_over_sqrtmab * eps_pred) + sqrt_beta_t * z
        all_samples.append(x.cpu())
        remaining -= bs
    return torch.cat(all_samples, dim=0)


# ---------------------------------------------------------------------------
# DDIM sampler
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_gaussians_ddim(model, ddpm, n, K, device, batch_size=64,
                          num_classes=0, cfg_scale=0.0,
                          ddim_steps=50, eta=0.0):
    """DDIM reverse diffusion. Returns [n, K, FEAT_DIM]."""
    model.eval()
    use_cfg = num_classes > 0 and cfg_scale > 0
    timesteps = torch.linspace(ddpm.n_T - 1, 0, ddim_steps + 1).round().long()
    alphabar = ddpm.alphabar_t.to(device)

    all_samples = []
    remaining = n
    while remaining > 0:
        bs = min(batch_size, remaining)
        x = torch.randn(bs, K, FEAT_DIM, device=device)
        if use_cfg:
            labels = torch.arange(num_classes, device=device).repeat(
                (bs + num_classes - 1) // num_classes)[:bs]
            null_labels = torch.full((bs,), num_classes, dtype=torch.long, device=device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            t_tensor = torch.full((bs,), t, dtype=torch.float32, device=device)

            if use_cfg:
                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t_tensor, t_tensor], dim=0)
                y_double = torch.cat([labels, null_labels], dim=0)
                eps_double = model(x_double, t_double, y=y_double)
                eps_cond, eps_uncond = eps_double.chunk(2, dim=0)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = model(x, t_tensor)

            ab_t = alphabar[t]
            ab_prev = alphabar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
            x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-4)
            x0_pred = x0_pred.clamp(-5, 5)

            if t_prev > 0 and eta > 0:
                sigma_t = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
            else:
                sigma_t = 0.0

            dir_xt = (1 - ab_prev - sigma_t ** 2).clamp(min=0).sqrt() * eps_pred
            noise = torch.randn_like(x) if (t_prev > 0 and eta > 0) else torch.zeros_like(x)
            x = ab_prev.sqrt() * x0_pred + dir_xt + sigma_t * noise

        all_samples.append(x.cpu())
        remaining -= bs
    return torch.cat(all_samples, dim=0)


# ---------------------------------------------------------------------------
# Render CIFAR-10 Gaussians to RGB images
# ---------------------------------------------------------------------------
def render_batch(W_norm, device="cpu"):
    """Render normalised CIFAR-10 Gaussians to [N, 3, H, W] float images."""
    W_phys = denormalize_parameters(W_norm.to(device), PARAM_RANGES)
    images = []
    for i in range(W_phys.shape[0]):
        w = W_phys[i]  # [K, 8]
        sigma_x = w[:, 0].clamp(1e-4, 1.0)
        sigma_y = w[:, 1].clamp(1e-4, 1.0)
        rho = w[:, 2].clamp(-0.999, 0.999)
        colours = w[:, 3:6].clamp(0, 1)  # [K, 3] RGB
        coords = w[:, 6:8]               # [K, 2]
        img = generate_2D_gaussian_splatting(
            kernel_size=KERNEL_SIZE,
            sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
            coords=coords, colours=colours,
            image_size=IMAGE_SIZE, channels=3, device=device,
            soft_clamp=SOFT_CLAMP,
        )  # [H, W, 3]
        images.append(img.permute(2, 0, 1))  # [3, H, W]
    return torch.stack(images, dim=0)  # [N, 3, H, W]


def main():
    parser = argparse.ArgumentParser(description="Sample from CIFAR-10 GaussianDiffusion")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Sampling batch size (keep small for K=500)")
    parser.add_argument("--out_dir", default="samples/cifar10/")
    parser.add_argument("--real_data_h5", default=None,
                        help="HDF5 file for real-data FID comparison")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="Override CFG scale (default: use checkpoint value)")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt.get("args", {})
    K = train_args.get("num_gaussians", 500)
    T = train_args.get("timesteps", 200)
    hidden_dim = train_args.get("hidden_dim", 256)
    num_blocks = train_args.get("num_blocks", 6)
    num_heads = train_args.get("num_heads", 16)
    num_classes = train_args.get("num_classes", NUM_CLASSES_DEFAULT)
    cfg_dropout = train_args.get("cfg_dropout", 0.1)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else train_args.get("cfg_scale", 1.5)
    schedule_s = train_args.get("schedule_s", 0.008)

    ddpm = DDPM(n_T=T, schedule_type="cosine", s=schedule_s)
    model = GaussianTransformer(
        input_dim=K,
        time_emb_dim=hidden_dim,
        feature_dim=FEAT_DIM,
        num_timestamps=T,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
        num_classes=num_classes,
        class_dropout_prob=cfg_dropout,
    ).to(args.device)

    def _strip_prefix(sd):
        return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    if "ema_shadow" in ckpt:
        ema_data = ckpt["ema_shadow"]
        if isinstance(ema_data, dict) and "shadow" in ema_data:
            ema_data = ema_data["shadow"]
        model.load_state_dict(_strip_prefix(ema_data))
        print(f"Loaded EMA weights from epoch {ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(_strip_prefix(ckpt["model_state_dict"]))
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (no EMA)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {num_blocks}B/{num_heads}H/{hidden_dim}D → {n_params/1e6:.2f}M params")

    # ---- Sample ----
    if args.sampler == "ddim":
        print(f"Sampling {args.n_samples} with DDIM ({args.ddim_steps} steps, eta={args.ddim_eta}), CFG w={cfg_scale}")
        W_norm = sample_gaussians_ddim(
            model, ddpm, n=args.n_samples, K=K, device=args.device,
            batch_size=args.batch_size, num_classes=num_classes, cfg_scale=cfg_scale,
            ddim_steps=args.ddim_steps, eta=args.ddim_eta,
        )
    else:
        print(f"Sampling {args.n_samples} with DDPM, CFG w={cfg_scale}")
        W_norm = sample_gaussians(
            model, ddpm, n=args.n_samples, K=K, device=args.device,
            batch_size=args.batch_size, num_classes=num_classes, cfg_scale=cfg_scale,
        )

    # ---- Render ----
    print("Rendering generated images …")
    rendered = render_batch(W_norm, device=args.device)

    grid = make_grid(rendered[:64], nrow=8, normalize=False)
    grid_path = os.path.join(args.out_dir, "sample_grid.png")
    save_image(grid, grid_path)
    print(f"Saved sample grid → {grid_path}")

    # ---- Metrics ----
    rendered_u8 = (rendered * 255).clamp(0, 255).to(torch.uint8)
    metrics = {}

    print("Computing IS …")
    is_metric = InceptionScore(normalize=True).to(args.device)
    for i in tqdm(range(0, len(rendered_u8), args.batch_size)):
        batch = rendered_u8[i:i + args.batch_size].to(args.device)
        is_metric.update(batch)
    is_mean, is_std = is_metric.compute()
    metrics["IS_mean"] = is_mean.item()
    metrics["IS_std"] = is_std.item()

    if args.real_data_h5:
        from src.dataset_v2 import GaussianDatasetV2

        print("Loading real data for FID/KID …")
        real_ds = GaussianDatasetV2(args.real_data_h5)
        real_W = torch.stack([real_ds[i][0] for i in range(len(real_ds))], dim=0)
        real_W_norm = normalize_parameters(real_W, PARAM_RANGES)
        print("Rendering real images …")
        real_rendered = render_batch(real_W_norm, device=args.device)
        real_u8 = (real_rendered * 255).clamp(0, 255).to(torch.uint8)

        print("Computing FID …")
        fid_metric = FrechetInceptionDistance(normalize=True).to(args.device)
        for i in tqdm(range(0, len(real_u8), args.batch_size)):
            fid_metric.update(real_u8[i:i + args.batch_size].to(args.device), real=True)
        for i in tqdm(range(0, len(rendered_u8), args.batch_size)):
            fid_metric.update(rendered_u8[i:i + args.batch_size].to(args.device), real=False)
        metrics["FID"] = fid_metric.compute().item()

        print("Computing KID …")
        kid_metric = KernelInceptionDistance(
            subset_size=min(1000, len(real_u8)), normalize=True).to(args.device)
        for i in tqdm(range(0, len(real_u8), args.batch_size)):
            kid_metric.update(real_u8[i:i + args.batch_size].to(args.device), real=True)
        for i in tqdm(range(0, len(rendered_u8), args.batch_size)):
            kid_metric.update(rendered_u8[i:i + args.batch_size].to(args.device), real=False)
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
