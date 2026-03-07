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

# Default parameter ranges — can be overridden via --config (see configs/)
# [sigma_x, sigma_y, rho, colour, x, y] — alpha dropped
DEFAULT_PARAM_RANGES = [
    (0.0, 1.0),    # sigma_x
    (0.0, 1.0),    # sigma_y
    (-1.0, 1.0),   # rho
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
    num_classes: int = 0,
    cfg_scale: float = 0.0,
    clip_x0: bool = False,
    clip_range: float = 5.0,
) -> torch.Tensor:
    """Generate n samples in normalised Gaussian space. Returns [n, K, feat_dim].

    For class-conditional models, generates n/num_classes samples per class.
    When clip_x0=True, clips the predicted x0 at each step (like DDIM's x0_pred clipping).
    """
    model.eval()
    use_cfg = num_classes > 0 and cfg_scale > 0

    all_samples = []
    remaining = n
    sample_idx = 0
    while remaining > 0:
        bs = min(batch_size, remaining)
        x = torch.randn(bs, K, feat_dim, device=device)

        # Generate class labels: cycle through classes
        if use_cfg:
            labels = torch.arange(num_classes, device=device).repeat(
                (bs + num_classes - 1) // num_classes
            )[:bs]
            null_labels = torch.full((bs,), num_classes, dtype=torch.long, device=device)

        for t in range(ddpm.n_T, 0, -1):
            t_tensor = torch.full((bs,), t, dtype=torch.float32, device=device)
            if use_cfg:
                # Two forward passes for CFG
                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t_tensor, t_tensor], dim=0)
                y_double = torch.cat([labels, null_labels], dim=0)
                eps_double = model(x_double, t_double, y=y_double)
                eps_cond, eps_uncond = eps_double.chunk(2, dim=0)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = model(x, t_tensor)

            # Optional x0_pred clipping: predict x0, clip, re-derive eps
            if clip_x0:
                ab_t = ddpm.alphabar_t[t].to(device)
                x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-4)
                x0_pred = x0_pred.clamp(-clip_range, clip_range)
                eps_pred = (x - ab_t.sqrt() * x0_pred) / (1 - ab_t).sqrt().clamp(min=1e-4)

            oneover_sqrta = ddpm.oneover_sqrta[t].to(device)
            mab_over_sqrtmab = ddpm.mab_over_sqrtmab[t].to(device)
            sqrt_beta_t = ddpm.sqrt_beta_t[t].to(device)
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            x = oneover_sqrta * (x - mab_over_sqrtmab * eps_pred) + sqrt_beta_t * z
        all_samples.append(x.cpu())
        remaining -= bs
        sample_idx += bs
    return torch.cat(all_samples, dim=0)


# ---------------------------------------------------------------------------
# DDIM reverse diffusion
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_gaussians_ddim(
    model: torch.nn.Module,
    ddpm: DDPM,
    n: int,
    K: int,
    feat_dim: int,
    device: str,
    batch_size: int = 64,
    num_classes: int = 0,
    cfg_scale: float = 0.0,
    ddim_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """Generate n samples using DDIM sampling. Returns [n, K, feat_dim].

    Args:
        ddim_steps: Number of denoising steps (can be < ddpm.n_T).
        eta: Stochasticity parameter. 0 = deterministic DDIM, 1 = DDPM-like.
    """
    model.eval()
    use_cfg = num_classes > 0 and cfg_scale > 0

    # Build decreasing timestep subsequence: [T-1, ..., 0]
    # NOTE: must start from T-1, not T. At t=T, alphabar≈0 which causes
    # x0_pred = (x - sqrt(1-ab)*eps) / sqrt(ab) to explode (division by ~0).
    # Starting from T-1 is standard in DDIM implementations (e.g., diffusers).
    timesteps = torch.linspace(ddpm.n_T - 1, 0, ddim_steps + 1).round().long()
    alphabar = ddpm.alphabar_t.to(device)

    all_samples = []
    remaining = n
    while remaining > 0:
        bs = min(batch_size, remaining)
        x = torch.randn(bs, K, feat_dim, device=device)

        if use_cfg:
            labels = torch.arange(num_classes, device=device).repeat(
                (bs + num_classes - 1) // num_classes
            )[:bs]
            null_labels = torch.full((bs,), num_classes, dtype=torch.long, device=device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]          # current timestep
            t_prev = timesteps[i + 1]  # next timestep (closer to 0)

            t_tensor = torch.full((bs,), t, dtype=torch.float32, device=device)

            # Predict noise (with optional CFG)
            if use_cfg:
                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t_tensor, t_tensor], dim=0)
                y_double = torch.cat([labels, null_labels], dim=0)
                eps_double = model(x_double, t_double, y=y_double)
                eps_cond, eps_uncond = eps_double.chunk(2, dim=0)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = model(x, t_tensor)

            # DDIM reverse step
            ab_t = alphabar[t]
            ab_prev = alphabar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)

            # Predict x_0 (clip to prevent extreme values at high noise levels)
            x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-4)
            x0_pred = x0_pred.clamp(-5, 5)

            # Compute sigma (stochasticity)
            if t_prev > 0 and eta > 0:
                sigma_t = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
            else:
                sigma_t = 0.0

            # Direction pointing to x_t
            dir_xt = (1 - ab_prev - sigma_t ** 2).clamp(min=0).sqrt() * eps_pred

            # Noise
            if t_prev > 0 and eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = ab_prev.sqrt() * x0_pred + dir_xt + sigma_t * noise

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
    """Render normalised Gaussians to [N, 3, H, W] float for metric computation."""
    W_phys = denormalize_parameters(W_norm.to(device), param_ranges)
    images = []
    for i in range(W_phys.shape[0]):
        w = W_phys[i]  # [K, 6]
        sigma_x = w[:, 0].clamp(1e-4, 1.0)
        sigma_y = w[:, 1].clamp(1e-4, 1.0)
        rho = w[:, 2].clamp(-0.999, 0.999)
        colour = w[:, 3].clamp(0, 1).unsqueeze(1)
        coords = torch.stack([w[:, 4], w[:, 5]], dim=1)
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
    parser.add_argument("--config", default=None,
                        help="Dataset config name (e.g. 'mnist'). See configs/.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", default="samples/")
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--real_data_dir", default=None,
                        help="Directory of .pt files for real-data FID comparison (optional)")
    parser.add_argument("--real_data_h5", default=None,
                        help="HDF5 file for real-data FID comparison (optional, overrides --real_data_dir)")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="CFG guidance scale (overrides training default)")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm",
                        help="Sampling method (default: ddpm)")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM denoising steps (default: 50)")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM stochasticity: 0=deterministic, 1=DDPM-like (default: 0)")
    parser.add_argument("--schedule_s", type=float, default=None,
                        help="Cosine schedule offset (default: use checkpoint value or 0.008)")
    parser.add_argument("--clip_x0", action="store_true",
                        help="Clip predicted x0 at each DDPM step (like DDIM's x0_pred clipping)")
    parser.add_argument("--clip_range", type=float, default=5.0,
                        help="Clipping range for x0_pred when --clip_x0 is used (default: 5.0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load param_ranges from config or use defaults
    if args.config:
        from configs import get_config
        cfg = get_config(args.config)
        param_ranges = cfg["param_ranges"]
    else:
        param_ranges = DEFAULT_PARAM_RANGES

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt.get("args", {})
    K = train_args.get("num_gaussians", 70)
    T = train_args.get("timesteps", 200)
    feat_dim = 6

    hidden_dim = train_args.get("hidden_dim", 512)
    num_blocks = train_args.get("num_blocks", 32)
    num_heads = train_args.get("num_heads", 64)
    num_classes = train_args.get("num_classes", 0)
    cfg_dropout = train_args.get("cfg_dropout", 0.0)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else train_args.get("cfg_scale", 3.0)
    schedule_s = args.schedule_s if args.schedule_s is not None else train_args.get("schedule_s", 0.008)

    ddpm = DDPM(n_T=T, schedule_type="cosine", s=schedule_s)
    model = GaussianTransformer(
        input_dim=K,
        time_emb_dim=hidden_dim,
        feature_dim=feat_dim,
        num_timestamps=T,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
        num_classes=num_classes,
        class_dropout_prob=cfg_dropout,
    ).to(args.device)
    # Prefer EMA weights for inference if available
    def _strip_compile_prefix(sd):
        """Strip '_orig_mod.' prefix added by torch.compile."""
        return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    if "ema_shadow" in ckpt:
        ema_data = ckpt["ema_shadow"]
        # New format: {"shadow": {...}, "step": N}; old format: state_dict directly
        if isinstance(ema_data, dict) and "shadow" in ema_data:
            ema_data = ema_data["shadow"]
        model.load_state_dict(_strip_compile_prefix(ema_data))
        print(f"Loaded EMA weights from epoch {ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(_strip_compile_prefix(ckpt["model_state_dict"]))
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (no EMA)")

    # ---- Sample ----
    sampler = args.sampler
    if sampler == "ddim":
        print(f"Sampling {args.n_samples} images with DDIM ({args.ddim_steps} steps, eta={args.ddim_eta})"
              f"{f', CFG scale={cfg_scale}' if num_classes > 0 else ''} …")
        W_norm = sample_gaussians_ddim(
            model, ddpm, n=args.n_samples, K=K, feat_dim=feat_dim,
            device=args.device, batch_size=args.batch_size,
            num_classes=num_classes, cfg_scale=cfg_scale,
            ddim_steps=args.ddim_steps, eta=args.ddim_eta,
        )
    else:
        clip_str = f", clip_x0=[-{args.clip_range},{args.clip_range}]" if args.clip_x0 else ""
        if num_classes > 0:
            print(f"Sampling {args.n_samples} images with CFG (scale={cfg_scale}, classes={num_classes}{clip_str}) …")
        else:
            print(f"Sampling {args.n_samples} images (unconditional{clip_str}) …")
        W_norm = sample_gaussians(model, ddpm, n=args.n_samples, K=K, feat_dim=feat_dim,
                                   device=args.device, batch_size=args.batch_size,
                                   num_classes=num_classes, cfg_scale=cfg_scale,
                                   clip_x0=args.clip_x0, clip_range=args.clip_range)

    # ---- Render all ----
    image_size = (args.image_size, args.image_size)
    print("Rendering …")
    rendered = render_batch(W_norm, param_ranges,
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
    has_real = args.real_data_h5 is not None or args.real_data_dir is not None
    if has_real:
        from src.utils.normalize import normalize_parameters

        print("Loading real data for FID/KID …")
        if args.real_data_h5 is not None:
            from src.dataset_v2 import GaussianDatasetV2
            real_ds = GaussianDatasetV2(args.real_data_h5)
            # GaussianDatasetV2 returns (W_tensor [K,6], label) — already physical space, alpha dropped
            real_W = torch.stack([real_ds[i][0] for i in range(len(real_ds))], dim=0)
        else:
            from src.dataset import GaussianDataset
            real_ds = GaussianDataset(args.real_data_dir, num_gaussians=K)
            real_W = real_ds.data  # [N, K, 6] — physical space
        # Normalize first so render_batch's denormalize recovers physical values
        real_W_norm = normalize_parameters(real_W, param_ranges)
        real_rendered = render_batch(real_W_norm, param_ranges,
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
