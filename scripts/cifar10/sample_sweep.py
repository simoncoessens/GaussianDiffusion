"""
Sweep sampling hyperparameters on a trained CIFAR-10 checkpoint.

Sweeps: DDIM eta, CFG scale, DDIM steps, x0 clip range.
Reports FID for each configuration. No retraining needed.

Usage:
    python scripts/cifar10/sample_sweep.py --checkpoint checkpoints/cifar10/std3_8h384d_lr1e3/best.pt
    python scripts/cifar10/sample_sweep.py --checkpoint checkpoints/cifar10/std3_lr5e4_bs128_500ep/best.pt --n_samples 2000
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.cifar10 import CIFAR10_CONFIG
from src.ddpm import DDPM
from src.models.transformer_model import GaussianTransformer
from src.utils.denormalize import denormalize_parameters
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting_batch
from src.utils.normalize import normalize_parameters

CFG = CIFAR10_CONFIG
PARAM_RANGES = CFG["param_ranges"]
FEAT_DIM = CFG["feature_dim"]
IMAGE_SIZE = (CFG["image_size"], CFG["image_size"])
KERNEL_SIZE = CFG["kernel_size"]
SOFT_CLAMP = CFG["soft_clamp"]
NUM_CLASSES = CFG["num_classes"]
K_DEFAULT = CFG["num_gaussians"]
DATA_MEAN = torch.tensor(CFG["data_mean"])
DATA_STD = torch.tensor(CFG["data_std"])


@torch.no_grad()
def ddim_sample(model, ddpm, n, K, device, labels=None, cfg_scale=0.0,
                steps=50, eta=0.0, clip_range=5.0):
    """DDIM reverse diffusion with configurable clip range."""
    model.eval()
    use_cfg = cfg_scale > 0 and labels is not None
    timesteps = torch.linspace(ddpm.n_T - 1, 0, steps + 1).round().long()
    alphabar = ddpm.alphabar_t.to(device)

    x = torch.randn(n, K, FEAT_DIM, device=device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        t_tensor = torch.full((n,), t, dtype=torch.float32, device=device)

        if use_cfg:
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            null_labels = torch.full_like(labels, model.num_classes)
            y_double = torch.cat([labels, null_labels], dim=0)
            eps_double = model(x_double, t_double, y=y_double)
            eps_cond, eps_uncond = eps_double.chunk(2, dim=0)
            eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps_pred = model(x, t_tensor, y=labels)

        ab_t = alphabar[t]
        ab_prev = alphabar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-4)
        x0_pred = x0_pred.clamp(-clip_range, clip_range)

        if t_prev > 0 and eta > 0:
            sigma_t = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
        else:
            sigma_t = 0.0

        dir_xt = (1 - ab_prev - sigma_t ** 2).clamp(min=0).sqrt() * eps_pred
        noise = torch.randn_like(x) if (t_prev > 0 and eta > 0) else torch.zeros_like(x)
        x = ab_prev.sqrt() * x0_pred + dir_xt + sigma_t * noise

    return x


def render_gaussians(W_std, device="cuda", chunk_size=32):
    """Render standardized Gaussians to [N, 3, H, W] float images."""
    W_std = W_std.to(device)
    W_norm = W_std * DATA_STD.to(device).view(1, 1, -1) + DATA_MEAN.to(device).view(1, 1, -1)
    W_phys = denormalize_parameters(W_norm, PARAM_RANGES)

    N = W_phys.shape[0]
    all_frames = []
    for start in range(0, N, chunk_size):
        batch = W_phys[start:start + chunk_size]
        sigma_x = batch[:, :, 0].clamp(1e-4, 1.0)
        sigma_y = batch[:, :, 1].clamp(1e-4, 1.0)
        rho = batch[:, :, 2].clamp(-0.999, 0.999)
        colours = batch[:, :, 3:6].clamp(0, 1)
        coords = batch[:, :, 6:8]
        imgs = generate_2D_gaussian_splatting_batch(
            kernel_size=KERNEL_SIZE, sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
            coords=coords, colours=colours, image_size=IMAGE_SIZE, channels=3,
            device=device, soft_clamp=SOFT_CLAMP,
        )
        all_frames.append(imgs.permute(0, 3, 1, 2))
    return torch.cat(all_frames, dim=0)


def load_real_images(n_real=10000):
    """Load real CIFAR-10 images for FID reference."""
    import torchvision.transforms as T
    import torchvision.datasets as datasets
    transform = T.Compose([T.ToTensor()])
    ds = datasets.CIFAR10(root="data/cifar10/raw", train=True, download=False,
                          transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)
    imgs = []
    for batch, _ in loader:
        imgs.append((batch * 255).to(torch.uint8))
        if sum(b.shape[0] for b in imgs) >= n_real:
            break
    return torch.cat(imgs, dim=0)[:n_real]


def compute_fid(model, ddpm, K, device, real_u8, n_samples=500,
                cfg_scale=1.5, ddim_steps=50, eta=0.0, clip_range=5.0,
                batch_size=64):
    """Sample, render, compute FID."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    labels = torch.arange(NUM_CLASSES, device=device).repeat(
        (n_samples + NUM_CLASSES - 1) // NUM_CLASSES)[:n_samples]

    all_samples = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        samples = ddim_sample(model, ddpm, end - start, K, device,
                              labels=labels[start:end], cfg_scale=cfg_scale,
                              steps=ddim_steps, eta=eta, clip_range=clip_range)
        all_samples.append(samples)
    W_std = torch.cat(all_samples, dim=0)

    rendered = render_gaussians(W_std, device=device)
    rendered_u8 = (rendered * 255).clamp(0, 255).to(torch.uint8)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    for i in range(0, len(real_u8), 256):
        fid.update(real_u8[i:i+256].to(device), real=True)
    for i in range(0, len(rendered_u8), 256):
        fid.update(rendered_u8[i:i+256].to(device), real=False)
    return fid.compute().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--n_samples", type=int, default=1000, help="Samples for FID")
    parser.add_argument("--n_real", type=int, default=10000, help="Real images for FID")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sweep_mode", default="full",
                        choices=["eta", "cfg", "steps", "clip", "full"],
                        help="Which parameter to sweep (or full grid)")
    args = parser.parse_args()

    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    hidden_dim = ckpt_args.get("hidden_dim", 256)
    num_blocks = ckpt_args.get("num_blocks", 6)
    num_heads = ckpt_args.get("num_heads", 16)
    timesteps = ckpt_args.get("timesteps", 200)
    K = ckpt_args.get("num_gaussians", K_DEFAULT)

    model = GaussianTransformer(
        input_dim=K,
        time_emb_dim=hidden_dim,
        feature_dim=FEAT_DIM,
        num_timestamps=timesteps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
        num_classes=NUM_CLASSES,
        class_dropout_prob=0.0,  # No dropout during sampling
    ).to(device)

    # Load weights (handle torch.compile prefix)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)

    # Load EMA weights if available
    if "ema_shadow" in ckpt:
        print("  Loading EMA weights")
        ema_data = ckpt["ema_shadow"]
        if isinstance(ema_data, dict) and "shadow" in ema_data:
            ema_data = ema_data["shadow"]
        ema_state = {k.replace("_orig_mod.", ""): v for k, v in ema_data.items()}
        model.load_state_dict(ema_state, strict=False)

    model.eval()
    epoch = ckpt.get("epoch", "?")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {num_blocks}B/{num_heads}H/{hidden_dim}D ({n_params:.1f}M), epoch={epoch}")

    ddpm = DDPM(n_T=timesteps, schedule_type="cosine")

    # Load real images
    print(f"Loading {args.n_real} real images for FID...")
    real_u8 = load_real_images(n_real=args.n_real)
    print(f"  Loaded {real_u8.shape[0]} real images")

    # Define sweeps
    if args.sweep_mode == "eta":
        configs = [{"eta": e, "cfg_scale": 2.0, "ddim_steps": 100, "clip_range": 5.0}
                   for e in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]]
    elif args.sweep_mode == "cfg":
        configs = [{"eta": 0.0, "cfg_scale": c, "ddim_steps": 100, "clip_range": 5.0}
                   for c in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]]
    elif args.sweep_mode == "steps":
        configs = [{"eta": 0.0, "cfg_scale": 2.0, "ddim_steps": s, "clip_range": 5.0}
                   for s in [25, 50, 100, 200]]
    elif args.sweep_mode == "clip":
        configs = [{"eta": 0.0, "cfg_scale": 2.0, "ddim_steps": 100, "clip_range": c}
                   for c in [2.0, 3.0, 5.0, 7.0, 10.0, 20.0]]
    else:  # full
        # Focused grid: eta × cfg × steps (skip 200 for speed)
        etas = [0.0, 0.3, 0.5, 0.8]
        cfgs = [1.5, 2.0, 2.5, 3.0]
        steps_list = [50, 100]
        configs = [{"eta": e, "cfg_scale": c, "ddim_steps": s, "clip_range": 5.0}
                   for e, c, s in itertools.product(etas, cfgs, steps_list)]

    print(f"\nSweeping {len(configs)} configurations with {args.n_samples} samples each:\n")
    print(f"{'eta':>5} {'cfg':>5} {'steps':>5} {'clip':>5} | {'FID':>8} | {'time':>6}")
    print("-" * 50)

    results = []
    for cfg in configs:
        t0 = time.time()
        fid = compute_fid(
            model, ddpm, K, device, real_u8,
            n_samples=args.n_samples,
            cfg_scale=cfg["cfg_scale"],
            ddim_steps=cfg["ddim_steps"],
            eta=cfg["eta"],
            clip_range=cfg["clip_range"],
        )
        dt = time.time() - t0
        results.append({**cfg, "fid": fid, "time": dt})
        print(f"{cfg['eta']:5.1f} {cfg['cfg_scale']:5.1f} {cfg['ddim_steps']:5d} "
              f"{cfg['clip_range']:5.1f} | {fid:8.1f} | {dt:5.0f}s")

    # Summary
    best = min(results, key=lambda x: x["fid"])
    print(f"\n{'='*50}")
    print(f"Best FID: {best['fid']:.1f}")
    print(f"  eta={best['eta']}, cfg={best['cfg_scale']}, "
          f"steps={best['ddim_steps']}, clip={best['clip_range']}")


if __name__ == "__main__":
    main()
