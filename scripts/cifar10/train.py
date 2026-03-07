"""
Train GaussianTransformer diffusion model on CIFAR-10 Gaussian representations.

CIFAR-10 specifics:
  - K=500 Gaussians, 8-dim features (alpha dropped): sigma_x, sigma_y, rho, r, g, b, x, y
  - 32x32 RGB images, kernel_size=32, soft_clamp=True
  - 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

Usage:
    python scripts/cifar10/train.py --data_h5 data/cifar10/cifar10_gaussians_K500.h5 --epochs 500
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.cifar10 import CIFAR10_CONFIG  # noqa: E402
from src.dataset_v2 import GaussianDatasetV2  # noqa: E402
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
NUM_CLASSES = CFG["num_classes"]  # 10
K_DEFAULT = CFG["num_gaussians"]  # 500


# ---------------------------------------------------------------------------
# Reverse diffusion sampler
# ---------------------------------------------------------------------------
@torch.no_grad()
def ddpm_sample(model, ddpm, n, K, device, labels=None, cfg_scale=0.0):
    """DDPM reverse diffusion. Returns [n, K, FEAT_DIM]."""
    model.eval()
    x = torch.randn(n, K, FEAT_DIM, device=device)
    use_cfg = cfg_scale > 0 and labels is not None
    for t in range(ddpm.n_T, 0, -1):
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
        oneover_sqrta = ddpm.oneover_sqrta[t].to(device)
        mab_over_sqrtmab = ddpm.mab_over_sqrtmab[t].to(device)
        sqrt_beta_t = ddpm.sqrt_beta_t[t].to(device)
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x = oneover_sqrta * (x - mab_over_sqrtmab * eps_pred) + sqrt_beta_t * z
    return x


# ---------------------------------------------------------------------------
# Render sample grid (RGB, soft clamp)
# ---------------------------------------------------------------------------
def render_sample_grid(W_norm, n_show=16, device="cuda"):
    """Render normalised Gaussians to a grid image for logging."""
    W_norm = W_norm[:n_show].to(device)
    W_phys = denormalize_parameters(W_norm, PARAM_RANGES)
    frames = []
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
        frames.append(img.permute(2, 0, 1))  # [3, H, W]
    grid = make_grid(torch.stack(frames), nrow=8, normalize=False)
    return grid


# ---------------------------------------------------------------------------
# EMA (same as src/train.py)
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(d).add_(v, alpha=1 - d)

    def apply(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return {"shadow": self.shadow, "step": self.step}

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "shadow" in state_dict:
            self.shadow = {k: v.clone().detach() for k, v in state_dict["shadow"].items()}
            self.step = state_dict.get("step", 0)
        else:
            self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = args.device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Data
    print("Loading dataset …")
    dataset = GaussianDatasetV2(args.data_h5, min_psnr=args.min_psnr)
    print(f"  {dataset}")

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(1, nw // 2), pin_memory=True,
                            persistent_workers=nw > 0)

    # Model
    model = GaussianTransformer(
        input_dim=args.num_gaussians,
        time_emb_dim=args.hidden_dim,
        feature_dim=FEAT_DIM,
        num_timestamps=args.timesteps,
        num_transformer_blocks=args.num_blocks,
        num_heads=args.num_heads,
        num_classes=NUM_CLASSES,
        class_dropout_prob=args.cfg_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {args.num_blocks}B/{args.num_heads}H/{args.hidden_dim}D "
          f"→ {n_params/1e6:.2f}M params, K={args.num_gaussians}, feat={FEAT_DIM}")

    if args.compile:
        print("  Compiling model with torch.compile …")
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    ddpm = DDPM(n_T=args.timesteps, schedule_type="cosine", s=args.schedule_s)

    # LR schedule: warmup → cosine decay
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 2)
    if warmup_steps > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6)
    print(f"  LR schedule: {warmup_steps} warmup, {total_steps} total steps")

    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema:
        print(f"  EMA enabled (decay={args.ema_decay})")

    use_amp = device.startswith("cuda") and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("  AMP (fp16) enabled")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 0

    # Resume
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_to_load.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            n_steps = ckpt["epoch"] * len(train_loader)
            for _ in range(n_steps):
                scheduler.step()
        if use_amp and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ema is not None and "ema_shadow" in ckpt:
            ema.load_state_dict(ckpt["ema_shadow"])
            if hasattr(model, "_orig_mod"):
                model_keys = set(model.state_dict().keys())
                shadow_keys = set(ema.shadow.keys())
                if shadow_keys != model_keys and not any(k.startswith("_orig_mod.") for k in shadow_keys):
                    ema.shadow = {"_orig_mod." + k: v for k, v in ema.shadow.items()}
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        start_epoch = ckpt["epoch"]
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            batch = batch_data[0].to(device, non_blocking=True)
            labels = batch_data[1].to(device, non_blocking=True)
            batch_norm = normalize_parameters(batch, PARAM_RANGES)
            t = torch.randint(1, args.timesteps + 1, (batch.shape[0],),
                              dtype=torch.long, device=device)
            x_t, _, noise = ddpm.get_noisy_images_and_noise(batch_norm, t)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x_t, t.float(), y=labels)
                loss = criterion(pred, noise)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                batch = batch_data[0].to(device, non_blocking=True)
                labels = batch_data[1].to(device, non_blocking=True)
                batch_norm = normalize_parameters(batch, PARAM_RANGES)
                t = torch.randint(1, args.timesteps + 1, (batch.shape[0],),
                                  dtype=torch.long, device=device)
                x_t, _, noise = ddpm.get_noisy_images_and_noise(batch_norm, t)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x_t, t.float(), y=labels)
                    val_loss += criterion(pred, noise).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # ---- Sample grid ----
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            if ema is not None:
                orig_state = copy.deepcopy(model.state_dict())
                ema.apply(model)
            sample_labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(2)[:16]
            samples_norm = ddpm_sample(model, ddpm, n=16, K=args.num_gaussians,
                                       device=device, labels=sample_labels,
                                       cfg_scale=args.cfg_scale)
            if ema is not None:
                model.load_state_dict(orig_state)
            grid = render_sample_grid(samples_norm, n_show=16, device=device)
            grid_path = os.path.join(args.sample_dir, f"samples_epoch{epoch:04d}.png")
            save_image(grid, grid_path)

        # ---- Checkpoints ----
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        if ema is not None:
            ckpt["ema_shadow"] = ema.state_dict()
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train GaussianDiffusion on CIFAR-10")
    parser.add_argument("--data_h5", required=True,
                        help="Path to CIFAR-10 HDF5 file")
    parser.add_argument("--min_psnr", type=float, default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints/cifar10/")
    parser.add_argument("--sample_dir", default="samples/cifar10/")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--num_gaussians", type=int, default=K_DEFAULT)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--schedule_s", type=float, default=0.008)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Store feature_dim in args for checkpoint metadata
    args.feature_dim = FEAT_DIM

    train(args)


if __name__ == "__main__":
    main()
