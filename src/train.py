"""
Train GaussianTransformer diffusion model on MNIST Gaussian representations.

Usage:
    python -m src.train \\
        --data_dir data/mnist_gaussian_representations/ \\
        --epochs 100
"""

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import wandb

from src.dataset import GaussianDataset
from src.dataset_v2 import GaussianDatasetV2
from src.ddpm import DDPM
from src.models.transformer_model import GaussianTransformer
from src.utils.normalize import normalize_parameters
from src.utils.denormalize import denormalize_parameters
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

# ---------------------------------------------------------------------------
# Default parameter ranges for 6-dim Gaussians (alpha dropped)
# [sigma_x, sigma_y, rho, colour, x, y]
# Can be overridden via --config (see configs/)
# ---------------------------------------------------------------------------
DEFAULT_PARAM_RANGES = [
    (0.0, 1.0),    # sigma_x
    (0.0, 1.0),    # sigma_y
    (-1.0, 1.0),   # rho
    (0.0, 1.0),    # colour
    (-1.0, 1.0),   # x
    (-1.0, 1.0),   # y
]


# ---------------------------------------------------------------------------
# Reverse diffusion sampler
# ---------------------------------------------------------------------------
@torch.no_grad()
def ddpm_sample(model: nn.Module, ddpm: DDPM, n: int, K: int, feat_dim: int,
                device: str, labels: torch.Tensor = None,
                cfg_scale: float = 0.0) -> torch.Tensor:
    """Run reverse diffusion for n samples. Returns [n, K, feat_dim].

    Args:
        labels: (n,) class labels for CFG. If None, unconditional sampling.
        cfg_scale: guidance scale w. 0 = no guidance, >0 = CFG.
    """
    model.eval()
    x = torch.randn(n, K, feat_dim, device=device)
    use_cfg = cfg_scale > 0 and labels is not None
    for t in range(ddpm.n_T, 0, -1):
        t_tensor = torch.full((n,), t, dtype=torch.float32, device=device)
        if use_cfg:
            # Two forward passes: conditional + unconditional
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            # Null class = num_classes (the extra embedding index)
            null_labels = torch.full_like(labels, model.num_classes if hasattr(model, 'num_classes') else labels.max() + 1)
            y_double = torch.cat([labels, null_labels], dim=0)
            eps_double = model(x_double, t_double, y=y_double)
            eps_cond, eps_uncond = eps_double.chunk(2, dim=0)
            eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps_pred = model(x, t_tensor, y=labels)
        # DDPM reverse step
        oneover_sqrta = ddpm.oneover_sqrta[t].to(device)
        mab_over_sqrtmab = ddpm.mab_over_sqrtmab[t].to(device)
        sqrt_beta_t = ddpm.sqrt_beta_t[t].to(device)
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x = oneover_sqrta * (x - mab_over_sqrtmab * eps_pred) + sqrt_beta_t * z
    return x


# ---------------------------------------------------------------------------
# Render a grid of sample images
# ---------------------------------------------------------------------------
def render_sample_grid(
    W_norm: torch.Tensor,    # [n, K, 7] normalised
    param_ranges: list,
    kernel_size: int,
    image_size: tuple,
    device: str,
    n_show: int = 16,
) -> torch.Tensor:
    """Denormalise, render, return [1, 3, H, W*n_show] grid tensor."""
    W_norm = W_norm[:n_show].to(device)
    W_phys = denormalize_parameters(W_norm, param_ranges)

    frames = []
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
        frames.append(img.permute(2, 0, 1))  # [1, H, W]

    grid = make_grid(torch.stack(frames), nrow=8, normalize=False)
    return grid


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
class EMA:
    """Maintains exponential moving average of model parameters with power ramp."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        # Power ramp: start with low decay (copy model), ramp up to target
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(d).add_(v, alpha=1 - d)

    def apply(self, model: nn.Module):
        """Copy EMA weights into model (for sampling/saving)."""
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return {"shadow": self.shadow, "step": self.step}

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and "shadow" in state_dict:
            self.shadow = {k: v.clone().detach() for k, v in state_dict["shadow"].items()}
            self.step = state_dict.get("step", 0)
        else:
            # Backwards compat: old checkpoints saved shadow directly
            self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = args.device

    # Load param_ranges from config or use defaults
    if args.config:
        from configs import get_config
        cfg = get_config(args.config)
        param_ranges = cfg["param_ranges"]
    else:
        param_ranges = DEFAULT_PARAM_RANGES

    # TF32 for A100+ GPUs (no-op on older GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # W&B setup
    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        wandb.login(key=api_key)
        wandb.init(project="GaussianDiffusion", config=vars(args))
        use_wandb = True
    else:
        print("WANDB_API_KEY not set — skipping W&B logging.")
        use_wandb = False

    # Data
    print("Loading dataset …")
    if args.data_h5:
        dataset = GaussianDatasetV2(args.data_h5, min_psnr=args.min_psnr)
        use_h5 = True
        print(f"  HDF5 dataset: {dataset}")
    else:
        dataset = GaussianDataset(args.data_dir, num_gaussians=args.num_gaussians)
        use_h5 = False

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

    sample_item = dataset[0]
    feat_dim = sample_item[0].shape[1] if use_h5 else sample_item.shape[1]

    # Model
    num_classes = args.num_classes if args.num_classes > 0 else 0
    model = GaussianTransformer(
        input_dim=args.num_gaussians,
        time_emb_dim=args.hidden_dim,
        feature_dim=feat_dim,
        num_timestamps=args.timesteps,
        num_transformer_blocks=args.num_blocks,
        num_heads=args.num_heads,
        num_classes=num_classes,
        class_dropout_prob=args.cfg_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg_str = f", CFG classes={num_classes}, dropout={args.cfg_dropout}" if num_classes > 0 else ""
    print(f"  Model: {args.num_blocks} blocks, {args.num_heads} heads, hidden={args.hidden_dim} → {n_params/1e6:.1f}M params{cfg_str}")

    if args.compile:
        print("  Compiling model with torch.compile …")
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    ddpm = DDPM(n_T=args.timesteps, schedule_type="cosine", s=args.schedule_s)

    # Per-step LR scheduling: warmup → cosine decay
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
    print(f"  LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema:
        print(f"  EMA enabled (decay={args.ema_decay})")

    # AMP setup
    use_amp = (device == "cuda" or device.startswith("cuda")) and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("  AMP (fp16) enabled")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 0

    # Resume from checkpoint
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Strip torch.compile prefix if needed
        state = ckpt["model_state_dict"]
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_to_load.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # Fast-forward scheduler to match resumed epoch
            n_steps = ckpt["epoch"] * len(train_loader)
            for _ in range(n_steps):
                scheduler.step()
        if use_amp and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ema is not None and "ema_shadow" in ckpt:
            ema.load_state_dict(ckpt["ema_shadow"])
            # If model is compiled, EMA shadow keys need _orig_mod. prefix
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
            if use_h5:
                batch = batch_data[0].to(device, non_blocking=True)
                labels = batch_data[1].to(device, non_blocking=True) if num_classes > 0 else None
            else:
                batch = batch_data.to(device, non_blocking=True)
                labels = None
            batch_norm = normalize_parameters(batch, param_ranges)
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
                if use_h5:
                    batch = batch_data[0].to(device, non_blocking=True)
                    labels = batch_data[1].to(device, non_blocking=True) if num_classes > 0 else None
                else:
                    batch = batch_data.to(device, non_blocking=True)
                    labels = None
                batch_norm = normalize_parameters(batch, param_ranges)
                t = torch.randint(1, args.timesteps + 1, (batch.shape[0],),
                                  dtype=torch.long, device=device)
                x_t, _, noise = ddpm.get_noisy_images_and_noise(batch_norm, t)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x_t, t.float(), y=labels)
                    val_loss += criterion(pred, noise).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # ---- Logging ----
        log_dict = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}

        # ---- Sample grid ----
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            # Use EMA weights for sampling if available
            if ema is not None:
                orig_state = copy.deepcopy(model.state_dict())
                ema.apply(model)
            # For CFG models, sample 2 per class (0-9) = 20 samples with guidance
            if num_classes > 0:
                sample_labels = torch.arange(num_classes, device=device).repeat_interleave(2)[:16]
                sample_cfg = args.cfg_scale
            else:
                sample_labels = None
                sample_cfg = 0.0
            samples_norm = ddpm_sample(model, ddpm, n=min(16, len(sample_labels) if sample_labels is not None else 16),
                                       K=args.num_gaussians, feat_dim=feat_dim, device=device,
                                       labels=sample_labels, cfg_scale=sample_cfg)
            if ema is not None:
                model.load_state_dict(orig_state)
            grid = render_sample_grid(samples_norm, param_ranges,
                                      kernel_size=11, image_size=(28, 28), device=device)
            grid_path = os.path.join(args.sample_dir, f"samples_epoch{epoch:04d}.png")
            save_image(grid, grid_path)
            if use_wandb:
                log_dict["samples"] = wandb.Image(grid_path)

        if use_wandb:
            wandb.log(log_dict)

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

    if use_wandb:
        wandb.finish()
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train GaussianDiffusion on MNIST")
    parser.add_argument("--config", default=None,
                        help="Dataset config name (e.g. 'mnist'). See configs/.")
    parser.add_argument("--data_dir", default="data/mnist_gaussian_representations/")
    parser.add_argument("--data_h5", default=None,
                        help="Path to HDF5 file (overrides --data_dir)")
    parser.add_argument("--min_psnr", type=float, default=None,
                        help="Min PSNR filter for HDF5 dataset")
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    parser.add_argument("--sample_dir", default="samples/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--num_gaussians", type=int, default=70)
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Transformer hidden dimension")
    parser.add_argument("--num_blocks", type=int, default=32,
                        help="Number of DiT blocks")
    parser.add_argument("--num_heads", type=int, default=64,
                        help="Number of attention heads")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (fp16)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for speedup")
    parser.add_argument("--use_ema", action="store_true",
                        help="Enable EMA of model weights")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="LR warmup steps (0 to disable)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Save sample grid every N epochs")
    parser.add_argument("--num_classes", type=int, default=0,
                        help="Number of classes for CFG (0 = unconditional)")
    parser.add_argument("--cfg_dropout", type=float, default=0.1,
                        help="Label dropout probability for CFG training")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="CFG guidance scale for sampling during training")
    parser.add_argument("--schedule_s", type=float, default=0.008,
                        help="Cosine noise schedule offset (default 0.008)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
