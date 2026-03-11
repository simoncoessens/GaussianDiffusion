"""
Train GaussianTransformer diffusion model on CIFAR-10 Gaussian representations.

CIFAR-10 specifics:
  - K=500 Gaussians, 8-dim features (alpha dropped): sigma_x, sigma_y, rho, r, g, b, x, y
  - 32x32 RGB images, kernel_size=32, soft_clamp=True
  - 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
  - Data standardized to ~N(0,1) per feature before diffusion

Usage:
    python scripts/cifar10/train.py --data_h5 data/cifar10/cifar10_gaussians_K500.h5 --epochs 500
"""

import argparse
import copy
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
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
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting_batch  # noqa: E402
from src.utils.normalize import normalize_parameters  # noqa: E402

CFG = CIFAR10_CONFIG
PARAM_RANGES = CFG["param_ranges"]
FEAT_DIM = CFG["feature_dim"]  # 8
IMAGE_SIZE = (CFG["image_size"], CFG["image_size"])  # (32, 32)
KERNEL_SIZE = CFG["kernel_size"]  # 32
SOFT_CLAMP = CFG["soft_clamp"]  # True
NUM_CLASSES = CFG["num_classes"]  # 10
K_DEFAULT = CFG["num_gaussians"]  # 500
DATA_MEAN = torch.tensor(CFG["data_mean"])  # per-feature mean (in [-1,1] space)
DATA_STD = torch.tensor(CFG["data_std"])    # per-feature std  (in [-1,1] space)


# ---------------------------------------------------------------------------
# Reverse diffusion sampler (DDIM for speed during eval)
# ---------------------------------------------------------------------------
def _model_predict_cfg(model, x, t_tensor, labels, cfg_scale, prediction_type="epsilon"):
    """Run model with optional CFG. Returns raw model output."""
    use_cfg = cfg_scale > 0 and labels is not None
    if use_cfg:
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t_tensor, t_tensor], dim=0)
        null_labels = torch.full_like(labels, model.num_classes)
        y_double = torch.cat([labels, null_labels], dim=0)
        out_double = model(x_double, t_double, y=y_double)
        out_cond, out_uncond = out_double.chunk(2, dim=0)
        return out_uncond + cfg_scale * (out_cond - out_uncond)
    else:
        return model(x, t_tensor, y=labels)


def _to_eps(model_out, x, t_long, ddpm, prediction_type):
    """Convert model output to epsilon prediction."""
    if prediction_type == "epsilon":
        return model_out
    elif prediction_type == "v":
        return ddpm.v_to_eps(model_out, x, t_long)
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")


@torch.no_grad()
def ddim_sample(model, ddpm, n, K, device, labels=None, cfg_scale=0.0,
                steps=50, eta=0.0, prediction_type="epsilon"):
    """DDIM reverse diffusion. Returns [n, K, FEAT_DIM] in standardized space."""
    model.eval()
    timesteps = torch.linspace(ddpm.n_T - 1, 0, steps + 1).round().long()
    alphabar = ddpm.alphabar_t.to(device)

    x = torch.randn(n, K, FEAT_DIM, device=device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        t_tensor = torch.full((n,), t, dtype=torch.float32, device=device)
        t_long = torch.full((n,), t, dtype=torch.long, device=device)

        model_out = _model_predict_cfg(model, x, t_tensor, labels, cfg_scale)
        eps_pred = _to_eps(model_out, x, t_long, ddpm, prediction_type)

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

    return x


@torch.no_grad()
def ddpm_sample(model, ddpm, n, K, device, labels=None, cfg_scale=0.0,
                prediction_type="epsilon"):
    """DDPM reverse diffusion. Returns [n, K, FEAT_DIM] in standardized space."""
    model.eval()
    x = torch.randn(n, K, FEAT_DIM, device=device)
    for t in range(ddpm.n_T, 0, -1):
        t_tensor = torch.full((n,), t, dtype=torch.float32, device=device)
        t_long = torch.full((n,), t, dtype=torch.long, device=device)
        model_out = _model_predict_cfg(model, x, t_tensor, labels, cfg_scale)
        eps_pred = _to_eps(model_out, x, t_long, ddpm, prediction_type)
        oneover_sqrta = ddpm.oneover_sqrta[t].to(device)
        mab_over_sqrtmab = ddpm.mab_over_sqrtmab[t].to(device)
        sqrt_beta_t = ddpm.sqrt_beta_t[t].to(device)
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        x = oneover_sqrta * (x - mab_over_sqrtmab * eps_pred) + sqrt_beta_t * z
    return x


# ---------------------------------------------------------------------------
# Render standardized Gaussians → RGB images
# ---------------------------------------------------------------------------
def render_gaussians(W_std, device="cuda", chunk_size=32):
    """Render standardized Gaussians to [N, 3, H, W] float images.

    Uses batched renderer for speed (processes chunk_size images at once).
    """
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
            kernel_size=KERNEL_SIZE,
            sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
            coords=coords, colours=colours,
            image_size=IMAGE_SIZE, channels=3, device=device,
            soft_clamp=SOFT_CLAMP,
        )  # [B, H, W, C]
        all_frames.append(imgs.permute(0, 3, 1, 2))  # [B, C, H, W]
    return torch.cat(all_frames, dim=0)


# ---------------------------------------------------------------------------
# In-training FID evaluation (lightweight)
# ---------------------------------------------------------------------------
def compute_fid_quick(model, ddpm, K, device, real_u8, n_samples=500,
                      cfg_scale=1.5, ddim_steps=50, prediction_type="epsilon"):
    """Sample n_samples, render, compute FID against cached real images."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    labels = torch.arange(NUM_CLASSES, device=device).repeat(
        (n_samples + NUM_CLASSES - 1) // NUM_CLASSES)[:n_samples]

    # Sample with DDIM (fast)
    all_samples = []
    bs = 64
    for start in range(0, n_samples, bs):
        end = min(start + bs, n_samples)
        batch_labels = labels[start:end]
        samples = ddim_sample(model, ddpm, end - start, K, device,
                              labels=batch_labels, cfg_scale=cfg_scale,
                              steps=ddim_steps, eta=0.0,
                              prediction_type=prediction_type)
        all_samples.append(samples)
    W_std = torch.cat(all_samples, dim=0)

    # Render
    rendered = render_gaussians(W_std, device=device)
    rendered_u8 = (rendered * 255).clamp(0, 255).to(torch.uint8)

    # Fresh FID metric each call (avoids stale state issues)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    for i in range(0, len(real_u8), 256):
        fid.update(real_u8[i:i + 256].to(device), real=True)
    for i in range(0, len(rendered_u8), 256):
        fid.update(rendered_u8[i:i + 256].to(device), real=False)

    return fid.compute().item()


def load_real_images(n_real=10000):
    """Load real CIFAR-10 images as uint8 tensor for FID computation."""
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=str(PROJECT_ROOT / "data"), train=True, download=False)
    indices = np.random.RandomState(42).choice(len(ds), size=n_real, replace=False)

    real_images = []
    for idx in indices:
        img, _ = ds[idx]
        real_images.append(torch.tensor(np.array(img)).permute(2, 0, 1))

    real_u8 = torch.stack(real_images)  # [n_real, 3, 32, 32] uint8
    print(f"  FID: cached {n_real} real CIFAR-10 images")
    return real_u8


# ---------------------------------------------------------------------------
# EMA
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
            raw = state_dict["shadow"]
            self.step = state_dict.get("step", 0)
        else:
            raw = state_dict
        # Build clean-key lookup from checkpoint
        clean_to_val = {}
        for k, v in raw.items():
            clean_to_val[k.replace("_orig_mod.", "")] = v.clone().detach()
        # Match shadow keys (which may have _orig_mod. prefix) to checkpoint
        for k in list(self.shadow.keys()):
            clean_k = k.replace("_orig_mod.", "")
            if clean_k in clean_to_val:
                self.shadow[k] = clean_to_val[clean_k]


# ---------------------------------------------------------------------------
# DDP utilities
# ---------------------------------------------------------------------------
def setup_ddp():
    """Auto-detect and initialize DDP from torchrun environment."""
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return 0, 1, False  # rank, world_size, is_distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, True


def unwrap_model(model):
    """Get the raw nn.Module from DDP/compile wrappers."""
    m = model
    if hasattr(m, "module"):       # DDP
        m = m.module
    if hasattr(m, "_orig_mod"):    # torch.compile
        m = m._orig_mod
    return m


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    # ---- DDP setup ----
    rank, world_size, is_distributed = setup_ddp()
    is_main = (rank == 0)
    if is_distributed:
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    else:
        device = args.device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Seed for reproducible data split (all ranks must agree)
    torch.manual_seed(42)

    # ---- Wandb (rank 0 only) ----
    run = None
    if args.wandb and is_main:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or args.tag,
            config=vars(args),
            tags=[f"{args.num_blocks}B{args.num_heads}H{args.hidden_dim}D"],
        )
        print(f"  Wandb: {run.url}")

    # ---- Data ----
    if is_main:
        print("Loading dataset …")
    dataset = GaussianDatasetV2(args.data_h5, min_psnr=args.min_psnr,
                                    digits=args.classes)
    if is_main:
        print(f"  {dataset}")

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(42)  # deterministic split across ranks
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    nw = args.num_workers
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler,
                            shuffle=False,
                            num_workers=max(1, nw // 2), pin_memory=True,
                            persistent_workers=nw > 0)

    # ---- Model ----
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

    raw_model = model  # keep reference before compile/DDP wrapping
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"  Model: {args.num_blocks}B/{args.num_heads}H/{args.hidden_dim}D "
              f"→ {n_params/1e6:.2f}M params, K={args.num_gaussians}, feat={FEAT_DIM}")

    if args.compile:
        if is_main:
            print("  Compiling model with torch.compile …")
        model = torch.compile(model)

    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = DDP(model, device_ids=[local_rank])
        if is_main:
            print(f"  DDP: {world_size} GPUs, per-GPU batch={args.batch_size}, "
                  f"global batch={args.batch_size * world_size}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ddpm = DDPM(n_T=args.timesteps, schedule_type="cosine", s=args.schedule_s)

    # Min-SNR weighting: reweight loss per timestep to focus on informative noise levels.
    # weight(t) = min(SNR(t), gamma) / SNR(t)  where SNR(t) = alphabar_t / (1 - alphabar_t)
    min_snr_weights = None
    if args.min_snr_gamma > 0:
        alphabar = ddpm.alphabar_t.to(device)  # [T+1]
        snr = alphabar / (1 - alphabar).clamp(min=1e-8)  # [T+1]
        min_snr_weights = (snr.clamp(max=args.min_snr_gamma) / snr.clamp(min=1e-8))  # [T+1]
        if is_main:
            print(f"  Min-SNR weighting: gamma={args.min_snr_gamma}")

    # Feature-weighted MSE loss: sigma/rho/coords are harder to learn than colors
    # Features: [sigma_x, sigma_y, rho, r, g, b, x, y]
    if args.loss_weights:
        feat_weights = torch.tensor(args.loss_weights, device=device).view(1, 1, -1)
        feat_weights = feat_weights / feat_weights.mean()  # normalize so mean=1
        if is_main:
            print(f"  Feature loss weights: {feat_weights.squeeze().tolist()}")
        def criterion(pred, target):
            return (feat_weights * (pred - target) ** 2).mean()
    else:
        criterion = nn.MSELoss()

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
    if is_main:
        print(f"  LR schedule: {warmup_steps} warmup, {total_steps} total steps")

    # EMA must track the raw (unwrapped) model
    ema = EMA(raw_model, decay=args.ema_decay) if args.use_ema else None
    if ema and is_main:
        print(f"  EMA enabled (decay={args.ema_decay})")

    use_amp = device.startswith("cuda") and args.amp
    # fp16 by default (better precision for K=500 attention); bf16 opt-in via --bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    if is_main:
        if use_amp:
            print(f"  AMP ({amp_dtype}) enabled")
        if args.prediction_type != "epsilon":
            print(f"  Prediction type: {args.prediction_type}")
        if args.hflip:
            print("  Horizontal flip augmentation enabled")

    # Data standardization tensors (broadcast over [B, K, 8])
    data_mean = DATA_MEAN.to(device).view(1, 1, -1)
    data_std = DATA_STD.to(device).view(1, 1, -1)

    # Cache real images for FID computation (rank 0 only)
    real_u8 = None
    if args.fid_every > 0 and is_main:
        real_u8 = load_real_images(n_real=args.fid_n_real)

    if is_main:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    best_val_loss = float("inf")
    best_fid = float("inf")
    start_epoch = 0

    # ---- Resume ----
    if args.resume:
        if is_main:
            print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        raw_model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if args.reset_scheduler:
            if is_main:
                print(f"  Resetting LR schedule for {args.epochs} total epochs (from epoch {ckpt['epoch']})")
            n_steps = ckpt["epoch"] * len(train_loader)
            for _ in range(n_steps):
                scheduler.step()
        elif "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            n_steps = ckpt["epoch"] * len(train_loader)
            for _ in range(n_steps):
                scheduler.step()
        if use_amp and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ema is not None and "ema_shadow" in ckpt:
            ema.load_state_dict(ckpt["ema_shadow"])
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        start_epoch = ckpt["epoch"]
        if is_main:
            print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ================================================================
    # Training loop
    # ================================================================
    nan_recovery_count = 0
    MAX_NAN_RECOVERIES = 10
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_t0 = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        train_loss = 0.0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]",
                               leave=False, disable=not is_main):
            batch = batch_data[0].to(device, non_blocking=True)
            labels = batch_data[1].to(device, non_blocking=True)
            batch_norm = normalize_parameters(batch, PARAM_RANGES)
            # Random horizontal flip: negate rho (col 2) and x (col 6)
            if args.hflip and random.random() < 0.5:
                batch_norm[:, :, 2] = -batch_norm[:, :, 2]  # rho
                batch_norm[:, :, 6] = -batch_norm[:, :, 6]  # x
            batch_std = (batch_norm - data_mean) / data_std
            t = torch.randint(1, args.timesteps + 1, (batch.shape[0],),
                              dtype=torch.long, device=device)
            x_t, _, noise = ddpm.get_noisy_images_and_noise(batch_std, t)
            # Compute target based on prediction type
            if args.prediction_type == "v":
                target = ddpm.get_velocity(batch_std, noise, t)
            else:
                target = noise
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(x_t, t.float(), y=labels)
                if min_snr_weights is not None:
                    per_sample = ((pred - target) ** 2).mean(dim=(1, 2))  # [B]
                    loss = (min_snr_weights[t] * per_sample).mean()
                else:
                    loss = criterion(pred, target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(raw_model)
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- NaN recovery ----
        if math.isnan(train_loss):
            best_path = os.path.join(args.checkpoint_dir, "best.pt")
            nan_recovery_count += 1
            if is_main:
                print(f"Epoch {epoch:4d} | NaN detected! Recovery #{nan_recovery_count}/{MAX_NAN_RECOVERIES}")
            if nan_recovery_count > MAX_NAN_RECOVERIES or not os.path.exists(best_path):
                if is_main:
                    print("  Too many NaN recoveries or no checkpoint. Stopping.")
                break
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            state = {k.replace("_orig_mod.", "").replace("module.", ""): v
                     for k, v in ckpt["model_state_dict"].items()}
            raw_model.load_state_dict(state)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ema is not None and "ema_shadow" in ckpt:
                ema.load_state_dict(ckpt["ema_shadow"])
            if use_amp and amp_dtype == torch.float16:
                scaler = torch.amp.GradScaler("cuda", init_scale=2**10)
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            if is_main:
                print(f"  Reloaded best.pt (epoch {ckpt['epoch']}, val_loss={best_val_loss:.4f}). "
                      f"Continuing from epoch {epoch+1} with lower LR ({optimizer.param_groups[0]['lr']:.2e})")
            continue

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]",
                                   leave=False, disable=not is_main):
                batch = batch_data[0].to(device, non_blocking=True)
                labels = batch_data[1].to(device, non_blocking=True)
                batch_norm = normalize_parameters(batch, PARAM_RANGES)
                batch_std = (batch_norm - data_mean) / data_std
                t = torch.randint(1, args.timesteps + 1, (batch.shape[0],),
                                  dtype=torch.long, device=device)
                x_t, _, noise = ddpm.get_noisy_images_and_noise(batch_std, t)
                if args.prediction_type == "v":
                    target = ddpm.get_velocity(batch_std, noise, t)
                else:
                    target = noise
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(x_t, t.float(), y=labels)
                    val_loss += criterion(pred, target).item()

        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_t0
        lr_now = optimizer.param_groups[0]["lr"]

        if is_main:
            print(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| lr {lr_now:.2e} | {epoch_time:.0f}s")

        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_now,
            "epoch_time_s": epoch_time,
        }

        # ---- Sample grid (rank 0 only) ----
        if is_main and (epoch % args.sample_every == 0 or epoch == args.epochs or epoch == 1):
            if ema is not None:
                orig_state = copy.deepcopy(raw_model.state_dict())
                ema.apply(raw_model)
            sample_labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(2)[:16]
            # Use raw_model for sampling (no DDP wrapper needed for inference)
            samples_std = ddim_sample(raw_model, ddpm, n=16, K=args.num_gaussians,
                                      device=device, labels=sample_labels,
                                      cfg_scale=args.cfg_scale, steps=50,
                                      prediction_type=args.prediction_type)
            if ema is not None:
                raw_model.load_state_dict(orig_state)

            rendered = render_gaussians(samples_std, device=device)
            grid = make_grid(rendered, nrow=8, normalize=False)
            grid_path = os.path.join(args.sample_dir, f"samples_epoch{epoch:04d}.png")
            save_image(grid, grid_path)

            if run is not None:
                import wandb
                log_dict["samples"] = wandb.Image(
                    grid.permute(1, 2, 0).cpu().numpy(),
                    caption=f"Epoch {epoch}"
                )

        # ---- FID evaluation (rank 0 only) ----
        if is_main and real_u8 is not None and (epoch % args.fid_every == 0 or epoch == 1):
            if ema is not None:
                orig_state = copy.deepcopy(raw_model.state_dict())
                ema.apply(raw_model)

            fid_t0 = time.time()
            fid_val = compute_fid_quick(
                raw_model, ddpm, args.num_gaussians, device, real_u8,
                n_samples=args.fid_n_samples, cfg_scale=args.cfg_scale,
                ddim_steps=args.fid_ddim_steps,
                prediction_type=args.prediction_type,
            )
            fid_time = time.time() - fid_t0

            if ema is not None:
                raw_model.load_state_dict(orig_state)

            log_dict["fid"] = fid_val
            log_dict["fid_time_s"] = fid_time
            print(f"  → FID = {fid_val:.1f} ({fid_time:.0f}s)")

            if fid_val < best_fid:
                best_fid = fid_val

        # ---- Wandb log (rank 0 only) ----
        if run is not None:
            import wandb
            run.log(log_dict, step=epoch)

        # ---- Checkpoints (rank 0 only) ----
        if is_main:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),  # clean keys, no module./orig_mod prefix
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "args": vars(args),
                "standardized": True,
                "data_mean": DATA_MEAN.tolist(),
                "data_std": DATA_STD.tolist(),
            }
            if ema is not None:
                ckpt["ema_shadow"] = ema.state_dict()
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            # Save periodic snapshot every 50 epochs
            if epoch % 50 == 0:
                torch.save(ckpt, os.path.join(args.checkpoint_dir, f"ep{epoch}_snapshot.pt"))
        if is_distributed:
            dist.barrier()

    if is_main:
        print("Training complete.")
    if run is not None:
        import wandb
        run.finish()
    if is_distributed:
        dist.destroy_process_group()


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
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 instead of fp16 for AMP (avoids NaN but less precise)")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--schedule_s", type=float, default=0.008)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--reset_scheduler", action="store_true",
                        help="Reset LR schedule to new --epochs instead of loading from checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="cifar10", help="Run tag for wandb and checkpoints")
    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", default="cifar10-gaussian-diffusion")
    parser.add_argument("--wandb_name", default=None, help="Wandb run name (default: --tag)")
    # In-training FID
    parser.add_argument("--fid_every", type=int, default=25,
                        help="Compute FID every N epochs (0=disabled)")
    parser.add_argument("--fid_n_samples", type=int, default=500,
                        help="Number of samples for FID evaluation")
    parser.add_argument("--fid_n_real", type=int, default=10000,
                        help="Number of real images for FID reference")
    parser.add_argument("--fid_ddim_steps", type=int, default=50,
                        help="DDIM steps for FID sampling")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Train on specific classes only (e.g. --classes 0 = airplane)")
    parser.add_argument("--loss_weights", type=float, nargs="+", default=None,
                        help="Per-feature loss weights [sigma_x,sigma_y,rho,r,g,b,x,y]")
    parser.add_argument("--prediction_type", default="epsilon", choices=["epsilon", "v"],
                        help="Prediction target: epsilon (noise) or v (velocity)")
    parser.add_argument("--min_snr_gamma", type=float, default=0,
                        help="Min-SNR weighting gamma (0=disabled, 5=recommended)")
    parser.add_argument("--hflip", action="store_true",
                        help="Random horizontal flip augmentation (negates x and rho)")
    args = parser.parse_args()

    # Store feature_dim in args for checkpoint metadata
    args.feature_dim = FEAT_DIM
    args.num_classes = NUM_CLASSES

    train(args)


if __name__ == "__main__":
    main()
