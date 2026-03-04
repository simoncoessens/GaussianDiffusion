"""
Train GaussianTransformer diffusion model on MNIST Gaussian representations.

Usage:
    python -m src.train \\
        --data_dir data/mnist_gaussian_representations/ \\
        --epochs 100
"""

import argparse
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
from src.ddpm import DDPM
from src.models.transformer_model import GaussianTransformer
from src.utils.normalize import normalize_parameters
from src.utils.denormalize import denormalize_parameters
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

# ---------------------------------------------------------------------------
# Parameter ranges for MNIST 7-dim Gaussians
# [sigma_x, sigma_y, rho, alpha, colour, x, y]
# ---------------------------------------------------------------------------
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
# Reverse diffusion sampler
# ---------------------------------------------------------------------------
@torch.no_grad()
def ddpm_sample(model: nn.Module, ddpm: DDPM, n: int, K: int, feat_dim: int,
                device: str) -> torch.Tensor:
    """Run reverse diffusion for n samples. Returns [n, K, feat_dim]."""
    model.eval()
    x = torch.randn(n, K, feat_dim, device=device)
    for t in range(ddpm.n_T, 0, -1):
        t_tensor = torch.full((n,), t, dtype=torch.float32, device=device)
        eps_pred = model(x, t_tensor)
        # DDPM reverse step
        alpha_t = ddpm.alpha_t[t].to(device)
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
        w = W_phys[i]  # [K, 7]
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
        frames.append(img.permute(2, 0, 1))  # [1, H, W]

    grid = make_grid(torch.stack(frames), nrow=8, normalize=False)
    return grid


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = args.device

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
    dataset = GaussianDataset(args.data_dir, num_gaussians=args.num_gaussians)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    feat_dim = dataset[0].shape[1]  # 7

    # Model
    model = GaussianTransformer(
        input_dim=args.num_gaussians,
        time_emb_dim=512,
        feature_dim=feat_dim,
        num_timestamps=args.timesteps,
        num_transformer_blocks=32,
        num_heads=64,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    ddpm = DDPM(n_T=args.timesteps, schedule_type="cosine")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            batch = batch.to(device, non_blocking=True)
            batch_norm = normalize_parameters(batch, PARAM_RANGES)
            t = torch.randint(1, args.timesteps + 1, (batch.shape[0],),
                              dtype=torch.long, device=device)
            x_t, _, noise = ddpm.get_noisy_images_and_noise(batch_norm, t)
            pred = model(x_t, t.float())
            loss = criterion(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                batch = batch.to(device, non_blocking=True)
                batch_norm = normalize_parameters(batch, PARAM_RANGES)
                t = torch.randint(1, args.timesteps + 1, (batch.shape[0],),
                                  dtype=torch.long, device=device)
                x_t, _, noise = ddpm.get_noisy_images_and_noise(batch_norm, t)
                pred = model(x_t, t.float())
                val_loss += criterion(pred, noise).item()

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # ---- Logging ----
        log_dict = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}

        # ---- Sample grid ----
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            samples_norm = ddpm_sample(model, ddpm, n=16,
                                       K=args.num_gaussians, feat_dim=feat_dim, device=device)
            grid = render_sample_grid(samples_norm, PARAM_RANGES,
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
            "val_loss": val_loss,
            "args": vars(args),
        }
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
    parser.add_argument("--data_dir", default="data/mnist_gaussian_representations/")
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    parser.add_argument("--sample_dir", default="samples/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--num_gaussians", type=int, default=70)
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Save sample grid every N epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
