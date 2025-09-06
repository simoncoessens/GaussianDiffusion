import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import random

# Custom modules
from src.ddpm import DDPM, ddpm_schedules
from src.models.transformer_model_old import GaussianTransformer
from src.utils.denormalize import denormalize_parameters
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from src.dataset import GaussianDatasetSpritesBigger

# Import the new latent-to-image function.
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

def diffusion_sampler(eps_model: nn.Module,
                      n_T: int,
                      oneover_sqrta: torch.Tensor,
                      mab_over_sqrtmab: torch.Tensor,
                      sqrt_beta_t: torch.Tensor,
                      n_sample: int,
                      size,
                      device: str = "cuda"):
    """
    Reverse-samples from the diffusion model and returns a list of latent samples at each timestep.
    """
    with torch.no_grad():
        x = torch.randn(n_sample, *size, device=device)
        x_history = []
        for t in range(n_T, 0, -1):
            t_tensor = torch.full((n_sample,), t, device=device, dtype=torch.float32)
            eps = eps_model(x, t_tensor)
            z = torch.randn(n_sample, *size, device=device) if t > 1 else torch.zeros(n_sample, *size, device=device)
            x = oneover_sqrta[t] * (x - eps * mab_over_sqrtmab[t]) + sqrt_beta_t[t] * z
            x_history.append(x.detach().clone())
        return x_history

def latent_to_image(latent: torch.Tensor,
                    param_ranges: list,
                    device: torch.device,
                    H: int,
                    W: int,
                    kernel_size: int = 17,
                    denormalize: bool = True) -> torch.Tensor:
    """
    Converts a latent vector into an RGB image using the generate_2D_gaussian_splatting function.

    The expected latent format ([num_gaussians, 8]) is:
      - Column 0: sigma_x
      - Column 1: sigma_y
      - Column 2: rho (rotation)
      - Columns 3-5: RGB features
      - Columns 6-7: x and y coordinates

    Args:
        latent (torch.Tensor): Latent vector of shape (num_gaussians, 8).
        param_ranges (list): Parameter ranges for denormalization.
        device (torch.device): Computation device.
        H (int): Height of the output image.
        W (int): Width of the output image.
        kernel_size (int): Kernel size for the splatting (default 17).
        denormalize (bool): Whether to apply denormalization.

    Returns:
        A tensor image of shape (C, H, W) in the [0, 1] range.
    """
    # Optionally denormalize the latent parameters.
    if denormalize:
        latent_processed = denormalize_parameters(latent.unsqueeze(0), param_ranges).squeeze(0)
    else:
        latent_processed = latent

    # Extract the parameters.
    sigma_x = latent_processed[:, 0]
    sigma_y = latent_processed[:, 1]
    rho = latent_processed[:, 2]
    alpha = latent_processed[:, 3]
    colours = latent_processed[:, 4:7]  # 3 channels for RGB.
    coords = latent_processed[:, 7:9]

    # Generate the image using the imported function.
    # Here we assume that generate_2D_gaussian_splatting returns an image tensor of shape (H, W, C).
    image = generate_2D_gaussian_splatting(
        kernel_size=kernel_size,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        rho=rho,
        coords=coords,
        colours=colours,
        image_size=(H, W),
        channels=3,  # Render as RGB.
        device=device
    )
    # Convert from (H, W, C) to (C, H, W).
    return image.permute(2, 0, 1).cpu()

def save_image_grid(images: list, grid_rows: int, grid_cols: int, save_path: str):
    """
    Saves a grid of images to a file.
    """
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2.5, grid_rows * 2.5))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            # Convert image tensor to numpy array.
            img_np = images[i].cpu().numpy()
            # If image is multi-channel, transpose from (C, H, W) to (H, W, C).
            if img_np.ndim == 3:
                img_np = img_np.transpose(1, 2, 0)
            ax.imshow(img_np)
            ax.set_title(f"Sample {i+1}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    # Set device and clear cache.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Global parameters and paths.
    # DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/datasets/Sprites/gaussian_representations_sprites_downsampled_200g.h5"
    DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/sprites_combined_32x32_old_encoding.h5"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/sprites_MHA_64h_32_blocks_downsampled_old_param/checkpoints/sprites_MHA_64h_32_blocks_downsampled_old_param/intermediate_e60_MHA_64h_32_blocks_old_parm.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/sprites_MHA_64h_32_blocks_downsampled_old_param/checkpoints/sprites_MHA_64h_32_blocks_downsampled_old_param/last_sprites_MHA_64h_32_blocks_old_parm_e100_ts200_mse_std_scaled.pth"
    model_path = "/gpfs/workdir/coessenss/gsplat/models/sprites_MHA_64h_32_blocks_downsampled/checkpoints/sprites_MHA_64h_32_blocks_downsampled/last_sprites_MHA_64h_32_blocks_new_parm_e100_ts200_mse_std_scaled.pth"
    betas = (1e-4, 0.02)
    n_T = 200

    # Image generation parameters.
    H, W = 32, 32            # Output image size.
    lr = 1e-3  # (Not used with the new latent image function)

    # Prepare diffusion schedules.
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    print("Schedules keys:", list(schedules.keys()))

    # Load diffusion model (GaussianTransformer).
    num_gaussians = 200
    feature_dim = 9
    time_emb_dim = 32
    num_blocks = 32
    num_heads = 64
    num_timestamps = n_T

    diffusion_model = GaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    diffusion_model.load_state_dict(checkpoint["model_state_dict"])
    diffusion_model.eval()
    diffusion_model.to(DEVICE)
    print(f"Model loaded from {model_path}")

    # Load dataset and determine latent dimensions automatically.
    print("Loading dataset...")
    dataset = GaussianDatasetSpritesBigger(DATA_FOLDER)
    all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    print("Shape of complete dataset:", all_data.shape)
    # Expected latent shape is [200, 8].
    latent_size = all_data.shape[1:]  # (200, 8)
    num_gaussians, feature_dim = latent_size
    print(f"Determined latent size: {latent_size}")

    # Compute parameter ranges for denormalization based on dataset statistics.
    param_ranges = [
        (all_data[:, :, i].min().item(), all_data[:, :, i].max().item())
        for i in range(feature_dim)
    ]
    print(f"Parameter ranges ({len(param_ranges)}): {param_ranges}\n")

    # Sample latent representations using the diffusion model.
    n_sample = 10000
    print(f"Sampling {n_sample} latent representations...")
    x_history = diffusion_sampler(
        eps_model=diffusion_model,
        n_T=n_T,
        oneover_sqrta=schedules["oneover_sqrta"],
        mab_over_sqrtmab=schedules["mab_over_sqrtmab"],
        sqrt_beta_t=schedules["sqrt_beta_t"],
        n_sample=n_sample,
        size=latent_size,
        device=DEVICE
    )
    final_latents = x_history[-1]

    # Set output folder based on the model name.
    model_name = f"Sprites_MHA_{num_heads}h_{num_blocks}blocks_{n_T}ts_new_param"
    output_folder = os.path.join("Metrics/Best_models", model_name)
    os.makedirs(output_folder, exist_ok=True)

    # Generate images from final latent samples and time the generation process.
    final_images = []
    start_time = time.time()
    for idx in range(n_sample):
        print(f"Processing generated sample {idx+1}...")
        latent = final_latents[idx]
        image_tensor = latent_to_image(latent, param_ranges, DEVICE, H, W,
                                       kernel_size=17, denormalize=True)
        final_images.append(image_tensor)
    end_time = time.time()
    avg_time = (end_time - start_time) / n_sample
    print(f"Average generation time per image: {avg_time:.4f} seconds")

    # Save an 8x8 grid (64 images) of generated samples.
    grid_size = 8
    grid_path = os.path.join(output_folder, f"grid_generated_ts_{n_T}_samples_{n_sample}.png")
    save_image_grid(final_images[:grid_size * grid_size], grid_size, grid_size, grid_path)
    print(f"Saved grid plot for generated images to {grid_path}")

    # ----------------------------------------
    # Process and save original images grid.
    # ----------------------------------------
    n_orig = 10000  # Number of original samples for metrics.
    original_images = []
    # Randomly choose indices from the dataset.
    orig_indices = random.sample(range(len(dataset)), n_orig)
    for idx in orig_indices:
        latent = dataset[idx]  # Each dataset item is a latent vector.
        img_tensor = latent_to_image(latent, param_ranges, DEVICE, H, W,
                                     kernel_size=17, denormalize=False)
        original_images.append(img_tensor)

    # Save an 8x8 grid (64 images) of original samples.
    grid_original_path = os.path.join(output_folder, f"grid_original_ts_{n_T}_samples_{n_orig}.png")
    save_image_grid(original_images[:grid_size * grid_size], grid_size, grid_size, grid_original_path)
    print(f"Saved grid plot for original images to {grid_original_path}")

    # ----------------------------------------
    # Compute metrics for the generated images.
    # ----------------------------------------
    # Process images for FID and other metrics:
    # 1. Convert grayscale images to RGB by repeating the channel if needed.
    # 2. Resize images to 299x299.
    processed_originals = []
    processed_generated = []

    for img in original_images:
        img_rgb = img if img.shape[0] == 3 else img.repeat(3, 1, 1)
        img_resized = F.interpolate(img_rgb.unsqueeze(0), size=(299, 299),
                                    mode='bilinear', align_corners=False).squeeze(0)
        processed_originals.append(img_resized)

    for img in final_images:
        img_rgb = img if img.shape[0] == 3 else img.repeat(3, 1, 1)
        img_resized = F.interpolate(img_rgb.unsqueeze(0), size=(299, 299),
                                    mode='bilinear', align_corners=False).squeeze(0)
        processed_generated.append(img_resized)

    processed_originals = torch.stack(processed_originals)
    processed_generated = torch.stack(processed_generated)

    # Calculate FID.
    fid_metric = FrechetInceptionDistance(normalize=True)
    fid_metric.update(processed_originals, real=True)
    fid_metric.update(processed_generated, real=False)
    fid_value = float(fid_metric.compute())
    print(f"FID: {fid_value}")

    # Calculate Inception Score.
    inception_metric = InceptionScore(normalize=True)
    inception_metric.update(processed_generated)
    is_mean, is_std = inception_metric.compute()
    print(f"Inception Score: {is_mean:.4f}")
    print(f"Inception Score std dev: {is_std:.4f}")

    # Calculate Kernel Inception Distance.
    kid_metric = KernelInceptionDistance(normalize=True, subset_size=100)
    kid_metric.update(processed_originals, real=True)
    kid_metric.update(processed_generated, real=False)
    kid_mean, kid_std = kid_metric.compute()
    print(f"Kernel Inception Distance: {kid_mean:.4f}")
    print(f"Kernel Inception Distance std dev: {kid_std:.4f}")

    # Save metrics to files.
    metrics = {
        "FID": fid_value,
        "Inception_Score": float(is_mean),
        "Inception_Score_StdDev": float(is_std),
        "KID": float(kid_mean),
        "KID_StdDev": float(kid_std),
        "Average_Generation_Time": avg_time
    }

    metrics_file = os.path.join(output_folder, f"metrics_ts_{n_T}_samples_{n_sample}.txt")
    with open(metrics_file, 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")

    json_file = os.path.join(output_folder, f"metrics_ts_{n_T}_samples_{n_sample}.json")
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to {metrics_file} and {json_file}")
