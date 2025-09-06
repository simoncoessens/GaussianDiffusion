import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import wandb

from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum

from src.ddpm import ddpm_schedules
from src.utils.denormalize import denormalize_parameters
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from data.encoding_scripts.GaussianImage.GaussianImage.gaussianimage_rs import GaussianImage_RS

def diffusion_sampler(eps_model, n_T, oneover_sqrta, mab_over_sqrtmab, sqrt_beta_t, n_sample, size, device="cuda"):
    """
    Generate samples from the diffusion model, returns the final samples.
    """
    with torch.no_grad():
        x = torch.randn(n_sample, *size, device=device)
        for t in range(n_T, 0, -1):
            t_tensor = torch.full((n_sample,), t, device=device, dtype=torch.float32)
            eps = eps_model(x, t_tensor)
            z = torch.randn(n_sample, *size, device=device) if t > 1 else torch.zeros(n_sample, *size, device=device)
            x = oneover_sqrta[t] * (x - eps * mab_over_sqrtmab[t]) + sqrt_beta_t[t] * z
        return x

def render_latent_to_image(latent, param_ranges, device, H, W, BLOCK_H, BLOCK_W, num_points, feature_dim, lr=1e-3, denormalize=True):
    """
    Renders an image from a latent vector using the GaussianImage_RS model.
    """
    # Optionally denormalize latent parameters
    if denormalize:
        latent_processed = denormalize_parameters(latent.unsqueeze(0), param_ranges).squeeze(0)
    else:
        latent_processed = latent

    sigma_x = latent_processed[:, 0]
    sigma_y = latent_processed[:, 1]
    rho = latent_processed[:, 2]
    colours = latent_processed[:, 3:6]  # 3 channels for RGB
    coords = latent_processed[:, 6:8]

    # Create the GaussianImage_RS instance only once
    if not hasattr(render_latent_to_image, "gaussian_image_model"):
        render_latent_to_image.gaussian_image_model = GaussianImage_RS(
            loss_type="L2",
            opt_type="adan",
            num_points=num_points,
            H=H,
            W=W,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            device=device,
            lr=lr,
            quantize=False,
            feature_dim=3
        ).to(device)
        render_latent_to_image.gaussian_image_model.eval()

    model_instance = render_latent_to_image.gaussian_image_model

    with torch.no_grad():
        # Restore coordinates using the inverse of tanh
        saved_xyz_tensor = torch.clamp(coords, -0.9999, 0.9999).to(device)
        model_instance._xyz.copy_(torch.atanh(saved_xyz_tensor))
        
        # Restore scaling parameters
        saved_scaling_tensor = torch.stack([sigma_x, sigma_y], dim=1).to(device)
        model_instance._scaling.copy_(saved_scaling_tensor - model_instance.bound)
        
        # Restore rotation
        saved_rotation_tensor = rho.to(device)
        normalized = saved_rotation_tensor / (2 * math.pi)
        normalized = torch.clamp(normalized, 1e-6, 1 - 1e-6)
        inv_sigmoid = lambda y: torch.log(y / (1 - y))
        model_instance._rotation.copy_(inv_sigmoid(normalized).unsqueeze(1))
        
        # Restore features (RGB)
        saved_features_tensor = colours.to(device)
        model_instance._features_dc.copy_(saved_features_tensor)
        
        # Set opacity to all ones
        model_instance._opacity.copy_(torch.ones((model_instance.init_num_points, 1), device=device))
        
        # Render the image
        output = model_instance.forward()
        render = output["render"]
        render_tensor = render.squeeze(0).cpu()
    return render_tensor

def plot_image_grid(images, grid_size=10, figsize=(10, 10)):
    """
    Creates a matplotlib figure with a grid of images
    Returns the figure for saving or display
    """
    n_samples = len(images)
    n_rows = int(np.ceil(n_samples / grid_size))
    n_cols = min(grid_size, n_samples)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < n_samples:
            # Convert tensor image to numpy for matplotlib
            img = images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused subplots
    
    plt.tight_layout()
    return fig

def sample_and_plot_images(
    model, dataset, param_ranges, device, 
    n_sample=100, n_T=200, grid_size=10, 
    H=32, W=32, BLOCK_H=16, BLOCK_W=16
):
    """
    Sample from the model, render images, create matplotlib grid, and calculate metrics
    Returns the figure and metrics
    """
    # Prepare the diffusion schedules
    betas = (1e-4, 0.02)
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    
    # Get the latent size from the dataset
    latent_size = dataset[0].shape
    
    # Sample latent representations and time it
    start_time = time.time()
    final_latents = diffusion_sampler(
        eps_model=model,
        n_T=n_T,
        oneover_sqrta=schedules["oneover_sqrta"],
        mab_over_sqrtmab=schedules["mab_over_sqrtmab"],
        sqrt_beta_t=schedules["sqrt_beta_t"],
        n_sample=n_sample,
        size=latent_size,
        device=device
    )
    
    # Generate images from final latent samples
    generated_images = []
    for idx in range(n_sample):
        latent = final_latents[idx]
        image_tensor = render_latent_to_image(
            latent, param_ranges, device, H, W, BLOCK_H, BLOCK_W,
            num_points=latent_size[0], feature_dim=3, lr=1e-3, denormalize=True
        )
        generated_images.append(image_tensor)
    
    sampling_time = time.time() - start_time
    
    # Create matplotlib figure with samples grid
    fig = plot_image_grid(
        generated_images[:grid_size*grid_size], 
        grid_size=grid_size,
        figsize=(10, 10)
    )
    
    # Get original images for metrics calculation
    original_indices = np.random.choice(len(dataset), n_sample, replace=len(dataset) < n_sample)
    original_images = []
    for idx in original_indices:
        latent = dataset[idx]
        img_tensor = render_latent_to_image(
            latent, param_ranges, device, H, W, BLOCK_H, BLOCK_W,
            num_points=latent_size[0], feature_dim=3, lr=1e-3, denormalize=False
        )
        original_images.append(img_tensor)
    
    # Calculate metrics
    metrics = calculate_metrics(original_images, generated_images, device)
    metrics["sampling_time"] = sampling_time
    metrics["samples_per_second"] = n_sample / sampling_time
    
    return fig, metrics

def calculate_metrics(original_images, generated_images, device):
    """
    Calculate FID, Inception Score, and KID metrics
    """
    # Process images for metrics calculation
    processed_originals = []
    processed_generated = []

    for img in original_images:
        img_rgb = img.repeat(3, 1, 1) if img.shape[0] == 1 else img
        img_resized = F.interpolate(img_rgb.unsqueeze(0), size=(299, 299),
                                    mode='bilinear', align_corners=False).squeeze(0)
        processed_originals.append(img_resized)

    for img in generated_images:
        img_rgb = img.repeat(3, 1, 1) if img.shape[0] == 1 else img
        img_resized = F.interpolate(img_rgb.unsqueeze(0), size=(299, 299),
                                    mode='bilinear', align_corners=False).squeeze(0)
        processed_generated.append(img_resized)

    processed_originals = torch.stack(processed_originals).to(device)
    processed_generated = torch.stack(processed_generated).to(device)

    # Calculate FID
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    fid_metric.update(processed_originals, real=True)
    fid_metric.update(processed_generated, real=False)
    fid_value = float(fid_metric.compute())

    # Calculate Inception Score
    inception_metric = InceptionScore(normalize=True).to(device)
    inception_metric.update(processed_generated)
    is_mean, is_std = inception_metric.compute()

    # Calculate KID
    kid_metric = KernelInceptionDistance(normalize=True, subset_size=min(50, len(processed_originals))).to(device)
    kid_metric.update(processed_originals, real=True)
    kid_metric.update(processed_generated, real=False)
    kid_mean, kid_std = kid_metric.compute()

    metrics = {
        "fid": float(fid_value),
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std)
    }
    
    return metrics