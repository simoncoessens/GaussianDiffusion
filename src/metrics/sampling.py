import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import os
from ddpm import DDPM
from enc_dec_model import GaussianSplatDiffusionModel

def diffusion_sampler(
    eps_model: nn.Module,
    n_T: int,
    oneover_sqrta: torch.Tensor,
    mab_over_sqrtmab: torch.Tensor,
    sqrt_beta_t: torch.Tensor,
    n_sample: int,
    size,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Args:
        eps_model: The trained noise predictor (GaussianSplatDiffusionModel).
        n_T: total number of timesteps in the diffusion process.
        oneover_sqrta, mab_over_sqrtmab, sqrt_beta_t:
            Tensors from your schedules dict (length = n_T+1).
        n_sample: Number of samples to generate.
        size: Shape of the samples.
        device: "cuda" or "cpu".
    """
    # Initialize Gaussian input with the correct noise distribution
    x_i = 0.4 * torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 0.4)

    for i in range(n_T, 0, -1):
        z = 0.4 * torch.randn(n_sample, *size).to(device) if i > 1 else 0
        eps = eps_model(
            x_i, torch.tensor(i / n_T, device=device).repeat(n_sample, 1)
        )
        x_i = (
            oneover_sqrta[i] * (x_i - eps * mab_over_sqrtmab[i])
            + sqrt_beta_t[i] * z
        )

    return x_i

# ---------------------------------------------------------
# 4) Denormalize parameters
# ---------------------------------------------------------
def denormalize_parameters(W_normalized, param_ranges):
    """
    Denormalizes parameters from the range [-1, 1] back to their original
    ranges with clipping.

    W_normalized: [batch_size, num_gaussians, 7]
    param_ranges: list of (min_val, max_val) for each of the 7 features
    """
    W_denormalized = torch.zeros_like(W_normalized)
    for i in range(W_normalized.shape[-1]):
        min_val, max_val = param_ranges[i]
        if min_val == max_val:
            # Edge case if range is zero
            W_denormalized[:, :, i] = min_val
        else:
            # Scale from [-1, 1] -> [min_val, max_val]
            W_denormalized[:, :, i] = (W_normalized[:, :, i] + 1) / 2.0
            W_denormalized[:, :, i] *= (max_val - min_val)
            W_denormalized[:, :, i] += min_val
            # Clip to [min_val, max_val]
            W_denormalized[:, :, i] = torch.clamp(W_denormalized[:, :, i], min_val, max_val)
    return W_denormalized

# ---------------------------------------------------------
# 5) 2D Gaussian Splatting
# ---------------------------------------------------------
def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho,
                                   coords, colours, image_size=(28, 28),
                                   channels=1, device="cuda"):
    """
    Converts a batch of Gaussians into a single 2D image by 'splatting' each Gaussian
    onto an image canvas of `image_size`. Summation is done across the batch dimension.
    """
    import math
    import numpy as np

    batch_size = sigma_x.shape[0]  # or colours.shape[0], etc.

    # Reshape sigma and rho
    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    # Create covariance matrices
    covariance = torch.stack([
        torch.stack([sigma_x**2, rho * sigma_x * sigma_y], dim=-1),
        torch.stack([rho * sigma_x * sigma_y, sigma_y**2], dim=-1)
    ], dim=-2)

    # Check positive semi-definiteness (this may throw an error if any determinant <= 0)
    determinant = sigma_x**2 * sigma_y**2 - (rho * sigma_x * sigma_y)**2

    # Print statements for debugging
    print("sigma_x:", sigma_x)
    print("sigma_y:", sigma_y)
    print("rho:", rho)
    print("sigma_x^2:", sigma_x**2)
    print("sigma_y^2:", sigma_y**2)
    print("(rho * sigma_x * sigma_y):", (rho * sigma_x * sigma_y))
    print("(rho * sigma_x * sigma_y)^2:", (rho * sigma_x * sigma_y)**2)
    print("determinant:", determinant)

    if (determinant <= 0).any():
        print('determinant', determinant)
        raise ValueError("Covariance matrix must be positive semi-definite")

    inv_covariance = torch.inverse(covariance)

    # Create grid for kernel in [-5, 5]
    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
    xy = torch.stack([xx, yy], dim=-1)  # [1, kernel_size, kernel_size, 2]

    # Compute Gaussian kernel for each batch item
    z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
    # denominator = 2*pi * sqrt(det(covariance)), repeated for each item
    denom = (2.0 * math.pi) * torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1)
    kernel = torch.exp(z) / denom

    # Normalize the kernel by its max
    kernel_max = kernel.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    kernel_normalized = kernel / kernel_max

    # Reshape to [batch_size, channels, kernel_size, kernel_size]
    kernel_reshaped = kernel_normalized.unsqueeze(1)  # -> [B, 1, K, K]
    kernel_reshaped = kernel_reshaped.repeat(1, channels, 1, 1)  # -> [B, C, K, K]

    # Pad to match desired image_size
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be <= image size.")

    padding = (
        pad_w // 2, pad_w // 2 + pad_w % 2,  # pad left, pad right
        pad_h // 2, pad_h // 2 + pad_h % 2   # pad top, pad bottom
    )
    kernel_padded = F.pad(kernel_reshaped, padding, "constant", 0)

    # Create affine transformation
    b, c, h, w = kernel_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    # coords is shape [B, 2] => place them in the translation slot
    theta[:, :, 2] = coords  # (x, y) in [-1,1] typically

    # Apply affine transformation to place the kernel at coords
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_transformed = F.grid_sample(kernel_padded, grid, align_corners=True)

    # Multiply by color
    # If 'colours' is shape [B], we unsqueeze for broadcast
    colours_reshaped = colours.view(batch_size, c, 1, 1)
    final_image_layers = colours_reshaped * kernel_transformed

    # Sum across batch dimension => single image
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)

    # final_image shape is [channels, height, width] => move to [height, width, channels]
    final_image = final_image.permute(1, 2, 0).contiguous()

    return final_image

# ---------------------------------------------------------
# 6) Main / End-to-End Example
# ---------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (A) Prepare Schedules
    betas = (1e-4, 0.02)
    n_T = 1000
    schedules = ddpm_schedules(betas[0], betas[1], n_T)

    # (B) Load Model
    num_gaussians = 70
    gaussian_features = 7
    depth = 1
    local_block_dim = [16, 32]
    global_block_dim = [32, 16, 7]
    time_emb_dim = 16

    eps_model = GaussianSplatDiffusionModel(
        num_gaussians=num_gaussians,
        gaussian_features=gaussian_features,
        depth=depth,
        local_dim_list=local_block_dim,
        global_dim_list=global_block_dim,
        time_emb_dim=time_emb_dim
    )

    checkpoint_path = "models/best_model_b32_e20.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    eps_model.load_state_dict(checkpoint["model_state_dict"])
    eps_model.to(device)
    eps_model.eval()
    print(f"Model loaded successfully from {checkpoint_path}")

    # (C) Instantiate Sampler
    sampler = DiffusionSampler(
        eps_model=eps_model,
        n_T=n_T,
        oneover_sqrta=schedules["oneover_sqrta"].to(device),
        mab_over_sqrtmab=schedules["mab_over_sqrtmab"].to(device),
        sqrt_beta_t=schedules["sqrt_beta_t"].to(device),
        device=device
    )

    # (D) Sample
    n_sample = 2  # e.g. create 2 samples
    samples = sampler.sample(n_sample=n_sample)  # shape [2, 70, 7]

    # (E) Denormalize
    param_ranges = [
        (0, 1),    # sigma_x
        (0, 1),    # sigma_y
        (-1, 1),   # rho
        (0, 1),    # alpha
        (0, 1),    # colours
        (-1, 1),   # x
        (-1, 1),   # y
    ]
    denormalized = denormalize_parameters(samples, param_ranges)  # [2, 70, 7]

    # Save histogram
    # Assuming `samples` has shape [2, 70, 7] and `denormalized` has shape [2, 70, 7]
    num_samples = samples.shape[0]  # Number of samples
    num_params = samples.shape[2]   # Number of parameters
    param_names = [
        "sigma_x",
        "sigma_y",
        "rho",
        "alpha",
        "colours",
        "x",
        "y"
    ]

    grid_size = int(np.ceil(np.sqrt(num_params)))  # Grid size for histograms

    # Generate histograms for each sample
    for sample_idx in range(num_samples):
        # Original data (before denormalization)
        sample_original = samples[sample_idx].cpu().numpy()  # Convert to NumPy
        # Denormalized data
        sample_denormalized = denormalized[sample_idx].cpu().numpy()  # Convert to NumPy

        # Plot Original Histograms
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))
        for i, ax in enumerate(axs.flat):
            if i < num_params:
                ax.hist(sample_original[:, i], bins=20, color="blue", alpha=0.7)
                ax.set_title(f"{param_names[i]} (Original - Sample {sample_idx})")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
            else:
                ax.axis('off')
        plt.tight_layout()
        original_filename = f"original_histograms_sample_{sample_idx}.png"
        plt.savefig(original_filename, dpi=300)
        plt.close(fig)
        print(f"Original histograms for sample {sample_idx} saved as '{original_filename}'")

        # Plot Denormalized Histograms
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))
        for i, ax in enumerate(axs.flat):
            if i < num_params:
                ax.hist(sample_denormalized[:, i], bins=20, color="green", alpha=0.7)
                ax.set_title(f"{param_names[i]} (Denormalized - Sample {sample_idx})")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
            else:
                ax.axis('off')
        plt.tight_layout()
        denormalized_filename = f"denormalized_histograms_sample_{sample_idx}.png"
        plt.savefig(denormalized_filename, dpi=300)
        plt.close(fig)
        print(f"Denormalized histograms for sample {sample_idx} saved as '{denormalized_filename}'")

    # (F) Convert each sample => 2D image
    image_size = (28, 28)
    kernel_size = 11

    images = []
    for i in range(n_sample):
        # shape: [70, 7]
        gaussians = denormalized[i]

        sigma_x = gaussians[:, 0]
        sigma_y = gaussians[:, 1]
        rho     = gaussians[:, 2]
        alpha   = gaussians[:, 3]  
        colour  = gaussians[:, 4]
        x       = gaussians[:, 5]
        y       = gaussians[:, 6]

        coords = torch.stack((x, y), dim=-1)

        final_image = generate_2D_gaussian_splatting(
            kernel_size=kernel_size,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rho=rho,
            coords=coords,       # shape [70, 2]
            colours=colour,      # shape [70]
            image_size=image_size,
            channels=1,
            device=device
        )
        # final_image is shape [28, 28, 1]
        images.append(final_image.cpu().numpy())

    print("Done! We have", len(images), "synthesized images.")

    # Folder where images will be saved
    output_folder = "sampling_images"
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all images and save them
    for idx, image in enumerate(images):
        # Extract a single grayscale channel (assuming images are stored in [height, width, channels])
        grayscale_image = image[:, :, 0]
        
        # Save the image
        output_path = os.path.join(output_folder, f"image_{idx:03d}.png")  # Save with a padded number
        plt.imsave(output_path, grayscale_image, cmap='gray')

    print(f"All images saved to the folder: {output_folder}")
