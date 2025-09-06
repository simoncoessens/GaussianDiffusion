import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from src.ddpm import ddpm_schedules
from src.models.transformer_model import GaussianTransformer
from src.dataset import GaussianDatasetSprites
from src.utils.denormalize import denormalize_parameters

# Data file containing the gaussian representations.
DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/datasets/Sprites/gaussian_representations_sprites_downsampled.h5"


def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho,
                                   coords, colours, image_size=(32, 32),
                                   channels=1, device="cuda"):
    """
    Generate an image by splatting 2D Gaussians.
    """
    batch_size = colours.shape[0]
    epsilon = 1e-6  # small regularization to ensure invertibility

    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    covariance = torch.stack(
        [
            torch.stack([sigma_x**2, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y**2], dim=-1)
        ],
        dim=-2
    )
    covariance[..., 0, 0] += epsilon
    covariance[..., 1, 1] += epsilon

    # Ensure the covariance is positive semi-definite.
    determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    if (determinant <= 0).any():
        covariance[..., 0, 0] = torch.clamp(covariance[..., 0, 0], min=epsilon)
        covariance[..., 1, 1] = torch.clamp(covariance[..., 1, 1], min=epsilon)

    try:
        inv_covariance = torch.inverse(covariance)
    except RuntimeError as e:
        raise ValueError("Covariance matrix inversion failed. Check input parameters.") from e

    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    # Create a coordinate grid.
    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)
    xy = torch.stack([ax_batch_expanded_x, ax_batch_expanded_y], dim=-1)

    # Compute the Gaussian kernel.
    z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) *
                             torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))
    # Normalize kernel
    kernel_max = kernel.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    kernel_normalized = kernel / kernel_max

    # Prepare kernel for multi-channel image.
    kernel_channels = kernel_normalized.repeat(1, channels, 1).view(batch_size * channels, kernel_size, kernel_size)
    kernel_channels = kernel_channels.unsqueeze(0).reshape(batch_size, channels, kernel_size, kernel_size)

    # Pad to desired image size.
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size must be smaller or equal to the image size.")
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    kernel_padded = F.pad(kernel_channels, padding, "constant", 0)

    # Prepare affine transformation grid using the provided coordinates.
    b, c, h, w = kernel_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_transformed = F.grid_sample(kernel_padded, grid, align_corners=True)

    # Apply colours (after reshaping) and sum channels.
    colours_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = colours_reshaped * kernel_transformed
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    return final_image.permute(1, 2, 0)  # shape: [H, W, channels]


def diffusion_sampler(eps_model: nn.Module,
                      n_T: int,
                      oneover_sqrta: torch.Tensor,
                      mab_over_sqrtmab: torch.Tensor,
                      sqrt_beta_t: torch.Tensor,
                      n_sample: int,
                      size,
                      device: str = "cuda"):
    """
    Reverse-sample from the diffusion model, returning a list of latent samples at each timestep.
    """
    with torch.no_grad():
        x = torch.randn(n_sample, *size, device=device)
        x_history = []
        for t in range(n_T, 0, -1):
            t_tensor = torch.full((n_sample,), t, device=device, dtype=torch.float32)
            eps = eps_model(x, t_tensor)
            z = torch.randn(n_sample, *size, device=device) if t > 1 else torch.zeros(n_sample, *size, device=device)
            x = oneover_sqrta[t] * (x - eps * mab_over_sqrtmab[t]) + sqrt_beta_t[t] * z
            x = x.detach()
            x_history.append(x.clone())
        return x_history


def process_latent_sample(latent, param_ranges, device, kernel_size=18, image_size=(32, 32)):
    """
    Convert a single latent sample (of shape [300, 9]) into an image tensor.
    """
    # Denormalize latent representation.
    latent_denorm = denormalize_parameters(latent, param_ranges)
    # Extract Gaussian parameters.
    sigma_x = torch.sigmoid(latent_denorm[:, 0])
    sigma_y = torch.sigmoid(latent_denorm[:, 1])
    rho = torch.tanh(latent_denorm[:, 2])
    alpha = torch.sigmoid(latent_denorm[:, 3])
    colours = torch.clamp(latent_denorm[:, 4:7], 0, 1)
    coords = latent_denorm[:, 7:9]

    final_image = generate_2D_gaussian_splatting(
        kernel_size=kernel_size,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        rho=rho,
        coords=coords,
        colours=colours,
        image_size=image_size,
        channels=3,
        device=device
    )
    # Permute to [C, H, W] for compatibility with torchvision.
    return final_image.permute(2, 0, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    #torch.cuda.empty_cache()

    # Prepare diffusion schedules.
    betas = (1e-4, 0.02)
    n_T = 200
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    
    # Load the pre-trained model.
    model_path = "/gpfs/workdir/coessenss/gsplat/models/sprites_MHA_64h_22_blocks_mlp_updated_200g/val_best_sprites_MHA_64h_22_blocks_200g_mlp_updated_e100_ts200_mse_std_scaled.pth"
    num_gaussians = 300
    feature_dim = 9
    time_emb_dim = 32
    num_blocks = 22
    num_heads = 64
    num_timestamps = n_T

    model = GaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = GaussianDatasetSprites(DATA_FOLDER)
    all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    param_ranges = [(all_data[:, :, i].min().item(), all_data[:, :, i].max().item())
                    for i in range(all_data.shape[2])]

    # Sample 100 latent representations.
    n_sample = 100
    latent_size = (300, 9)
    x_history = diffusion_sampler(
        eps_model=model,
        n_T=n_T,
        oneover_sqrta=schedules["oneover_sqrta"],
        mab_over_sqrtmab=schedules["mab_over_sqrtmab"],
        sqrt_beta_t=schedules["sqrt_beta_t"],
        n_sample=n_sample,
        size=latent_size,
        device=device
    )
    # Use only the final timestep (x₀).
    x_final = x_history[-1]

    # Create output folder at the specified path.
    output_folder = "/gpfs/workdir/coessenss/gsplat/models/sprites_MHA_64h_22_blocks_mlp_updated_200g/samples"
    os.makedirs(output_folder, exist_ok=True)

    # Process each latent sample into an image and save separately.
    for i in range(n_sample):
        latent = x_final[i]
        img_tensor = process_latent_sample(latent, param_ranges, device)
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        image_path = os.path.join(output_folder, f"sample_{i:03d}.png")
        img_pil.save(image_path)
        print(f"Saved image {i} to {image_path}")
