import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ddpm import DDPM, ddpm_schedules
from transformer_model import GaussianTransformer
# from transformers_model_sprites import GaussianTransformer
# from set_transformer_model import GaussianTransformer
# from transformer_pn_emb_model import GaussianTransformer
# from transformer_model_large import GaussianTransformer
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import os
from dataset import GaussianDatasetSprites
from utils.denormalize import denormalize_parameters


DATA_FOLDER = "Sprites/gaussian_representations_sprites_downsampled.h5"

# ================================================
# Function to convert gaussian splatting to image
# ================================================
def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho,
                                   coords, colours, image_size=(28, 28),
                                   channels=1, device="cuda"):
    batch_size = colours.shape[0]
    epsilon = 1e-6  # Small regularization term to ensure invertibility

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

    determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    if (determinant <= 0).any():
        print("Warning: Adjusting covariance matrix to ensure positive semi-definiteness.")
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

    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
    xy = torch.stack([xx, yy], dim=-1)

    z = torch.einsum(
        'b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy
    )
    kernel = (
        torch.exp(z) /
        (2 * torch.tensor(np.pi, device=device) *
         torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))
    )

    kernel_max = kernel.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    kernel_normalized = kernel / kernel_max

    kernel_reshaped = kernel_normalized.repeat(1, channels, 1).view(batch_size * channels, kernel_size, kernel_size)
    kernel_channels = kernel_reshaped.unsqueeze(0).reshape(batch_size, channels, kernel_size, kernel_size)

    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    padding = (
        pad_w // 2, pad_w // 2 + pad_w % 2,
        pad_h // 2, pad_h // 2 + pad_h % 2
    )

    kernel_padded = F.pad(kernel_channels, padding, "constant", 0)

    b, c, h, w = kernel_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_transformed = F.grid_sample(kernel_padded, grid, align_corners=True)

    colours_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = colours_reshaped * kernel_transformed

    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)  # Shape: [H, W, channels]

    return final_image

# ---------------------------------------------------------
# Diffusion Sampler Function (Updated)
# ---------------------------------------------------------
def diffusion_sampler(
    eps_model: nn.Module,
    n_T: int,
    oneover_sqrta: torch.Tensor,
    mab_over_sqrtmab: torch.Tensor,
    sqrt_beta_t: torch.Tensor,
    n_sample: int,
    size,
    device: str = "cuda"
):
    """
    Reverse-samples from the diffusion model and returns a list of latent samples at each timestep.
    
    Args:
        eps_model: The trained noise predictor.
        n_T: Total number of timesteps in the diffusion process.
        oneover_sqrta, mab_over_sqrtmab, sqrt_beta_t: Tensors from your schedules (length = n_T+1).
        n_sample: Number of samples to generate.
        size: Shape of the latent (Gaussian) representation (e.g. (70, 7)).
        device: "cuda" or "cpu".
        
    Returns:
        x_history (List[Tensor]): List of intermediate latent samples (including the final x₀).
    """
    with torch.no_grad():
        # Initialize with standard normal noise.
        x = torch.randn(n_sample, *size, device=device)
        x_history = []  # To store intermediate latent samples

        # Reverse diffusion: from t = n_T down to t = 1.
        for t in range(n_T, 0, -1):
            t_tensor = torch.full((n_sample,), t, device=device, dtype=torch.float32)
            eps = eps_model(x, t_tensor)
            z = torch.randn(n_sample, *size, device=device) if t > 1 else torch.zeros(n_sample, *size, device=device)
            x = oneover_sqrta[t] * (x - eps * mab_over_sqrtmab[t]) + sqrt_beta_t[t] * z
            x = x.detach()
            x_history.append(x.clone())
        return x_history

# ---------------------------------------------------------
# Main Sampling Routine (Updated)
# ---------------------------------------------------------
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Prepare Schedules (using n_T = 200 timesteps here)
    betas = (1e-4, 0.02)
    n_T = 200
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    print("Schedules keys:", list(schedules.keys()))
    
    # Folder containing multiple model files (update this path as needed)
    folder_path = "/gpfs/workdir/coessenss/gsplat/models/sprites_MHA_64h_downsampled"
    model_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pth')]
    print(f"Found {len(model_files)} model files.")

    # Define the data folder for the dataset (adjust DATA_FOLDER as needed)
    dataset = GaussianDatasetSprites(DATA_FOLDER)
    
    # Calculate min-max for each feature over the entire dataset
    all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    print("Shape of complete dataset:", all_data.shape)
    param_ranges = [(all_data[:, :, i].min().item(), all_data[:, :, i].max().item()) for i in range(all_data.shape[2])]
    print(f"Parameter ranges length: {len(param_ranges)}")
    print(f"Parameter ranges: {param_ranges}\n")
    
    # Common model parameters
    num_gaussians = 100
    feature_dim = 9
    time_emb_dim = 32
    num_blocks = 16
    num_heads = 64
    num_timestamps = n_T
    
    # Create output folder for saving grid images
    output_folder = f"sprites_MHA_64h_sampled_scaled_ts_{n_T}"
    os.makedirs(output_folder, exist_ok=True)
    
    # For each model file, sample 4 images and save them in a grid.
    for model_file in model_files:
        print(f"Processing model: {model_file}")
        # Initialize model architecture
        model = GaussianTransformer(
            input_dim=num_gaussians,
            time_emb_dim=time_emb_dim,
            feature_dim=feature_dim,
            num_timestamps=num_timestamps,
            num_transformer_blocks=num_blocks,
            num_heads=num_heads,
        ).to(DEVICE)
        
        checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Diffusion sampling: sample 4 images
        n_sample = 4
        size = (100, 9)
        x_history = diffusion_sampler(
            eps_model=model,
            n_T=n_T,
            oneover_sqrta=schedules["oneover_sqrta"],
            mab_over_sqrtmab=schedules["mab_over_sqrtmab"],
            sqrt_beta_t=schedules["sqrt_beta_t"],
            n_sample=n_sample,
            size=size,
            device=DEVICE
        )
        
        # Use only the final latent sample (x₀)
        final_latent = x_history[-1]  # Shape: [4, 100, 9]
        final_images = []
        
        for i in range(n_sample):
            # Process each latent sample individually.
            latent_sample = final_latent[i].unsqueeze(0)  # Add batch dimension.
            latent_denorm = denormalize_parameters(latent_sample, param_ranges)
            latent_denorm = latent_denorm.squeeze(0)  # Shape: [100, 9]
            
            # Extract parameters.
            sigma_x = torch.sigmoid(latent_denorm[:, 0])
            sigma_y = torch.sigmoid(latent_denorm[:, 1])
            rho = torch.tanh(latent_denorm[:, 2])
            alpha = torch.sigmoid(latent_denorm[:, 3])
            colours = torch.clamp(latent_denorm[:, 4:7], 0, 1)
            coords = latent_denorm[:, 7:9]
            colours = alpha.unsqueeze(1) * colours
            
            # Generate final image.
            final_image = generate_2D_gaussian_splatting(
                kernel_size=18,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                rho=rho,
                coords=coords,
                colours=colours,
                image_size=(32, 32),
                channels=3,
                device=DEVICE
            )
            # Rearrange to [C, H, W]
            final_image = final_image.permute(2, 0, 1)
            final_images.append(final_image)
            print(f"Processed sample {i+1} for model {os.path.basename(model_file)}")
        
        # Create a grid of the 4 final images (2x2 grid).
        grid_image = make_grid(final_images, nrow=2)
        transform = transforms.ToPILImage()
        grid_image_pil = transform(grid_image.cpu())
        
        # Save the grid image using the model filename as part of the output name.
        model_filename = os.path.basename(model_file).split('.')[0]
        output_path = os.path.join(output_folder, f"{model_filename}_grid.png")
        grid_image_pil.save(output_path)
        print(f"Saved grid image to {output_path}\n")
