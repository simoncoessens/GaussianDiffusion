import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ddpm import DDPM, ddpm_schedules
from transformer_model import GaussianTransformer
# from set_transformer_model import GaussianTransformer
# from transformer_pn_emb_model import GaussianTransformer
#from transformer_model_large import GaussianTransformer
import torchvision.transforms as transforms
from PIL import Image
import os


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

##############################################
# Normalization and Denormalization Functions
##############################################
def normalize_parameters(W, param_ranges):
    """Normalizes parameters to the range [-1, 1]."""
    W_normalized = torch.zeros_like(W)
    for i in range(W.shape[-1]):
        min_val, max_val = param_ranges[i]
        if min_val == max_val:
            W_normalized[..., i] = 0.0
        else:
            W_normalized[..., i] = 2 * (W[..., i] - min_val) / (max_val - min_val) - 1
    return W_normalized

def denormalize_parameters(W_normalized, param_ranges):
    """Denormalizes parameters from [-1, 1] back to the original range."""
    W_denormalized = torch.zeros_like(W_normalized)
    for i in range(W_normalized.shape[-1]):
        min_val, max_val = param_ranges[i]
        if min_val == max_val:
            W_denormalized[..., i] = min_val
        else:
            denorm_param = (W_normalized[..., i] + 1) / 2 * (max_val - min_val) + min_val
            W_denormalized[..., i] = torch.clamp(denorm_param, min_val, max_val)
    return W_denormalized

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
            # Create a timestep tensor with value t (consistent with training).
            t_tensor = torch.full((n_sample,), t, device=device, dtype=torch.float32)
            # Predict the noise epsilon using the model.
            eps = eps_model(x, t_tensor)
            # For t > 1, sample Gaussian noise; for t == 1, use zero noise.
            z = torch.randn(n_sample, *size, device=device) if t > 1 else torch.zeros(n_sample, *size, device=device)
            # Reverse update:
            x = oneover_sqrta[t] * (x - eps * mab_over_sqrtmab[t]) + sqrt_beta_t[t] * z
            x = x.detach()  # Detach from the computation graph.
            x_history.append(x.clone())

        # Return the full list of latent samples (from t = n_T to t = 1, where x_history[-1] is x₀)
        return x_history

# ---------------------------------------------------------
# Main Sampling Routine
# ---------------------------------------------------------
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Prepare Schedules (using n_T = 1000 timesteps, for example)
    betas = (1e-4, 0.02)
    n_T = 1000
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    print("Schedules keys:", list(schedules.keys()))
    
    # Load Model
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_64h_pn_e20_ts1000_mse_std_scaled.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_64h_e100_ts200_mse_std_scaled.pth"
    #model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_large_256h_e100_ts200_mse_std_scaled.pth"
    model_path = "/gpfs/workdir/coessenss/gsplat/models/sprites_MHA_64h_e20_ts1000_mse_std_scaled.pth"
    # num_gaussians = 70
    # feature_dim = 7
    # time_emb_dim = 32
    # num_blocks = 12  
    # num_heads = 32  

    # model = GaussianTransformer(
    #     input_dim=num_gaussians,
    #     time_emb_dim=time_emb_dim,
    #     feature_dim=feature_dim,
    #     num_transformer_blocks=num_blocks,
    #     num_heads=num_heads
    # ).to(DEVICE)
    
    num_gaussians = 500
    feature_dim = 9
    time_emb_dim = 512
    num_blocks = 5
    num_heads = 64
    num_timestamps = n_T

    model = GaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    ).to(DEVICE)
    
    # feature_dim = 7    # Each Gaussian is represented by 7 features.
    # time_emb_dim = 32  # Hidden dimension for time embedding.
    # num_heads = 32

    # # Specify the number of repeated blocks and model dimension.
    # num_mab_layers = 2
    # num_isab_layers = 1
    # num_sab_layers = 2
    # model_dim = 512 
    # dropout = 0.1

    # model = GaussianTransformer(feature_dim=feature_dim, time_emb_dim=time_emb_dim, num_heads=num_heads,
    #                             num_mab_layers=num_mab_layers, num_isab_layers=num_isab_layers,
    #                             num_sab_layers=num_sab_layers, model_dim=model_dim, T=n_T, dropout=dropout).to(DEVICE) 
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()    
    # For sampling, set the parameter ranges.
    # Here we use (-1, 1) for coordinates as desired.
    dataset = GaussianDatasetSprites(DATA_FOLDER)
    # dataset = torch.utils.data.Subset(dataset, range(1000))

    # Calculate min-max for each feature over the entire dataset
    all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    param_ranges = [(all_data[:, i].min().item(), all_data[:, i].max().item()) for i in range(all_data.shape[1])]
    
    # Sample latent representations from t = n_T down to t = 1.
    # x_history is a list of latent samples (each of shape: (n_sample, 70, 7))
    n_sample = 1   # Number of samples
    size = (500, 9)  # Shape of the Gaussian representation
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
    
    # Create folder for saving images
    # output_folder = "MHA_32h_sampled_scaled_ts_{n_T}"
    output_folder = f"sprites_MHA_large_64h_sampled_scaled_ts_{n_T}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each latent sample to generate and save an image.
    # Each latent sample in x_history is processed independently.
    for idx, latent in enumerate(x_history):
        # Denormalize latent representation if needed.
        latent_denorm = denormalize_parameters(latent, param_ranges)
        # Remove the sample dimension (n_sample == 1)
        latent_denorm = latent_denorm.squeeze(0)
        print(f"Latent sample {idx+1} shape: {latent_denorm.shape}")
        
        # Extract parameters using appropriate activation functions.
        sigma_x = torch.sigmoid(latent_denorm[:, 0])
        sigma_y = torch.sigmoid(latent_denorm[:, 1])
        rho = torch.tanh(latent_denorm[:, 2])
        alpha = torch.sigmoid(latent_denorm[:, 3])
        colours = torch.clamp(latent_denorm[:, 4:5], 0, 1)
        coords = latent_denorm[:, 5:7]
        
        # Generate the final image from the latent Gaussian parameters.
        final_image = generate_2D_gaussian_splatting(
            kernel_size=25,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rho=rho,
            coords=coords,
            colours=colours,
            image_size=(64, 64),
            channels=3,
            device=DEVICE
        )
        final_image = final_image.permute(2, 0, 1).to(DEVICE)
        print(f"Final image {idx+1} shape: {final_image.shape}")
        
        # Save the image.
        transform = transforms.ToPILImage()
        final_image_pil = transform(final_image.cpu())
        image_path = os.path.join(output_folder, f"image_{idx+1}.png")
        final_image_pil.save(image_path)
        print(f"Saved image {idx+1} to {image_path}")