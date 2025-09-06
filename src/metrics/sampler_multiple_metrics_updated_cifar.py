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

# Custom modules
from src.ddpm import DDPM, ddpm_schedules
from src.models.transformer_model import GaussianTransformer
from src.utils.denormalize import denormalize_parameters
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from src.dataset import GaussianDatasetCIFAR10
from data.encoding_scripts.GaussianImage.GaussianImage.gaussianimage_rs import GaussianImage_RS


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


def render_latent_to_image(latent: torch.Tensor,
                           param_ranges: list,
                           device: torch.device,
                           H: int,
                           W: int,
                           BLOCK_H: int,
                           BLOCK_W: int,
                           num_points: int,
                           feature_dim: int,
                           lr: float = 1e-3,
                           denormalize: bool = True) -> torch.Tensor:
    """
    Renders an image from a latent vector using the GaussianImage_RS model.
    
    Expected latent format ([200, 8]):
      - Index 0: sigma_x
      - Index 1: sigma_y
      - Index 2: rho (rotation)
      - Indices 3-5: RGB features
      - Indices 6-7: x and y coordinates
      
    Args:
        latent (torch.Tensor): Latent vector.
        param_ranges (list): Parameter ranges for denormalization.
        device (torch.device): Computation device.
        H (int): Height of the output image.
        W (int): Width of the output image.
        BLOCK_H (int): Block height.
        BLOCK_W (int): Block width.
        num_points (int): Number of points for the Gaussian image.
        feature_dim (int): Feature dimensions.
        lr (float): Learning rate.
        denormalize (bool): Whether to apply denormalization (default True).
        
    Returns:
        A tensor image of shape (C, H, W) in the [0, 1] range.
    """
    # Optionally denormalize latent parameters.
    if denormalize:
        latent_processed = denormalize_parameters(latent.unsqueeze(0), param_ranges).squeeze(0)
    else:
        latent_processed = latent

    sigma_x = latent_processed[:, 0]
    sigma_y = latent_processed[:, 1]
    rho = latent_processed[:, 2]
    colours = latent_processed[:, 3:6]  # 3 channels for RGB
    coords = latent_processed[:, 6:8]

    # Create the GaussianImage_RS instance only once.
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
            quantize=False,  # Adjust if quantization was used during training
            feature_dim=3
        ).to(device)
        render_latent_to_image.gaussian_image_model.eval()

    model_instance = render_latent_to_image.gaussian_image_model

    with torch.no_grad():
        # Restore coordinates using the inverse of tanh.
        saved_xyz_tensor = torch.clamp(coords, -0.9999, 0.9999).to(device)
        model_instance._xyz.copy_(torch.atanh(saved_xyz_tensor))
        
        # Restore scaling parameters: stack sigma_x and sigma_y.
        saved_scaling_tensor = torch.stack([sigma_x, sigma_y], dim=1).to(device)
        model_instance._scaling.copy_(saved_scaling_tensor - model_instance.bound)
        
        # Restore rotation: inverse of sigmoid for rho (assumed in [0, 2π]).
        saved_rotation_tensor = rho.to(device)
        normalized = saved_rotation_tensor / (2 * math.pi)
        normalized = torch.clamp(normalized, 1e-6, 1 - 1e-6)
        inv_sigmoid = lambda y: torch.log(y / (1 - y))
        print("model_instance._rotation.shape:", model_instance._rotation.shape)
        print(inv_sigmoid(normalized).shape)
        model_instance._rotation.copy_(inv_sigmoid(normalized).unsqueeze(1))
        
        # Restore features: colours (RGB).
        saved_features_tensor = colours.to(device)
        print("model_instance._features.shape:", model_instance._features_dc.shape)
        print(saved_features_tensor.shape)
        model_instance._features_dc.copy_(saved_features_tensor)
        
        # Set opacity to all ones (no alpha channel).
        model_instance._opacity.copy_(torch.ones((model_instance.init_num_points, 1), device=device))
        
        # Render the image.
        output = model_instance.forward()
        render = output["render"]
        # Convert the render (assumed shape [1, C, H, W]) to a PIL image then to a tensor.
        render_image = to_pil_image(render.squeeze(0).cpu())
        render_tensor = transforms.ToTensor()(render_image)
    return render_tensor



def save_image_grid(images: list, grid_rows: int, grid_cols: int, save_path: str):
    """
    Saves a grid of images to a file.
    """
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2.5, grid_rows * 2.5))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            # Convert image tensor to numpy array.
            img_np = images[i].cpu().numpy()
            # If image is single-channel, display it in grayscale.
            cmap = 'gray' if (img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[0] == 1)) else None
            # If multi-channel, transpose from (C, H, W) to (H, W, C).
            if img_np.ndim == 3:
                img_np = img_np.transpose(1, 2, 0)
            ax.imshow(img_np, cmap=cmap)
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
    # DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/datasets/Sprites/sprites_results_combined.h5"
    DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/datasets/CIFAR10/cifar_first50k.h5"
    # model_path = "/gpfs/workdir/coessenss/gsplat/src/models/CIFAR10/CIFAR_MHA_64h_32blocks_mha_classic/val_best_CIFAR_MHA_64h_32blocks_mha_classic_e100_ts200_mse_std_scaled.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/src/models/Sprites/SPRITES_MHA_64h_10blocks_200ts_mha_classic/val_best_SPRITES_MHA_64h_10blocks_200ts_mha_classic_e100_ts200_mse_std_scaled.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/src/models/CIFAR10/CIFAR_MHA_64h_32blocks_mha_classic/last_CIFAR_MHA_64h_32blocks_mha_classic_e100_ts200_mse_std_scaled.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/src/models/CIFAR10/CIFAR_MHA_64h_32blocks_mha_classic_1000s/val_best_CIFAR_MHA_64h_32blocks_mha_classic_1000s_e100_ts50_mse_std_scaled.pth"
    model_path = "/gpfs/workdir/coessenss/gsplat/src/models/CIFAR10/CIFAR_MHA_64h_32blocks_mha_classic_1000s/val_best_CIFAR_MHA_64h_32blocks_mha_classic_1000s_e100_ts100_mse_std_scaled.pth"
    betas = (1e-4, 0.02)
    # n_T = 200
    # n_T = 50
    n_T = 100

    # Image generation parameters.
    H, W = 32, 32            # Output image size.
    BLOCK_H, BLOCK_W = 16, 16  # Fixed block parameters.
    lr = 1e-3

    # Prepare diffusion schedules.
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    print("Schedules keys:", list(schedules.keys()))

    # Load diffusion model (GaussianTransformer).
    # Use default parameters; latent dimensions will be updated from the dataset.
    num_gaussians = 200
    feature_dim = 8
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

    # Load dataset and determine latent dimensions automatically.
    print("Loading dataset...")
    dataset = GaussianDatasetCIFAR10(DATA_FOLDER)
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
    n_sample = 100
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
    # model_name = os.path.basename(os.path.dirname(model_path))
    model_name = f"CIFAR_MHA_64h_32blocks_{n_T}ts_classic"
    output_folder = os.path.join("Metrics", model_name)
    os.makedirs(output_folder, exist_ok=True)

    # Generate images from final latent samples and time the generation process.
    final_images = []
    start_time = time.time()
    for idx in range(n_sample):
        print(f"Processing generated sample {idx+1}...")
        latent = final_latents[idx]
        image_tensor = render_latent_to_image(
            latent, param_ranges, DEVICE, H, W, BLOCK_H, BLOCK_W,
            num_points=num_gaussians, feature_dim=3, lr=lr, denormalize=True
        )
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
    n_orig = 100  # Number of original samples for metrics.
    original_images = []
    for i in range(n_orig):
        latent = dataset[i]  # Each dataset item is a latent vector.
        img_tensor = render_latent_to_image(
            latent, param_ranges, DEVICE, H, W, BLOCK_H, BLOCK_W,
            num_points=num_gaussians, feature_dim=3, lr=lr, denormalize=False
        )
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
        img_rgb = img.repeat(3, 1, 1) if img.shape[0] == 1 else img
        img_resized = F.interpolate(img_rgb.unsqueeze(0), size=(299, 299),
                                    mode='bilinear', align_corners=False).squeeze(0)
        processed_originals.append(img_resized)

    for img in final_images:
        img_rgb = img.repeat(3, 1, 1) if img.shape[0] == 1 else img
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
