import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ddpm import DDPM, ddpm_schedules
from transformer_model import GaussianTransformer
from dataset import GaussianDataset
import torchvision.transforms as transforms
from torchvision.models import inception_v3 
from torchvision.models import Inception_V3_Weights
from PIL import Image
from scipy.linalg import sqrtm  # used internally by FID metric in ignite
from ignite.metrics import InceptionScore, FID  # Ignite metrics
from utils.normalize import normalize_parameters
from utils.denormalize import denormalize_parameters
from utils.gaussian_to_image import generate_2D_gaussian_splatting, generate_2D_gaussian_splatting_gray

###############################################
# Custom Inception model for FID feature extraction
###############################################
class InceptionV3ForFID(nn.Module):
    def __init__(self):
        super(InceptionV3ForFID, self).__init__()
        self.inception = inception_v3(aux_logits=True, transform_input=False, weights=Inception_V3_Weights.DEFAULT)
        self.inception.eval()

    def forward(self, x):
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        x = self.inception.avgpool(x)  # shape: [N, 2048, 1, 1]
        return x

###############################################
# Wrapper for InceptionV3 for FID (as recommended)
###############################################
class WrapperInceptionV3(nn.Module):
    def __init__(self, fid_incv3):
        super(WrapperInceptionV3, self).__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        return y[:, :, 0, 0]

###############################################
# Diffusion Sampler Function (Updated for batch_size = 32)
###############################################
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
    print(f"Starting diffusion sampling for {n_sample} images...")
    with torch.no_grad():
        x = torch.randn(n_sample, *size, device=device)
        x_history = []  # To store intermediate latent samples
        for t in range(n_T, 0, -1):
            t_tensor = torch.full((n_sample,), t, device=device, dtype=torch.float32)
            eps = eps_model(x, t_tensor)
            z = torch.randn(n_sample, *size, device=device) if t > 1 else torch.zeros(n_sample, *size, device=device)
            x = oneover_sqrta[t] * (x - eps * mab_over_sqrtmab[t]) + sqrt_beta_t[t] * z
            x = x.detach()
            x_history.append(x.clone())
            if t % 50 == 0:
                print(f"  Sampling timestep: {t}")
        print("Diffusion sampling completed for current batch.")
        return x_history

###############################################
# Preprocessing for Inception models (for both IS & FID)
###############################################
def fid_preprocess(img):
    img = F.interpolate(img.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False).squeeze(0)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return normalize(img)

###############################################
# Compute KID (Kernel Inception Distance)
###############################################
def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x @ y.T + coef0) ** degree

def compute_kid(features_real, features_fake, degree=3, gamma=None, coef0=1):
    K_rr = polynomial_kernel(features_real, features_real, degree, gamma, coef0)
    K_ff = polynomial_kernel(features_fake, features_fake, degree, gamma, coef0)
    K_rf = polynomial_kernel(features_real, features_fake, degree, gamma, coef0)
    
    n = features_real.shape[0]
    m = features_fake.shape[0]
    sum_rr = (K_rr.sum() - torch.diag(K_rr).sum()) / (n * (n - 1))
    sum_ff = (K_ff.sum() - torch.diag(K_ff).sum()) / (m * (m - 1))
    sum_rf = K_rf.mean()
    return sum_rr + sum_ff - 2 * sum_rf

###############################################
# Compute Precision and Recall for Generative Models
###############################################
def compute_precision_recall(features_real, features_fake, threshold_percentile=95):
    print("Computing Precision and Recall...")
    with torch.no_grad():
        dist_real = torch.cdist(features_real, features_real, p=2)
        diag_indices = torch.eye(dist_real.shape[0], device=dist_real.device).bool()
        dist_real[diag_indices] = float('inf')
        nn_distances = torch.min(dist_real, dim=1)[0]
        threshold = torch.quantile(nn_distances, threshold_percentile / 100.0).item()

        dist_fake_real = torch.cdist(features_fake, features_real, p=2)
        min_dist_fake = torch.min(dist_fake_real, dim=1)[0]
        precision = (min_dist_fake < threshold).float().mean().item()

        dist_real_fake = torch.cdist(features_real, features_fake, p=2)
        min_dist_real = torch.min(dist_real_fake, dim=1)[0]
        recall = (min_dist_real < threshold).float().mean().item()

    print("Precision and Recall computed.")
    return precision, recall

###############################################
# Main Sampling Routine
###############################################
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print("Device:", DEVICE)

    DATA_FOLDER = "mnist_gaussian_representations/"

    # Prepare Schedules
    betas = (1e-4, 0.02)
    n_T = 200
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    print("Schedules keys:", list(schedules.keys()))
    
    # Load the diffusion model
    print("Loading diffusion model...")
    # Load and instantiate the diffusion model (modify as needed)
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_64h_old_param/val_best_MHA_64h_old_parm_e100_ts200_mse_std_scaled.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/comb_loss/MHA_64h/val_best_MHA_64h_e100_ts200.pth"
    model_path = "/gpfs/workdir/coessenss/gsplat/models/comb_loss_reg/MHA_64h/val_best_MHA_64h_e100_ts200.pth"
    num_gaussians = 70
    feature_dim = 7
    time_emb_dim = 32
    num_blocks = 16
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
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()    
    print("Model loaded.")

    # Parameter ranges for denormalization.
    param_ranges = [
        (0, 1),    # sigma_x
        (0, 1),    # sigma_y
        (-1, 1),   # rho
        (0, 1),    # alpha
        (0, 1),    # colours
        (-1, 1),   # x (coords)
        (-1, 1)    # y (coords)
    ]
    
    # --- Set sample sizes ---
    n_sample_metric = 10000         # Total number of generated samples for metrics
    batch_size_sampling = 100        # Process 100 images at a time
    n_batches = math.ceil(n_sample_metric / batch_size_sampling)
    n_display = 100                 # Number of samples for visualization
    latent_size = (70, 7)

    print(f"Sampling diffusion process in {n_batches} batches (each of {batch_size_sampling} images)...")
    start_time = time.time()
    final_latents_list = []
    for b in tqdm(range(n_batches), desc="Sampling batches"):
        x_history_batch = diffusion_sampler(
            eps_model=model,
            n_T=n_T,
            oneover_sqrta=schedules["oneover_sqrta"],
            mab_over_sqrtmab=schedules["mab_over_sqrtmab"],
            sqrt_beta_t=schedules["sqrt_beta_t"],
            n_sample=batch_size_sampling,
            size=latent_size,
            device=DEVICE
        )
        final_latents_list.append(x_history_batch[-1])
        torch.cuda.empty_cache()  # Clear cache between batches
    final_latents = torch.cat(final_latents_list, dim=0)
    total_time = time.time() - start_time
    time_per_image = total_time / n_sample_metric
    print(f"Sampling complete: {n_sample_metric} images in {total_time:.2f} seconds.")
    print(f"Average time per image: {time_per_image:.4f} seconds.")

    # Generate images for visualization from the first n_display samples.
    print(f"Generating images for visualization from first {n_display} samples...")
    final_images = []
    for idx in tqdm(range(n_sample_metric), desc="Generating visualization images"):
        latent = final_latents[idx]
        latent_denorm = denormalize_parameters(latent.unsqueeze(0), param_ranges).squeeze(0)
        sigma_x = torch.sigmoid(latent_denorm[:, 0])
        sigma_y = torch.sigmoid(latent_denorm[:, 1])
        rho = torch.tanh(latent_denorm[:, 2])
        alpha = torch.sigmoid(latent_denorm[:, 3])
        colours = torch.clamp(latent_denorm[:, 4:5], 0, 1)
        coords = latent_denorm[:, 5:7]
        
        final_image = generate_2D_gaussian_splatting(
            kernel_size=17,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rho=rho,
            coords=coords,
            colours=colours,
            image_size=(28, 28),
            channels=1,
            device=DEVICE
        )
        final_images.append(final_image.permute(2, 0, 1).cpu())
    print("Visualization image generation complete.")

    # Save a grid of n_display images.
    print(f"Saving grid of {n_display} images for visualization...")
    grid_images = final_images[:n_display]
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i, ax in enumerate(axes.flatten()):
        img_np = grid_images[i].squeeze(0).numpy()  # (28, 28)
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")
    plt.tight_layout()
    output_folder = "comb_loss_reg_res/MHA_64h_old_param"
    os.makedirs(output_folder, exist_ok=True)
    grid_path = os.path.join(output_folder, f"grid2_ts_{num_timestamps}_samples_{n_display}.png")
    plt.savefig(grid_path)
    plt.show()
    print(f"Saved grid plot to {grid_path}")
    
    # ----------------------------------------
    # Compute metrics using Ignite
    # ----------------------------------------
    print("Loading original dataset for metric computation...")
    original_dataset = GaussianDataset(DATA_FOLDER)
    n_orig_metric = 10000  # Use 10,000 original samples
    original_images = []
    print("Processing original images...")
    for i in tqdm(range(n_orig_metric), desc="Processing original images"):
        latent = original_dataset[i]
        latent_denorm = denormalize_parameters(latent.unsqueeze(0), param_ranges).squeeze(0)
        sigma_x = torch.sigmoid(latent_denorm[:, 0])
        sigma_y = torch.sigmoid(latent_denorm[:, 1])
        rho = torch.tanh(latent_denorm[:, 2])
        alpha = torch.sigmoid(latent_denorm[:, 3])
        colours = torch.clamp(latent_denorm[:, 4:5], 0, 1)
        coords = latent_denorm[:, 5:7]
        orig_image = generate_2D_gaussian_splatting(
            kernel_size=17,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rho=rho,
            coords=coords,
            colours=colours,
            image_size=(28, 28),
            channels=1,
            device=DEVICE
        )
        original_images.append(orig_image.permute(2, 0, 1).cpu())
    print("Original images processed.")

    print("Preprocessing images for metrics (IS, FID, KID)...")
    fid_generated = [fid_preprocess(img) for img in final_images]
    fid_original = [fid_preprocess(img) for img in original_images]
    
    batch_size = 32

    # --- Compute Inception Score ---
    print("Computing Inception Score (IS)...")
    is_metric = InceptionScore(device=DEVICE)
    for i in tqdm(range(0, len(fid_generated), batch_size), desc="Computing IS"):
        batch = torch.stack(fid_generated[i:i+batch_size]).to(DEVICE)
        is_metric.update(batch)
    is_value = is_metric.compute()
    print("Inception Score computed.")

    # --- Compute FID ---
    print("Computing FID...")
    dims = 2048
    fid_incv3_model = InceptionV3ForFID().to(DEVICE)
    wrapper_model = WrapperInceptionV3(fid_incv3_model)
    wrapper_model.eval()
    
    fid_metric = FID(num_features=dims, feature_extractor=wrapper_model, device=DEVICE)
    for i in tqdm(range(0, len(fid_generated), batch_size), desc="Computing FID"):
        batch_fake = torch.stack(fid_generated[i:i+batch_size]).to(DEVICE)
        batch_real = torch.stack(fid_original[i:i+batch_size]).to(DEVICE)
        fid_metric.update((batch_fake, batch_real))
    fid_value = fid_metric.compute()
    print("FID computed.")

    # --- Compute KID ---
    print("Computing KID...")
    def get_inception_features(images, feature_extractor, device, batch_size=32):
        features = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i+batch_size]).to(device)
            with torch.no_grad():
                feat = feature_extractor(batch)
            features.append(feat)
        return torch.cat(features, dim=0)
    
    features_real = get_inception_features(fid_original, wrapper_model, DEVICE, batch_size=32)
    features_fake = get_inception_features(fid_generated, wrapper_model, DEVICE, batch_size=32)
    kid_value = compute_kid(features_real, features_fake)
    print("KID computed.")

    # --- Compute Precision and Recall ---
    precision, recall = compute_precision_recall(features_real, features_fake, threshold_percentile=95)
    
    # Format and save the metrics dictionary.
    metrics = {
        "FID": float(fid_value) if torch.is_tensor(fid_value) else fid_value,
        "IS": float(is_value) if torch.is_tensor(is_value) else is_value,
        "KID": float(kid_value) if torch.is_tensor(kid_value) else kid_value,
        "Precision": float(precision) if torch.is_tensor(precision) else precision,
        "Recall": float(recall) if torch.is_tensor(recall) else recall,
        "Total_time": float(total_time),
        "Time_per_image": float(time_per_image)
    }
    
    metrics_path = os.path.join(output_folder, "metrics.txt")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics computed and saved to", metrics_path)
    print("Computed Metrics:")
    print(json.dumps(metrics, indent=4))
    print("All computations complete.")