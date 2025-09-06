import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ddpm import DDPM, ddpm_schedules
from transformer_model import GaussianTransformer
from dataset import GaussianDataset
import torchvision.transforms as transforms
from torchvision.models import inception_v3  # used to load inception_v3
from PIL import Image
from scipy.linalg import sqrtm  # used internally by FID metric in ignite
from ignite.metrics import InceptionScore, FID  # Ignite metrics
from utils.normalize import normalize_parameters
from utils.denormalize import denormalize_parameters
from utils.gaussian_to_image import generate_2D_gaussian_splatting, generate_2D_gaussian_splatting_gray
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

###############################################
# Custom Inception model for FID feature extraction
###############################################
class InceptionV3ForFID(nn.Module):
    def __init__(self):
        super(InceptionV3ForFID, self).__init__()
        self.inception = inception_v3(pretrained=True, aux_logits=True, transform_input=False)
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
        x = self.inception.avgpool(x)
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
# Diffusion Sampler Function (Updated)
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
    """
    Reverse-samples from the diffusion model and returns a list of latent samples at each timestep.
    """
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
    img = normalize(img)
    return img

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
    kid = sum_rr + sum_ff - 2 * sum_rf
    return kid.item()

###############################################
# Main Sampling Routine (Modified for 10,000 images)
###############################################
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    DATA_FOLDER = "mnist_gaussian_representations/"

    # Prepare Schedules
    betas = (1e-4, 0.02)
    n_T = 200
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    logging.info("Schedules keys: %s", list(schedules.keys()))
    
    # Load and instantiate the diffusion model (modify as needed)
    model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_64h_old_param_22_blocks/val_best_MHA_64h_new_parm_e100_ts200_mse_std_scaled.pth"
    num_gaussians = 70
    feature_dim = 7
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
    ).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()   
    
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
    
    # Sampling latent representations for 10,000 images in 100 batches of 100 images.
    total_batches = 100
    batch_size = 100
    latent_size = (70, 7)  # Shape of the Gaussian representation
    final_images = []
    logging.info("Starting generation of 10,000 images in %d batches...", total_batches)
    
    for batch_idx in tqdm(range(total_batches), desc="Generating images"):
        x_history = diffusion_sampler(
            eps_model=model,
            n_T=n_T,
            oneover_sqrta=schedules["oneover_sqrta"],
            mab_over_sqrtmab=schedules["mab_over_sqrtmab"],
            sqrt_beta_t=schedules["sqrt_beta_t"],
            n_sample=batch_size,
            size=latent_size,
            device=DEVICE
        )
        final_latents = x_history[-1]
        # Process each latent in the batch
        for idx in range(batch_size):
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
        logging.info("Completed batch %d/%d", batch_idx+1, total_batches)
    
    end_time = time.time()
    total_generation_time = end_time  # using entire time elapsed for generation
    avg_time = total_generation_time / (total_batches * batch_size)
    logging.info("Average generation time per image: %.4f seconds", avg_time)
    
    # Display a grid of the first 100 generated images.
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i, ax in enumerate(axes.flatten()):
        img_np = final_images[i].squeeze(0).numpy()  # shape: (28, 28)
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")
    plt.tight_layout()
    output_folder = f"Metrics/MHA_64h_22_blocks_10000_samples"
    os.makedirs(output_folder, exist_ok=True)
    grid_path = os.path.join(output_folder, f"grid_ts_{num_timestamps}_samples_{total_batches*batch_size}.png")
    plt.savefig(grid_path)
    plt.show()
    logging.info("Saved grid plot to %s", grid_path)
    
    # ----------------------------------------
    # Compute Inception Score and FID Metrics using Ignite
    # ----------------------------------------
    # Load original data from dataset and convert to images.
    original_dataset = GaussianDataset(DATA_FOLDER)
    n_orig = total_batches * batch_size  # 10000 original samples
    original_images = []
    logging.info("Processing %d original images...", n_orig)
    for i in tqdm(range(n_orig), desc="Processing original images"):
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
    
    # Preprocess images for metrics (resize to 299x299, replicate channels if needed, normalize)
    fid_generated = [fid_preprocess(img) for img in final_images]
    fid_original = [fid_preprocess(img) for img in original_images]
    
    batch_size_metric = 32

    # --- Setup Ignite Inception Score ---
    is_metric = InceptionScore(device=DEVICE)
    for i in range(0, len(fid_generated), batch_size_metric):
        batch = torch.stack(fid_generated[i:i+batch_size_metric]).to(DEVICE)
        is_metric.update(batch)
    is_value = is_metric.compute()  # returns the inception score
    logging.info("Inception Score: %s", is_value)

    # --- Setup Ignite FID ---
    dims = 2048  # feature dimension
    fid_incv3_model = InceptionV3ForFID().to(DEVICE)
    wrapper_model = WrapperInceptionV3(fid_incv3_model)
    wrapper_model.eval()
    
    fid_metric = FID(num_features=dims, feature_extractor=wrapper_model, device=DEVICE)
    for i in range(0, len(fid_generated), batch_size_metric):
        batch_fake = torch.stack(fid_generated[i:i+batch_size_metric]).to(DEVICE)
        batch_real = torch.stack(fid_original[i:i+batch_size_metric]).to(DEVICE)
        fid_metric.update((batch_fake, batch_real))
    fid_value = fid_metric.compute()
    logging.info("FID: %s", fid_value)
    
    # --- Compute KID ---
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
    logging.info("KID: %s", kid_value)
    
    # Save all computed metrics
    metrics = {
        "fid": fid_value,
        "is": is_value,
        "kid": kid_value,
        "time": avg_time
    }
    metrics_path = os.path.join(output_folder, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(str(metrics))
    logging.info("Metrics computed and saved to %s", metrics_path)
    logging.info("Computed Metrics: %s", metrics)