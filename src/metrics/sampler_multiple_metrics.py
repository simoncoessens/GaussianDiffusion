import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ddpm import DDPM, ddpm_schedules
from src.models.transformer_model import GaussianTransformer
from dataset import GaussianDataset
import torchvision.transforms as transforms
from torchvision.models import inception_v3  # used to load inception_v3
from PIL import Image
from scipy.linalg import sqrtm  # used internally by FID metric in ignite
# from ignite.metrics import InceptionScore, FID  # Ignite metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from utils.normalize import normalize_parameters
from utils.denormalize import denormalize_parameters
from utils.gaussian_to_image import generate_2D_gaussian_splatting, generate_2D_gaussian_splatting_gray

###############################################
# Custom Inception model for FID feature extraction
###############################################
class InceptionV3ForFID(nn.Module):
    def __init__(self):
        super(InceptionV3ForFID, self).__init__()
        # Load the inception_v3 model with aux_logits=True (as in your original code) and no transform_input.
        self.inception = inception_v3(pretrained=True, aux_logits=True, transform_input=False)
        self.inception.eval()

    def forward(self, x):
        # x is expected to be [N, 3, 299, 299]
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
        # fid_incv3 returns a tensor of shape [N, 2048, 1, 1]
        y = self.fid_incv3(x)
        # Squeeze spatial dimensions to get shape [N, 2048]
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
    """
    Given an image tensor of shape [C, H, W] in [0, 1], resize to 299x299,
    replicate channel if necessary, and normalize.
    """
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
    # x: [N, d], y: [M, d]
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x @ y.T + coef0) ** degree

def compute_kid(features_real, features_fake, degree=3, gamma=None, coef0=1):
    # features_real: [N, d]; features_fake: [M, d]
    K_rr = polynomial_kernel(features_real, features_real, degree, gamma, coef0)
    K_ff = polynomial_kernel(features_fake, features_fake, degree, gamma, coef0)
    K_rf = polynomial_kernel(features_real, features_fake, degree, gamma, coef0)
    
    n = features_real.shape[0]
    m = features_fake.shape[0]
    # Use unbiased estimator: remove diagonal for same-set kernels.
    sum_rr = (K_rr.sum() - torch.diag(K_rr).sum()) / (n * (n - 1))
    sum_ff = (K_ff.sum() - torch.diag(K_ff).sum()) / (m * (m - 1))
    sum_rf = K_rf.mean()
    kid = sum_rr + sum_ff - 2 * sum_rf
    return kid.item()

###############################################
# Main Sampling Routine
###############################################
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    DATA_FOLDER = "mnist_gaussian_representations/"

    # Prepare Schedules
    betas = (1e-4, 0.02)
    n_T = 200
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    print("Schedules keys:", list(schedules.keys()))
    
    # Load and instantiate the diffusion model (modify as needed)
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_64h_old_param/val_best_MHA_64h_old_parm_e100_ts200_mse_std_scaled.pth"
    model_path = "/gpfs/workdir/coessenss/gsplat/models/comb_loss_reg/MHA_64h/val_best_MHA_64h_e100_ts200.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/comb_loss/MHA_64h/val_best_MHA_64h_e100_ts200.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/comb_loss_reg_cov_mean/MHA_64h/val_best_MHA_64h_e100_ts200.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/comb_loss_reg_noise/MHA_64h/last_MHA_64h_e100_ts200.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_64h_new_param/val_best_MHA_64h_new_parm_e100_ts200_mse_std_scaled.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_64h_old_param_22_blocks/val_best_MHA_64h_new_parm_e100_ts200_mse_std_scaled.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_128h_old_param/val_best_MHA_128h_old_param_e100_ts200_mse_std_scaled.pth"
    # model_path = "/gpfs/workdir/coessenss/gsplat/models/MHA_256h_old_param/val_best_MHA_256h_old_param_e100_ts200_mse_std_scaled.pth"
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
    
    # # Calculate min-max for each feature over the entire dataset
    # all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    # print("shape of complete dataset: ", all_data.shape)
    # param_ranges = [(all_data[:, :, i].min().item(), all_data[:, :, i].max().item()) for i in range(all_data.shape[2])]
    # print(f"Parameter ranges length: {len(param_ranges)}")
    # print( f"Parameter ranges: {param_ranges}\n") 
    
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
    
    # Sampling latent representations for 100 images (sampled all together).
    n_sample = 1000
    latent_size = (70, 7)  # Shape of the Gaussian representation
    x_history = diffusion_sampler(
        eps_model=model,
        n_T=n_T,
        oneover_sqrta=schedules["oneover_sqrta"],
        mab_over_sqrtmab=schedules["mab_over_sqrtmab"],
        sqrt_beta_t=schedules["sqrt_beta_t"],
        n_sample=n_sample,
        size=latent_size,
        device=DEVICE
    )
    
    final_latents = x_history[-1]  # x₀ for each sample
    
    output_folder = f"Metrics/updated/comb_loss_reg_MHA_64h_1000_samples"
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate images from the final latent samples and time the generation process.
    final_images = []
    start_time = time.time()
    for idx in range(n_sample):
        print(f"Processing generated sample {idx+1}...")
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
        # Permute to [C, H, W] and append to list
        final_images.append(final_image.permute(2, 0, 1).cpu())
    end_time = time.time()
    avg_time = (end_time - start_time) / n_sample
    print(f"Average generation time per image: {avg_time:.4f} seconds")
    
    # Display a grid of generated images.
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i, ax in enumerate(axes.flatten()):
        img_np = final_images[i].squeeze(0).numpy()  # shape: (28, 28)
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")
    plt.tight_layout()
    grid_path = os.path.join(output_folder, f"grid_ts_{num_timestamps}_samples_{n_sample}.png")
    plt.savefig(grid_path)
    plt.show()
    print(f"Saved grid plot to {grid_path}")
    
    # ----------------------------------------
    # Compute Inception Score and FID Metrics using Ignite
    # ----------------------------------------
    # Load original data from dataset and convert to images.
    original_dataset = GaussianDataset(DATA_FOLDER)
    n_orig = 1000  # number of original samples
    original_images = []
    for i in range(n_orig):
        latent_denorm = original_dataset[i]  # shape: (70, 7)
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
    
    batch_size = 32

    # --- Setup Ignite Inception Score ---
    is_metric = InceptionScore(device=DEVICE)
    for i in range(0, len(fid_generated), batch_size):
        batch = torch.stack(fid_generated[i:i+batch_size]).to(DEVICE)
        is_metric.update(batch)
    is_value = is_metric.compute()  # returns the inception score

    # --- Setup Ignite FID ---
    dims = 2048  # feature dimension from the final pooling layer
    fid_incv3_model = InceptionV3ForFID().to(DEVICE)
    wrapper_model = WrapperInceptionV3(fid_incv3_model)
    wrapper_model.eval()
    
    fid_metric = FID(num_features=dims, feature_extractor=wrapper_model, device=DEVICE)
    for i in range(0, len(fid_generated), batch_size):
        batch_fake = torch.stack(fid_generated[i:i+batch_size]).to(DEVICE)
        batch_real = torch.stack(fid_original[i:i+batch_size]).to(DEVICE)
        fid_metric.update((batch_fake, batch_real))
    fid_value = fid_metric.compute()
    
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
    
    # Save all computed metrics with keys: fid, is, kis, time.
    metrics = {
        "fid": fid_value,
        "is": is_value,
        "kid": kid_value,
        "time": avg_time
    }
    metrics_path = os.path.join(output_folder, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(str(metrics))
    print("Metrics computed and saved to", metrics_path)
    print("Computed Metrics:", metrics)