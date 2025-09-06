import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import time
import wandb
from torch.nn import functional as F
from transformer_model import GaussianTransformer
# from transformer_model_large import GaussianTransformer
# from transformer_pn_emb_model import GaussianTransformer
from dataset import GaussianDataset
from ddpm import DDPM, ddpm_schedules
from utils.gaussian_to_image import generate_2D_gaussian_splatting
from utils.normalize import normalize_parameters
from utils.denormalize import denormalize_parameters
from skimage.metrics import structural_similarity as ssim

# ==============================
# Configurations
# ==============================
print("getting the wandb key")
# Set the API key as an environment variable
os.environ["WANDB_API_KEY"] = "d30c12b44a4235f945b74e0026b737863ffbb044"

# Log in to Weights & Biases
wandb.login()

DATA_FOLDER = "mnist_gaussian_representations/"
SAVE_PATH = 'models/comb_loss_reg_cov_mean/MHA_64h'
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4         # Try a lower learning rate
WEIGHT_DECAY = 1e-4          # Weight decay for AdamW
T = 200
SAVE_FILENAME = f'MHA_64h_old_param_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================
# Saving the model
# ==============================
def save_model(model, optimizer, epoch, loss, save_path=SAVE_PATH, filename=SAVE_FILENAME):
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_loss': loss,
    }
    torch.save(checkpoint, full_path)
    print(f"Model saved to {full_path} (Val Loss: {loss:.4f})")
    
# ==============================
# Definig the new loss
# ==============================    
# Revised reverse update function using DDPM schedule
def get_predicted_g_t_minus_1(g_t, predicted_noise, t, oneover_sqrta, mab_over_sqrtmab, sqrt_beta_t, noise=0):
    """
    Computes g_{t-1} from g_t using the DDPM reverse update:
      g_{t-1} = 1/sqrt(alpha_t) * [ g_t - predicted_noise * ((1 - alpha_t)/sqrt(1 - bar_alpha_t)) ] + sqrt(beta_t)*noise
    For a deterministic reverse update (e.g. for SSIM loss), use noise=0.
    """
    # return oneover_sqrta[t] * (g_t - predicted_noise * mab_over_sqrtmab[t]) + sqrt_beta_t[t] * noise
    # Unsqueeze to allow proper broadcasting: shape (batch,) -> (batch, 1, 1)
    scale = oneover_sqrta[t].view(-1, 1, 1)
    mab = mab_over_sqrtmab[t].view(-1, 1, 1)
    sqrt_beta = sqrt_beta_t[t].view(-1, 1, 1)
    return scale * (g_t - predicted_noise * mab) + sqrt_beta * noise

# Revised image generation: return tensor shape [batch, 1, H, W]
def compute_predicted_image_t_minus_1(predicted_g_t_minus_1, param_ranges, kernel_size=17, device="cuda"):
    """
    Generates predicted image at t-1 from latent predicted_g_t_minus_1.
    Denormalizes parameters and uses gaussian splatting to generate an image.
    Returns a tensor of shape [batch, 1, H, W].
    """
    predicted_g_t_minus_1 = denormalize_parameters(predicted_g_t_minus_1, param_ranges)
    batch_size, num_gaussians, _ = predicted_g_t_minus_1.shape
    final_images_list = []
    for i in range(batch_size):
        single_gaussian = predicted_g_t_minus_1[i]
        sigma_x = torch.sigmoid(single_gaussian[:, 0])
        sigma_y = torch.sigmoid(single_gaussian[:, 1])
        rho = torch.tanh(single_gaussian[:, 2])
        colours = torch.clamp(single_gaussian[:, 4:5], 0, 1)
        coords = single_gaussian[:, 5:7]
        
        # Generate image with shape [H, W, 1]
        img = generate_2D_gaussian_splatting(
            kernel_size,
            sigma_x,
            sigma_y,
            rho,
            coords,
            colours,
            image_size=(28, 28),
            channels=1,
            device=device
        )
        # Permute to [1, H, W]
        img = img.permute(2, 0, 1)
        final_images_list.append(img)
    final_image = torch.stack(final_images_list, dim=0)  # [batch, 1, H, W]
    return final_image

def combined_loss(real_g_t_minus_1, g_t, noise, predicted_noise, t, oneover_sqrta,
                  mab_over_sqrtmab, sqrt_beta_t, param_ranges, kernel_size=17,
                  device="cuda", alpha=0.5, reg_lambda=1e-4, lambda_cov=1e-3, lambda_center=1e-3):
    """
    Computes a combined loss consisting of:
      - L2 loss between true noise and predicted noise,
      - SSIM loss between the images generated from the real and predicted g_{t-1},
      - L2 regularization on the predicted noise,
      - Regularization that forces the Gaussians to be circular (σₓ and σ_y close to 1)
        and centered (x, y close to 0).
    
    Args:
        real_g_t_minus_1: ground-truth latent at t-1.
        g_t: latent at time t.
        noise: ground-truth noise at time t.
        predicted_noise: noise predicted by the model.
        t: current timestep (as an integer tensor) used for schedule indexing.
        oneover_sqrta, mab_over_sqrtmab, sqrt_beta_t: schedule tensors.
        param_ranges: parameter ranges for denormalization.
        kernel_size: kernel size for the Gaussian splatting.
        device: computation device.
        alpha: weight for the noise L2 loss (the SSIM loss gets weight 1 - alpha).
        reg_lambda: coefficient for L2 regularization on predicted_noise.
        lambda_cov: coefficient for forcing σₓ and σ_y to be near 1.
        lambda_center: coefficient for forcing the coordinates (x, y) to be near 0.
    
    Returns:
        A combined loss (scalar tensor).
    """
    # Compute predicted g_{t-1} using the deterministic reverse update (noise=0)
    predicted_g_t_minus_1 = get_predicted_g_t_minus_1(
        g_t, predicted_noise, t, oneover_sqrta, mab_over_sqrtmab, sqrt_beta_t, noise=0
    )
    
    # L2 loss between true noise and predicted noise
    l2_loss_noise = nn.MSELoss()(noise, predicted_noise)
    
    # Generate images at time t-1 from the real and predicted latent representations
    real_image_t_minus_1 = compute_predicted_image_t_minus_1(real_g_t_minus_1, param_ranges, kernel_size, device)
    pred_image_t_minus_1 = compute_predicted_image_t_minus_1(predicted_g_t_minus_1, param_ranges, kernel_size, device)
    
    # Compute SSIM loss (1 - SSIM value) over the batch.
    batch_size = real_image_t_minus_1.shape[0]
    ssim_total = 0
    for i in range(batch_size):
        real_np = real_image_t_minus_1[i, 0].detach().cpu().numpy()
        pred_np = pred_image_t_minus_1[i, 0].detach().cpu().numpy()
        data_range = pred_np.max() - pred_np.min()
        ssim_total += ssim(real_np, pred_np, data_range=data_range)
    ssim_loss = 1 - (ssim_total / batch_size)
    
    # L2 regularization on predicted_noise (Penalize large noise values)
    l2_reg = reg_lambda * torch.mean(predicted_noise ** 2)
    
    # Regularization for Gaussian parameters:
    # parameter order: [sigma_x, sigma_y, rho, alpha, colours, x, y]
    predicted_params = denormalize_parameters(predicted_g_t_minus_1, param_ranges)  # shape: [B, num_gaussians, features]
    sigma_x = predicted_params[:, :, 0]
    sigma_y = predicted_params[:, :, 1]
    coords = predicted_params[:, :, 5:7]  # x and y coordinates

    # Force sigma_x and sigma_y to be close to 1 (i.e. covariance close to 1)
    reg_cov = lambda_cov * torch.mean((sigma_x - 1) ** 2 + (sigma_y - 1) ** 2)
    
    # Force coordinates to be close to 0 (i.e. centers close to 0)
    reg_center = lambda_center * torch.mean(coords ** 2)
    
    # Combine the losses: weight alpha for noise L2 loss, (1 - alpha) for SSIM loss, plus regularizations
    combined_loss_val = alpha * l2_loss_noise + (1 - alpha) * ssim_loss + reg_cov + reg_center
    return combined_loss_val

# ==============================
# Model Training and Evaluation
# ==============================
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
    
def train_model(train_loader, val_loader, model, scheduler, optimizer, num_epochs, param_ranges):
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    # Create an instance of the DDPM class
    betas = (1e-4, 0.02)
    n_T = T
    ddpm = DDPM(betas=betas, n_T=n_T)
    schedules = ddpm_schedules(betas[0], betas[1], n_T)
    
    # Move schedule tensors to DEVICE (GPU)
    oneover_sqrta = schedules["oneover_sqrta"].to(DEVICE)
    mab_over_sqrtmab = schedules["mab_over_sqrtmab"].to(DEVICE)
    sqrt_beta_t = schedules["sqrt_beta_t"].to(DEVICE)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start time of epoch

        model.train()
        train_loss = 0
        for images in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            train_batch_size = images.shape[0]
            
            # Sample timesteps (keep integer version for schedule indexing)
            timesteps_int = torch.randint(low=1, high=n_T+1, size=(train_batch_size,), dtype=torch.long, device=DEVICE)
            timesteps_float = timesteps_int.float().to(DEVICE)
            
            gaussian_splat = images
            gaussian_splat_normalized = normalize_parameters(gaussian_splat, param_ranges)
            
            # Get noisy images and noise
            g_t, g_t_minus_1, noise = ddpm.get_noisy_images_and_noise(gaussian_splat_normalized, timesteps_int)
            g_t = g_t.to(DEVICE)
            
            # Forward pass without mixed precision
            predicted_noise = model(g_t, timesteps_float)
            
            loss = combined_loss(g_t_minus_1, g_t, noise, predicted_noise, timesteps_int,
                     oneover_sqrta, mab_over_sqrtmab, sqrt_beta_t, param_ranges,
                     kernel_size=17, device="cuda", alpha=0.5, reg_lambda=1e-4, lambda_cov=1e-3, lambda_center=1e-3)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop (without mixed precision)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                images = images.to(DEVICE, non_blocking=True)
                gaussian_splat = images
                gaussian_splat_normalized = normalize_parameters(gaussian_splat, param_ranges)
                batch_size_val = images.shape[0]

                # Sample timesteps (keep integer version for schedule indexing)
                timesteps_int = torch.randint(low=1, high=n_T+1, size=(batch_size_val,), dtype=torch.long, device=DEVICE)
                timesteps_float = timesteps_int.float().to(DEVICE)
                
                g_t, g_t_minus_1, noise = ddpm.get_noisy_images_and_noise(gaussian_splat_normalized, timesteps_int)
                g_t = g_t.to(DEVICE)
                
                predicted_noise = model(g_t, timesteps_float)
                
                loss = combined_loss(g_t_minus_1, g_t, noise, predicted_noise, timesteps_int,
                     oneover_sqrta, mab_over_sqrtmab, sqrt_beta_t, param_ranges,
                     kernel_size=17, device="cuda", alpha=0.5, reg_lambda=1e-4, lambda_cov=1e-3, lambda_center=1e-3)
                
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f}s")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch_time": epoch_duration})

        # Save the model only if the validation loss decreases
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, filename=f'val_best_MHA_64h_e{NUM_EPOCHS}_ts{T}.pth')
        if train_loss <= best_train_loss:   
            best_train_loss = train_loss
            save_model(model, optimizer, epoch, train_loss, save_path=SAVE_PATH, filename=f'train_best_MHA_64h_old_param_e{NUM_EPOCHS}_ts{T}.pth')
        save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, filename=f'last_MHA_64h_e{NUM_EPOCHS}_ts{T}.pth')
    
    return train_losses, val_losses

def main():
    wandb.init(project='Transformers_MHA_comb_loss', config={
        "learning_rate": LEARNING_RATE, 
        "epochs": NUM_EPOCHS, 
        "batch_size": BATCH_SIZE
    })
    
    print("Loading dataset...")
    dataset = GaussianDataset(DATA_FOLDER)
    
    # Here we use (-1, 1) for coordinates as desired.
    param_ranges = [
        (0, 1),    # sigma_x
        (0, 1),    # sigma_y
        (-1, 1),   # rho
        (-1, 1),    # alpha
        (0, 1),    # colours
        (-1, 1),   # x (coords)
        (-1, 1)    # y (coords)
    ]
    
    print("Splitting dataset...")
    num_samples = len(dataset)
    train_size = int(0.7 * num_samples)
    val_size = num_samples - train_size  # Using all remaining data for validation

    print("Creating data loaders...")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define model parameters
    num_gaussians = 70
    feature_dim = 7
    time_emb_dim = 32
    num_blocks = 16
    num_heads = 64
    num_timestamps = T

    model = GaussianTransformer(
        input_dim=num_gaussians,
        time_emb_dim=time_emb_dim,
        feature_dim=feature_dim,
        num_timestamps=num_timestamps,
        num_transformer_blocks=num_blocks,
        num_heads=num_heads,
    ).to(DEVICE)
    
    # # Enable multi-GPU training if multiple GPUs are available
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    print("Training model...")
    train_losses, val_losses = train_model(train_loader, val_loader, model, scheduler, optimizer, NUM_EPOCHS, param_ranges)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_PATH, f"MHA_64h_e{NUM_EPOCHS}_ts{T}.png"))
    plt.show()

if __name__ == "__main__":
    main()