import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb
from torch.nn import functional as F
from set_transformer_model import GaussianTransformer
from dataset import GaussianDataset
from ddpm import DDPM

# ==============================
# Configurations
# ==============================
print("getting the wandb key")
# Set the API key as an environment variable
os.environ["WANDB_API_KEY"] = "d30c12b44a4235f945b74e0026b737863ffbb044"

# Log in to Weights & Biases
wandb.login()

DATA_FOLDER = "mnist_gaussian_representations/"
SAVE_PATH = 'models/'
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4         # Try a lower learning rate
WEIGHT_DECAY = 1e-4          # Weight decay for AdamW
T = 1000
SAVE_FILENAME = f'SET_MHA_128h_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================
# Function to convert gaussian splatting to image
# (Same as before)
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
            W_normalized[..., i] = 0.0  # Using ... allows for more general shapes.
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
       
# ==============================
# Model Training and Evaluation
# ==============================

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# saving the best model
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
    
def train_model(train_loader, val_loader, model, scheduler, optimizer, num_epochs):
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # Initialize best loss to infinity
    
    # Define parameter ranges for normalization
    param_ranges = [
        (0, 1),   # sigma_x
        (0, 1),   # sigma_y
        (-1, 1),  # rho
        (0, 1),   # alpha
        (0, 1),   # colours
        (-1, 1),  # pixel_coords x
        (-1, 1)   # pixel_coords y
    ]
    
    # Create an instance of the DDPM class
    betas = (1e-4, 0.02)
    n_T = T
    ddpm = DDPM(betas=betas, n_T=n_T)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            current_batch_size = images.shape[0]
            
            # Sample timesteps for the current batch
            timesteps = torch.randint(low=1, high=n_T+1, size=(current_batch_size,), dtype=torch.long, device=DEVICE)
            gaussian_splat = images
            gaussian_splat_normalized = normalize_parameters(gaussian_splat, param_ranges)
            # gaussian_splat_normalized = gaussian_splat
            
            g_t, g_t_minus_1, noise = ddpm.get_noisy_images_and_noise(gaussian_splat_normalized, timesteps)
            g_t = g_t.to(DEVICE)
            timesteps = timesteps.float().to(DEVICE)
            predicted_noise = model(g_t, timesteps)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            
            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                images = images.to(DEVICE)
                gaussian_splat = images
                gaussian_splat_normalized = normalize_parameters(gaussian_splat, param_ranges)
                # gaussian_splat_normalized = gaussian_splat
                timesteps = torch.randint(low=1, high=n_T+1, size=(len(images),), dtype=torch.long, device=DEVICE)
                g_t, g_t_minus_1, noise = ddpm.get_noisy_images_and_noise(gaussian_splat_normalized, timesteps)
                g_t = g_t.to(DEVICE)
                timesteps = timesteps.float().to(DEVICE)
                predicted_noise = model(g_t, timesteps)
                loss = criterion(predicted_noise, noise)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        # Save the model only if the validation loss decreases
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss)
    
    return train_losses, val_losses

def main():
    wandb.init(project='SET_MHA', config={"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE})
    
    print("Loading dataset...")
    dataset = GaussianDataset(DATA_FOLDER)
    
    print("Splitting dataset...")
    num_samples = len(dataset)
    train_size = int(0.6 * num_samples)
    val_size = int(0.4 * num_samples)
    # test_size = num_samples - train_size - val_size

    print("Creating data loaders...")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_gaussians = 70
    feature_dim = 7    # Each Gaussian is represented by 7 features.
    time_emb_dim = 256  # Increased time embedding dimension for expressiveness.
    num_heads = 128
    num_mab_layers = 6
    num_isab_layers = 4
    num_sab_layers = 6
    model_dim = 1024   # Increased model (embedding) dimension.
    timesteps = T          # Total timesteps.
    dropout = 0.1
    num_inducing = 32  # More inducing points in ISAB.

    model = GaussianTransformer(feature_dim=feature_dim, time_emb_dim=time_emb_dim, num_heads=num_heads,
                                num_mab_layers=num_mab_layers, num_isab_layers=num_isab_layers,
                                num_sab_layers=num_sab_layers, model_dim=model_dim, T=timesteps, dropout=dropout,
                                num_inducing=num_inducing).to(DEVICE) 
    
    # Wrap model for multi-GPU using DataParallel
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    print("Training model...")
    train_losses, val_losses = train_model(train_loader, val_loader, model, scheduler, optimizer, NUM_EPOCHS)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"SET_MHA_128h_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.png")
    plt.show()

if __name__ == "__main__":
    main()