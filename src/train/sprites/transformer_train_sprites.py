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
from transformer_model import GaussianTransformer
# from transformers_model_sprites import GaussianTransformer
# from transformer_model_large import GaussianTransformer
# from transformer_pn_emb_model import GaussianTransformer
from dataset import GaussianDatasetSpritesBigger
from ddpm import DDPM
from utils.normalize import normalize_parameters

# ==============================
# Configurations
# ==============================
print("getting the wandb key")
# Set the API key as an environment variable
os.environ["WANDB_API_KEY"] = "d30c12b44a4235f945b74e0026b737863ffbb044"

# Log in to Weights & Biases
wandb.login()

# DATA_FOLDER = "Sprites/gaussian_representations_sprites_downsampled.h5"
DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/Sprites/gaussian_representations_sprites_downsampled_200g.h5"
SAVE_PATH = 'models/sprites_MHA_64h_22_blocks_mlp_updated_200g'
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4         # Try a lower learning rate
WEIGHT_DECAY = 1e-4          # Weight decay for AdamW
T = 200
SAVE_FILENAME = f'sprites_MHA_64h_downsampled_new_param_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    
def train_model(train_loader, val_loader, model, scheduler, optimizer, num_epochs, param_ranges):
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # Initialize best loss to infinity
    best_train_loss = float('inf')
    intermediate_model_loss = float('inf')
    
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
            save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, filename=f'val_best_sprites_MHA_64h_22_blocks_200g_mlp_updated_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth')
        # save the model only if the training loss decreases
        if train_loss <= best_train_loss:
            best_train_loss = train_loss
            save_model(model, optimizer, epoch, train_loss, save_path=SAVE_PATH, filename=f'train_best_sprites_MHA_64h_22_blocks_200g_mlp_updated_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth')
        # Save intermediate model every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1 :
            intermediate_model_loss = val_loss
            save_model(model, optimizer, epoch, intermediate_model_loss, save_path=SAVE_PATH, filename=f'intermediate_e{epoch+1}_MHA_64h_22_blocks_200g_mlp_updated.pth')
        
        save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, filename=f'last_sprites_MHA_64h_22_blocks_200g_mlp_updated_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth')
    
    return train_losses, val_losses

def main():
    wandb.init(project='Sprites_Transformers_MHA', config={"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE})
    
    print("Loading dataset...")
    dataset = GaussianDatasetSpritesBigger(DATA_FOLDER)
    # dataset = torch.utils.data.Subset(dataset, range(1000))

    # Calculate min-max for each feature over the entire dataset
    all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    print("shape of complete dataset: ", all_data.shape)
    param_ranges = [(all_data[:, :, i].min().item(), all_data[:, :, i].max().item()) for i in range(all_data.shape[2])]
    print(f"Parameter ranges length: {len(param_ranges)}")
    print( f"Parameter ranges: {param_ranges}\n")
    
    # # Here we use (-1, 1) for coordinates as desired.
    # param_ranges = [
    #     (0, 1),    # sigma_x
    #     (0, 1),    # sigma_y
    #     (-1, 1),   # rho
    #     (0, 1),    # alpha
    #     (0, 1),    # colours(R)
    #     (0, 1),    # colours(G)
    #     (0, 1),    # colours(B)
    #     (-1, 1),   # x (coords)
    #     (-1, 1)    # y (coords)
    # ]
    
    print("Splitting dataset...")
    num_samples = len(dataset)
    print(f"Number of samples: {num_samples}")
    train_size = int(0.7 * num_samples)
    val_size = num_samples - train_size  # Ensure total sum equals num_samples

    print("Creating data loaders...")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("shape of sprites gaussian: ", train_dataset[0].shape)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # num_gaussians = 70
    # feature_dim = 7   # Each Gaussian is represented by 7 features.
    # time_emb_dim = 32
    # num_blocks = 16
    # num_heads = 64
    # num_timesteps = T  # Total timesteps

    # model = GaussianTransformer(
    #     input_dim=num_gaussians,  # not used directly here
    #     time_emb_dim=time_emb_dim,
    #     feature_dim=feature_dim,
    #     num_transformer_blocks=num_blocks,
    #     num_heads=num_heads,
    #     num_timesteps=num_timesteps
    # ).to(DEVICE)
    
    # num_gaussians = 100
    num_gaussians = 300
    feature_dim = 9
    time_emb_dim = 32
    num_blocks = 22
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
    
    # # Wrap model for multi-GPU using DataParallel
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
    plt.savefig(os.path.join(SAVE_PATH, f"sprites_MHA_64h_mlp_updated_200g_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.png"))
    plt.show()

if __name__ == "__main__":
    main()