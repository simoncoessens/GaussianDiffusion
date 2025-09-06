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
from models.flash_transformer_model import GaussianTransformer
from dataset import GaussianDataset
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

DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/datasets/MNIST/mnist_gaussian_representations/"
SAVE_PATH = '/gpfs/workdir/coessenss/gsplat/models/GaussianImage/flash_MHA_64h_32_blocks/'
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4         # Try a lower learning rate
WEIGHT_DECAY = 1e-4          # Weight decay for AdamW
T = 200
SAVE_FILENAME = f'flash_MHA_64h_32_blocks_old_param_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
# ==============================
# Model Training and Evaluation
# ==============================

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

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
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    # Create an instance of the DDPM class
    betas = (1e-4, 0.02)
    n_T = T
    ddpm = DDPM(betas=betas, n_T=n_T)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start time of epoch

        model.train()
        train_loss = 0
        for images in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            current_batch_size = images.shape[0]
            
            # Sample timesteps for the current batch
            timesteps = torch.randint(low=1, high=n_T+1, size=(current_batch_size,), dtype=torch.long, device=DEVICE)
            gaussian_splat = images
            gaussian_splat_normalized = normalize_parameters(gaussian_splat, param_ranges)
            
            # Get noisy images and noise
            g_t, g_t_minus_1, noise = ddpm.get_noisy_images_and_noise(gaussian_splat_normalized, timesteps)
            g_t = g_t.to(DEVICE)
            timesteps = timesteps.float().to(DEVICE)
            
            # Forward pass without mixed precision
            predicted_noise = model(g_t, timesteps)
            loss = criterion(predicted_noise, noise)
            
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
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f}s")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch_time": epoch_duration})

        # Save the model only if the validation loss decreases
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, filename=f'flash_val_best_MHA_64h_32_blocks_old_param_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth')
        if train_loss <= best_train_loss:   
            best_train_loss = train_loss
            save_model(model, optimizer, epoch, train_loss, save_path=SAVE_PATH, filename=f'flash_train_best_MHA_64h_32_blocks_old_param_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth')
         # Save intermediate model every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1 :
            intermediate_model_loss = val_loss
            save_model(model, optimizer, epoch, intermediate_model_loss, save_path=SAVE_PATH, filename=f'flash_intermediate_e{epoch+1}_MHA_64h_32_blocks_old_parm.pth')
        save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, filename=f'flash_last_MHA_64h_32_blocks_old_param_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth')
    
    return train_losses, val_losses

def main():
    wandb.init(project='Flash_transformers_MHA', config={
        "learning_rate": LEARNING_RATE, 
        "epochs": NUM_EPOCHS, 
        "batch_size": BATCH_SIZE
    })
    
    print("Loading dataset...")
    dataset = GaussianDataset(DATA_FOLDER)

    # # Calculate min-max for each feature over the entire dataset
    # all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    # print("shape of complete dataset: ", all_data.shape)
    # param_ranges = [(all_data[:, :, i].min().item(), all_data[:, :, i].max().item()) for i in range(all_data.shape[2])]
    # print(f"Parameter ranges length: {len(param_ranges)}")
    # print( f"Parameter ranges: {param_ranges}\n")
    
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
    num_blocks = 32
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
    plt.savefig(os.path.join(SAVE_PATH, f"flash_MHA_64h_32_blocks_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.png"))
    plt.show()

if __name__ == "__main__":
    main()