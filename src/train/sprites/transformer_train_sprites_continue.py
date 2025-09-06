import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from torch.nn import functional as F
from src.models.transformer_model import GaussianTransformer
from src.dataset import GaussianDatasetSprites
from src.ddpm import DDPM
from src.utils.normalize import normalize_parameters

# ==============================
# Configurations
# ==============================
os.environ["WANDB_API_KEY"] = "d30c12b44a4235f945b74e0026b737863ffbb044"
wandb.login()

DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/datasets/Sprites/gaussian_representations_sprites_downsampled.h5"
SAVE_PATH = 'models/sprites_MHA_64h_22_blocks_mlp_updated_200g'
BATCH_SIZE = 16
# Number of additional epochs you wish to train after resuming
ADDITIONAL_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
T = 200
SAVE_FILENAME = f'sprites_MHA_64h_200g{T}_mse_std_scaled.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================
# Checkpoint Loading Helper
# ==============================
def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads a checkpoint and restores model and optimizer states.
    Returns the epoch to resume from.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    print(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {start_epoch}.")
    return start_epoch

# ==============================
# Training Function (Resumable)
# ==============================
def resume_training(train_loader, val_loader, model, scheduler, optimizer, param_ranges, start_epoch, additional_epochs):
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    # Create an instance of the DDPM class
    betas = (1e-4, 0.02)
    n_T = T
    ddpm = DDPM(betas=betas, n_T=n_T)
    
    total_epochs = start_epoch + additional_epochs
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0
        for images in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            current_batch_size = images.shape[0]
            # Sample timesteps for the current batch
            timesteps = torch.randint(low=1, high=n_T+1, size=(current_batch_size,), dtype=torch.long, device=DEVICE)
            gaussian_splat_normalized = normalize_parameters(images, param_ranges)
            g_t, g_t_minus_1, noise = ddpm.get_noisy_images_and_noise(gaussian_splat_normalized, timesteps)
            g_t = g_t.to(DEVICE)
            timesteps = timesteps.float().to(DEVICE)
            predicted_noise = model(g_t, timesteps)
            loss = criterion(predicted_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                images = images.to(DEVICE)
                gaussian_splat_normalized = normalize_parameters(images, param_ranges)
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
        print(f"Epoch {epoch + 1}/{total_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        
        # Save checkpoint only if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_filename = f"epoch_{epoch + 1}_best_{SAVE_FILENAME}"
            checkpoint_full_path = os.path.join(SAVE_PATH, checkpoint_filename)
            os.makedirs(SAVE_PATH, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': val_loss,
            }
            torch.save(checkpoint, checkpoint_full_path)
            print(f"Best checkpoint saved to {checkpoint_full_path}")
        
    return train_losses, val_losses

# ==============================
# Main Function
# ==============================
def main():
    # Resume the same run by using your previous run id if desired
    wandb.init(project='Sprites_Transformers_MHA', resume='allow', id='pgdjedfi')

    print("Loading dataset...")
    dataset = GaussianDatasetSprites(DATA_FOLDER)
    # Calculate parameter ranges over the complete dataset
    all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    param_ranges = [(all_data[:, :, i].min().item(), all_data[:, :, i].max().item()) for i in range(all_data.shape[2])]
    
    num_samples = len(dataset)
    train_size = int(0.7 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define model parameters
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
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Specify the path to your last checkpoint file
    checkpoint_path = "/gpfs/workdir/coessenss/gsplat/models/sprites_MHA_64h_22_blocks_mlp_updated_200g/val_best_sprites_MHA_64h_22_blocks_200g_mlp_updated_e100_ts200_mse_std_scaled.pth"
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    
    # Resume training for the additional epochs
    train_losses, val_losses = resume_training(train_loader, val_loader, model, scheduler, optimizer, param_ranges, start_epoch, ADDITIONAL_EPOCHS)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch (Resumed Training)")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_PATH, "sprites_MHA_64h_resume_training_loss.png"))
    plt.show()

if __name__ == "__main__":
    main()
