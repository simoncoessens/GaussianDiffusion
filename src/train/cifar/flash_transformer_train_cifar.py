import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb
from torch.nn import functional as F
#from models.flash_transformer_model import GaussianTransformer
from models.transformer_model import GaussianTransformer
from dataset import GaussianDatasetCIFAR10
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

DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/datasets/CIFAR10/cifar_first50k.h5"

# Model parameters (used in file naming and folder structure)
NUM_HEADS = 64
NUM_BLOCKS = 32

# Build a file tag based on the number of heads and blocks
FILE_TAG = f"CIFAR_MHA_{NUM_HEADS}h_{NUM_BLOCKS}blocks_mha_classic_1000s"

# Build a safe save path that includes the parameterized file tag
SAVE_PATH = os.path.join("models", "CIFAR10", FILE_TAG)

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4         # Try a lower learning rate
WEIGHT_DECAY = 1e-4          # Weight decay for AdamW
T = 50                      # Number of timesteps for DDPM
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_filename(checkpoint_type, epoch=None):
    """
    Build a filename for checkpoint saving based on the checkpoint type.
    checkpoint_type: one of "last", "train_best", "val_best", or "intermediate"
    For "intermediate", the epoch must be provided.
    """
    if checkpoint_type == "last":
        return f"last_{FILE_TAG}_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth"
    elif checkpoint_type == "train_best":
        return f"train_best_{FILE_TAG}_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth"
    elif checkpoint_type == "val_best":
        return f"val_best_{FILE_TAG}_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.pth"
    elif checkpoint_type == "intermediate":
        if epoch is None:
            raise ValueError("Epoch must be provided for intermediate checkpoint")
        return f"intermediate_{FILE_TAG}_e{epoch}_checkpoint.pth"
    else:
        raise ValueError("Unknown checkpoint type")

# Resume training flag and checkpoint path (using the new naming scheme)
RESUME_TRAINING = False
RESUME_CHECKPOINT = "/gpfs/workdir/coessenss/gsplat/src/models/CIFAR10/Flash_MHA_8h_4_blocks/last_Flash_MHA_64h_16_blocks_e100_ts200_mse_std_scaled.pth"

# Define the wandb run id to resume the same run (set this to your existing run id)
WANDB_RUN_ID = "a5120nae"

# ==============================
# Model Saving and Loading Functions
# ==============================
def save_model(model, optimizer, epoch, loss, save_path=SAVE_PATH, filename=None):
    os.makedirs(save_path, exist_ok=True)
    if filename is None:
        filename = build_filename("last")
    full_path = os.path.join(save_path, filename)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_loss': loss,
    }
    torch.save(checkpoint, full_path)
    print(f"Model saved to {full_path} (Loss: {loss:.4f})")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    val_loss = checkpoint.get('validation_loss', None)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (Loss: {val_loss:.4f})")
    return start_epoch

# ==============================
# Training Function
# ==============================
def train_model(train_loader, val_loader, model, scheduler, optimizer, num_epochs, param_ranges, start_epoch=0):
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    betas = (1e-4, 0.02)
    n_T = T
    ddpm = DDPM(betas=betas, n_T=n_T)

    for epoch in range(start_epoch, num_epochs):
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

        # Save checkpoints based on performance and schedule
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, 
                       filename=build_filename("val_best"))
        if train_loss <= best_train_loss:
            best_train_loss = train_loss
            save_model(model, optimizer, epoch, train_loss, save_path=SAVE_PATH, 
                       filename=build_filename("train_best"))
        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
            save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, 
                       filename=build_filename("intermediate", epoch=epoch+1))
        
        # Always save the last checkpoint
        save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, 
                   filename=build_filename("last"))
    
    return train_losses, val_losses

# ==============================
# Main Function
# ==============================
def main():
    # If resuming training, initialize wandb with the same run id to update the same run.
    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT):
        wandb.init(project='Flash_CIFAR_MHA',
                   config={"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE},
                   id=WANDB_RUN_ID,
                   resume="allow")
    else:
        wandb.init(project='Flash_CIFAR_MHA',
                   config={"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE})
    
    print("Loading dataset...")
    dataset = GaussianDatasetCIFAR10(DATA_FOLDER)
    all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    print("shape of complete dataset: ", all_data.shape)
    param_ranges = [(all_data[:, :, i].min().item(), all_data[:, :, i].max().item()) for i in range(all_data.shape[2])]
    print(f"Parameter ranges length: {len(param_ranges)}")
    print(f"Parameter ranges: {param_ranges}\n")
    
    print("Splitting dataset...")
    num_samples = len(dataset)
    print(f"Number of samples: {num_samples}")
    train_size = int(0.7 * num_samples)
    val_size = num_samples - train_size

    print("Creating data loaders...")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("shape of sprites gaussian: ", train_dataset[0].shape)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Use the parameterized values for the model
    model = GaussianTransformer(
        input_dim=200,  # assuming num_gaussians is 200
        time_emb_dim=32,
        feature_dim=8,
        num_timestamps=T,
        num_transformer_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    start_epoch = 0
    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT):
        start_epoch = load_checkpoint(model, optimizer, RESUME_CHECKPOINT)
    
    print("Training model...")
    train_losses, val_losses = train_model(train_loader, val_loader, model, scheduler, optimizer, NUM_EPOCHS, param_ranges, start_epoch=start_epoch)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses")
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(SAVE_PATH, f"{FILE_TAG}_e{NUM_EPOCHS}_ts{T}_mse_std_scaled.png")
    plt.savefig(plot_filename)
    plt.show()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting training.")
        sys.exit(1)
    main()
