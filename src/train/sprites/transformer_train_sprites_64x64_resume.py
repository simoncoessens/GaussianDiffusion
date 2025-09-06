import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse  # Add this import
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb
from torch.nn import functional as F
from models.transformer_model import GaussianTransformer
from dataset import GaussianDatasetSprites64x64
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

DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/datasets/Sprites/sprites_results_combined.h5"

# Model parameters (used in file naming and folder structure)
NUM_HEADS = 64
NUM_BLOCKS = 12

BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4         
WEIGHT_DECAY = 1e-4          # Weight decay for AdamW
T = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build a file tag based on the number of heads and blocks
FILE_TAG = f"SPRITES_64x64_MHA_{NUM_HEADS}h_{NUM_BLOCKS}blocks_{T}ts"

# Build a safe save path that includes the parameterized file tag
SAVE_PATH = os.path.join("models", "Sprites_64x64", FILE_TAG)

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

# Find the latest checkpoint file
def find_latest_checkpoint():
    """Find the most recent checkpoint in the save path."""
    if not os.path.exists(SAVE_PATH):
        return None
        
    last_checkpoint = os.path.join(SAVE_PATH, build_filename("last"))
    if os.path.exists(last_checkpoint):
        return last_checkpoint
    return None

# ==============================
# Model Saving and Loading Functions
# ==============================
def save_model(model, optimizer, scheduler, epoch, loss, save_path=SAVE_PATH, filename=None):
    os.makedirs(save_path, exist_ok=True)
    if filename is None:
        filename = build_filename("last")
    full_path = os.path.join(save_path, filename)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'validation_loss': loss,
        'wandb_run_id': wandb.run.id if wandb.run else None,
    }
    torch.save(checkpoint, full_path)
    
    # Also save wandb run ID to a separate file for easier access
    with open(os.path.join(save_path, "wandb_run_id.txt"), "w") as f:
        f.write(f"{wandb.run.id if wandb.run else ''}")
    
    print(f"Model saved to {full_path} (Loss: {loss:.4f})")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    val_loss = checkpoint.get('validation_loss', None)
    wandb_id = checkpoint.get('wandb_run_id', None)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (Loss: {val_loss:.4f})")
    return start_epoch, wandb_id

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
            save_model(model, optimizer, scheduler, epoch, val_loss, save_path=SAVE_PATH, 
                       filename=build_filename("val_best"))
        if train_loss <= best_train_loss:
            best_train_loss = train_loss
            save_model(model, optimizer, scheduler, epoch, train_loss, save_path=SAVE_PATH, 
                       filename=build_filename("train_best"))
        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
            save_model(model, optimizer, scheduler, epoch, val_loss, save_path=SAVE_PATH, 
                       filename=build_filename("intermediate", epoch=epoch+1))
        
        # Always save the last checkpoint
        save_model(model, optimizer, scheduler, epoch, val_loss, save_path=SAVE_PATH, 
                   filename=build_filename("last"))
    
    return train_losses, val_losses

# ==============================
# Main Function
# ==============================
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train transformer model on Sprites dataset')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    args = parser.parse_args()
    
    # Check for resuming training
    resume_training = args.resume
    resume_checkpoint = None
    wandb_run_id = None
    
    if resume_training:
        # Find latest checkpoint
        resume_checkpoint = find_latest_checkpoint()
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Found checkpoint to resume from: {resume_checkpoint}")
            
            # Check if we have a saved wandb run ID
            wandb_id_file = os.path.join(SAVE_PATH, "wandb_run_id.txt")
            if os.path.exists(wandb_id_file):
                with open(wandb_id_file, "r") as f:
                    wandb_run_id = f.read().strip()
                if wandb_run_id:
                    print(f"Will resume wandb run: {wandb_run_id}")
        else:
            print("No checkpoint found. Starting training from scratch.")
            resume_training = False
    
    # Initialize wandb with appropriate run ID if resuming
    if resume_training and wandb_run_id:
        wandb.init(project='Sprites_64x64_MHA',
                   config={"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE},
                   id=wandb_run_id,
                   resume="allow")
    else:
        wandb.init(project='Sprites_64x64_MHA',
                   config={"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE})
    
    print("Loading dataset...")
    dataset = GaussianDatasetSprites64x64(DATA_FOLDER)
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
        input_dim=500,  # assuming num_gaussians is 500
        time_emb_dim=32,
        feature_dim=8,
        num_timestamps=T,
        num_transformer_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    start_epoch = 0
    if resume_training and resume_checkpoint:
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, resume_checkpoint)
    
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
    plot_filename = os.path.join(SAVE_PATH, f"{FILE_TAG}_e{NUM_EPOCHS}_ts{T}_mse.png")
    plt.savefig(plot_filename)
    plt.show()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting training.")
        sys.exit(1)
    main()