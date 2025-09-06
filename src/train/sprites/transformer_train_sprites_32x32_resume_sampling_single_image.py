import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
import signal
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb
from torch.nn import functional as F
from src.models.transformer_model import GaussianTransformer
from src.dataset import GaussianDatasetSprites32x32Single
from src.ddpm import DDPM
from src.utils.normalize import normalize_parameters
from src.train.sampler_resume import sample_and_plot_images

# ==============================
# Configurations
# ==============================
print("Getting the wandb key")
# Set the API key as an environment variable
os.environ["WANDB_API_KEY"] = "d30c12b44a4235f945b74e0026b737863ffbb044"

# Log in to Weights & Biases
wandb.login()

DATA_FOLDER = "/gpfs/workdir/coessenss/gsplat/data/encoding_scripts/GaussianImage/sprites32x32/sprites_20_images.h5"

# Model parameters (used in file naming and folder structure)
NUM_HEADS = 8
NUM_BLOCKS = 8

BATCH_SIZE = 8
NUM_EPOCHS = 10000  # Train for 1000 epochs total
LEARNING_RATE = 1e-3         
WEIGHT_DECAY = 1e-4          # Weight decay for AdamW
T = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set a time limit for training (in minutes) before saving and exiting
# 23 hours in minutes (leaving 1 hour buffer for a 24-hour job)
TIME_LIMIT_MINUTES = 23 * 60

# Sampling parameters
SAMPLE_BATCH_SIZE = 100  # Number of samples to generate for metrics
GRID_SIZE = 10          # Will create a 10x10 grid of images
SAMPLE_EVERY_N_EPOCHS = 10  # Sample every N epochs to save time
H, W = 32, 32           # Output image dimensions
BLOCK_H, BLOCK_W = 16, 16

# Build a file tag based on the number of heads and blocks
FILE_TAG = f"SPRITES_MHA_{NUM_HEADS}h_{NUM_BLOCKS}blocks_{T}ts_mha_classic_1000e"

# Build a save path that includes the file tag
SAVE_PATH = os.path.join("models", "Sprites_32x32", FILE_TAG)
os.makedirs(SAVE_PATH, exist_ok=True)

# Path to store run ID for resuming the same wandb run
RUN_ID_FILE = os.path.join(SAVE_PATH, "run_id.txt")

# Path to store the current epoch for resuming
CURRENT_EPOCH_FILE = os.path.join(SAVE_PATH, "current_epoch.txt")

def build_filename(checkpoint_type, epoch=None):
    """
    Build a filename for checkpoint saving based on the checkpoint type.
    checkpoint_type: one of "last", "train_best", "val_best", "intermediate" or "timeout"
    For "intermediate" or "timeout", the epoch must be provided.
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
    elif checkpoint_type == "timeout":
        if epoch is None:
            raise ValueError("Epoch must be provided for timeout checkpoint")
        return f"timeout_{FILE_TAG}_e{epoch}_checkpoint.pth"
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

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
    return full_path

def load_checkpoint(model, optimizer, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file {checkpoint_path} not found.")
        return 0
        
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
def train_model(train_loader, val_loader, model, scheduler, optimizer, num_epochs, param_ranges, start_epoch=0, dataset=None):
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    betas = (1e-4, 0.02)
    n_T = T
    ddpm = DDPM(betas=betas, n_T=n_T)
    
    # Set start time for time limit tracking
    start_time = time.time()
    time_limit_seconds = TIME_LIMIT_MINUTES * 60
    
    # Setup signal handler for graceful termination
    def signal_handler(sig, frame):
        print("Received termination signal. Saving checkpoint and exiting.")
        save_model(model, optimizer, epoch, val_loss, 
                  save_path=SAVE_PATH, filename=build_filename("timeout", epoch=epoch))
        # Save current epoch to resume from later
        with open(CURRENT_EPOCH_FILE, 'w') as f:
            f.write(str(epoch))
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    for epoch in range(start_epoch, num_epochs):
        # Save current epoch for potential resume
        with open(CURRENT_EPOCH_FILE, 'w') as f:
            f.write(str(epoch))
        
        model.train()
        train_loss = 0
        for images in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Check if we're approaching the time limit
            elapsed = time.time() - start_time
            if elapsed > time_limit_seconds - 900:  # 15 minutes before time limit
                print(f"Approaching time limit. Saving checkpoint at epoch {epoch + 1} and exiting.")
                checkpoint_path = save_model(model, optimizer, epoch, train_loss / len(train_loader), 
                                           save_path=SAVE_PATH, filename=build_filename("timeout", epoch=epoch))
                # Save current epoch to resume from later
                with open(CURRENT_EPOCH_FILE, 'w') as f:
                    f.write(str(epoch))
                # Log the early exit to wandb
                wandb.log({"early_exit_epoch": epoch + 1})
                print("Exiting with code 0 to trigger automatic resubmission")
                sys.exit(0)
            
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
            for images in tqdm(val_loader, desc=f"Validation {epoch + 1}/{num_epochs}"):
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

        # Adjust learning rate
        scheduler.step()


        # Log basic metrics to wandb
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

            # Save model every 10 epochs (overwriting the same file)
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(SAVE_PATH, "latest_model.pth")
            save_model(model, optimizer, epoch, val_loss, save_path=SAVE_PATH, filename="latest_model.pth")
            print(f"Saved model at epoch {epoch + 1} to {model_path}")

        # Sample and calculate metrics every N epochs
        if (epoch + 1) % SAMPLE_EVERY_N_EPOCHS == 0:
            print(f"Sampling images at epoch {epoch + 1}...")

            # Load the latest saved model from disk
            checkpoint = torch.load(os.path.join(SAVE_PATH, "latest_model.pth"), map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Sample and plot images
            fig, metrics = sample_and_plot_images(
                model, dataset, param_ranges, DEVICE,
                n_sample=SAMPLE_BATCH_SIZE, n_T=T,
                grid_size=GRID_SIZE, H=H, W=W,
                BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
            )

            # Log the sampled image grid to wandb
            wandb.log({
                "epoch": epoch + 1,
                "samples": wandb.Image(fig, caption=f"Generated Samples (Epoch {epoch+1})")
            })
            plt.close(fig)

            # Log metrics to wandb
            metric_dict = {
                "epoch": epoch + 1,
                "metrics/fid": metrics["fid"],
                "metrics/inception_score": metrics["inception_score_mean"],
                "metrics/kid": metrics["kid_mean"],
                "metrics/sampling_time_seconds": metrics["sampling_time"],
                "metrics/samples_per_second": metrics["samples_per_second"]
            }
            wandb.log(metric_dict)

            print(f"Epoch {epoch+1} Metrics:")
            print(f"  FID: {metrics['fid']:.4f}")
            print(f"  Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
            print(f"  KID: {metrics['kid_mean']:.4f} ± {metrics['kid_std']:.4f}")
            print(f"  Sampling Time: {metrics['sampling_time']:.2f}s ({metrics['samples_per_second']:.2f} samples/s)")

    
    return train_losses, val_losses

# ==============================
# Main Function
# ==============================
def main():
    wandb.init(project='sprites_single_image_test', 
               config={"learning_rate": LEARNING_RATE, 
                       "epochs": NUM_EPOCHS, 
                       "batch_size": 100,
                       "num_heads": NUM_HEADS,
                       "num_blocks": NUM_BLOCKS,
                       "time_steps": T},
               name="SingleImageSanityCheck")

    
    print("Loading dataset...")
    dataset = GaussianDatasetSprites32x32Single(DATA_FOLDER)
    all_data = torch.cat([dataset[i].unsqueeze(0) for i in range(len(dataset))], dim=0)
    print("Shape of complete dataset: ", all_data.shape)
    param_ranges = [(all_data[:, :, i].min().item(), all_data[:, :, i].max().item()) for i in range(all_data.shape[2])]
    print(f"Parameter ranges length: {len(param_ranges)}")
    print(f"Parameter ranges: {param_ranges}\n")
    
    print("Splitting dataset...")
    num_samples = len(dataset)
    print(f"Number of samples: {num_samples}")
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size

    print("Creating data loaders...")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print("Shape of sprites gaussian: ", train_dataset[0].shape)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create the model
    model = GaussianTransformer(
        input_dim=150,  # Number of gaussians
        time_emb_dim=32,
        feature_dim=8,
        num_timestamps=T,
        num_transformer_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-5
    )

    
    start_epoch = 0
    print("Starting training from scratch...")

    train_losses, val_losses = train_model(
        train_loader, val_loader, model, scheduler, optimizer, 
        NUM_EPOCHS, param_ranges, start_epoch=start_epoch, dataset=dataset
    )

    # Plot and save the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses")
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(SAVE_PATH, f"{FILE_TAG}_loss_plot.png")
    plt.savefig(plot_filename)
    
    # Log the final loss plot to wandb
    wandb.log({"final_loss_plot": wandb.Image(plot_filename)})
    print("Training completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Aborting training.")
        sys.exit(1)
    main()