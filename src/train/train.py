import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import wandb
from model import GaussianSplatDiffusionModel
from dataset import GaussianDataset
from forward_diffusion import (
    forward_diffusion_sample,
    linear_beta_schedule,
    get_index_from_list
)

wandb.login(key="d30c12b44a4235f945b74e0026b737863ffbb044")

# ==============================
# Hyperparameters and Configurations
# ==============================
# Paths and Data
DATA_FOLDER = "mnist_gaussian_representations"
SAVE_PATH = 'models/'
SAVE_FILENAME = 'best_model.pth'

# Training Parameters
BATCH_SIZE = 32           # Adjust as needed
NUM_EPOCHS = 20         # Adjust as needed
LEARNING_RATE = 1e-3
T = 1000                 # Number of timesteps

# Model Parameters
INPUT_DIM = 7
HIDDEN_DIM = 32
OUTPUT_DIM = 7
POOLING_METHOD = 'max'   # 'mean' or 'max'

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)


def save_model(model, optimizer, epoch, loss,
               save_path=SAVE_PATH, filename=SAVE_FILENAME):
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_loss': loss,
    }
    torch.save(checkpoint, full_path)
    print(f"Model saved to {full_path}")


def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho,
                                   coords, colours, image_size=(28, 28),
                                   channels=1, device="cuda"):
    batch_size = colours.shape[0]

    print("Generating 2D gaussian splatting with kernel size:", kernel_size)
    # Reshape sigma and rho
    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    # Create covariance matrices
    covariance = torch.stack(
        [
            torch.stack([sigma_x**2, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y**2], dim=-1)
        ],
        dim=-2
    )

    # Check for positive semi-definiteness
    determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    if (determinant <= 0).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    inv_covariance = torch.inverse(covariance)

    # Create grid for kernel
    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    # Create meshgrid
    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
    xy = torch.stack([xx, yy], dim=-1)

    # Compute Gaussian kernel
    z = torch.einsum(
        'b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy
    )
    kernel = (
        torch.exp(z) /
        (2 * torch.tensor(np.pi, device=device) *
         torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))
    )

    # Normalize the kernel
    kernel_max = kernel.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(
        batch_size, 1, 1
    )
    kernel_normalized = kernel / kernel_max

    # Reshape kernel to match channels
    kernel_reshaped = kernel_normalized.repeat(1, channels, 1).view(
        batch_size * channels, kernel_size, kernel_size
    )
    kernel_channels = kernel_reshaped.unsqueeze(0).reshape(
        batch_size, channels, kernel_size, kernel_size
    )

    # Calculate padding
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            "Kernel size should be smaller or equal to the image size."
        )

    padding = (
        pad_w // 2, pad_w // 2 + pad_w % 2,
        pad_h // 2, pad_h // 2 + pad_h % 2
    )

    kernel_padded = F.pad(kernel_channels, padding, "constant", 0)

    # Create affine transformation
    b, c, h, w = kernel_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    # Apply affine transformation
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_transformed = F.grid_sample(kernel_padded, grid, align_corners=True)

    # Multiply by color values
    colours_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = colours_reshaped * kernel_transformed

    # Sum layers to form final image
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)  # Shape: [H, W, channels]

    return final_image


##############################################
# Normalization and Denormalization Functions
##############################################

def normalize_parameters(W, param_ranges):
    """Normalizes parameters to the range [-1, 1]."""
    print("Normalizing parameters...")
    W_normalized = torch.zeros_like(W)
    for i in range(W.shape[-1]):
        min_val, max_val = param_ranges[i]
        if min_val == max_val:
            W_normalized[:, :, i] = 0.0
        else:
            W_normalized[:, :, i] = (
                2 * (W[:, :, i] - min_val) / (max_val - min_val) - 1
            )
    return W_normalized


def denormalize_parameters(W_normalized, param_ranges):
    """Denormalizes parameters from the range [-1, 1] back to their original
    ranges with clipping.
    """
    print("Denormalizing parameters...")
    W_denormalized = torch.zeros_like(W_normalized)
    for i in range(W_normalized.shape[-1]):
        min_val, max_val = param_ranges[i]
        if min_val == max_val:
            W_denormalized[:, :, i] = min_val
        else:
            W_denormalized[:, :, i] = (
                (W_normalized[:, :, i] + 1) / 2 *
                (max_val - min_val) + min_val
            )
            W_denormalized[:, :, i] = torch.clamp(
                W_denormalized[:, :, i], min_val, max_val
            )
    return W_denormalized


def pixel_loss(x, y):
    """
    Compute the L2 loss (Euclidean distance) between two images.
    """
    return torch.sqrt(torch.sum((x - y)**2))


def main():
    print("Initializing Weights & Biases (W&B)...")
    wandb.init(
        project='bdrp',
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "timesteps": T,
        }
    )
    print("W&B initialized with config:", wandb.config)

    # --- Define parameter ranges for normalization and denormalization ---
    param_ranges = [
        (0, 1),   # sigma_x (assuming a reasonable range)
        (0, 1),   # sigma_y (assuming a reasonable range)
        (-1, 1),  # rho
        (0, 1),   # alpha
        (0, 1),   # colours
        (-2, 2),  # pixel_coords x
        (-2, 2)   # pixel_coords y
    ]

    # Select percentage of dataset to use
    percent_to_use = 0.01

    print("Creating dataset from:", DATA_FOLDER)
    dataset = GaussianDataset(DATA_FOLDER)
    print("Original dataset size:", len(dataset))

    if percent_to_use == 1.0:
        print("Using the full dataset.")
        num_samples = len(dataset)
        subset = dataset  # No need to create a subset
    else:
        # Calculate the reduced dataset size
        num_samples = int(len(dataset) * percent_to_use)
        print(f"Using {num_samples} samples ({percent_to_use * 100:.2f}%) out of {len(dataset)}")

        # Create the reduced dataset
        subset, _ = random_split(dataset, [num_samples, len(dataset) - num_samples])

    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
    print("Split sizes -> Train:", train_size, "Val:", val_size, "Test:", test_size)

    # Final split
    train_dataset, val_dataset, test_dataset = random_split(
        subset, [train_size, val_size, test_size]
    )


    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Training dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    betas = linear_beta_schedule(timesteps=T).to(DEVICE)
    alphas = (1.0 - betas).to(DEVICE)
    alphas_cumprod = torch.cumprod(alphas, axis=0).to(DEVICE)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(DEVICE)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(
        1.0 - alphas_cumprod
    ).to(DEVICE)

    # Load the model
    print("Initializing GaussianSplatDiffusionModel...")
    batch_size = 32  
    num_gaussians = 70
    gaussian_features = 7
    depth = 1
    local_block_dim = [16, 32]
    global_block_dim = [32, 16, 7]
    time_emb_dim = 8

    model = GaussianSplatDiffusionModel(
        num_gaussians, gaussian_features, depth,
        local_block_dim, global_block_dim, time_emb_dim
    ).to(DEVICE)
    print("Model initialized:", model)

    wandb.watch(model, log="all")
    print("Model is now watched by W&B.")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # epoch_train_losses = []
    # epoch_val_losses = []
    # best_val_loss = float('inf')

    # # DDPM Training Loop
    # print("Starting training loop...")
    # for epoch in range(NUM_EPOCHS):
    #     print(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}...")
    #     # ======= Training Phase =======
    #     model.train()
    #     train_loss = 0.0
    #     num_train_samples = 0

    #     with tqdm(
    #         range(len(train_loader)),
    #         unit="batch",
    #         desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training"
    #     ) as tepoch:
    #         for _ in tepoch:
    #             # Sample a random batch
    #             batch = next(iter(train_loader))
    #             gaussian_ground_truth = batch.to(DEVICE)  # (batch_size, 70, 7)
    #             gaussian_ground_truth = normalize_parameters(
    #                 gaussian_ground_truth, param_ranges
    #             )

    #             # Sample a random timestep t
    #             t_step = torch.randint(
    #                 0, T, (gaussian_ground_truth.size(0),), device=DEVICE
    #             )

    #             # Forward diffusion at timestep t
    #             noisy_xt, added_noise = forward_diffusion_sample(
    #                 gaussian_ground_truth,
    #                 t_step,
    #                 sqrt_alphas_cumprod,
    #                 sqrt_one_minus_alphas_cumprod,
    #                 device=DEVICE
    #             )
    #             # Print statements to check shapes
    #             # print("noisy_xt shape:", noisy_xt.shape, "added_noise shape:", added_noise.shape)

    #             # Predict the noise from noisy_x0
    #             predicted_noise = model(noisy_xt, t_step)
    #             # print("predicted_noise shape:", predicted_noise.shape)

    #             # Compute loss
    #             loss = loss_fn(predicted_noise, added_noise)
    #             # print("Training loss value:", loss.item())

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             train_loss += loss.item()
    #             num_train_samples += 1
    #             tepoch.set_postfix(loss=loss.item())

    #     avg_train_loss = train_loss / num_train_samples
    #     epoch_train_losses.append(avg_train_loss)
    #     wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

    #     # ======= Validation Phase =======
    #     print("Starting Validation Phase for epoch", epoch + 1)
    #     model.eval()
    #     val_loss = 0.0
    #     num_val_samples = 0

    #     with torch.no_grad():
    #         with tqdm(
    #             range(len(val_loader)),
    #             unit="batch",
    #             desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Validation"
    #         ) as tepoch_val:
    #             for _ in tepoch_val:
    #                 # Sample a random batch
    #                 batch = next(iter(val_loader))
    #                 gaussian_ground_truth = batch.to(DEVICE)
    #                 gaussian_ground_truth = normalize_parameters(
    #                     gaussian_ground_truth, param_ranges
    #                 )

    #                 # Sample a random timestep t
    #                 t_step = torch.randint(
    #                     0, T, (gaussian_ground_truth.size(0),), device=DEVICE
    #                 )

    #                 # Forward diffusion at timestep t
    #                 noisy_xt, added_noise = forward_diffusion_sample(
    #                     gaussian_ground_truth,
    #                     t_step,
    #                     sqrt_alphas_cumprod,
    #                     sqrt_one_minus_alphas_cumprod,
    #                     device=DEVICE
    #                 )

    #                 # Predict the noise
    #                 predicted_noise = model(noisy_xt, t_step)

    #                 # Compute validation loss
    #                 loss = loss_fn(predicted_noise, added_noise)
    #                 val_loss += loss.item()
    #                 num_val_samples += 1
    #                 tepoch_val.set_postfix(loss=loss.item())

    #     avg_val_loss = val_loss / num_val_samples
    #     epoch_val_losses.append(avg_val_loss)
    #     wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})

    #     print(
    #         f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
    #         f"Training Loss: {avg_train_loss:.4f} | "
    #         f"Validation Loss: {avg_val_loss:.4f}"
    #     )

    #     # Checkpointing
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         save_model(model, optimizer, epoch + 1,
    #                    best_val_loss, filename=SAVE_FILENAME)

    # wandb.finish()
    # print("Finished training. Now plotting losses...")
    
    # Training Loop
    print("Starting training loop...")
    best_val_loss = float("inf")
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}...")

        # ======= Training Phase =======
        model.train()
        train_loss = 0.0
        with tqdm(
            train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training"
        ) as tepoch:
            for batch in tepoch:
                gaussian_ground_truth = batch[0].to(DEVICE)  # Only the image tensor
                gaussian_ground_truth = normalize_parameters(gaussian_ground_truth, param_ranges)

                # Sample a random timestep t
                t_step = torch.randint(0, T, (gaussian_ground_truth.size(0),), device=DEVICE)

                # Forward diffusion at timestep t
                noisy_xt, added_noise = forward_diffusion_sample(
                    gaussian_ground_truth,
                    t_step,
                    sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod,
                    device=DEVICE
                )

                # Predict the noise from noisy_x0
                predicted_noise = model(noisy_xt, t_step)

                # Compute loss
                loss = loss_fn(predicted_noise, added_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # ======= Validation Phase =======
        print("Starting Validation Phase for epoch", epoch + 1)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            with tqdm(
                val_loader, unit="batch", desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Validation"
            ) as tepoch_val:
                for batch in tepoch_val:
                    gaussian_ground_truth = batch[0].to(DEVICE)
                    gaussian_ground_truth = normalize_parameters(
                        gaussian_ground_truth, param_ranges
                    )

                    # Sample a random timestep t
                    t_step = torch.randint(0, T, (gaussian_ground_truth.size(0),), device=DEVICE)

                    # Forward diffusion at timestep t
                    noisy_xt, added_noise = forward_diffusion_sample(
                        gaussian_ground_truth,
                        t_step,
                        sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod,
                        device=DEVICE
                    )

                    # Predict the noise
                    predicted_noise = model(noisy_xt, t_step)

                    # Compute validation loss
                    loss = loss_fn(predicted_noise, added_noise)
                    val_loss += loss.item()
                    tepoch_val.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)
        wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Training Loss: {avg_train_loss:.4f} | "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, optimizer, epoch + 1, best_val_loss, filename=SAVE_FILENAME)

    wandb.finish()
    print("Finished training. Now plotting losses...")

    # Plotting training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, NUM_EPOCHS + 1),
        epoch_train_losses,
        label='Training Loss',
        marker='o',
        linestyle='-'
    )
    plt.plot(
        range(1, NUM_EPOCHS + 1),
        epoch_val_losses,
        label='Validation Loss',
        marker='s',
        linestyle='--'
    )
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()
    print("Loss plot saved as loss_plot.png")


def new_func(num_gaussians, gaussian_features, depth,
             local_block_dim, global_block_dim, time_emb_dim):
    return (
        num_gaussians,
        gaussian_features,
        depth,
        local_block_dim,
        global_block_dim,
        time_emb_dim
    )


if __name__ == "__main__":
    print("Entering main function...")
    main()
    print("Execution completed.")
