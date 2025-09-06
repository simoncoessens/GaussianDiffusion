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
from model_updated import GaussianSplatDiffusionModel
# from model_attention import GaussianSplatDiffusionModelAttention
from dataset import GaussianDataset
from ddpm import DDPM
from skimage.metrics import structural_similarity as ssim

# ==============================
# Configurations
# ==============================
print("getting the wandb key")
wandb.login(key="d30c12b44a4235f945b74e0026b737863ffbb044")

DATA_FOLDER = "mnist_gaussian_representations/"
SAVE_PATH = 'models/'
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
T = 2000
SAVE_FILENAME = f'best_model_b{BATCH_SIZE}_e{NUM_EPOCHS}_t{T}.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================
# Function to convert gaussian splatting to image
# ================================================
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
    # print("Normalizing parameters...")
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
    # print("Denormalizing parameters...")
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


# def pixel_loss(x, y):
#     """
#     Compute the L2 loss (Euclidean distance) between two images.
#     """
#     return torch.sqrt(torch.sum((x - y)**2))

# def get_predicted_g_t_minus_1(g_t, predicted_noise):
#     """
#     Simple reverse step to get g_{t-1} using the predicted noise.
#     Adjust this if you need the exact DDPM reverse formula.
#     """
#     return g_t - predicted_noise


# def compute_predicted_image_t_minus_1(predicted_g_t_minus_1, kernel_size=17, device="cuda"):
#     """
#     Generates the predicted image at t-1 from predicted g_{t-1}.
#     """
#     sigma_x = predicted_g_t_minus_1[:, :, 0]
#     sigma_y = predicted_g_t_minus_1[:, :, 1]
#     rho = predicted_g_t_minus_1[:, :, 2]
#     colours = predicted_g_t_minus_1[:, :, 4]
#     coords = predicted_g_t_minus_1[:, :, 5:7]
#     return generate_2D_gaussian_splatting(
#         kernel_size,
#         sigma_x,
#         sigma_y,
#         rho,
#         coords,
#         colours,
#         image_size=(28, 28),
#         channels=1,
#         device=device
#     )

# def pixel_loss_t_minus_1(real_g_t_minus_1, predicted_g_t_minus_1, kernel_size=17, device="cuda"):
#     """
#     Computes the pixel loss between the real and predicted t-1 images.
#     """
#     real_image_t_minus_1 = generate_2D_gaussian_splatting(
#         kernel_size,
#         real_g_t_minus_1[:, :, 0],
#         real_g_t_minus_1[:, :, 1],
#         real_g_t_minus_1[:, :, 2],
#         real_g_t_minus_1[:, :, 5:7],
#         real_g_t_minus_1[:, :, 4],
#         image_size=(28, 28),
#         channels=1,
#         device=device
#     )
#     pred_image_t_minus_1 = compute_predicted_image_t_minus_1(predicted_g_t_minus_1, kernel_size, device)
#     return pixel_loss(real_image_t_minus_1, pred_image_t_minus_1)

# def combined_loss(real_g_t_minus_1, g_t, noise, predicted_noise, kernel_size=17, device="cuda", alpha=0.5, beta=0.25):
#     """
#     Compute a combined loss of L2 loss of gaussians, L2 loss in pixel space, and SSIM loss.
#     alpha: weight for the L2 loss of gaussians.
#     beta: weight for the L2 loss in pixel space.
#     (1 - alpha - beta) will be the weight for the SSIM loss.
#     """
#     # Compute L2 loss of gaussians
#     predicted_g_t_minus_1 = get_predicted_g_t_minus_1(g_t, predicted_noise)
#     # l2_loss_gaussians = nn.MSELoss()(real_g_t_minus_1, predicted_g_t_minus_1)
#     l2_loss_noises = nn.MSELoss()(noise, predicted_noise)

#     # Compute L2 loss in pixel space
#     l2_loss_pixels = pixel_loss_t_minus_1(real_g_t_minus_1, predicted_g_t_minus_1, kernel_size, device)


#     # Compute SSIM loss in pixel space
#     real_image_t_minus_1 = generate_2D_gaussian_splatting(
#         kernel_size,
#         real_g_t_minus_1[:, 0],
#         real_g_t_minus_1[:, :, 1],
#         real_g_t_minus_1[:, :, 2],
#         real_g_t_minus_1[:, :, 5:7],
#         real_g_t_minus_1[:, :, 4],
#         image_size=(28, 28),
#         channels=1,
#         device=device
#     )
#     pred_image_t_minus_1 = compute_predicted_image_t_minus_1(predicted_g_t_minus_1, kernel_size, device)

#     real_image_np = real_image_t_minus_1.detach().cpu().numpy()
#     pred_image_np = pred_image_t_minus_1.detach().cpu().numpy()

#     ssim_value = 0
#     for i in range(real_image_np.shape[0]):  # Iterate over batch
#         ssim_value += ssim(real_image_np[i, 0], pred_image_np[i, 0], data_range=pred_image_np[i, 0].max() - pred_image_np[i, 0].min())

#     ssim_loss = 1 - (ssim_value / real_image_np.shape[0])  # Return 1 - SSIM to use as a loss (lower is better)

#     # Combine the losses
#     combined_loss = alpha * l2_loss_noises + beta * l2_loss_pixels + (1 - alpha - beta) * ssim_loss

#     return combined_loss

# ==============================
# Model Training and Evaluation
# ==============================
def train_model(train_loader, val_loader, model, scheduler, optimizer, num_epochs):
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []
    
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

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(DEVICE)
            optimizer.zero_grad()

            current_batch_size = images.shape[0]  # Dynamic batch size
    
            # Adjust timesteps size for the current batch
            timesteps = torch.randint(0, T, (current_batch_size,), device=DEVICE)
            print(timesteps)
            # print("timesteps shape: ", timesteps.shape)
            
            gaussian_splat= images
            gaussian_splat_normalized = normalize_parameters(gaussian_splat, param_ranges)
            # gaussian_splat_normalized = gaussian_splat
            # print("normalized gaussian splat shape: ", gaussian_splat_normalized.shape)

            # creating the DDPM instance
            # Define the parameters
            betas = (1e-4, 0.02)
            n_T = T
            
            # Create an instance of the DDPM class
            ddpm = DDPM(betas=betas, n_T=n_T)
            g_t, g_t_minus_1, noise = ddpm.get_noisy_images_and_noise(gaussian_splat_normalized, timesteps)
            print("g_t shape:", g_t.shape)
            print("g_t-1 shape:", g_t_minus_1.shape)
            print("noise shape: ", noise.shape)

            predicted_noise = model(g_t, timesteps)
            loss = criterion(predicted_noise, noise)

            # loss = combined_loss(g_t_minus_1, g_t, noise, predicted_noise, kernel_size=17)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(DEVICE)

                gaussian_splat= images
                gaussian_splat_normalized = normalize_parameters(gaussian_splat, param_ranges)
                # gaussian_splat_normalized = gaussian_splat
                # print("normalized gaussian splat shape: ", gaussian_splat_normalized.shape)

                timesteps = torch.randint(low=0, high=T, size=(len(images),), device=DEVICE)
                
                # creating the DDPM instance
                # Define the parameters
                betas = (1e-4, 0.02)
                n_T = 1000
                
                # Create an instance of the DDPM class
                ddpm = DDPM(betas=betas, n_T=n_T)
                g_t, g_t_minus_1, noise = ddpm.get_noisy_images_and_noise(gaussian_splat_normalized, timesteps)
                # print("g_t shape:", g_t.shape)
                # print("g_t-1 shape:", g_t_minus_1.shape)
                # print("noise shape: ", noise.shape)

                predicted_noise = model(g_t, timesteps)
                loss = criterion(predicted_noise, noise)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        save_model(model, optimizer, epoch, val_loss)

    return train_losses, val_losses


def save_model(model, optimizer, epoch, loss, save_path=SAVE_PATH, filename=SAVE_FILENAME):
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


def main():
    wandb.init(project=f'GSD_Epoch{NUM_EPOCHS}_Batch{BATCH_SIZE}', config={"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE})

    dataset = GaussianDataset(DATA_FOLDER)
    num_samples = len(dataset)
    train_size = int(0.6 * num_samples)
    val_size = int(0.2 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_gaussians = 70
    gaussian_features = 7
    depth = 1
    local_block_dim = [16, 32]
    global_block_dim = [32, 16, 7]
    time_emb_dim = 8
    model = GaussianSplatDiffusionModel(num_gaussians, gaussian_features, depth, local_block_dim, global_block_dim, time_emb_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    train_losses, val_losses = train_model(train_loader, val_loader, model, scheduler, optimizer, NUM_EPOCHS)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.title("Train and Validation Losses")
    plt.legend()
    plt.show()
    plt.savefig(f"loss_plot_b{BATCH_SIZE}_e{NUM_EPOCHS}_temb{time_emb_dim}_maxdim{local_block_dim[-1]}.png")


if __name__ == "__main__":
    main()