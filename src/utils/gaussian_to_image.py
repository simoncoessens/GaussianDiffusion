#!/usr/bin/env python
"""
Utility function for converting Gaussian splatting to an image.
"""

import torch
import torch.nn.functional as F
import numpy as np


def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho,
                                   coords, colours, image_size=(28, 28),
                                   channels=1, device="cuda"):
    """
    Generate an image via Gaussian splatting based on the provided parameters.

    Args:
        kernel_size (int): Size of the Gaussian kernel.
        sigma_x (torch.Tensor): Tensor of standard deviations along x, shape [batch_size].
        sigma_y (torch.Tensor): Tensor of standard deviations along y, shape [batch_size].
        rho (torch.Tensor): Tensor of correlation coefficients, shape [batch_size].
        coords (torch.Tensor): Tensor of coordinates for affine translation, shape [batch_size, 2].
        colours (torch.Tensor): Tensor of colour/intensity values, shape [batch_size, 1].
        image_size (tuple, optional): Final output image dimensions (H, W). Defaults to (28, 28).
        channels (int, optional): Number of image channels. Defaults to 1.
        device (str, optional): Device for computation. Defaults to "cuda".

    Returns:
        torch.Tensor: Generated image tensor of shape [H, W, channels] with values in [0, 1].
    """
    
    # Ensure all input tensors are on the correct device.
    sigma_x = sigma_x.to(device)
    sigma_y = sigma_y.to(device)
    rho = rho.to(device)
    coords = coords.to(device)
    colours = colours.to(device)
    
    batch_size = colours.shape[0]

    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    # Build the covariance matrix for each Gaussian
    covariance = torch.stack(
        [
            torch.stack([sigma_x**2, rho * sigma_x * sigma_y], dim=-1),
            torch.stack([rho * sigma_x * sigma_y, sigma_y**2], dim=-1)
        ],
        dim=-2
    )

    determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    epsilon = 1e-6  # Small value to ensure positive semi-definiteness
    if (determinant <= 0).any():
        #print("Determinant:", determinant)
        covariance[..., 0, 0] += epsilon
        covariance[..., 1, 1] += epsilon
    try:
        inv_covariance = torch.inverse(covariance)
    except RuntimeError as e:
        raise ValueError("Covariance matrix inversion failed. Check input parameters.") from e

    # Create a coordinate grid for the kernel
    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
    xy = torch.stack([xx, yy], dim=-1)

    # Calculate the Gaussian kernel
    z = torch.einsum(
        'b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy
    )
    kernel = (
        torch.exp(z) /
        (2 * torch.tensor(np.pi, device=device) *
         torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))
    )

    # Normalize the kernel
    kernel_max = kernel.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    kernel_normalized = kernel / kernel_max

    # Prepare kernel for channel repetition and later transformation
    kernel_reshaped = kernel_normalized.repeat(1, channels, 1).view(batch_size * channels, kernel_size, kernel_size)
    kernel_channels = kernel_reshaped.unsqueeze(0).reshape(batch_size, channels, kernel_size, kernel_size)

    # Compute required padding
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    padding = (
        pad_w // 2, pad_w // 2 + pad_w % 2,
        pad_h // 2, pad_h // 2 + pad_h % 2
    )

    # Pad the kernel to match the target image size
    kernel_padded = F.pad(kernel_channels, padding, "constant", 0)

    # Apply an affine transformation to translate the kernel based on provided coordinates
    b, c, h, w = kernel_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_transformed = F.grid_sample(kernel_padded, grid, align_corners=True)

    # Combine the kernel with the colour/intensity values to form the final image
    colours_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = colours_reshaped * kernel_transformed

    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)  # Shape: [H, W, channels]

    return final_image

def generate_2D_gaussian_splatting_gray(kernel_size, sigma_x, sigma_y, rho,
                                   coords, colours, image_size=(28, 28),
                                   channels=1, device="cuda"):
    """
    Generate an image via Gaussian splatting based on the provided parameters.
    Pixels outside the Gaussian(s) (i.e. not covered by the transformed kernel)
    will have a value of 0.5, while pixels inside will show the Gaussian’s colour.

    Args:
        kernel_size (int): Size of the Gaussian kernel.
        sigma_x (torch.Tensor): Standard deviations along x, shape [batch_size].
        sigma_y (torch.Tensor): Standard deviations along y, shape [batch_size].
        rho (torch.Tensor): Correlation coefficients, shape [batch_size].
        coords (torch.Tensor): Coordinates for affine translation, shape [batch_size, 2].
        colours (torch.Tensor): Colour/intensity values, shape [batch_size, 1].
        image_size (tuple, optional): Final output image dimensions (H, W). Defaults to (28, 28).
        channels (int, optional): Number of image channels. Defaults to 1.
        device (str, optional): Computation device. Defaults to "cuda".

    Returns:
        torch.Tensor: Final image tensor of shape [H, W, channels] with values in [0, 1].
    """
    batch_size = colours.shape[0]
    epsilon = 1e-6  # Regularization to ensure invertibility

    # Reshape parameters to [batch_size, 1, 1]
    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    # Build the covariance matrix for each Gaussian:
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

    # Create a coordinate grid for the kernel.
    # The grid spans from -5 to 5 in both axes.
    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    # Create two expanded axes and stack to form a grid [kernel_size, kernel_size, 2]
    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y
    xy = torch.stack([xx, yy], dim=-1)

    # Calculate the Gaussian kernel.
    z = torch.einsum(
        'b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy
    )
    kernel = (
        torch.exp(z) /
        (2 * torch.tensor(np.pi, device=device) *
         torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))
    )

    # Normalize the kernel (maximum value becomes 1).
    kernel_max = kernel.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    kernel_normalized = kernel / kernel_max

    # Prepare kernel for channel repetition.
    # Resulting shape: [batch_size, channels, kernel_size, kernel_size]
    kernel_reshaped = kernel_normalized.repeat(1, channels, 1).view(batch_size * channels, kernel_size, kernel_size)
    kernel_channels = kernel_reshaped.unsqueeze(0).reshape(batch_size, channels, kernel_size, kernel_size)

    # Compute padding to embed the kernel into an image of size image_size.
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    # Padding format: (left, right, top, bottom)
    padding = (
        pad_w // 2, pad_w // 2 + pad_w % 2,
        pad_h // 2, pad_h // 2 + pad_h % 2
    )

    # Pad the kernel with zeros.
    kernel_padded = F.pad(kernel_channels, padding, "constant", 0)

    # Also create a binary mask from the unpadded kernel: 1 where kernel is present, 0 elsewhere.
    mask_channels = (kernel_channels > 0).float()
    mask_padded = F.pad(mask_channels, padding, "constant", 0)

    # Prepare affine transformation to translate the kernel based on provided coordinates.
    b, c, h, w = kernel_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords  # translation parameters

    # Generate the sampling grid.
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)

    # Apply the affine transformation.
    kernel_transformed = F.grid_sample(kernel_padded, grid, align_corners=True)
    # Use nearest-neighbor for the mask so it stays binary.
    mask_transformed = F.grid_sample(mask_padded, grid, align_corners=True, mode='nearest')

    # Compute the Gaussian contribution.
    # Multiply by the colour/intensity values.
    colours_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    gaussian_contribution = colours_reshaped * kernel_transformed  # shape: [batch_size, channels, h, w]

    # Combine contributions from all Gaussians.
    # For the Gaussian values, we sum over the batch dimension.
    gaussian_sum = gaussian_contribution.sum(dim=0)  # shape: [channels, h, w]
    # For the mask, if any Gaussian covers a pixel, mark it as 1.
    mask_combined = mask_transformed.max(dim=0)[0]     # shape: [channels, h, w]

    # Rearrange dimensions to [H, W, channels].
    gaussian_sum = gaussian_sum.permute(1, 2, 0)
    mask_combined = mask_combined.permute(1, 2, 0)

    # Composite the final image:
    # For pixels where mask==1, use the Gaussian value; otherwise, use the background value 0.5.
    final_image = mask_combined * gaussian_sum + (1 - mask_combined) * 0.5
    final_image = torch.clamp(final_image, 0, 1)

    return final_image


if __name__ == "__main__":
    # Example usage:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    kernel_size = 17
    image_size = (28, 28)
    channels = 1

    # Simulate latent parameters (make sure the tensor dimensions match expected sizes)
    latent_denorm = torch.randn(batch_size, 7, device=DEVICE)
    sigma_x = torch.sigmoid(latent_denorm[:, 0])
    sigma_y = torch.sigmoid(latent_denorm[:, 1])
    rho = torch.tanh(latent_denorm[:, 2])
    # Note: alpha is extracted but not used in the function.
    alpha = torch.sigmoid(latent_denorm[:, 3])
    colours = torch.clamp(latent_denorm[:, 4:5], 0, 1)
    coords = latent_denorm[:, 5:7]

    # Generate the final image from the latent Gaussian parameters.
    final_image = generate_2D_gaussian_splatting(
        kernel_size=kernel_size,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        rho=rho,
        coords=coords,
        colours=colours,
        image_size=image_size,
        channels=channels,
        device=DEVICE
    )

    print("Generated image shape:", final_image.shape)
