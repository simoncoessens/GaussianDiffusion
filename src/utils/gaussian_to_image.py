#!/usr/bin/env python
"""
Utility function for converting Gaussian splatting to an image.
"""

import math

import torch
import torch.nn.functional as F


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
        inv_covariance = torch.linalg.inv(covariance)
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
        (2 * math.pi *
         torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))
    )

    # Normalize the kernel
    kernel_max = kernel.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    kernel_max = kernel_max.clamp(min=1e-8)
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
