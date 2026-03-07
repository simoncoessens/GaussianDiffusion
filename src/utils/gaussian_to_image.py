#!/usr/bin/env python
"""
Utility function for converting Gaussian splatting to an image.
"""

import math

import torch
import torch.nn.functional as F


def _generate_direct(sigma_x, sigma_y, rho, coords, colours,
                     image_size, channels, device, soft_clamp=False):
    """Direct Gaussian evaluation at pixel positions — no affine_grid/grid_sample.

    Faster than the pad+translate approach when kernel covers the full image.
    Uses analytical 2x2 covariance inverse and efficient matmul for colour mixing.

    Valid when kernel_size >= min(image_size) (no padding needed).
    """
    K = sigma_x.shape[0]
    H, W = image_size

    # Pixel grid in [-1, 1]
    gx = torch.linspace(-1, 1, W, device=device)
    gy = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [H, W]

    # Shifted positions scaled to kernel coords [-5, 5]: kc = 5 * (pixel + coord)
    # Then divide by sigma to get normalized displacement: ux = 5*(p+t)/sx
    sx = sigma_x.view(-1, 1, 1)
    sy = sigma_y.view(-1, 1, 1)
    r = rho.view(-1, 1, 1)

    ux = (grid_x.unsqueeze(0) + coords[:, 0].view(-1, 1, 1)) * (5.0 / sx)  # [K, H, W]
    uy = (grid_y.unsqueeze(0) + coords[:, 1].view(-1, 1, 1)) * (5.0 / sy)

    # Simplified quadratic form: z = -0.5/(1-r²) * (ux² - 2r·ux·uy + uy²)
    one_minus_r2 = (1 - r * r).clamp(min=1e-6)
    z = (-0.5 / one_minus_r2) * (ux * ux - 2 * r * ux * uy + uy * uy)

    # Numerically stable normalization: subtract max before exp so peak = 1.0
    z_max = z.view(K, -1).max(dim=1)[0].view(-1, 1, 1)
    kernel = torch.exp(z - z_max)  # [K, H, W], max value = 1.0

    # Efficient colour combination via matmul: [C, K] @ [K, H*W] → [C, H*W]
    final = torch.mm(colours.T, kernel.view(K, -1)).view(channels, H, W)

    if soft_clamp:
        final = 1.0 - torch.exp(-final.clamp(min=0))
    else:
        final = torch.clamp(final, 0, 1)

    return final.permute(1, 2, 0)  # [H, W, C]


def generate_2D_gaussian_splatting_batch(kernel_size, sigma_x, sigma_y, rho,
                                         coords, colours, image_size=(32, 32),
                                         channels=3, device="cuda",
                                         soft_clamp=False):
    """Batched direct Gaussian evaluation for multiple images in parallel.

    Args:
        sigma_x: [B, K]
        sigma_y: [B, K]
        rho:     [B, K]
        coords:  [B, K, 2]
        colours: [B, K, C]

    Returns:
        [B, H, W, C] image tensor with values in [0, 1].
    """
    B, K = sigma_x.shape
    H, W = image_size
    C = channels

    gx = torch.linspace(-1, 1, W, device=device)
    gy = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [H, W]

    sx = sigma_x.view(B, K, 1, 1)
    sy = sigma_y.view(B, K, 1, 1)
    r = rho.view(B, K, 1, 1)

    ux = (grid_x + coords[:, :, 0].view(B, K, 1, 1)) * (5.0 / sx)  # [B, K, H, W]
    uy = (grid_y + coords[:, :, 1].view(B, K, 1, 1)) * (5.0 / sy)

    one_minus_r2 = (1 - r * r).clamp(min=1e-6)
    z = (-0.5 / one_minus_r2) * (ux * ux - 2 * r * ux * uy + uy * uy)

    z_max = z.view(B, K, -1).max(dim=2)[0].view(B, K, 1, 1)
    kernel = torch.exp(z - z_max)  # [B, K, H, W]

    # Batched matmul: [B, C, K] @ [B, K, H*W] → [B, C, H*W]
    final = torch.bmm(
        colours.permute(0, 2, 1),       # [B, C, K]
        kernel.view(B, K, -1),          # [B, K, H*W]
    ).view(B, C, H, W)

    if soft_clamp:
        final = 1.0 - torch.exp(-final.clamp(min=0))
    else:
        final = torch.clamp(final, 0, 1)

    return final.permute(0, 2, 3, 1)  # [B, H, W, C]


def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho,
                                   coords, colours, image_size=(28, 28),
                                   channels=1, device="cuda",
                                   soft_clamp=False):
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

    # Fast path: direct evaluation when kernel covers the full image
    if kernel_size >= min(image_size[0], image_size[1]):
        return _generate_direct(sigma_x, sigma_y, rho, coords, colours,
                                image_size, channels, device, soft_clamp)

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
    if soft_clamp:
        final_image = 1.0 - torch.exp(-final_image.clamp(min=0))
    else:
        final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)  # Shape: [H, W, channels]

    return final_image
