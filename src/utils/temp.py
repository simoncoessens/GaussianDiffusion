def generate_2D_gaussian_splatting(kernel_size, sigma_x, sigma_y, rho, coords, colours, image_size=(256, 256, 3), device="cpu"):

    batch_size = colours.shape[0]

    sigma_x = sigma_x.view(batch_size, 1, 1)
    sigma_y = sigma_y.view(batch_size, 1, 1)
    rho = rho.view(batch_size, 1, 1)

    covariance = torch.stack(
        [torch.stack([sigma_x**2, rho*sigma_x*sigma_y], dim=-1),
        torch.stack([rho*sigma_x*sigma_y, sigma_y**2], dim=-1)],
        dim=-2
    )

    # Check for positive semi-definiteness
    determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    if (determinant <= 0).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    inv_covariance = torch.inverse(covariance)

    # Choosing quite a broad range for the distribution [-5,5] to avoid any clipping
    start = torch.tensor([-5.0], device=device).view(-1, 1)
    end = torch.tensor([5.0], device=device).view(-1, 1)
    base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
    ax_batch = start + (end - start) * base_linspace

    # Expanding dims for broadcasting
    ax_batch_expanded_x = ax_batch.unsqueeze(-1).expand(-1, -1, kernel_size)
    ax_batch_expanded_y = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1)

    # Creating a batch-wise meshgrid using broadcasting
    xx, yy = ax_batch_expanded_x, ax_batch_expanded_y

    xy = torch.stack([xx, yy], dim=-1)
    z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))


    kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
    kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
    kernel_normalized = kernel / kernel_max_2

    kernel_reshaped = kernel_normalized.repeat(1, 3, 1).view(batch_size * 3, kernel_size, kernel_size)
    kernel_rgb = kernel_reshaped.unsqueeze(0).reshape(batch_size, 3, kernel_size, kernel_size)

    # Calculating the padding needed to match the image size
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    # Adding padding to make kernel size equal to the image size
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2,  # padding left and right
               pad_h // 2, pad_h // 2 + pad_h % 2)  # padding top and bottom

    kernel_rgb_padded = torch.nn.functional.pad(kernel_rgb, padding, "constant", 0)

    # Extracting shape information
    b, c, h, w = kernel_rgb_padded.shape

    # Create a batch of 2D affine matrices
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    # Creating grid and performing grid sampling
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    rgb_values_reshaped = colours.unsqueeze(-1).unsqueeze(-1)

    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1,2,0)

    return final_image