"""CelebA-64 dataset configuration for GaussianDiffusion."""

CELEBA64_CONFIG = {
    "name": "celeba64",
    "image_size": 64,
    "channels": 3,
    "num_classes": 0,  # unconditional generation
    "num_gaussians": 1000,
    "feature_dim": 8,  # after alpha drop: sigma_x, sigma_y, rho, r, g, b, x, y
    "param_ranges": [
        (0.0, 1.0),    # sigma_x
        (0.0, 1.0),    # sigma_y
        (-1.0, 1.0),   # rho
        (0.0, 1.0),    # r
        (0.0, 1.0),    # g
        (0.0, 1.0),    # b
        (-1.0, 1.0),   # x
        (-1.0, 1.0),   # y
    ],
    # Renderer settings (must match encoding)
    "kernel_size": 32,
    "soft_clamp": True,
    # CelebA-specific
    "n_total": 202599,
    "n_train": 162770,
    "n_val": 19867,
    "n_test": 19962,
    "crop_size": 140,  # center crop before resize to 64x64
}
