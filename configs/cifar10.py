"""CIFAR-10 dataset configuration for GaussianDiffusion."""

CIFAR10_CONFIG = {
    "name": "cifar10",
    "image_size": 32,
    "channels": 3,
    "num_classes": 10,
    "num_gaussians": 500,
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
}
