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
    # Per-feature standardization (computed from 60k training data, normalized to [-1,1])
    # After normalizing to [-1,1], apply: z = (x - mean) / std
    # This makes data ~N(0,1) which DDPM expects.
    "data_mean": [-0.4648, -0.4656, -0.0004, -0.1467, -0.1723, -0.2139, 0.0005, 0.0246],
    "data_std":  [ 0.1957,  0.1933,  0.3157,  0.5278,  0.5036,  0.5626,  0.6028,  0.6082],
}
