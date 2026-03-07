"""MNIST dataset configuration for GaussianDiffusion."""

MNIST_CONFIG = {
    "name": "mnist",
    "image_size": 28,
    "num_classes": 10,
    "num_gaussians": 70,
    "feature_dim": 6,  # after alpha drop
    "param_ranges": [
        (0.0, 1.0),    # sigma_x
        (0.0, 1.0),    # sigma_y
        (-1.0, 1.0),   # rho
        (0.0, 1.0),    # colour
        (-1.0, 1.0),   # x
        (-1.0, 1.0),   # y
    ],
}
