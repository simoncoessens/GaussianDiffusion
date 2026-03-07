"""Dataset configuration registry."""

from configs.mnist import MNIST_CONFIG

CONFIGS = {
    "mnist": MNIST_CONFIG,
}


def get_config(name: str) -> dict:
    """Return config dict by dataset name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
