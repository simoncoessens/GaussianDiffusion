"""Dataset configuration registry."""

from configs.mnist import MNIST_CONFIG
from configs.cifar10 import CIFAR10_CONFIG
from configs.celeba64 import CELEBA64_CONFIG

CONFIGS = {
    "mnist": MNIST_CONFIG,
    "cifar10": CIFAR10_CONFIG,
    "celeba64": CELEBA64_CONFIG,
}


def get_config(name: str) -> dict:
    """Return config dict by dataset name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
