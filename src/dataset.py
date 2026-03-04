import os
import torch
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    """
    Dataset of 2D Gaussian splatting representations for MNIST.
    Each sample is a tensor of shape [K, 7] loaded from a .pt file
    saved by src/encode.py.
    """

    def __init__(self, data_dir: str, num_gaussians: int = 70):
        self.data_dir = data_dir
        self.files = sorted(
            f for f in os.listdir(data_dir) if f.endswith(".pt")
        )
        if not self.files:
            raise ValueError(f"No .pt files found in {data_dir}")

        samples = []
        for fname in self.files:
            path = os.path.join(data_dir, fname)
            try:
                w = torch.load(path, weights_only=True)["W"]
            except (KeyError, Exception):
                continue
            if w.shape[0] == num_gaussians:
                samples.append(w)

        if not samples:
            raise ValueError(
                f"No valid samples with {num_gaussians} Gaussians found in {data_dir}"
            )

        self.data = torch.stack(samples, dim=0)  # [N, K, 7]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
