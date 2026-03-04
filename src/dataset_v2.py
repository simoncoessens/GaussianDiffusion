"""
GaussianDatasetV2 — PyTorch Dataset backed by a single HDF5 file.

Replaces the legacy GaussianDataset (which loads from a directory of .pt files)
with a memory-efficient HDF5-based loader that supports:
  - Preloaded mode (default): reads all W + labels into RAM at init → fast __getitem__
  - Lazy mode (preload=False): keeps file open, reads per-index (memory-efficient)
  - Filtering: by split, minimum PSNR, and digit class

Same __getitem__ signature as GaussianDataset:
  Returns (W_tensor [K, 7], label: int)

Typical usage:
    from src.dataset_v2 import GaussianDatasetV2

    ds = GaussianDatasetV2("data/mnist_gaussians_K70.h5")
    ds_train = GaussianDatasetV2("data/mnist_gaussians_K70.h5", split="train")
    ds_clean = GaussianDatasetV2("data/mnist_gaussians_K70.h5", min_psnr=35.0)
    ds_digits = GaussianDatasetV2("data/mnist_gaussians_K70.h5", digits=[0, 1, 2])

Switching from GaussianDataset (one-liner):
    # Old:  dataset = GaussianDataset("data/mnist_gaussian_representations/")
    # New:  dataset = GaussianDatasetV2("data/mnist_gaussians_K70.h5")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class GaussianDatasetV2(Dataset):
    """
    PyTorch Dataset that reads Gaussian splatting representations from an HDF5 file.

    Parameters
    ----------
    h5_path : str | Path
        Path to the merged HDF5 file (e.g. data/mnist_gaussians_K70.h5).
    split : {'train', 'test', None}
        If 'train', only include original MNIST training images (orig_split=0).
        If 'test',  only include original MNIST test images (orig_split=1).
        If None (default), include all images.
    min_psnr : float | None
        If set, exclude images with PSNR < min_psnr dB.
    digits : list[int] | None
        If set, only include images from the specified digit classes.
    preload : bool
        If True (default), load W and labels arrays fully into RAM at init.
        If False, keep HDF5 file open and read each sample on demand (slower
        __getitem__, lower memory).
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        split: Optional[str] = None,
        min_psnr: Optional[float] = None,
        digits: Optional[list[int]] = None,
        preload: bool = True,
    ):
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for GaussianDatasetV2. "
                "Install it: pip install h5py"
            )

        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.preload = preload
        self._h5_handle = None  # used in lazy mode

        # ------------------------------------------------------------------
        # Build index mask from filtering criteria
        # ------------------------------------------------------------------
        with h5py.File(self.h5_path, "r") as hf:
            n_total = hf["W"].shape[0]
            labels_np    = hf["labels"][:]       # uint8 [N]
            orig_split   = hf["orig_split"][:]   # uint8 [N] (0=train, 1=test)
            psnr_np      = hf["psnr"][:]         # float32 [N]

            mask = np.ones(n_total, dtype=bool)

            if split == "train":
                mask &= (orig_split == 0)
            elif split == "test":
                mask &= (orig_split == 1)
            elif split is not None:
                raise ValueError(f"split must be 'train', 'test', or None. Got: {split!r}")

            if min_psnr is not None:
                mask &= (psnr_np >= min_psnr)

            if digits is not None:
                digit_mask = np.zeros(n_total, dtype=bool)
                for d in digits:
                    digit_mask |= (labels_np == d)
                mask &= digit_mask

            self._indices = np.where(mask)[0].astype(np.int32)
            self._labels = labels_np[self._indices]  # always keep labels in RAM

            # Metadata attributes (useful for logging/display)
            self.K             = int(hf.attrs.get("K", hf["W"].shape[1]))
            self.encoder_version = str(hf.attrs.get("encoder_version", "unknown"))
            self.global_mean_psnr = float(psnr_np.mean()) if len(psnr_np) else 0.0

            if preload:
                self._W = hf["W"][self._indices]  # float32 [N_filtered, K, 7]

        self._n = len(self._indices)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        """
        Returns (W_tensor [K, 7], label: int)

        Matches the signature of the legacy GaussianDataset.__getitem__.
        """
        if self.preload:
            W_np = self._W[idx]
        else:
            import h5py
            if self._h5_handle is None or not self._h5_handle.id.valid:
                self._h5_handle = h5py.File(self.h5_path, "r")
            global_idx = int(self._indices[idx])
            W_np = self._h5_handle["W"][global_idx]

        W_tensor = torch.from_numpy(W_np.copy()).float()  # [K, 7]
        label = int(self._labels[idx])
        return W_tensor, label

    def __del__(self):
        if self._h5_handle is not None:
            try:
                self._h5_handle.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_psnr_stats(self) -> dict:
        """Return PSNR statistics for the filtered subset."""
        import h5py
        with h5py.File(self.h5_path, "r") as hf:
            psnr = hf["psnr"][self._indices]
        return {
            "n":    len(psnr),
            "mean": float(psnr.mean()),
            "std":  float(psnr.std()),
            "min":  float(psnr.min()),
            "max":  float(psnr.max()),
            "pct_below_30": float((psnr < 30.0).mean()) * 100,
            "pct_below_35": float((psnr < 35.0).mean()) * 100,
        }

    def __repr__(self) -> str:
        return (
            f"GaussianDatasetV2(n={self._n}, K={self.K}, "
            f"preload={self.preload}, h5='{self.h5_path.name}')"
        )
