"""
Tests for GaussianDataset (src/dataset.py).
"""

import os
import tempfile

import pytest
import torch

from src.dataset import GaussianDataset

K = 10


def _make_dataset(tmpdir, n=5, k=K):
    """Write n valid .pt files with shape [k, 7]."""
    for i in range(n):
        W = torch.rand(k, 7)
        torch.save({"W": W}, os.path.join(tmpdir, f"img_{i:04d}.pt"))


# 1
def test_loads_all_valid_files():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=5)
        ds = GaussianDataset(d, num_gaussians=K)
        assert len(ds) == 5


# 2
def test_sample_shape():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=3)
        ds = GaussianDataset(d, num_gaussians=K)
        assert ds[0].shape == (K, 7)


# 3
def test_full_data_shape():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=4)
        ds = GaussianDataset(d, num_gaussians=K)
        assert ds.data.shape == (4, K, 7)


# 4
def test_filters_wrong_k():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=3, k=K)
        # Add a file with wrong K
        torch.save({"W": torch.rand(K + 5, 7)}, os.path.join(d, "wrong_k.pt"))
        ds = GaussianDataset(d, num_gaussians=K)
        assert len(ds) == 3, f"Expected 3 valid files, got {len(ds)}"


# 5
def test_raises_on_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            GaussianDataset(d, num_gaussians=K)


# 6
def test_missing_w_key_skips():
    """File without 'W' key should be skipped silently, not raise KeyError."""
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=3)
        # Save a file with a different key
        torch.save({"X": torch.rand(K, 7)}, os.path.join(d, "bad_key.pt"))
        ds = GaussianDataset(d, num_gaussians=K)
        assert len(ds) == 3, "Bad-key file should have been skipped"


# 7
def test_getitem_returns_float_tensor():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=2)
        ds = GaussianDataset(d, num_gaussians=K)
        assert ds[0].dtype == torch.float32


# 8
def test_values_preserved():
    with tempfile.TemporaryDirectory() as d:
        W_saved = torch.rand(K, 7)
        torch.save({"W": W_saved}, os.path.join(d, "img_0000.pt"))
        ds = GaussianDataset(d, num_gaussians=K)
        assert (ds[0] - W_saved).abs().max() < 1e-7


# 9
def test_deterministic_ordering():
    with tempfile.TemporaryDirectory() as d:
        _make_dataset(d, n=5)
        ds1 = GaussianDataset(d, num_gaussians=K)
        ds2 = GaussianDataset(d, num_gaussians=K)
        assert torch.allclose(ds1.data, ds2.data), "Two loads of same dir gave different ordering"
