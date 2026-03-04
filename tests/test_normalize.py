"""
Tests for normalize_parameters and denormalize_parameters
(src/utils/normalize.py, src/utils/denormalize.py).
"""

import pytest
import torch

from src.utils.normalize import normalize_parameters
from src.utils.denormalize import denormalize_parameters

# MNIST physical param ranges: [sigma_x, sigma_y, rho, alpha, colour, x, y]
PARAM_RANGES = [
    (0.0, 1.0),   # sigma_x
    (0.0, 1.0),   # sigma_y
    (-1.0, 1.0),  # rho
    (0.0, 1.0),   # alpha
    (0.0, 1.0),   # colour
    (-1.0, 1.0),  # x
    (-1.0, 1.0),  # y
]

K = 70


def _make_W():
    """Random physically valid [K, 7] tensor."""
    torch.manual_seed(42)
    W = torch.zeros(K, 7)
    W[:, 0] = torch.rand(K)           # sigma_x ∈ (0,1)
    W[:, 1] = torch.rand(K)           # sigma_y ∈ (0,1)
    W[:, 2] = torch.rand(K) * 2 - 1  # rho ∈ (-1,1)
    W[:, 3] = torch.rand(K)           # alpha ∈ (0,1)
    W[:, 4] = torch.rand(K)           # colour ∈ (0,1)
    W[:, 5] = torch.rand(K) * 2 - 1  # x ∈ (-1,1)
    W[:, 6] = torch.rand(K) * 2 - 1  # y ∈ (-1,1)
    return W


# 1
def test_normalize_output_range():
    W = _make_W()
    W_norm = normalize_parameters(W, PARAM_RANGES)
    assert W_norm.min() >= -1.0 - 1e-6, f"Min below -1: {W_norm.min():.4f}"
    assert W_norm.max() <= 1.0 + 1e-6, f"Max above +1: {W_norm.max():.4f}"


# 2
def test_round_trip_identity():
    W = _make_W()
    W_norm = normalize_parameters(W, PARAM_RANGES)
    W_rec = denormalize_parameters(W_norm, PARAM_RANGES)
    assert torch.allclose(W, W_rec, atol=1e-6), "Round-trip denorm(norm(W)) ≠ W"


# 3
def test_boundary_values():
    """Min-value maps to -1; max-value maps to +1."""
    W = _make_W()
    for i, (mn, mx) in enumerate(PARAM_RANGES):
        W[0, i] = mn
        W[1, i] = mx
    W_norm = normalize_parameters(W, PARAM_RANGES)
    for i in range(7):
        assert abs(W_norm[0, i].item() - (-1.0)) < 1e-5, f"Col {i}: min val → {W_norm[0,i]:.4f} ≠ -1"
        assert abs(W_norm[1, i].item() - 1.0) < 1e-5, f"Col {i}: max val → {W_norm[1,i]:.4f} ≠ +1"


# 4
def test_degenerate_range():
    """When min==max, norm should yield 0 and denorm should yield min_val."""
    degenerate_ranges = [(0.5, 0.5)] * 7
    W = torch.full((K, 7), 0.5)
    W_norm = normalize_parameters(W, degenerate_ranges)
    assert torch.all(W_norm == 0.0), "Degenerate range should normalize to 0"
    W_rec = denormalize_parameters(W_norm, degenerate_ranges)
    assert torch.allclose(W_rec, torch.full((K, 7), 0.5)), "Degenerate denorm should give min_val"


# 5
def test_gradient_flows():
    W = _make_W().requires_grad_(True)
    W_norm = normalize_parameters(W, PARAM_RANGES)
    W_rec = denormalize_parameters(W_norm, PARAM_RANGES)
    loss = W_rec.sum()
    loss.backward()
    assert W.grad is not None, "Gradient did not flow through normalize→denormalize"


# 6
def test_shape_preserved_2d():
    W = _make_W()
    assert normalize_parameters(W, PARAM_RANGES).shape == (K, 7)
    assert denormalize_parameters(W, PARAM_RANGES).shape == (K, 7)


# 7
def test_shape_preserved_3d():
    N = 32
    W3 = _make_W().unsqueeze(0).expand(N, -1, -1).clone()
    assert normalize_parameters(W3, PARAM_RANGES).shape == (N, K, 7)
    assert denormalize_parameters(W3, PARAM_RANGES).shape == (N, K, 7)
