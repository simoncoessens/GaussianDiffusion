"""
Tests for ssim_loss and _gaussian_kernel (src/encode.py).
"""

import pytest
import torch

from src.encode import ssim_loss, _gaussian_kernel

DEVICE = "cpu"


# 1
def test_identical_images_zero_loss():
    x = torch.rand(28, 28)
    assert ssim_loss(x, x).item() < 1e-5


# 2
def test_loss_nonnegative():
    torch.manual_seed(0)
    for _ in range(10):
        a = torch.rand(28, 28)
        b = torch.rand(28, 28)
        assert ssim_loss(a, b).item() >= 0.0


# 3
def test_black_vs_white_high_loss():
    zeros = torch.zeros(28, 28)
    ones = torch.ones(28, 28)
    assert ssim_loss(zeros, ones).item() > 0.9


# 4
def test_symmetry():
    torch.manual_seed(1)
    a = torch.rand(28, 28)
    b = torch.rand(28, 28)
    assert abs(ssim_loss(a, b).item() - ssim_loss(b, a).item()) < 1e-6


# 5
def test_no_nan_on_uniform_image():
    zeros = torch.zeros(28, 28)
    loss = ssim_loss(zeros, zeros)
    assert torch.isfinite(loss), "SSIM produced NaN/Inf on uniform image"


# 6
def test_noise_increases_loss():
    torch.manual_seed(2)
    x = torch.rand(28, 28)
    eps = 0.1 * torch.rand(28, 28)
    loss_clean = ssim_loss(x, x).item()
    loss_noisy = ssim_loss(x + eps, x).item()
    assert loss_noisy > loss_clean


# 7
def test_gradient_flows():
    pred = torch.rand(28, 28, requires_grad=True)
    target = torch.rand(28, 28)
    loss = ssim_loss(pred, target)
    loss.backward()
    assert pred.grad is not None


# 8
def test_shape_agnostic():
    torch.manual_seed(3)
    x = torch.rand(28, 28)
    loss_2d = ssim_loss(x, x)
    loss_3d = ssim_loss(x.unsqueeze(0), x.unsqueeze(0))
    assert abs(loss_2d.item() - loss_3d.item()) < 1e-6
