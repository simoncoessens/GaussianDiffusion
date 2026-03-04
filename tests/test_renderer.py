"""
Tests for generate_2D_gaussian_splatting (src/utils/gaussian_to_image.py).
"""

import pytest
import torch

from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

DEVICE = "cpu"
IMAGE_SIZE = (28, 28)
KERNEL_SIZE = 9


def _single_gaussian(sigma_x=0.3, sigma_y=0.3, rho=0.0, colour=0.8,
                     x=0.0, y=0.0, kernel_size=KERNEL_SIZE, image_size=IMAGE_SIZE):
    K = 1
    return generate_2D_gaussian_splatting(
        kernel_size=kernel_size,
        sigma_x=torch.tensor([sigma_x]),
        sigma_y=torch.tensor([sigma_y]),
        rho=torch.tensor([rho]),
        coords=torch.tensor([[x, y]]),
        colours=torch.tensor([[colour]]),
        image_size=image_size,
        channels=1,
        device=DEVICE,
    )


# 1
def test_output_shape():
    img = _single_gaussian()
    assert img.shape == (28, 28, 1)


# 2
def test_output_range():
    img = _single_gaussian()
    assert img.min() >= 0.0 - 1e-6
    assert img.max() <= 1.0 + 1e-6


# 3
def test_centered_gaussian_peaks_at_center():
    img = _single_gaussian(x=0.0, y=0.0)  # [28, 28, 1]
    flat = img[:, :, 0]
    peak_idx = flat.argmax()
    cy, cx = divmod(peak_idx.item(), 28)
    assert abs(cy - 13) <= 2, f"Peak y={cy} too far from center"
    assert abs(cx - 13) <= 2, f"Peak x={cx} too far from center"


# 4
def test_position_effect():
    img_center = _single_gaussian(x=0.0, y=0.0)[:, :, 0]
    img_shifted = _single_gaussian(x=0.3, y=0.0)[:, :, 0]
    cx_center = img_center.argmax() % 28
    cx_shifted = img_shifted.argmax() % 28
    assert abs(cx_shifted.item() - cx_center.item()) >= 3, "Position shift had no effect"


# 5
def test_sigma_controls_spread():
    def entropy(img_2d):
        p = img_2d / (img_2d.sum() + 1e-8)
        p = p.clamp(min=1e-10)
        return -(p * p.log()).sum()

    img_small = _single_gaussian(sigma_x=0.1, sigma_y=0.1)[:, :, 0]
    img_large = _single_gaussian(sigma_x=0.5, sigma_y=0.5)[:, :, 0]
    assert entropy(img_large) > entropy(img_small), "Larger sigma should have more spread"


# 6
def test_colour_scales_output():
    img_half = _single_gaussian(colour=0.5, sigma_x=0.15, sigma_y=0.15)[:, :, 0]
    img_full = _single_gaussian(colour=1.0, sigma_x=0.15, sigma_y=0.15)[:, :, 0]
    ratio = img_half.max() / (img_full.max() + 1e-8)
    assert abs(ratio.item() - 0.5) < 0.05, f"Colour ratio {ratio:.3f} not ≈ 0.5"


# 7
def test_additive_compositing():
    K = 2
    img_both = generate_2D_gaussian_splatting(
        kernel_size=KERNEL_SIZE,
        sigma_x=torch.tensor([0.1, 0.1]),
        sigma_y=torch.tensor([0.1, 0.1]),
        rho=torch.zeros(2),
        coords=torch.tensor([[-0.6, 0.0], [0.6, 0.0]]),
        colours=torch.tensor([[0.5], [0.5]]),
        image_size=IMAGE_SIZE,
        channels=1,
        device=DEVICE,
    )[:, :, 0]

    img_left = _single_gaussian(sigma_x=0.1, sigma_y=0.1, colour=0.5, x=-0.6)[:, :, 0]
    img_right = _single_gaussian(sigma_x=0.1, sigma_y=0.1, colour=0.5, x=0.6)[:, :, 0]

    combined = (img_left + img_right).clamp(0, 1)
    assert torch.allclose(img_both, combined, atol=1e-5), "Additive compositing failed"


# 8
def test_gradient_flows():
    sigma_x = torch.tensor([0.3], requires_grad=True)
    sigma_y = torch.tensor([0.3], requires_grad=True)
    rho = torch.tensor([0.0], requires_grad=True)
    coords = torch.tensor([[0.0, 0.0]], requires_grad=True)
    colours = torch.tensor([[0.8]], requires_grad=True)

    img = generate_2D_gaussian_splatting(
        kernel_size=KERNEL_SIZE,
        sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
        coords=coords, colours=colours,
        image_size=IMAGE_SIZE, channels=1, device=DEVICE,
    )
    loss = img.sum()
    loss.backward()
    assert sigma_x.grad is not None, "Gradient did not flow to sigma_x"


# 9
def test_near_singular_rho_no_nan():
    img = _single_gaussian(rho=0.99)
    assert torch.isfinite(img).all(), "NaN/Inf with near-singular rho=0.99"


# 10
def test_small_sigma_no_crash():
    """Validates Bug 2 fix: near-zero sigma should not produce NaN via kernel_max."""
    img = _single_gaussian(sigma_x=0.01, sigma_y=0.01)
    assert torch.isfinite(img).all(), "NaN/Inf with small sigma=0.01"


# 11
def test_kernel_equals_image_size():
    img = generate_2D_gaussian_splatting(
        kernel_size=28,
        sigma_x=torch.tensor([0.3]),
        sigma_y=torch.tensor([0.3]),
        rho=torch.zeros(1),
        coords=torch.zeros(1, 2),
        colours=torch.ones(1, 1) * 0.5,
        image_size=(28, 28),
        channels=1,
        device=DEVICE,
    )
    assert img.shape == (28, 28, 1)


# 12
def test_kernel_larger_than_image_raises():
    with pytest.raises(ValueError):
        generate_2D_gaussian_splatting(
            kernel_size=32,
            sigma_x=torch.tensor([0.3]),
            sigma_y=torch.tensor([0.3]),
            rho=torch.zeros(1),
            coords=torch.zeros(1, 2),
            colours=torch.ones(1, 1) * 0.5,
            image_size=(28, 28),
            channels=1,
            device=DEVICE,
        )
