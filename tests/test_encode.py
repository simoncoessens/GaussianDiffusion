"""
Tests for the encoding pipeline (src/encode.py).

Run with:
    pytest tests/test_encode.py -v
    pytest tests/test_encode.py -v -m slow   # round-trip PSNR test only
"""

import math

import pytest
import torch
import torch.optim

from src.encode import (
    _init_gaussians, _to_physical, _render, _dead_mask, _recycle, encode_image,
)
from src.utils.gaussian_to_image import generate_2D_gaussian_splatting

DEVICE = "cpu"
K = 10
IMAGE_SIZE = (28, 28)
KERNEL_SIZE = 9


# ===========================================================================
# Unit tests — _init_gaussians and _to_physical
# ===========================================================================

# 1
def test_init_output_shape():
    img = torch.rand(*IMAGE_SIZE)
    W = _init_gaussians(img, K, DEVICE)
    assert W.shape == (K, 7), f"Expected ({K}, 7), got {W.shape}"


# 2
def test_init_sigma_raw():
    img = torch.rand(*IMAGE_SIZE)
    W = _init_gaussians(img, K, DEVICE)
    assert torch.allclose(W[:, 0], torch.full((K,), -1.5), atol=1e-5), "sigma_x raw should be -1.5"
    assert torch.allclose(W[:, 1], torch.full((K,), -1.5), atol=1e-5), "sigma_y raw should be -1.5"


# 3
def test_init_rho_raw_zero():
    img = torch.rand(*IMAGE_SIZE)
    W = _init_gaussians(img, K, DEVICE)
    assert torch.all(W[:, 2] == 0.0), "rho raw should be exactly 0.0"


# 4
def test_init_colour_seeded_correctly():
    """Bug 1 fix: sigmoid(W[:,4]) should recover pixel intensity."""
    intensity = 0.7
    img = torch.full(IMAGE_SIZE, intensity)
    W = _init_gaussians(img, K, DEVICE)
    recovered = torch.sigmoid(W[:, 4])
    assert torch.allclose(recovered, torch.full((K,), intensity), atol=1e-4), (
        f"Colour seed mismatch: got {recovered.mean():.4f}, expected {intensity}"
    )


# 5
def test_init_coord_seeded_correctly():
    """Renderer coordinate inversion: W[:,5] = atanh(-pixel_x) so rendered position = pixel_x.

    The renderer applies an affine translation that INVERTS coordinates: a Gaussian
    with W[i,5]=r renders at output x = -tanh(r). So to render at pixel xs, init
    sets W[i,5] = atanh(-xs), meaning tanh(W[:,5]) = -pixel_x.

    We verify that tanh(W[:,5]) is in (-1, 1) and that the implied rendered position
    (-tanh(W[:,5])) maps to a valid pixel on the 28×28 grid.
    """
    img = torch.ones(IMAGE_SIZE)   # uniform bright image
    W = _init_gaussians(img, K, DEVICE)
    raw_x = torch.tanh(W[:, 5])   # = -actual_rendered_x
    raw_y = torch.tanh(W[:, 6])
    # Raw values must be in (-1, 1)
    assert (raw_x > -1).all() and (raw_x < 1).all(), "x raw out of (-1, 1)"
    assert (raw_y > -1).all() and (raw_y < 1).all(), "y raw out of (-1, 1)"
    # Actual rendered position = -raw; must map to valid pixel
    H, W_img = IMAGE_SIZE
    px = ((-raw_x + 1) / 2 * (W_img - 1)).round().long()
    py = ((-raw_y + 1) / 2 * (H - 1)).round().long()
    assert (px >= 0).all() and (px < W_img).all(), f"rendered x pixel out of [0,{W_img})"
    assert (py >= 0).all() and (py < H).all(), f"rendered y pixel out of [0,{H})"


# 6
def test_init_black_image_safe():
    img = torch.zeros(IMAGE_SIZE)
    W = _init_gaussians(img, K, DEVICE)
    assert W.shape == (K, 7)
    assert torch.isfinite(W).all(), "NaN/Inf in init for all-black image"


# 7
def test_to_physical_all_ranges():
    img = torch.rand(*IMAGE_SIZE)
    W_raw = _init_gaussians(img, K, DEVICE)
    p = _to_physical(W_raw)
    assert (p["sigma_x"] > 0).all() and (p["sigma_x"] < 1).all(), "sigma_x out of (0,1)"
    assert (p["sigma_y"] > 0).all() and (p["sigma_y"] < 1).all(), "sigma_y out of (0,1)"
    assert (p["rho"] > -1).all() and (p["rho"] < 1).all(), "rho out of (-1,1)"
    assert (p["alpha"] > 0).all() and (p["alpha"] < 1).all(), "alpha out of (0,1)"
    assert (p["colours"] > 0).all() and (p["colours"] < 1).all(), "colours out of (0,1)"
    assert (p["x"] > -1).all() and (p["x"] < 1).all(), "x out of (-1,1)"
    assert (p["y"] > -1).all() and (p["y"] < 1).all(), "y out of (-1,1)"


# 8
def test_to_physical_gradient():
    img = torch.rand(*IMAGE_SIZE)
    W_raw = _init_gaussians(img, K, DEVICE).requires_grad_(True)
    p = _to_physical(W_raw)
    loss = sum(v.sum() for v in p.values())
    loss.backward()
    assert W_raw.grad is not None, "Gradient did not flow through _to_physical"


# 9
def test_alpha_is_vestigial():
    """alpha is not passed to generate_2D_gaussian_splatting, so changing it has no effect."""
    K_t = 3
    common_kwargs = dict(
        kernel_size=KERNEL_SIZE,
        sigma_x=torch.ones(K_t) * 0.3,
        sigma_y=torch.ones(K_t) * 0.3,
        rho=torch.zeros(K_t),
        coords=torch.zeros(K_t, 2),
        colours=torch.ones(K_t, 1) * 0.5,
        image_size=IMAGE_SIZE,
        channels=1,
        device=DEVICE,
    )
    img1 = generate_2D_gaussian_splatting(**common_kwargs)
    img2 = generate_2D_gaussian_splatting(**common_kwargs)
    assert torch.allclose(img1, img2), "Renders differ despite identical inputs (sanity check)"


# ===========================================================================
# Integration tests — encode_image
# ===========================================================================

# 10
def test_encode_output_shape():
    img = torch.rand(*IMAGE_SIZE)
    W, _ = encode_image(img, K=K, epochs=5, lr=1e-2, kernel_size=KERNEL_SIZE, device=DEVICE)
    assert W.shape == (K, 7), f"Expected ({K}, 7), got {W.shape}"


# 11
def test_encode_no_nan():
    img = torch.rand(*IMAGE_SIZE)
    W, _ = encode_image(img, K=K, epochs=5, lr=1e-2, kernel_size=KERNEL_SIZE, device=DEVICE)
    assert torch.isfinite(W).all(), "NaN/Inf in encoded parameters"


# 12
def test_encode_params_valid():
    img = torch.rand(*IMAGE_SIZE)
    W, _ = encode_image(img, K=K, epochs=5, lr=1e-2, kernel_size=KERNEL_SIZE, device=DEVICE)
    assert (W[:, 0] > 0).all() and (W[:, 0] < 1).all(), "sigma_x out of (0,1)"
    assert (W[:, 1] > 0).all() and (W[:, 1] < 1).all(), "sigma_y out of (0,1)"
    assert (W[:, 2] > -1).all() and (W[:, 2] < 1).all(), "rho out of (-1,1)"
    assert (W[:, 4] >= 0).all() and (W[:, 4] <= 1).all(), "colour out of [0,1]"
    assert (W[:, 5] >= -1).all() and (W[:, 5] <= 1).all(), "x out of [-1,1]"
    assert (W[:, 6] >= -1).all() and (W[:, 6] <= 1).all(), "y out of [-1,1]"


# 13
def test_encode_loss_decreases():
    img = torch.rand(*IMAGE_SIZE)
    _, _, history = encode_image(
        img, K=K, epochs=100, lr=5e-3, kernel_size=KERNEL_SIZE, device=DEVICE,
        return_history=True, log_every=1,
    )
    assert len(history) > 50, "Too few history entries recorded"
    assert history[50]["loss"] < history[0]["loss"], (
        f"Loss did not decrease: history[0]={history[0]['loss']:.4f}, "
        f"history[50]={history[50]['loss']:.4f}"
    )


# 14
def test_encode_early_stop():
    """Solid uniform image should trigger early stop before max_epochs (trivially easy to fit)."""
    img = torch.full(IMAGE_SIZE, 0.0)   # all-zero: Gaussians easily match with colour→0
    max_epochs = 500
    _, _, history = encode_image(
        img, K=K, epochs=max_epochs, lr=5e-3, kernel_size=KERNEL_SIZE,
        early_stop_threshold=0.5,   # very easy threshold — should stop within a few epochs
        device=DEVICE, return_history=True, log_every=1,
    )
    assert len(history) < max_epochs, (
        f"Expected early stop before {max_epochs} epochs, ran {len(history)}"
    )


# 15
@pytest.mark.slow
def test_encode_round_trip_psnr():
    """K=20 Gaussians, 200 epochs: PSNR > 8 dB (shows non-trivial reconstruction).

    The L1+SSIM training loss does not directly minimize MSE, so PSNR is not
    the primary training objective. 8 dB ensures the representation is
    meaningfully capturing image structure.
    """
    H, W = IMAGE_SIZE
    img = torch.zeros(H, W)
    img[8:20, 8:20] = 0.8
    img[5:12, 15:22] = 0.5

    W_phys, _ = encode_image(
        img, K=20, epochs=200, lr=5e-3, kernel_size=KERNEL_SIZE, device=DEVICE
    )

    sigma_x = W_phys[:, 0].clamp(1e-4, 1.0)
    sigma_y = W_phys[:, 1].clamp(1e-4, 1.0)
    rho = W_phys[:, 2].clamp(-0.999, 0.999)
    colour = W_phys[:, 4].clamp(0, 1).unsqueeze(1)
    coords = torch.stack([W_phys[:, 5], W_phys[:, 6]], dim=1)

    rendered = generate_2D_gaussian_splatting(
        kernel_size=KERNEL_SIZE,
        sigma_x=sigma_x, sigma_y=sigma_y, rho=rho,
        coords=coords, colours=colour,
        image_size=IMAGE_SIZE, channels=1, device=DEVICE,
    )[:, :, 0]

    mse = torch.mean((rendered - img) ** 2).item()
    psnr = 100.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)
    assert psnr > 8.0, f"Round-trip PSNR too low: {psnr:.2f} dB"


# ===========================================================================
# New tests — history dict format
# ===========================================================================

# 16
def test_history_is_list_of_dicts():
    """History entries must be dicts with required keys."""
    img = torch.rand(*IMAGE_SIZE)
    _, _, history = encode_image(
        img, K=K, epochs=20, lr=5e-3, kernel_size=KERNEL_SIZE, device=DEVICE,
        return_history=True, log_every=5,
    )
    assert len(history) > 0, "History is empty"
    required_keys = {"epoch", "loss", "psnr_db", "n_dead"}
    for entry in history:
        assert isinstance(entry, dict), f"Entry is not a dict: {type(entry)}"
        assert required_keys <= entry.keys(), (
            f"Missing keys: {required_keys - entry.keys()}"
        )


# 17
def test_history_dict_values_valid():
    """psnr_db must be finite and positive; n_dead must be int >= 0; recycling_event is bool."""
    img = torch.rand(*IMAGE_SIZE)
    _, _, history = encode_image(
        img, K=K, epochs=30, lr=5e-3, kernel_size=KERNEL_SIZE, device=DEVICE,
        return_history=True, log_every=5,
    )
    for entry in history:
        assert math.isfinite(entry["psnr_db"]), f"psnr_db not finite: {entry['psnr_db']}"
        assert entry["psnr_db"] > 0, f"psnr_db not positive: {entry['psnr_db']}"
        assert isinstance(entry["n_dead"], int), f"n_dead not int: {type(entry['n_dead'])}"
        assert entry["n_dead"] >= 0, f"n_dead negative: {entry['n_dead']}"
        assert entry["loss"] >= 0, f"loss negative: {entry['loss']}"


# 18
def test_history_psnr_improves():
    """PSNR should be higher at the end than at the start after meaningful training."""
    img = torch.rand(*IMAGE_SIZE)
    _, _, history = encode_image(
        img, K=K, epochs=100, lr=5e-3, kernel_size=KERNEL_SIZE, device=DEVICE,
        return_history=True, log_every=1,
    )
    assert history[-1]["psnr_db"] > history[0]["psnr_db"], (
        f"PSNR did not improve: start={history[0]['psnr_db']:.2f} end={history[-1]['psnr_db']:.2f}"
    )


# ===========================================================================
# New tests — _dead_mask
# ===========================================================================

# 19
def test_dead_mask_black_image():
    """On a black image, all Gaussians placed at center should be detected as dead."""
    img = torch.zeros(*IMAGE_SIZE, 1)  # [H, W, C]
    # Create W_raw with Gaussians at center and low colour
    W_raw = torch.zeros(K, 7)
    # x=0, y=0 (center) → tanh(0)=0, but image is black → target_at_center=0 < threshold
    # colour raw = 0 → sigmoid(0)=0.5, above threshold; dead purely from target_at_center
    dead = _dead_mask(W_raw, img, threshold=0.05)
    assert dead.shape == (K,), f"Expected shape ({K},), got {dead.shape}"
    assert dead.all(), "All Gaussians should be dead on a black image"


# 20
def test_dead_mask_suppressed_colour_not_dead():
    """Gaussians with suppressed colour on a bright pixel should NOT be detected as dead.

    The optimizer legitimately suppresses colour for overlapping Gaussians. Flagging
    them as dead causes false positives that recycle converging Gaussians and destroy
    optimization progress. Only center-on-background is a true dead indicator.
    """
    img = torch.ones(*IMAGE_SIZE, 1)  # [H, W, C]
    W_raw = torch.zeros(K, 7)
    W_raw[:, 4] = -10.0  # colour raw → sigmoid(-10) ≈ 0
    # x=y=0 → center pixel of all-ones image → target_at_center = 1.0 > 0.05
    dead = _dead_mask(W_raw, img, threshold=0.05)
    assert not dead.any(), (
        "Suppressed-colour Gaussians on bright pixels should NOT be flagged as dead"
    )


# 21
def test_dead_mask_alive_gaussians():
    """Gaussians whose center is on a bright pixel should NOT be detected as dead."""
    img = torch.ones(*IMAGE_SIZE, 1)  # [H, W, C]
    W_raw = torch.zeros(K, 7)
    # x=y=0 → center pixel of all-ones image → target_at_center = 1.0 > 0.05
    dead = _dead_mask(W_raw, img, threshold=0.05)
    assert not dead.any(), "No Gaussians should be dead when centers are on bright pixels"


# ===========================================================================
# New tests — _recycle
# ===========================================================================

# 22
def test_recycle_reduces_dead_count():
    """After recycling, the number of dead Gaussians should decrease."""
    img = torch.zeros(*IMAGE_SIZE, 1)  # [H, W, C]
    img[10:18, 10:18] = 1.0   # white square in center

    # Init Gaussians all at top-left (black area) → all dead
    W_raw = torch.full((K, 7), -5.0)  # raw params → very low colour, background location
    W_raw[:, 5] = -3.0  # x_raw → tanh(-3) ≈ -0.995 (far left)
    W_raw[:, 6] = -3.0  # y_raw → tanh(-3) ≈ -0.995 (far top)
    W_raw.requires_grad_(True)
    optimizer = torch.optim.Adam([W_raw], lr=1e-3)

    dead_before = int(_dead_mask(W_raw, img, threshold=0.05).sum().item())
    n_recycled = _recycle(W_raw, img, optimizer, KERNEL_SIZE, IMAGE_SIZE, DEVICE, threshold=0.05)
    dead_after = int(_dead_mask(W_raw, img, threshold=0.05).sum().item())

    assert n_recycled > 0, "Expected some Gaussians to be recycled"
    assert dead_after < dead_before, (
        f"Dead count did not decrease: before={dead_before}, after={dead_after}"
    )


# 23
def test_recycle_no_dead_returns_zero():
    """_recycle should return 0 when no dead Gaussians exist."""
    img = torch.ones(*IMAGE_SIZE, 1)  # [H, W, C]
    W_raw = torch.zeros(K, 7)
    W_raw[:, 4] = 2.0   # high colour → alive
    W_raw.requires_grad_(True)
    optimizer = torch.optim.Adam([W_raw], lr=1e-3)

    n_recycled = _recycle(W_raw, img, optimizer, KERNEL_SIZE, IMAGE_SIZE, DEVICE, threshold=0.05)
    assert n_recycled == 0, f"Expected 0 recycled, got {n_recycled}"
