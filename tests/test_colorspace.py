"""Tests for RGB ↔ YIQ colorspace conversions."""

import torch
import pytest
from evm.magnification._colorspace import rgb_to_yiq, yiq_to_rgb


@pytest.fixture
def rgb_batch():
    torch.manual_seed(1)
    return torch.rand(4, 3, 32, 32, dtype=torch.float64)


def test_round_trip(rgb_batch):
    """RGB → YIQ → RGB should recover the original within float precision."""
    recovered = yiq_to_rgb(rgb_to_yiq(rgb_batch))
    assert torch.allclose(rgb_batch, recovered, atol=1e-6), \
        f"Max abs diff: {(rgb_batch - recovered).abs().max().item()}"


def test_output_shape(rgb_batch):
    yiq = rgb_to_yiq(rgb_batch)
    assert yiq.shape == rgb_batch.shape


def test_pure_white():
    """White (1, 1, 1) → Y channel should be ~1."""
    white = torch.ones(1, 3, 1, 1, dtype=torch.float64)
    yiq = rgb_to_yiq(white)
    assert abs(yiq[0, 0, 0, 0].item() - 1.0) < 1e-5, "Y of white should be ≈ 1"


def test_pure_black():
    """Black (0, 0, 0) → all YIQ channels should be 0."""
    black = torch.zeros(1, 3, 1, 1, dtype=torch.float64)
    yiq = rgb_to_yiq(black)
    assert torch.allclose(yiq, torch.zeros_like(yiq), atol=1e-7)


def test_luma_range():
    """Y channel of any valid RGB in [0,1] must lie in [0, 1]."""
    torch.manual_seed(99)
    x = torch.rand(10, 3, 8, 8, dtype=torch.float64)
    yiq = rgb_to_yiq(x)
    y = yiq[:, 0]
    assert y.min().item() >= -0.01, "Y should not be strongly negative"
    assert y.max().item() <= 1.01, "Y should not exceed 1"
