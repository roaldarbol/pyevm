"""Shared pytest fixtures for EVM tests."""

import pytest
import torch


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def small_frame(cpu_device) -> torch.Tensor:
    """A single (1, 3, 64, 64) RGB frame, values in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 64, 64, device=cpu_device)


@pytest.fixture
def small_video(cpu_device) -> torch.Tensor:
    """A tiny (30, 3, 64, 64) RGB video tensor, values in [0, 1]."""
    torch.manual_seed(0)
    return torch.rand(30, 3, 64, 64, device=cpu_device)


@pytest.fixture
def luma_frame(cpu_device) -> torch.Tensor:
    """A single (64, 64) greyscale frame."""
    torch.manual_seed(7)
    return torch.rand(64, 64, device=cpu_device)
