"""Tests for device detection utilities."""

import pytest
import torch

from pyevm.device import device_info, get_device


def test_get_device_cpu_forced():
    device = get_device(force="cpu")
    assert device.type == "cpu"


def test_get_device_auto_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")


def test_get_device_invalid_force():
    with pytest.raises(RuntimeError):
        get_device(force="nonexistent_device")


def test_device_info_cpu():
    device = torch.device("cpu")
    info = device_info(device)
    assert info["device"] == "cpu"
    assert "name" in info


def test_device_info_auto():
    device = get_device()
    info = device_info(device)
    assert "device" in info
    assert "name" in info
