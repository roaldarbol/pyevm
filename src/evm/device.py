"""Device detection and management for GPU/MPS/CPU compute."""

from __future__ import annotations

import torch
from loguru import logger


def get_device(force: str | None = None) -> torch.device:
    """Return the best available compute device.

    Priority: CUDA > MPS > CPU, unless *force* overrides.

    Args:
        force: One of ``"cuda"``, ``"mps"``, or ``"cpu"``. When ``None``
               the best available device is selected automatically.

    Returns:
        A :class:`torch.device` ready to use.
    """
    if force is not None:
        device = torch.device(force)
        logger.debug(f"Forced compute device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected — using CPU")

    return device


def device_info(device: torch.device) -> dict[str, str]:
    """Return a human-readable summary of *device* capabilities."""
    info: dict[str, str] = {"device": str(device)}
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info["vram_gb"] = f"{props.total_memory / 1e9:.1f}"
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"
    else:
        info["name"] = "CPU"
    return info
