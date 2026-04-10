"""RGB ↔ YIQ colorspace conversions (pure PyTorch, GPU-compatible).

YIQ is used by the original Matlab EVM code (Wu 2012 / Wadhwa 2013).
"""

from __future__ import annotations

import torch

# Row vectors for the 3×3 matrix (applied along the channel dim)
_RGB_TO_YIQ = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312],
    ],
    dtype=torch.float64,
)

_YIQ_TO_RGB = torch.linalg.inv(_RGB_TO_YIQ)


def rgb_to_yiq(x: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 3, H, W)`` RGB tensor to YIQ."""
    mat = _RGB_TO_YIQ.to(device=x.device, dtype=x.dtype)
    # x: (..., 3, H, W) → (..., H, W, 3) for matmul, then back
    x_perm = x.permute(*range(x.dim() - 3), -2, -1, -3)  # (..., H, W, 3)
    y_perm = x_perm @ mat.T  # (..., H, W, 3)
    return y_perm.permute(*range(x.dim() - 3), -1, -3, -2)  # (..., 3, H, W)


def yiq_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 3, H, W)`` YIQ tensor to RGB."""
    mat = _YIQ_TO_RGB.to(device=x.device, dtype=x.dtype)
    x_perm = x.permute(*range(x.dim() - 3), -2, -1, -3)
    y_perm = x_perm @ mat.T
    return y_perm.permute(*range(x.dim() - 3), -1, -3, -2)
