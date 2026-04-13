"""Gaussian pyramid — build and collapse via PyTorch convolutions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger


def _gaussian_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """5×5 Gaussian blur kernel matching the reference MATLAB implementation."""
    k = torch.tensor([1, 4, 6, 4, 1], dtype=dtype, device=device)
    k = torch.outer(k, k)
    k = k / k.sum()
    # Shape: (1, 1, 5, 5) — applied per-channel via groups
    return k.unsqueeze(0).unsqueeze(0)


def _blur_downsample(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Blur with Gaussian kernel then downsample by 2.

    Args:
        x: ``(B, C, H, W)`` tensor.
        kernel: ``(1, 1, 5, 5)`` Gaussian kernel.

    Returns:
        ``(B, C, H//2, W//2)`` tensor.
    """
    B, C, H, W = x.shape
    # Apply per-channel by repeating kernel across channel groups
    k = kernel.expand(C, 1, 5, 5)
    x = F.conv2d(x, k, padding=2, groups=C)
    return x[:, :, ::2, ::2]


def _upsample_blur(
    x: torch.Tensor, kernel: torch.Tensor, target_h: int, target_w: int
) -> torch.Tensor:
    """Upsample by 2 then blur.

    Args:
        x: ``(B, C, H, W)`` tensor.
        kernel: ``(1, 1, 5, 5)`` Gaussian kernel.
        target_h: Target height (may be odd).
        target_w: Target width (may be odd).

    Returns:
        ``(B, C, target_h, target_w)`` tensor.
    """
    B, C, H, W = x.shape
    # Bilinear upsample to exact target size
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    k = kernel.expand(C, 1, 5, 5)
    return F.conv2d(x, k, padding=2, groups=C)


class GaussianPyramid:
    """Multi-scale Gaussian pyramid.

    Frames are expected as ``(B, C, H, W)`` float tensors, values in ``[0, 1]``.

    Args:
        n_levels: Number of pyramid levels (including the original).
        device: Compute device.
        dtype: Floating-point dtype (default ``torch.float32``).
    """

    def __init__(
        self,
        n_levels: int = 6,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_levels = n_levels
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self._kernel = _gaussian_kernel(self.device, self.dtype)
        logger.debug(f"GaussianPyramid: {n_levels} levels on {self.device}")

    def build(self, frame: torch.Tensor) -> list[torch.Tensor]:
        """Decompose *frame* into a Gaussian pyramid.

        Args:
            frame: ``(B, C, H, W)`` or ``(C, H, W)`` tensor.

        Returns:
            List of tensors from finest (level 0 = original) to coarsest.
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        frame = frame.to(device=self.device, dtype=self.dtype)
        levels: list[torch.Tensor] = [frame]

        current = frame
        for i in range(1, self.n_levels):
            current = _blur_downsample(current, self._kernel)
            levels.append(current)
            logger.debug(f"Gaussian level {i}: {tuple(current.shape)}")

        return levels

    def collapse(self, levels: list[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from pyramid by upsampling the coarsest level.

        This simply returns the upsampled coarsest level (level 0 = original
        resolution).  For reconstruction with residuals use
        :class:`LaplacianPyramid`.

        Returns:
            ``(B, C, H, W)`` tensor at original resolution.
        """
        result = levels[-1]
        for i in range(len(levels) - 2, -1, -1):
            target = levels[i]
            result = _upsample_blur(result, self._kernel, target.shape[2], target.shape[3])
        return result
