"""Laplacian pyramid — build and collapse via PyTorch convolutions."""

from __future__ import annotations

import torch
from loguru import logger

from pyevm.pyramids.gaussian import _blur_downsample, _gaussian_kernel, _upsample_blur


class LaplacianPyramid:
    """Multi-scale Laplacian pyramid (difference-of-Gaussians).

    Each level stores the *band-pass* detail image; the coarsest level stores
    the low-pass residual (a Gaussian level).

    Args:
        n_levels: Number of pyramid levels.
        device: Compute device.
        dtype: Floating-point dtype.
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
        logger.debug(f"LaplacianPyramid: {n_levels} levels on {self.device}")

    def build(self, frame: torch.Tensor) -> list[torch.Tensor]:
        """Decompose *frame* into a Laplacian pyramid.

        Args:
            frame: ``(B, C, H, W)`` or ``(C, H, W)`` tensor.

        Returns:
            List of ``n_levels`` tensors. Levels 0 … n-2 are band-pass detail
            images; level n-1 is the low-pass Gaussian residual.
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        frame = frame.to(device=self.device, dtype=self.dtype)

        # Build Gaussian pyramid first
        gaussian: list[torch.Tensor] = [frame]
        current = frame
        for _ in range(1, self.n_levels):
            current = _blur_downsample(current, self._kernel)
            gaussian.append(current)

        # Laplacian levels = Gaussian[i] − upsample(Gaussian[i+1])
        laplacian: list[torch.Tensor] = []
        for i in range(self.n_levels - 1):
            g_up = _upsample_blur(
                gaussian[i + 1],
                self._kernel,
                gaussian[i].shape[2],
                gaussian[i].shape[3],
            )
            lap = gaussian[i] - g_up
            laplacian.append(lap)
            logger.debug(f"Laplacian level {i}: {tuple(lap.shape)}")

        # Append coarsest Gaussian as residual
        laplacian.append(gaussian[-1])
        logger.debug(f"Laplacian residual (level {self.n_levels - 1}): {tuple(gaussian[-1].shape)}")

        return laplacian

    def collapse(self, levels: list[torch.Tensor]) -> torch.Tensor:
        """Reconstruct frame from Laplacian pyramid.

        Args:
            levels: Pyramid returned by :meth:`build` (possibly modified).

        Returns:
            ``(B, C, H, W)`` tensor at original resolution.
        """
        result = levels[-1]
        for i in range(len(levels) - 2, -1, -1):
            result = _upsample_blur(
                result, self._kernel, levels[i].shape[2], levels[i].shape[3]
            )
            result = result + levels[i]
        return result
