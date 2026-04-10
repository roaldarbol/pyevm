"""Motion-based Eulerian Video Magnification (Wu et al., SIGGRAPH 2012).

Algorithm
---------
1. Convert video to YIQ; process all three channels.
2. Build a Laplacian spatial pyramid per frame.
3. Apply a Butterworth bandpass IIR filter temporally at each pyramid level.
4. Apply a *spatially-adaptive* amplification factor per level:
   ``alpha_eff = min(alpha, lambda_c / (8 * (1 + alpha)))``,
   where ``lambda_c`` is the spatial wavelength cutoff.
5. Collapse modified pyramid back to full resolution and add to original.

This algorithm is best for subtle *motion* (e.g. vibrations, breathing).
"""

from __future__ import annotations

import math

import torch
from loguru import logger

from pyevm.filters.temporal import ButterworthBandpass, IdealBandpass
from pyevm.magnification._colorspace import rgb_to_yiq, yiq_to_rgb
from pyevm.pyramids.laplacian import LaplacianPyramid


class MotionMagnifier:
    """Motion-based EVM magnifier.

    Args:
        alpha: Nominal amplification factor (may be reduced per level).
        freq_low: Temporal bandpass lower frequency (Hz).
        freq_high: Temporal bandpass upper frequency (Hz).
        n_levels: Laplacian pyramid levels.
        lambda_c: Spatial wavelength cutoff (pixels) for adaptive scaling
            (default 16, matching the reference MATLAB code).
        filter_type: ``"butterworth"`` (default, streaming) or ``"ideal"``.
        device: Compute device.
        dtype: Tensor dtype.
    """

    def __init__(
        self,
        alpha: float = 20.0,
        freq_low: float = 0.4,
        freq_high: float = 3.0,
        n_levels: int = 6,
        lambda_c: float = 16.0,
        filter_type: str = "butterworth",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.alpha = alpha
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.n_levels = n_levels
        self.lambda_c = lambda_c
        self.filter_type = filter_type
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self._pyramid = LaplacianPyramid(n_levels=n_levels, device=self.device, dtype=dtype)

        logger.info(
            f"MotionMagnifier: alpha={alpha}, band=[{freq_low}, {freq_high}] Hz, "
            f"lambda_c={lambda_c}, filter={filter_type}"
        )

    def _alpha_for_level(self, level: int) -> float:
        """Compute spatially-adaptive alpha for *level*.

        The spatial wavelength at level *i* is ``2^i`` pixels (at full
        resolution).  We clamp alpha so that amplified motion doesn't alias.
        """
        lambda_at_level = 2 ** (level + 1)
        alpha_max = self.lambda_c / (8.0 * lambda_at_level) - 1.0
        alpha_eff = min(self.alpha, alpha_max) if alpha_max > 0 else 0.0
        logger.debug(f"  Level {level}: lambda={lambda_at_level}, alpha_eff={alpha_eff:.2f}")
        return alpha_eff

    def process(self, frames: torch.Tensor, fps: float) -> torch.Tensor:
        """Run motion EVM on a video tensor.

        Args:
            frames: ``(T, C, H, W)`` RGB tensor with values in ``[0, 1]``.
            fps: Frames per second.

        Returns:
            Amplified ``(T, C, H, W)`` RGB tensor, clamped to ``[0, 1]``.
        """
        frames = frames.to(device=self.device, dtype=self.dtype)
        T, C, H, W = frames.shape
        logger.info(f"MotionMagnifier.process: {T} frames @ {fps} fps, shape {(C, H, W)}")

        # --- Convert to YIQ ---
        yiq = rgb_to_yiq(frames)  # (T, 3, H, W)

        # --- Build Laplacian pyramid per frame ---
        logger.debug("Building Laplacian pyramids…")
        # pyramids[level] = list of (3, h, w) tensors, length T
        pyramids: list[list[torch.Tensor]] = [[] for _ in range(self.n_levels)]
        for t in range(T):
            levels = self._pyramid.build(yiq[t])  # list of (1, 3, h, w)
            for lvl, lev in enumerate(levels):
                pyramids[lvl].append(lev.squeeze(0))  # (3, h, w)

        # --- Temporally filter each level and amplify ---
        filtered_pyramids: list[list[torch.Tensor]] = []
        for lvl in range(self.n_levels):
            # Stack → (T, 3, h, w)
            level_tensor = torch.stack(pyramids[lvl], dim=0)
            logger.debug(f"Level {lvl}: shape {tuple(level_tensor.shape)}")

            if self.filter_type == "ideal":
                filt = IdealBandpass(fps, self.freq_low, self.freq_high)
                filtered = filt.apply(level_tensor)
            else:
                filt = ButterworthBandpass(fps, self.freq_low, self.freq_high)
                filtered = filt.apply(level_tensor)

            alpha_eff = self._alpha_for_level(lvl)
            filtered = filtered * alpha_eff

            # Back to list of (3, h, w)
            filtered_pyramids.append([filtered[t] for t in range(T)])

        # --- Reconstruct each frame ---
        logger.debug("Collapsing pyramids…")
        output_frames: list[torch.Tensor] = []
        for t in range(T):
            # Modify original pyramid by adding filtered amplification
            orig_levels = self._pyramid.build(yiq[t])
            modified_levels = [
                orig_levels[lvl] + filtered_pyramids[lvl][t].unsqueeze(0)
                for lvl in range(self.n_levels)
            ]
            reconstructed = self._pyramid.collapse(modified_levels)  # (1, 3, H, W)
            output_frames.append(reconstructed.squeeze(0))  # (3, H, W)

        result_yiq = torch.stack(output_frames, dim=0)  # (T, 3, H, W)

        # --- Convert back to RGB and clamp ---
        result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)
        logger.info("MotionMagnifier.process complete")
        return result_rgb
