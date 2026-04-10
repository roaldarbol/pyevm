"""Color-based Eulerian Video Magnification (Wu et al., SIGGRAPH 2012).

Algorithm
---------
1. Convert video frames to YIQ colorspace.
2. Build a Gaussian spatial pyramid for each frame.
3. Accumulate the selected pyramid level across all frames → temporal signal.
4. Apply an ideal bandpass filter in time.
5. Amplify by *alpha*; attenuate chrominance (I, Q) by *chrom_attenuation*.
6. Collapse amplified pyramid back to full resolution and add to original.

This algorithm is best for subtle *colour* changes (e.g. skin-tone pulse).
"""

from __future__ import annotations

import torch
from loguru import logger

from evm.filters.temporal import ButterworthBandpass, IdealBandpass
from evm.magnification._colorspace import rgb_to_yiq, yiq_to_rgb
from evm.pyramids.gaussian import GaussianPyramid


class ColorMagnifier:
    """Colour-based EVM magnifier.

    Args:
        alpha: Luminance amplification factor.
        freq_low: Temporal bandpass lower frequency (Hz).
        freq_high: Temporal bandpass upper frequency (Hz).
        n_levels: Gaussian pyramid levels (typically 4–6).
        chrom_attenuation: Scale applied to amplified I, Q channels
            (0 = no chrominance amplification, 1 = same as luma).
        pyramid_level: Which Gaussian level to temporally filter (default
            ``n_levels - 1``; coarsest, lowest spatial frequency).
        filter_type: ``"ideal"`` (FFT bandpass) or ``"butterworth"`` (IIR,
            lower memory).
        device: Compute device (auto-selected if ``None``).
        dtype: Tensor dtype.
    """

    def __init__(
        self,
        alpha: float = 50.0,
        freq_low: float = 0.4,
        freq_high: float = 3.0,
        n_levels: int = 6,
        chrom_attenuation: float = 0.1,
        pyramid_level: int | None = None,
        filter_type: str = "ideal",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.alpha = alpha
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.n_levels = n_levels
        self.chrom_attenuation = chrom_attenuation
        self.pyramid_level = pyramid_level if pyramid_level is not None else n_levels - 1
        self.filter_type = filter_type
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self._pyramid = GaussianPyramid(n_levels=n_levels, device=self.device, dtype=dtype)

        logger.info(
            f"ColorMagnifier: alpha={alpha}, band=[{freq_low}, {freq_high}] Hz, "
            f"pyramid_level={self.pyramid_level}, filter={filter_type}"
        )

    def process(self, frames: torch.Tensor, fps: float) -> torch.Tensor:
        """Run colour EVM on a video tensor.

        Args:
            frames: ``(T, C, H, W)`` RGB tensor with values in ``[0, 1]``.
            fps: Frames per second.

        Returns:
            Amplified ``(T, C, H, W)`` RGB tensor, clamped to ``[0, 1]``.
        """
        frames = frames.to(device=self.device, dtype=self.dtype)
        T, C, H, W = frames.shape
        logger.info(f"ColorMagnifier.process: {T} frames @ {fps} fps, shape {(C, H, W)}")

        # --- Convert to YIQ ---
        yiq = rgb_to_yiq(frames)  # (T, 3, H, W)

        # --- Build pyramid per frame, extract target level ---
        logger.debug(f"Building Gaussian pyramids (level {self.pyramid_level})…")
        level_stack: list[torch.Tensor] = []
        for t in range(T):
            levels = self._pyramid.build(yiq[t])  # list of (1, 3, h, w)
            level_stack.append(levels[self.pyramid_level].squeeze(0))  # (3, h, w)

        # Stack → (T, 3, h, w)
        level_tensor = torch.stack(level_stack, dim=0)
        logger.debug(f"Pyramid level tensor: {tuple(level_tensor.shape)}")

        # --- Temporal filtering ---
        if self.filter_type == "ideal":
            filt = IdealBandpass(fps, self.freq_low, self.freq_high)
            filtered = filt.apply(level_tensor)  # (T, 3, h, w)
        else:
            filt = ButterworthBandpass(fps, self.freq_low, self.freq_high)
            filtered = filt.apply(level_tensor)

        # --- Amplify: Y × alpha, I/Q × alpha × chrom_attenuation ---
        amplified = filtered.clone()
        amplified[:, 0] *= self.alpha
        amplified[:, 1:] *= self.alpha * self.chrom_attenuation
        logger.debug("Applied amplification to filtered signal")

        # --- Upsample amplified signal back to original resolution ---
        # We upsample the coarsest-level amplified signal to full resolution
        upsampled = torch.nn.functional.interpolate(
            amplified.view(T * 3, 1, *amplified.shape[2:]),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).view(T, 3, H, W)

        # --- Add to original YIQ ---
        result_yiq = yiq + upsampled

        # --- Convert back to RGB and clamp ---
        result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)
        logger.info("ColorMagnifier.process complete")
        return result_rgb
