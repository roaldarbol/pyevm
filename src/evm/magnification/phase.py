"""Phase-based Eulerian Video Magnification (Wadhwa et al., SIGGRAPH 2013).

Algorithm
---------
1. Convert video frames to YIQ; work on the luminance channel (Y).
2. Build a complex steerable pyramid for each frame.
3. For each sub-band (scale × orientation):
   a. Extract phase:          φ(t) = angle(coeff(t))
   b. Compute phase delta:    Δφ(t) = φ(t) − φ(0)  (remove DC)
   c. Temporally filter Δφ to isolate the motion frequency band.
   d. Amplify:                Δφ_amp(t) = factor × Δφ_filtered(t)
   e. Shift coefficient:      coeff'(t) = |coeff(t)| × exp(j·(φ(t) + Δφ_amp(t)))
4. Reconstruct Y from modified pyramid for each frame.
5. Recombine with I, Q channels and convert back to RGB.

This algorithm handles *larger* amplifications with far fewer artefacts than
the linear (colour / motion) method.
"""

from __future__ import annotations

import torch
from loguru import logger

from evm.filters.temporal import ButterworthBandpass, IdealBandpass
from evm.magnification._colorspace import rgb_to_yiq, yiq_to_rgb
from evm.pyramids.steerable import SteerablePyramid


class PhaseMagnifier:
    """Phase-based EVM magnifier.

    Args:
        factor: Phase amplification factor.
        freq_low: Temporal bandpass lower frequency (Hz).
        freq_high: Temporal bandpass upper frequency (Hz).
        n_scales: Number of pyramid scales.
        n_orientations: Number of orientation bands per scale.
        sigma: Spatial phase smoothing sigma (pixels, ``0`` = disabled).
            Smoothing the phase before amplification reduces spatial noise.
        filter_type: ``"ideal"`` (FFT) or ``"butterworth"`` (IIR).
        device: Compute device.
        dtype: Real tensor dtype (sub-band coefficients are complex).
    """

    def __init__(
        self,
        factor: float = 10.0,
        freq_low: float = 0.4,
        freq_high: float = 3.0,
        n_scales: int = 4,
        n_orientations: int = 6,
        sigma: float = 3.0,
        filter_type: str = "ideal",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.factor = factor
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.sigma = sigma
        self.filter_type = filter_type
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self._pyramid = SteerablePyramid(
            n_scales=n_scales,
            n_orientations=n_orientations,
            device=self.device,
            dtype=dtype,
        )
        logger.info(
            f"PhaseMagnifier: factor={factor}, band=[{freq_low}, {freq_high}] Hz, "
            f"scales={n_scales}, orientations={n_orientations}, filter={filter_type}"
        )

    def _smooth_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Spatial Gaussian smoothing of a ``(H, W)`` phase map."""
        if self.sigma <= 0:
            return phase
        # Kernel radius
        radius = int(3 * self.sigma)
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=self.dtype, device=self.device) - radius
        g1d = torch.exp(-coords ** 2 / (2 * self.sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = torch.outer(g1d, g1d).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        ph = phase.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return torch.nn.functional.conv2d(ph, kernel, padding=radius).squeeze(0).squeeze(0)

    def process(self, frames: torch.Tensor, fps: float) -> torch.Tensor:
        """Run phase-based EVM on a video tensor.

        Args:
            frames: ``(T, C, H, W)`` RGB tensor with values in ``[0, 1]``.
            fps: Frames per second.

        Returns:
            Amplified ``(T, C, H, W)`` RGB tensor, clamped to ``[0, 1]``.
        """
        frames = frames.to(device=self.device, dtype=self.dtype)
        T, C, H, W = frames.shape
        logger.info(f"PhaseMagnifier.process: {T} frames @ {fps} fps, shape {(C, H, W)}")

        # --- Convert to YIQ ---
        yiq = rgb_to_yiq(frames)  # (T, 3, H, W)
        luma = yiq[:, 0, :, :]   # (T, H, W)

        # --- Build steerable pyramid for every frame ---
        logger.debug("Building steerable pyramids for all frames…")
        pyr_list: list[dict] = []
        for t in range(T):
            pyr = self._pyramid.build(luma[t])
            pyr_list.append(pyr)

        # --- Process each sub-band ---
        logger.debug("Amplifying phase in each sub-band…")
        for scale in range(self.n_scales):
            for orient in range(self.n_orientations):
                # Collect complex coefficients across time: list of (H_s, W_s) complex
                coeffs = [pyr_list[t]["subbands"][scale][orient] for t in range(T)]
                # Stack → (T, H_s, W_s) complex
                coeffs_stack = torch.stack(coeffs, dim=0)

                # Extract phase
                phase = torch.angle(coeffs_stack)  # (T, H_s, W_s) real

                # Remove DC (reference = first frame)
                delta_phase = phase - phase[0:1]

                # Spatial smoothing of phase before filtering
                if self.sigma > 0:
                    smoothed: list[torch.Tensor] = []
                    for t in range(T):
                        smoothed.append(self._smooth_phase(delta_phase[t]))
                    delta_phase = torch.stack(smoothed, dim=0)

                # Temporal filter
                if self.filter_type == "ideal":
                    filt = IdealBandpass(fps, self.freq_low, self.freq_high)
                    filtered_phase = filt.apply(delta_phase)
                else:
                    filt_bw = ButterworthBandpass(fps, self.freq_low, self.freq_high)
                    filtered_phase = filt_bw.apply(delta_phase)

                # Amplify
                amp_phase = filtered_phase * self.factor

                # Shift original coefficients by amplified phase
                magnitude = coeffs_stack.abs()
                original_phase = phase
                new_phase = original_phase + amp_phase
                new_coeffs = torch.polar(magnitude, new_phase)

                # Write back
                for t in range(T):
                    pyr_list[t]["subbands"][scale][orient] = new_coeffs[t]

                logger.debug(
                    f"Scale {scale}, orientation {orient}: "
                    f"phase amp range [{amp_phase.min():.3f}, {amp_phase.max():.3f}]"
                )

        # --- Reconstruct Y from modified pyramids ---
        logger.debug("Collapsing modified pyramids…")
        recon_luma: list[torch.Tensor] = []
        for t in range(T):
            y_hat = self._pyramid.collapse(pyr_list[t])  # (H, W)
            recon_luma.append(y_hat)

        luma_out = torch.stack(recon_luma, dim=0)  # (T, H, W)

        # --- Recombine with I, Q and convert to RGB ---
        result_yiq = yiq.clone()
        result_yiq[:, 0, :, :] = luma_out
        result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)
        logger.info("PhaseMagnifier.process complete")
        return result_rgb
