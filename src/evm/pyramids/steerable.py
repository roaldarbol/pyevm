"""Complex steerable pyramid (Simoncelli & Freeman, 1995).

Implemented entirely in PyTorch so it runs on CUDA / MPS / CPU.

Reference
---------
Wadhwa et al., "Phase-Based Video Motion Processing", SIGGRAPH 2013.
Simoncelli & Freeman, "The Steerable Pyramid", ICIP 1995.

Design notes
------------
The pyramid is constructed in the *frequency domain*:
  1. Split the spectrum into a high-pass residual and a low-pass envelope.
  2. At each scale, multiply the low-pass envelope by oriented band-pass
     filters (one per orientation) to produce complex sub-band coefficients.
  3. Downsample the low-pass envelope by 2 and repeat.

The oriented filters are raised-cosine windows in polar frequency coordinates.
Each sub-band coefficient is complex, so its angle encodes local phase (motion).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# Frequency-domain filter construction
# ---------------------------------------------------------------------------

def _polar_grid(height: int, width: int, device: torch.device, dtype: torch.dtype):
    """Normalised polar frequency grid in ``[-π, π]``.

    Returns:
        radius: ``(H, W)`` tensor with values in ``[0, π√2]``.
        angle:  ``(H, W)`` tensor with values in ``(-π, π]``.
    """
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype) * 2 * math.pi
    fx = torch.fft.fftfreq(width, device=device, dtype=dtype) * 2 * math.pi
    # Grid: note meshgrid uses (rows, cols) → (fy, fx)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    angle = torch.atan2(grid_y, grid_x)
    return radius, angle


def _raised_cosine(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Smooth raised-cosine transition from 0 to 1 between *low* and *high*."""
    x = x.clamp(low, high)
    return 0.5 * (1.0 - torch.cos(math.pi * (x - low) / (high - low)))


def _highpass_filter(radius: torch.Tensor) -> torch.Tensor:
    """High-pass radial window centred at Nyquist (π)."""
    return _raised_cosine(radius, math.pi / 2, math.pi)


def _lowpass_filter(radius: torch.Tensor) -> torch.Tensor:
    """Low-pass radial window."""
    return _raised_cosine(radius, math.pi / 4, math.pi / 2)


def _oriented_filter(
    radius: torch.Tensor,
    angle: torch.Tensor,
    orientation_idx: int,
    n_orientations: int,
) -> torch.Tensor:
    """Complex oriented band-pass filter for one orientation.

    Returns a complex tensor (real + imag) representing a steerable filter
    at angle ``orientation_idx * π / n_orientations``.
    """
    target_angle = math.pi * orientation_idx / n_orientations
    # Angular distance (wrapped to [-π, π])
    angle_diff = angle - target_angle
    # Wrap
    angle_diff = ((angle_diff + math.pi) % (2 * math.pi)) - math.pi

    # Raised-cosine in angle: support = ±π/n_orientations
    half_bw = math.pi / n_orientations
    angular_window = _raised_cosine(angle_diff.abs(), 0.0, half_bw + 1e-9)
    # Also apply the antipodal lobe (angle + π)
    angle_diff2 = ((angle - target_angle + 2 * math.pi) % (2 * math.pi)) - math.pi
    angle_diff2 = ((angle_diff2 + math.pi) % (2 * math.pi)) - math.pi
    angular_window2 = _raised_cosine(angle_diff2.abs(), 0.0, half_bw + 1e-9)
    angular_window = torch.maximum(angular_window, angular_window2)

    # Radial band-pass: between π/4 and π/2 (octave band)
    radial_window = _raised_cosine(radius, math.pi / 4, math.pi / 2)

    magnitude = radial_window * angular_window
    # Make analytic (one-sided): zero out negative-frequency half
    # The imaginary part is a π/2-phase-shifted version → use Hilbert trick
    # For the steerable pyramid we encode as real filter × exp(j·target_angle·k)
    # Simple approach: return as complex with phase = target_angle
    real_part = magnitude * torch.cos(torch.tensor(target_angle))
    imag_part = magnitude * torch.sin(torch.tensor(target_angle))
    return torch.complex(real_part, imag_part)


# ---------------------------------------------------------------------------
# Pyramid class
# ---------------------------------------------------------------------------

class SteerablePyramid:
    """Complex steerable pyramid decomposition/reconstruction.

    Args:
        n_scales: Number of scale levels (octaves).
        n_orientations: Number of orientation bands per scale (typically 4 or 8).
        device: Compute device.
        dtype: Real floating-point dtype used for computation.
    """

    def __init__(
        self,
        n_scales: int = 4,
        n_orientations: int = 6,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        logger.debug(
            f"SteerablePyramid: {n_scales} scales × {n_orientations} orientations on {self.device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, frame: torch.Tensor) -> dict:
        """Decompose a single-channel frame into the steerable pyramid.

        Args:
            frame: ``(H, W)`` or ``(1, 1, H, W)`` real tensor (one channel).

        Returns:
            Dictionary with keys:

            * ``"highpass"`` – ``(H, W)`` complex tensor.
            * ``"lowpass"``  – ``(H', W')`` real tensor (coarsest).
            * ``"subbands"`` – list of length *n_scales*, each a list of
              *n_orientations* complex ``(H_s, W_s)`` tensors.
            * ``"sizes"``    – list of ``(H_s, W_s)`` tuples for reconstruction.
        """
        if frame.dim() == 4:
            frame = frame.squeeze(0).squeeze(0)
        elif frame.dim() == 3:
            frame = frame.squeeze(0)

        frame = frame.to(device=self.device, dtype=self.dtype)
        H, W = frame.shape

        # Full-frame FFT
        dft = torch.fft.fft2(frame)

        # High-pass residual
        radius, angle = _polar_grid(H, W, self.device, self.dtype)
        hp_filter = _highpass_filter(radius)
        highpass = torch.fft.ifft2(dft * hp_filter)
        logger.debug(f"Highpass residual: {tuple(highpass.shape)}")

        subbands: list[list[torch.Tensor]] = []
        sizes: list[tuple[int, int]] = []
        current_dft = dft
        current_h, current_w = H, W

        for scale in range(self.n_scales):
            radius_s, angle_s = _polar_grid(current_h, current_w, self.device, self.dtype)

            # Oriented sub-bands at this scale
            scale_bands: list[torch.Tensor] = []
            for orient in range(self.n_orientations):
                filt = _oriented_filter(radius_s, angle_s, orient, self.n_orientations)
                subband = torch.fft.ifft2(current_dft * filt)
                scale_bands.append(subband)
                logger.debug(
                    f"Scale {scale}, orientation {orient}: {tuple(subband.shape)}"
                )

            subbands.append(scale_bands)
            sizes.append((current_h, current_w))

            # Low-pass at this scale → input to next scale
            lp_filter = _lowpass_filter(radius_s)
            lp_dft = current_dft * lp_filter

            # Downsample by 2 in frequency domain (crop + scale)
            current_h, current_w = current_h // 2, current_w // 2
            current_dft = self._downsample_dft(lp_dft, current_h, current_w)

        # Coarsest low-pass residual (real)
        lowpass = torch.fft.ifft2(current_dft).real
        logger.debug(f"Lowpass residual: {tuple(lowpass.shape)}")

        return {
            "highpass": highpass,
            "lowpass": lowpass,
            "subbands": subbands,
            "sizes": sizes,
        }

    def collapse(self, pyramid: dict) -> torch.Tensor:
        """Reconstruct a frame from a (modified) steerable pyramid.

        Args:
            pyramid: Dictionary as returned by :meth:`build`.

        Returns:
            Reconstructed ``(H, W)`` real tensor.
        """
        subbands = pyramid["subbands"]
        sizes = pyramid["sizes"]
        highpass = pyramid["highpass"]
        lowpass = pyramid["lowpass"]

        # Start from coarsest low-pass
        current_h, current_w = lowpass.shape
        current_dft = torch.fft.fft2(lowpass.to(dtype=self.dtype))

        for scale in range(self.n_scales - 1, -1, -1):
            target_h, target_w = sizes[scale]
            # Upsample low-pass DFT
            current_dft = self._upsample_dft(current_dft, target_h, target_w)
            radius_s, angle_s = _polar_grid(target_h, target_w, self.device, self.dtype)
            lp_filter = _lowpass_filter(radius_s)
            current_dft = current_dft * lp_filter

            # Add oriented sub-bands
            for orient in range(self.n_orientations):
                filt = _oriented_filter(radius_s, angle_s, orient, self.n_orientations)
                sb_dft = torch.fft.fft2(subbands[scale][orient])
                current_dft = current_dft + sb_dft * filt.conj()

        # Add high-pass residual
        H, W = sizes[0]
        radius0, _ = _polar_grid(H, W, self.device, self.dtype)
        hp_filter = _highpass_filter(radius0)
        hp_dft = torch.fft.fft2(highpass)
        current_dft = current_dft + hp_dft * hp_filter

        result = torch.fft.ifft2(current_dft).real
        logger.debug(f"Collapsed pyramid: {tuple(result.shape)}")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample_dft(
        dft: torch.Tensor, new_h: int, new_w: int
    ) -> torch.Tensor:
        """Crop DFT to ``(new_h, new_w)`` — equivalent to spatial downsampling."""
        H, W = dft.shape
        dft_shifted = torch.fft.fftshift(dft)
        ch, cw = H // 2, W // 2
        half_h, half_w = new_h // 2, new_w // 2
        cropped = dft_shifted[
            ch - half_h : ch - half_h + new_h,
            cw - half_w : cw - half_w + new_w,
        ]
        return torch.fft.ifftshift(cropped)

    @staticmethod
    def _upsample_dft(
        dft: torch.Tensor, new_h: int, new_w: int
    ) -> torch.Tensor:
        """Zero-pad DFT to ``(new_h, new_w)`` — equivalent to spatial upsampling."""
        H, W = dft.shape
        dft_shifted = torch.fft.fftshift(dft)
        padded = torch.zeros(new_h, new_w, dtype=dft.dtype, device=dft.device)
        # Centre the existing spectrum in the larger canvas
        ph = (new_h - H) // 2
        pw = (new_w - W) // 2
        padded[ph : ph + H, pw : pw + W] = dft_shifted
        # Scale to preserve energy
        scale = (new_h * new_w) / (H * W)
        return torch.fft.ifftshift(padded) * scale
