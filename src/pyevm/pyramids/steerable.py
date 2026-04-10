"""Complex steerable pyramid (Simoncelli & Freeman, 1995).

Implemented entirely in PyTorch so it runs on CUDA / MPS / CPU.

Reference
---------
Wadhwa et al., "Phase-Based Video Motion Processing", SIGGRAPH 2013.
Simoncelli & Freeman, "The Steerable Pyramid", ICIP 1995.

Filter design (tight-frame / perfect reconstruction)
-----------------------------------------------------
The pyramid uses a two-level filter structure:

**Outer split** (full resolution):
    lo0(r)  = low-pass,  smooth from 1→0 in [π/2, π]
    hi0(r)  = sqrt(1 − lo0²), the complementary high-pass

**Inner split** (at each recursive scale, in current-resolution coordinates):
    lp(r)   = low-pass,  smooth from 1→0 in [π/4, π/2]
    bp(r)   = sqrt(1 − lp²), the complementary band-pass

**Oriented filters** (at each scale × orientation):
    H_k(r, θ) = bp(r) · g_k(θ)

where g_k are Simoncelli binomial angular filters normalised so that
    Σ_k g_k(θ)² = 1  for all θ   (antipodal symmetry)

This gives the partition of unity at each scale:
    lp² + Σ_k |H_k|² = lp² + bp² · Σ_k g_k² = lp² + (1−lp²)·1 = 1

And at the outer level:
    lo0² + hi0² = 1

So the overall frame bound is 1 → near-perfect reconstruction.

Reconstruction formula
----------------------
    collapse = lo0 · (lo0 · LP_scales + hp_contrib)  [outer LP applied twice]

where LP_scales is reconstructed from the inner levels via:
    X_s = lp · X_{s+1,up} + Σ_k sb_k · bp · g_k
        = lp² · X_{s,orig} + bp² · X_{s,orig} = X_{s,orig}  ✓
"""

from __future__ import annotations

import math

import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Frequency-domain filter helpers
# ---------------------------------------------------------------------------

def _polar_grid(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised polar frequency grid.

    Returns:
        radius: ``(H, W)`` tensor in ``[0, π√2]``.
        angle:  ``(H, W)`` tensor in ``(−π, π]``.
    """
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype) * 2 * math.pi
    fx = torch.fft.fftfreq(width, device=device, dtype=dtype) * 2 * math.pi
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    angle = torch.atan2(grid_y, grid_x)
    return radius, angle


def _raised_cosine(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Smooth transition from 0 → 1 between *lo* and *hi*."""
    x = x.clamp(lo, hi)
    return 0.5 * (1.0 - torch.cos(math.pi * (x - lo) / (hi - lo)))


# -- Outer LP/HP (full-resolution split) ------------------------------------

def _lo0(radius: torch.Tensor) -> torch.Tensor:
    """Outer low-pass: 1 for r < π/2, smooth to 0 at r = π."""
    return 1.0 - _raised_cosine(radius, math.pi / 2, math.pi)


def _hi0(lo0_vals: torch.Tensor) -> torch.Tensor:
    """Outer high-pass: sqrt(1 − lo0²)."""
    return torch.sqrt((1.0 - lo0_vals ** 2).clamp(min=0.0))


# -- Inner LP/BP (at each scale) --------------------------------------------

def _lp(radius: torch.Tensor) -> torch.Tensor:
    """Inner low-pass: 1 for r < π/4, smooth to 0 at r = π/2."""
    return 1.0 - _raised_cosine(radius, math.pi / 4, math.pi / 2)


def _bp(lp_vals: torch.Tensor) -> torch.Tensor:
    """Inner band-pass: sqrt(1 − lp²), complementary to _lp."""
    return torch.sqrt((1.0 - lp_vals ** 2).clamp(min=0.0))


# -- Angular filters (Simoncelli binomial) ----------------------------------

def _alpha_simoncelli(n: int) -> float:
    """Normalisation constant so that Σ_k g_k(θ)² = 1 (with antipodal lobes)."""
    return math.sqrt(
        2 ** (2 * (n - 1)) * math.factorial(n - 1) ** 2
        / (n * math.factorial(2 * (n - 1)))
    )


def _angular_filter(
    angle: torch.Tensor, orient_idx: int, n_orientations: int
) -> torch.Tensor:
    """Simoncelli angular filter g_k = α · |cos(θ − k·π/n)|^{n−1}.

    Both the primary lobe and its antipodal lobe are included so that the
    partition of unity holds for all θ.

    Returns a **real** tensor; the sub-band becomes complex because it is
    multiplied with a complex (FFT) spectrum.
    """
    alpha = _alpha_simoncelli(n_orientations)
    n = n_orientations
    target = math.pi * orient_idx / n

    def _lobe(diff: torch.Tensor) -> torch.Tensor:
        diff = ((diff + math.pi) % (2 * math.pi)) - math.pi
        inside = diff.abs() <= math.pi / 2 + 1e-7
        cos_val = torch.cos(diff).clamp(0.0, 1.0)
        return torch.where(inside, cos_val.pow(n - 1), torch.zeros_like(diff))

    primary = _lobe(angle - target)
    antipodal = _lobe(angle - (target + math.pi))
    return alpha * (primary + antipodal)


def _oriented_filter(
    radius: torch.Tensor,
    angle: torch.Tensor,
    orient_idx: int,
    n_orientations: int,
    lp_vals: torch.Tensor,
) -> torch.Tensor:
    """Real oriented band-pass filter: bp(r) · g_k(θ).

    Args:
        lp_vals: Pre-computed inner LP values at current scale/resolution.
    """
    bp_vals = _bp(lp_vals)
    angular = _angular_filter(angle, orient_idx, n_orientations)
    return bp_vals * angular  # real tensor


# ---------------------------------------------------------------------------
# Pyramid class
# ---------------------------------------------------------------------------

class SteerablePyramid:
    """Complex steerable pyramid (tight frame, near-perfect reconstruction).

    Args:
        n_scales: Number of octave levels.
        n_orientations: Oriented sub-bands per scale (2, 4, 6, or 8).
        device: Compute device.
        dtype: Real floating-point dtype.
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
            f"SteerablePyramid: {n_scales} scales × {n_orientations} "
            f"orientations on {self.device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, frame: torch.Tensor) -> dict:
        """Decompose a single-channel frame into the complex steerable pyramid.

        Args:
            frame: ``(H, W)`` or ``(1, 1, H, W)`` real tensor.

        Returns:
            Dictionary with keys:

            * ``"highpass"``  – ``(H, W)`` complex tensor (outer HP residual).
            * ``"lowpass"``   – ``(H', W')`` real tensor (coarsest LP residual).
            * ``"subbands"``  – ``[n_scales][n_orientations]`` complex tensors.
            * ``"sizes"``     – ``(H_s, W_s)`` at each scale.
        """
        if frame.dim() == 4:
            frame = frame.squeeze(0).squeeze(0)
        elif frame.dim() == 3:
            frame = frame.squeeze(0)

        frame = frame.to(device=self.device, dtype=self.dtype)
        H, W = frame.shape
        dft = torch.fft.fft2(frame)

        # --- Outer LP/HP split ---
        radius0, _ = _polar_grid(H, W, self.device, self.dtype)
        lo0_vals = _lo0(radius0)
        hi0_vals = _hi0(lo0_vals)

        highpass = torch.fft.ifft2(dft * hi0_vals)
        logger.debug(f"Highpass residual: {tuple(highpass.shape)}")

        # Scale loop starts from the LP component only
        current_dft = dft * lo0_vals
        current_h, current_w = H, W

        subbands: list[list[torch.Tensor]] = []
        sizes: list[tuple[int, int]] = []

        for scale in range(self.n_scales):
            radius_s, angle_s = _polar_grid(current_h, current_w, self.device, self.dtype)
            lp_vals = _lp(radius_s)

            scale_bands: list[torch.Tensor] = []
            for orient in range(self.n_orientations):
                filt = _oriented_filter(radius_s, angle_s, orient, self.n_orientations, lp_vals)
                subband = torch.fft.ifft2(current_dft * filt)
                scale_bands.append(subband)
                logger.debug(
                    f"Scale {scale}, orientation {orient}: {tuple(subband.shape)}"
                )

            subbands.append(scale_bands)
            sizes.append((current_h, current_w))

            # Pass LP component to next scale (downsample)
            current_h //= 2
            current_w //= 2
            current_dft = self._downsample_dft(current_dft * lp_vals, current_h, current_w)

        lowpass = torch.fft.ifft2(current_dft).real
        logger.debug(f"Lowpass residual: {tuple(lowpass.shape)}")

        return {
            "highpass": highpass,
            "lowpass": lowpass,
            "subbands": subbands,
            "sizes": sizes,
        }

    def collapse(self, pyramid: dict) -> torch.Tensor:
        """Reconstruct a frame from a (possibly phase-modified) pyramid.

        Args:
            pyramid: Dictionary as returned by :meth:`build`.

        Returns:
            Reconstructed ``(H, W)`` real tensor.
        """
        subbands = pyramid["subbands"]
        sizes = pyramid["sizes"]
        highpass = pyramid["highpass"]
        lowpass = pyramid["lowpass"]

        # --- Inner scale reconstruction (coarse → fine) ---
        current_dft = torch.fft.fft2(lowpass.to(dtype=self.dtype))

        for scale in range(self.n_scales - 1, -1, -1):
            target_h, target_w = sizes[scale]
            current_dft = self._upsample_dft(current_dft, target_h, target_w)

            radius_s, angle_s = _polar_grid(target_h, target_w, self.device, self.dtype)
            lp_vals = _lp(radius_s)

            # LP² component: apply inner LP to the upsampled LP signal
            current_dft = current_dft * lp_vals

            # BP component: sum oriented sub-band contributions
            # sb_dft = fft2(ifft2(X_s * H_k)) = X_s * H_k  →  sb_dft * H_k = X_s * H_k²
            # Σ_k X_s · H_k² = X_s · bp² · Σ_k g_k² = X_s · (1 − lp²)
            for orient in range(self.n_orientations):
                filt = _oriented_filter(
                    radius_s, angle_s, orient, self.n_orientations, lp_vals
                )
                sb_dft = torch.fft.fft2(subbands[scale][orient])
                current_dft = current_dft + sb_dft * filt  # real filt → conj == self

        # --- Outer LP² contribution ---
        # Scale reconstruction gave dft·lo0; apply lo0 again → dft·lo0²
        H, W = sizes[0]
        radius0, _ = _polar_grid(H, W, self.device, self.dtype)
        lo0_vals = _lo0(radius0)
        current_dft = current_dft * lo0_vals

        # --- Outer HP² contribution: dft·hi0² = dft·(1 − lo0²) ---
        hi0_vals = _hi0(lo0_vals)
        current_dft = current_dft + torch.fft.fft2(highpass) * hi0_vals

        result = torch.fft.ifft2(current_dft).real
        logger.debug(f"Collapsed pyramid: {tuple(result.shape)}")
        return result

    # ------------------------------------------------------------------
    # DFT up/downsample helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample_dft(dft: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
        """Crop DFT spectrum to ``(new_h, new_w)`` — bandlimited downsampling."""
        H, W = dft.shape
        shifted = torch.fft.fftshift(dft)
        ch, cw = H // 2, W // 2
        half_h, half_w = new_h // 2, new_w // 2
        cropped = shifted[
            ch - half_h: ch - half_h + new_h,
            cw - half_w: cw - half_w + new_w,
        ]
        return torch.fft.ifftshift(cropped)

    @staticmethod
    def _upsample_dft(dft: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
        """Zero-pad DFT spectrum to ``(new_h, new_w)`` — bandlimited upsampling.

        No amplitude scaling: for a band-limited signal the crop→pad roundtrip
        is the identity, so the DFT values themselves need no rescaling.
        """
        H, W = dft.shape
        shifted = torch.fft.fftshift(dft)
        padded = torch.zeros(new_h, new_w, dtype=dft.dtype, device=dft.device)
        ph = (new_h - H) // 2
        pw = (new_w - W) // 2
        padded[ph: ph + H, pw: pw + W] = shifted
        return torch.fft.ifftshift(padded)
