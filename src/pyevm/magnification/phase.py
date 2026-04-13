"""Phase-based Eulerian Video Magnification (Wadhwa et al., SIGGRAPH 2013).

Algorithm
---------
1. Convert video frames to YIQ; work on the luminance channel (Y).
2. Build a complex steerable pyramid for each frame.
3. For each sub-band (scale × orientation):
   a. Extract amplitude and phase: A = |coeff|, φ = angle(coeff)
   b. Phase delta:  Δφ(t) = φ(t) − φ(0)  (remove reference-frame offset)
   c. Temporal bandpass filter Δφ → B(t)  (isolate motion frequency band)
   d. Amplitude-weighted spatial smoothing of B(t)  (Eq. 17, optional)
      Smoothing increases phase SNR; amplitude-weighting prevents low-amplitude
      (noisy-phase) regions from corrupting neighbouring signal.
   e. Amplify:   amp(t) = factor × B(t)
   f. Shift coeff:  coeff'(t) = A(t) × exp(j·(φ(t) + amp(t)))
4. Reconstruct Y from modified pyramid for each frame.
5. Recombine with I, Q channels and convert back to RGB.

Key correctness notes (vs. a naïve implementation)
----------------------------------------------------
* **Amplitude-weighted smoothing** (Eq. 17): the Gaussian blur is weighted by
  the local coefficient amplitude A.  Pixels where A ≈ 0 (flat/uniform
  regions, e.g. blue sky) have meaningless phase; equal-weight blurring
  spreads that noise onto high-contrast neighbours, which is catastrophic at
  large amplification factors.

* **Smoothing order**: temporal filtering comes *first*, then spatial
  smoothing is applied to the temporally-bandpassed phase B(t).  Smoothing
  the raw Δφ first lets broadband spatial noise contaminate B(t) before the
  temporal filter can attenuate it.

Reference: Wadhwa et al., Fig. 2 and Sect. 3.4.
"""

from __future__ import annotations

import math
import time
from collections.abc import Generator, Iterable

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from pyevm.filters.temporal import ButterworthBandpass, IdealBandpass
from pyevm.magnification._colorspace import rgb_to_yiq, yiq_to_rgb
from pyevm.pyramids.steerable import SteerablePyramid


class PhaseMagnifier:
    """Phase-based EVM magnifier.

    Args:
        factor: Phase amplification factor.
        freq_low: Temporal bandpass lower frequency (Hz).
        freq_high: Temporal bandpass upper frequency (Hz).
        n_scales: Number of pyramid scales.
        n_orientations: Number of orientation bands per scale.
        sigma: Spatial phase smoothing sigma (pixels, ``0`` = disabled).
            Uses amplitude-weighted Gaussian blur (Eq. 17) so that low-amplitude
            regions do not corrupt the phase of high-amplitude neighbours.
        filter_type: ``"ideal"`` (FFT) or ``"butterworth"`` (IIR).
        attenuate_motion: If ``True``, amplified phase changes that exceed
            ``attenuate_mag`` are **wrapped** back into ``[−lim, +lim]`` rather
            than applied directly.  Large motions (e.g. camera shake) produce
            large uniform phase changes after amplification and are effectively
            attenuated, while subtle local motions with ``|amp| < lim`` pass
            through unmodified.  Corresponds to the "Attenuate" mode in Fig. 11
            of Wadhwa et al. 2013.
        attenuate_mag: Threshold for large-motion attenuation (radians).
            Default ``π`` — the largest unambiguous single-step phase change.
        device: Compute device.
        dtype: Real tensor dtype (sub-band coefficients are complex).
    """

    def __init__(
        self,
        factor: float = 10.0,
        freq_low: float = 0.4,
        freq_high: float = 3.0,
        n_scales: int = 6,
        n_orientations: int = 8,
        sigma: float = 0.0,
        filter_type: str = "ideal",
        attenuate_motion: bool = False,
        attenuate_mag: float = math.pi,
        notch_freqs: list[float] | None = None,
        notch_width: float = 1.0,
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
        self.attenuate_motion = attenuate_motion
        self.attenuate_mag = attenuate_mag
        self.notch_freqs = notch_freqs or []
        self.notch_width = notch_width
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
            + (f", attenuate_mag={attenuate_mag:.3f}" if attenuate_motion else "")
        )

    def _smooth_phase(self, phase: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        """Amplitude-weighted spatial Gaussian smoothing (Eq. 17, Wadhwa et al. 2013).

        Computes:   (φ · A) ∗ K_σ  /  (A ∗ K_σ)

        where K_σ is a Gaussian kernel.  Pixels with near-zero amplitude
        contribute negligible weight, preventing their noisy phase values from
        corrupting high-amplitude neighbours.

        Args:
            phase:     ``(H, W)`` or ``(T, H, W)`` real phase tensor.
            amplitude: Same shape — local coefficient amplitude (``abs(coeff)``).

        Returns:
            Smoothed phase with the same shape.
        """
        if self.sigma <= 0:
            return phase

        radius = int(3 * self.sigma)
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=self.dtype, device=self.device) - radius
        g1d = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g1d = g1d / g1d.sum()
        kernel = torch.outer(g1d, g1d).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

        if phase.dim() == 2:
            ph = phase.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            am = amplitude.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        else:  # (T, H, W) or (N, H, W) — leading dim treated as batch
            ph = phase.unsqueeze(1)  # (T,1,H,W)
            am = amplitude.unsqueeze(1)  # (T,1,H,W)

        numer = F.conv2d(ph * am, kernel, padding=radius)
        denom = F.conv2d(am, kernel, padding=radius).clamp(min=1e-8)
        result = numer / denom

        if phase.dim() == 2:
            return result.squeeze(0).squeeze(0)
        return result.squeeze(1)

    def _apply_attenuation(self, amp_phase: torch.Tensor) -> torch.Tensor:
        """Wrap amplified phase into ``[−lim, +lim]`` to attenuate large motions.

        Implements the "Attenuate" mode from Fig. 11 of Wadhwa et al. 2013.

        The wrapping formula is:
            ``mod(amp_phase + lim, 2·lim) − lim``

        Behaviour by region (lim = π by default):
            * ``|amp| ≤ π``: returned unchanged — small motions amplified normally.
            * ``|amp| = 2π``: maps to 0 — motion fully cancelled.
            * ``|amp| > 2π``: wraps back into [−π, π] — motion attenuated.

        Camera shake produces large uniform phase changes across the frame;
        after amplification these exceed π and are wrapped back, effectively
        cancelling the global motion.  Subtle local vibrations with
        ``|amp| < π`` are unaffected.

        Args:
            amp_phase: Already-amplified phase tensor (any shape).

        Returns:
            Phase tensor with large-motion contributions attenuated.
        """
        lim = self.attenuate_mag
        return (amp_phase + lim) % (2 * lim) - lim

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
        luma = yiq[:, 0, :, :]  # (T, H, W)

        # --- Build steerable pyramid for ALL frames in one batched GPU call ---
        logger.debug("Building steerable pyramids for all frames…")
        pyramid = self._pyramid.build(luma)  # subbands[s][o]: (T, H_s, W_s) complex

        # --- Process each sub-band ---
        n_bands = self.n_scales * self.n_orientations
        with tqdm(total=n_bands, desc="Magnifying", unit="band", leave=False) as bar:
            for scale in range(self.n_scales):
                for orient in range(self.n_orientations):
                    coeffs = pyramid["subbands"][scale][orient]  # (T, H_s, W_s) complex
                    amplitude = coeffs.abs()  # (T, H_s, W_s)
                    phase = torch.angle(coeffs)  # (T, H_s, W_s)
                    # Circular wrapping: mod(π + Δ, 2π) − π  (matches MATLAB reference)
                    delta_phase = (phase - phase[0:1] + math.pi) % (2 * math.pi) - math.pi

                    # Step 1: temporal filter — isolate motion frequency band
                    if self.filter_type == "ideal":
                        filt = IdealBandpass(
                            fps,
                            self.freq_low,
                            self.freq_high,
                            notch_freqs=self.notch_freqs,
                            notch_width=self.notch_width,
                        )
                        filtered_phase = filt.apply(delta_phase)
                    else:
                        filt_bw = ButterworthBandpass(
                            fps,
                            self.freq_low,
                            self.freq_high,
                            notch_freqs=self.notch_freqs,
                            notch_width=self.notch_width,
                        )
                        filtered_phase = filt_bw.apply(delta_phase)

                    # Step 2: amplitude-weighted spatial smoothing (Eq. 17)
                    # Applied AFTER temporal filtering (Fig. 2 order) so that
                    # broadband phase noise is attenuated before smoothing.
                    if self.sigma > 0:
                        filtered_phase = self._smooth_phase(filtered_phase, amplitude)

                    # Step 3: amplify (and optionally attenuate large motions) then shift
                    amp_phase = filtered_phase * self.factor
                    if self.attenuate_motion:
                        amp_phase = self._apply_attenuation(amp_phase)
                    pyramid["subbands"][scale][orient] = torch.polar(amplitude, phase + amp_phase)
                    logger.debug(
                        f"Scale {scale}, orientation {orient}: "
                        f"phase amp range [{amp_phase.min():.3f}, {amp_phase.max():.3f}]"
                    )
                    bar.update(1)

        # --- Reconstruct Y from all modified pyramids in one batched GPU call ---
        logger.debug("Collapsing modified pyramids…")
        luma_out = self._pyramid.collapse(pyramid)  # (T, H, W)

        # --- Recombine with I, Q and convert to RGB ---
        result_yiq = yiq.clone()
        result_yiq[:, 0, :, :] = luma_out
        result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)
        logger.info("PhaseMagnifier.process complete")
        return result_rgb

    def process_stream(
        self,
        frames: Iterable[torch.Tensor],
        fps: float,
        n_frames: int | None = None,
        chunk_size: int = 64,
    ) -> Generator[torch.Tensor, None, None]:
        """Process frames in chunks, yielding each output frame.

        Buffers *chunk_size* frames, then runs the batched pyramid build and
        collapse in one GPU call.  The Butterworth IIR filter state carries
        across chunk boundaries via its ``zi`` parameter.

        Uses Butterworth IIR regardless of *filter_type* (the ideal FFT filter
        requires all frames at once and cannot be used in streaming mode).

        Args:
            frames: Iterable of ``(C, H, W)`` float32 RGB tensors on any device.
            fps: Frames per second.
            n_frames: Total frame count (optional, used for the progress bar).
            chunk_size: Number of frames to process per GPU batch.

        Yields:
            Amplified ``(C, H, W)`` float32 RGB tensors, clamped to ``[0, 1]``.
        """
        filters: dict[tuple[int, int], ButterworthBandpass] = {
            (s, o): ButterworthBandpass(
                fps,
                self.freq_low,
                self.freq_high,
                notch_freqs=self.notch_freqs,
                notch_width=self.notch_width,
            )
            for s in range(self.n_scales)
            for o in range(self.n_orientations)
        }
        ref_phase: dict[tuple[int, int], torch.Tensor] = {}

        def _process_chunk(chunk: list[torch.Tensor]) -> Generator[torch.Tensor, None, None]:
            N = len(chunk)
            batch = torch.stack(chunk)  # (N, C, H, W)

            t0 = time.perf_counter()
            yiq = rgb_to_yiq(batch)  # (N, 3, H, W)
            luma = yiq[:, 0, :, :]  # (N, H, W)
            pyramid = self._pyramid.build(luma)  # subbands: (N, H_s, W_s)
            t_build = time.perf_counter() - t0

            t1 = time.perf_counter()
            for s in range(self.n_scales):
                for o in range(self.n_orientations):
                    coeffs = pyramid["subbands"][s][o]  # (N, H_s, W_s) complex
                    amplitude = coeffs.abs()  # (N, H_s, W_s)
                    phase = torch.angle(coeffs)  # (N, H_s, W_s)

                    key = (s, o)
                    if key not in ref_phase:
                        ref_phase[key] = phase[0].clone()

                    # Circular wrapping: mod(π + Δ, 2π) − π  (matches MATLAB reference)
                    delta = (phase - ref_phase[key] + math.pi) % (2 * math.pi) - math.pi

                    # Temporal filter first, then amplitude-weighted spatial smooth
                    filtered = filters[key].apply_chunk(delta)
                    if self.sigma > 0:
                        filtered = self._smooth_phase(filtered, amplitude)

                    amp_phase = filtered * self.factor
                    if self.attenuate_motion:
                        amp_phase = self._apply_attenuation(amp_phase)
                    pyramid["subbands"][s][o] = torch.polar(amplitude, phase + amp_phase)
            t_filter = time.perf_counter() - t1

            t2 = time.perf_counter()
            luma_out = self._pyramid.collapse(pyramid)  # (N, H, W)
            result_yiq = yiq.clone()
            result_yiq[:, 0, :, :] = luma_out
            result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)  # (N, C, H, W)
            t_collapse = time.perf_counter() - t2

            logger.debug(
                f"  [phase chunk N={N}]  "
                f"build={t_build * 1000:.1f}ms  "
                f"filter={t_filter * 1000:.1f}ms  "
                f"collapse={t_collapse * 1000:.1f}ms"
            )
            yield from result_rgb  # yield N individual (C, H, W) frames

        chunk: list[torch.Tensor] = []
        with tqdm(total=n_frames, desc="Magnifying", unit="frame", position=1, leave=True) as bar:
            for frame in frames:
                chunk.append(frame.to(device=self.device, dtype=self.dtype))
                if len(chunk) == chunk_size:
                    t0 = time.perf_counter()
                    for out_frame in _process_chunk(chunk):
                        yield out_frame
                        bar.update(1)
                    elapsed = time.perf_counter() - t0
                    logger.debug(
                        f"Chunk {len(chunk)} frames: {elapsed:.2f}s ({len(chunk) / elapsed:.1f} fps)"
                    )
                    chunk = []
            if chunk:  # final partial chunk
                t0 = time.perf_counter()
                for out_frame in _process_chunk(chunk):
                    yield out_frame
                    bar.update(1)
                elapsed = time.perf_counter() - t0
                logger.debug(
                    f"Chunk {len(chunk)} frames: {elapsed:.2f}s ({len(chunk) / elapsed:.1f} fps)"
                )
