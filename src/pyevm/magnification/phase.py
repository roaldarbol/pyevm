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

import time
from collections.abc import Generator, Iterable

import torch
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
        """Spatial Gaussian smoothing of a ``(H, W)`` or ``(T, H, W)`` phase map."""
        if self.sigma <= 0:
            return phase
        radius = int(3 * self.sigma)
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=self.dtype, device=self.device) - radius
        g1d = torch.exp(-coords ** 2 / (2 * self.sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = torch.outer(g1d, g1d).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        if phase.dim() == 2:
            ph = phase.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            return torch.nn.functional.conv2d(ph, kernel, padding=radius).squeeze(0).squeeze(0)
        else:  # (T, H, W) — treat T as batch dim for conv2d
            ph = phase.unsqueeze(1)  # (T,1,H,W)
            return torch.nn.functional.conv2d(ph, kernel, padding=radius).squeeze(1)  # (T,H,W)

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

        # --- Build steerable pyramid for ALL frames in one batched GPU call ---
        logger.debug("Building steerable pyramids for all frames…")
        pyramid = self._pyramid.build(luma)  # subbands[s][o]: (T, H_s, W_s) complex

        # --- Process each sub-band ---
        n_bands = self.n_scales * self.n_orientations
        with tqdm(total=n_bands, desc="Magnifying", unit="band", leave=False) as bar:
            for scale in range(self.n_scales):
                for orient in range(self.n_orientations):
                    coeffs = pyramid["subbands"][scale][orient]  # (T, H_s, W_s) complex

                    # Extract phase and remove DC (reference = first frame)
                    phase = torch.angle(coeffs)       # (T, H_s, W_s) real
                    delta_phase = phase - phase[0:1]

                    # Spatial smoothing (batched over T)
                    if self.sigma > 0:
                        delta_phase = self._smooth_phase(delta_phase)

                    # Temporal filter
                    if self.filter_type == "ideal":
                        filt = IdealBandpass(fps, self.freq_low, self.freq_high)
                        filtered_phase = filt.apply(delta_phase)
                    else:
                        filt_bw = ButterworthBandpass(fps, self.freq_low, self.freq_high)
                        filtered_phase = filt_bw.apply(delta_phase)

                    # Amplify and write modified coefficients back
                    amp_phase = filtered_phase * self.factor
                    pyramid["subbands"][scale][orient] = torch.polar(
                        coeffs.abs(), phase + amp_phase
                    )
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
        collapse in one GPU call (much better utilisation than frame-by-frame).
        The Butterworth IIR filter state carries across chunk boundaries via
        its ``zi`` parameter, so the result is numerically identical to true
        frame-by-frame streaming.

        Memory scales with *chunk_size*, not total video length.  At 1080p
        each chunk occupies roughly ``chunk_size × 165 MB`` of VRAM.

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
            (s, o): ButterworthBandpass(fps, self.freq_low, self.freq_high)
            for s in range(self.n_scales)
            for o in range(self.n_orientations)
        }
        ref_phase: dict[tuple[int, int], torch.Tensor] = {}

        def _process_chunk(chunk: list[torch.Tensor]) -> Generator[torch.Tensor, None, None]:
            N = len(chunk)
            batch = torch.stack(chunk)                  # (N, C, H, W)

            t0 = time.perf_counter()
            yiq   = rgb_to_yiq(batch)                   # (N, 3, H, W)
            luma  = yiq[:, 0, :, :]                     # (N, H, W)
            pyramid = self._pyramid.build(luma)         # subbands: (N, H_s, W_s)
            t_build = time.perf_counter() - t0

            t1 = time.perf_counter()
            for s in range(self.n_scales):
                for o in range(self.n_orientations):
                    coeffs = pyramid["subbands"][s][o]  # (N, H_s, W_s) complex
                    phase  = torch.angle(coeffs)        # (N, H_s, W_s)

                    key = (s, o)
                    if key not in ref_phase:
                        ref_phase[key] = phase[0].clone()

                    delta = phase - ref_phase[key]

                    if self.sigma > 0:
                        delta = self._smooth_phase(delta)

                    filtered = filters[key].apply_chunk(delta)
                    pyramid["subbands"][s][o] = torch.polar(
                        coeffs.abs(), phase + filtered * self.factor
                    )
            t_filter = time.perf_counter() - t1

            t2 = time.perf_counter()
            luma_out   = self._pyramid.collapse(pyramid)        # (N, H, W)
            result_yiq = yiq.clone()
            result_yiq[:, 0, :, :] = luma_out
            result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)  # (N, C, H, W)
            t_collapse = time.perf_counter() - t2

            logger.debug(
                f"  [phase chunk N={N}]  "
                f"build={t_build*1000:.1f}ms  "
                f"filter={t_filter*1000:.1f}ms  "
                f"collapse={t_collapse*1000:.1f}ms"
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
                    logger.debug(f"Chunk {len(chunk)} frames: {elapsed:.2f}s ({len(chunk)/elapsed:.1f} fps)")
                    chunk = []
            if chunk:  # final partial chunk
                t0 = time.perf_counter()
                for out_frame in _process_chunk(chunk):
                    yield out_frame
                    bar.update(1)
                elapsed = time.perf_counter() - t0
                logger.debug(f"Chunk {len(chunk)} frames: {elapsed:.2f}s ({len(chunk)/elapsed:.1f} fps)")
