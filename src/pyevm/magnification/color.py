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

import time
from collections.abc import Generator, Iterable

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from pyevm.filters.temporal import ButterworthBandpass, IdealBandpass
from pyevm.magnification._colorspace import rgb_to_yiq, yiq_to_rgb
from pyevm.pyramids.gaussian import GaussianPyramid


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
        notch_freqs: list[float] | None = None,
        notch_width: float = 1.0,
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
        self.notch_freqs = notch_freqs or []
        self.notch_width = notch_width
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
            filt = IdealBandpass(
                fps,
                self.freq_low,
                self.freq_high,
                notch_freqs=self.notch_freqs,
                notch_width=self.notch_width,
            )
            filtered = filt.apply(level_tensor)  # (T, 3, h, w)
        else:
            filt = ButterworthBandpass(
                fps,
                self.freq_low,
                self.freq_high,
                notch_freqs=self.notch_freqs,
                notch_width=self.notch_width,
            )
            filtered = filt.apply(level_tensor)

        # --- Amplify: Y × alpha, I/Q × alpha × chrom_attenuation ---
        amplified = filtered.clone()
        amplified[:, 0] *= self.alpha
        amplified[:, 1:] *= self.alpha * self.chrom_attenuation
        logger.debug("Applied amplification to filtered signal")

        # --- Upsample amplified signal back to original resolution ---
        # We upsample the coarsest-level amplified signal to full resolution
        upsampled = torch.nn.functional.interpolate(
            amplified.reshape(T * 3, 1, *amplified.shape[2:]),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).reshape(T, 3, H, W)

        # --- Add to original YIQ ---
        result_yiq = yiq + upsampled

        # --- Convert back to RGB and clamp ---
        result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)
        logger.info("ColorMagnifier.process complete")
        return result_rgb

    def process_stream(
        self,
        frames: Iterable[torch.Tensor],
        fps: float,
        n_frames: int | None = None,
        chunk_size: int = 64,
    ) -> Generator[torch.Tensor, None, None]:
        """Process frames in chunks, yielding each output frame.

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
        filt = ButterworthBandpass(
            fps,
            self.freq_low,
            self.freq_high,
            notch_freqs=self.notch_freqs,
            notch_width=self.notch_width,
        )

        def _process_chunk(chunk: list[torch.Tensor]) -> Generator[torch.Tensor, None, None]:
            batch = torch.stack(chunk)  # (N, C, H, W)
            N, C, H, W = batch.shape

            t0 = time.perf_counter()
            yiq = rgb_to_yiq(batch)  # (N, 3, H, W)
            levels = self._pyramid.build(yiq)  # list of (N, 3, h, w)
            level_t = levels[self.pyramid_level]  # (N, 3, h, w)
            t_build = time.perf_counter() - t0

            # Temporal filter (state carried across chunks via filt._zi)
            t1 = time.perf_counter()
            filtered = filt.apply_chunk(level_t)  # (N, 3, h, w)
            t_filter = time.perf_counter() - t1

            # Amplify: Y × alpha, I/Q × alpha × chrom_attenuation
            t2 = time.perf_counter()
            amplified = filtered.clone()
            amplified[:, 0] *= self.alpha
            amplified[:, 1:] *= self.alpha * self.chrom_attenuation

            # Upsample amplified signal back to original resolution
            _, _, h, w = amplified.shape
            upsampled = F.interpolate(
                amplified.reshape(N * 3, 1, h, w),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).reshape(N, 3, H, W)

            result_yiq = yiq + upsampled
            result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)  # (N, C, H, W)
            t_reconstruct = time.perf_counter() - t2

            logger.debug(
                f"  [color chunk N={N}]  "
                f"build={t_build * 1000:.1f}ms  "
                f"filter={t_filter * 1000:.1f}ms  "
                f"reconstruct={t_reconstruct * 1000:.1f}ms"
            )
            yield from result_rgb

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
            if chunk:
                t0 = time.perf_counter()
                for out_frame in _process_chunk(chunk):
                    yield out_frame
                    bar.update(1)
                elapsed = time.perf_counter() - t0
                logger.debug(
                    f"Chunk {len(chunk)} frames: {elapsed:.2f}s ({len(chunk) / elapsed:.1f} fps)"
                )
