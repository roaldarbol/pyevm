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

import time
from collections.abc import Generator, Iterable

import torch
from loguru import logger
from tqdm import tqdm

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
        notch_freqs: list[float] | None = None,
        notch_width: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.alpha = alpha
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.n_levels = n_levels
        self.lambda_c = lambda_c
        self.filter_type = filter_type
        self.notch_freqs = notch_freqs or []
        self.notch_width = notch_width
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self._pyramid = LaplacianPyramid(n_levels=n_levels, device=self.device, dtype=dtype)

        logger.info(
            f"MotionMagnifier: alpha={alpha}, band=[{freq_low}, {freq_high}] Hz, "
            f"lambda_c={lambda_c}, filter={filter_type}"
        )

    def _alpha_for_level(self, level: int) -> float:
        """Compute spatially-adaptive alpha for *level*.

        The spatial wavelength at level *i* is ``2^(i+1)`` pixels at full
        resolution (level 0 = finest, λ = 2 px; level 5 = coarsest, λ = 64 px).

        Fine levels (λ < λ_c) are suppressed to avoid amplifying noise and
        aliasing artefacts.  Coarser levels receive progressively higher
        amplification up to *alpha*.  The crossover is at λ = λ_c / 8; above
        that the cap grows linearly.
        """
        lambda_at_level = 2 ** (level + 1)
        alpha_max = 8.0 * lambda_at_level / self.lambda_c - 1.0
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
                filt = IdealBandpass(
                    fps,
                    self.freq_low,
                    self.freq_high,
                    notch_freqs=self.notch_freqs,
                    notch_width=self.notch_width,
                )
                filtered = filt.apply(level_tensor)
            else:
                filt = ButterworthBandpass(
                    fps,
                    self.freq_low,
                    self.freq_high,
                    notch_freqs=self.notch_freqs,
                    notch_width=self.notch_width,
                )
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
        # One filter per pyramid level; state carries across chunk boundaries
        filters = [
            ButterworthBandpass(
                fps,
                self.freq_low,
                self.freq_high,
                notch_freqs=self.notch_freqs,
                notch_width=self.notch_width,
            )
            for _ in range(self.n_levels)
        ]
        # Precompute per-level alpha
        alphas = [self._alpha_for_level(lvl) for lvl in range(self.n_levels)]

        def _process_chunk(chunk: list[torch.Tensor]) -> Generator[torch.Tensor, None, None]:
            N = len(chunk)
            batch = torch.stack(chunk)  # (N, C, H, W)

            t0 = time.perf_counter()
            yiq = rgb_to_yiq(batch)  # (N, 3, H, W)
            levels = self._pyramid.build(yiq)  # list of n_levels × (N, 3, h_l, w_l)
            t_build = time.perf_counter() - t0

            t_filter_lvl: list[float] = []
            modified_levels = []
            for lvl in range(self.n_levels):
                level_t = levels[lvl]  # (N, 3, h_l, w_l)
                if alphas[lvl] == 0.0:
                    # Skip IIR call entirely; still advance state so timing is consistent
                    modified_levels.append(level_t)
                    t_filter_lvl.append(0.0)
                else:
                    tf = time.perf_counter()
                    filtered = filters[lvl].apply_chunk(level_t)  # (N, 3, h_l, w_l)
                    t_filter_lvl.append(time.perf_counter() - tf)
                    modified_levels.append(level_t + filtered * alphas[lvl])

            t1 = time.perf_counter()
            result_yiq = self._pyramid.collapse(modified_levels)  # (N, 3, H, W)
            result_rgb = yiq_to_rgb(result_yiq).clamp(0.0, 1.0)  # (N, C, H, W)
            t_collapse = time.perf_counter() - t1

            total_filter = sum(t_filter_lvl)
            per_lvl = "  ".join(f"L{i}={ms * 1000:.1f}ms" for i, ms in enumerate(t_filter_lvl))
            logger.debug(
                f"  [motion chunk N={N}]  "
                f"build={t_build * 1000:.1f}ms  "
                f"filter_total={total_filter * 1000:.1f}ms ({per_lvl})  "
                f"collapse={t_collapse * 1000:.1f}ms"
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
