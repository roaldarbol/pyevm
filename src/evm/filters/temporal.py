"""Temporal bandpass filters for video magnification.

Two strategies are provided:

IdealBandpass
    FFT-based brick-wall filter.  Requires all frames in memory at once but
    gives perfectly flat pass-band.  Best for colour/phase EVM.

ButterworthBandpass
    IIR filter applied frame-by-frame via ``scipy.signal.sosfilt``.
    O(1) memory — suitable for long videos and streaming.  Used for motion EVM
    and as a memory-efficient alternative to the ideal filter.
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from scipy.signal import butter, sosfilt, sosfilt_zi


class IdealBandpass:
    """FFT-based ideal bandpass filter over the time axis.

    Args:
        fps: Frames per second of the video.
        freq_low: Lower cut-off frequency in Hz.
        freq_high: Upper cut-off frequency in Hz.
    """

    def __init__(self, fps: float, freq_low: float, freq_high: float) -> None:
        self.fps = fps
        self.freq_low = freq_low
        self.freq_high = freq_high
        logger.debug(
            f"IdealBandpass: {freq_low}–{freq_high} Hz @ {fps} fps"
        )

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Filter *signal* along its first (time) dimension.

        Args:
            signal: ``(T, ...)`` tensor where ``T`` is the number of frames.

        Returns:
            Bandpass-filtered tensor with the same shape.
        """
        T = signal.shape[0]
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps, device=signal.device)

        # FFT along time axis (dim 0)
        spectrum = torch.fft.rfft(signal.float(), dim=0)

        # Build mask
        mask = ((freqs >= self.freq_low) & (freqs <= self.freq_high)).float()
        # Broadcast mask over all spatial dims
        for _ in range(signal.dim() - 1):
            mask = mask.unsqueeze(-1)

        filtered_spectrum = spectrum * mask
        result = torch.fft.irfft(filtered_spectrum, n=T, dim=0)
        logger.debug(
            f"IdealBandpass: filtered signal shape {tuple(signal.shape)} "
            f"→ {tuple(result.shape)}"
        )
        return result.to(dtype=signal.dtype)


class ButterworthBandpass:
    """Butterworth IIR bandpass filter applied causally frame-by-frame.

    Suitable for long videos where loading all frames at once is not feasible.
    Uses ``scipy.signal.sosfilt`` under the hood; filter state is maintained
    between :meth:`step` calls so it can be used in a streaming loop.

    Args:
        fps: Frames per second of the video.
        freq_low: Lower cut-off frequency in Hz.
        freq_high: Upper cut-off frequency in Hz.
        order: Filter order (default 1 — matches reference MATLAB code).
    """

    def __init__(
        self,
        fps: float,
        freq_low: float,
        freq_high: float,
        order: int = 1,
    ) -> None:
        self.fps = fps
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.order = order

        nyq = fps / 2.0
        low = freq_low / nyq
        high = min(freq_high / nyq, 1.0 - 1e-6)
        self._sos = butter(order, [low, high], btype="bandpass", output="sos")
        self._zi: np.ndarray | None = None
        logger.debug(
            f"ButterworthBandpass order={order}: {freq_low}–{freq_high} Hz @ {fps} fps"
        )

    # ------------------------------------------------------------------
    # Batch mode (all frames at once)
    # ------------------------------------------------------------------

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Filter *signal* along its first (time) dimension (batch mode).

        Args:
            signal: ``(T, ...)`` tensor.

        Returns:
            Filtered tensor with the same shape.
        """
        original_shape = signal.shape
        T = signal.shape[0]
        flat = signal.float().cpu().numpy().reshape(T, -1)  # (T, N)

        # sosfilt expects (N, T) → transpose
        flat_t = flat.T  # (N, T)
        filtered_t = sosfilt(self._sos, flat_t, axis=-1)
        filtered = filtered_t.T  # (T, N)

        result = torch.from_numpy(filtered.reshape(original_shape)).to(
            device=signal.device, dtype=signal.dtype
        )
        logger.debug(f"ButterworthBandpass: filtered {tuple(signal.shape)}")
        return result

    # ------------------------------------------------------------------
    # Streaming mode (one frame at a time)
    # ------------------------------------------------------------------

    def reset(self, signal_shape: tuple[int, ...]) -> None:
        """Initialise filter state for streaming on a signal of *signal_shape*.

        Call once before the first :meth:`step` call.
        """
        n_signals = int(np.prod(signal_shape))
        zi_base = np.zeros((self._sos.shape[0], 2))  # (n_sections, 2)
        self._zi = np.stack([zi_base] * n_signals, axis=-1)  # (sections, 2, N)
        logger.debug(f"ButterworthBandpass: reset state for {n_signals} signals")

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        """Filter a single frame, updating internal state.

        Args:
            frame: ``(...)`` tensor (one frame of the signal, no time dim).

        Returns:
            Filtered frame with the same shape.
        """
        if self._zi is None:
            self.reset(frame.shape)

        original_shape = frame.shape
        flat = frame.float().cpu().numpy().flatten()  # (N,)
        # sosfilt with zi: input shape (N,) treated as (N, 1)
        filtered_flat, self._zi = sosfilt(
            self._sos, flat[np.newaxis, :], axis=0, zi=self._zi
        )
        result = torch.from_numpy(filtered_flat[0].reshape(original_shape)).to(
            device=frame.device, dtype=frame.dtype
        )
        return result
