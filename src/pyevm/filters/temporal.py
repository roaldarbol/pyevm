"""Temporal bandpass filters for video magnification.

Two strategies are provided:

IdealBandpass
    FFT-based brick-wall filter.  Requires all frames in memory at once but
    gives perfectly flat pass-band.  Best for colour/phase EVM.

ButterworthBandpass
    IIR filter applied causally via scipy (CPU) or a torch.jit.script loop
    (GPU).  The GPU path eliminates the PCIe roundtrip that dominates runtime
    for large sub-bands — e.g. a 64-frame 1080 p sub-band is ~1.6 GB; moving
    that to CPU and back costs ~100 ms on PCIe 5, while the GPU loop takes
    ~2–5 ms.  O(1) memory — suitable for long videos and streaming.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from loguru import logger
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

# ---------------------------------------------------------------------------
# GPU IIR filter — compiled once at import time
# ---------------------------------------------------------------------------
# Direct Form II transposed, one second-order section (SOS).
# State variables s1/s2 are (N,) GPU tensors that persist across chunks.

_GPU_IIR_AVAILABLE: bool = False
_sos_step_gpu: Callable | None = None


def _try_compile_gpu_filter() -> tuple[Callable | None, bool]:
    """Attempt to JIT-compile the GPU IIR filter kernel."""
    try:

        @torch.jit.script
        def _fn(
            x: torch.Tensor,  # (T, N) float32
            b0: float,
            b1: float,
            b2: float,
            a1: float,
            a2: float,
            s1: torch.Tensor,  # (N,) float32 — in/out state
            s2: torch.Tensor,  # (N,) float32 — in/out state
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """One SOS section, Direct Form II transposed — runs on any device."""
            T = x.shape[0]
            y = torch.empty_like(x)
            for t in range(T):
                xt = x[t]
                yt = b0 * xt + s1
                y[t] = yt
                new_s1 = b1 * xt - a1 * yt + s2
                s2 = b2 * xt - a2 * yt
                s1 = new_s1
            return y, s1, s2

        logger.debug("GPU IIR filter compiled via torch.jit.script")
        return _fn, True
    except Exception as exc:
        logger.debug(f"GPU IIR filter unavailable ({exc}); falling back to scipy")
        return None, False


_sos_step_gpu, _GPU_IIR_AVAILABLE = _try_compile_gpu_filter()


# ---------------------------------------------------------------------------
# Ideal (FFT) bandpass
# ---------------------------------------------------------------------------


class IdealBandpass:
    """FFT-based ideal bandpass filter over the time axis, with optional notch stops.

    Args:
        fps: Frames per second of the video.
        freq_low: Lower cut-off frequency in Hz.
        freq_high: Upper cut-off frequency in Hz.
        notch_freqs: Frequencies to notch out (Hz).  Each notch zeros a
            symmetric window of width ``notch_width`` centred on the frequency.
        notch_width: Width of each notch in Hz (default 1.0).
    """

    def __init__(
        self,
        fps: float,
        freq_low: float,
        freq_high: float,
        notch_freqs: list[float] | None = None,
        notch_width: float = 1.0,
    ) -> None:
        self.fps = fps
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.notch_freqs = notch_freqs or []
        self.notch_width = notch_width
        logger.debug(
            f"IdealBandpass: {freq_low}–{freq_high} Hz @ {fps} fps"
            + (f", notches={self.notch_freqs} ±{notch_width / 2} Hz" if self.notch_freqs else "")
        )

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Filter *signal* along its first (time) dimension.

        Args:
            signal: ``(T, ...)`` tensor where ``T`` is the number of frames.

        Returns:
            Bandpass-filtered (and notch-filtered) tensor with the same shape.
        """
        T = signal.shape[0]
        nyq = self.fps / 2.0
        if self.freq_low >= nyq:
            raise ValueError(
                f"freq_low={self.freq_low} Hz is at or above the Nyquist frequency "
                f"({nyq} Hz for fps={self.fps}). Use a lower frequency or a higher-fps video."
            )
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps, device=signal.device)

        # FFT along time axis (dim 0)
        spectrum = torch.fft.rfft(signal.float(), dim=0)

        # Build combined mask: bandpass × notch stops
        mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
        half = self.notch_width / 2.0
        for nf in self.notch_freqs:
            mask = mask & ~((freqs >= nf - half) & (freqs <= nf + half))
        mask_f = mask.float()
        for _ in range(signal.dim() - 1):
            mask_f = mask_f.unsqueeze(-1)

        filtered_spectrum = spectrum * mask_f
        result = torch.fft.irfft(filtered_spectrum, n=T, dim=0)
        logger.debug(
            f"IdealBandpass: filtered signal shape {tuple(signal.shape)} → {tuple(result.shape)}"
        )
        return result.to(dtype=signal.dtype)


# ---------------------------------------------------------------------------
# Butterworth IIR bandpass
# ---------------------------------------------------------------------------


class ButterworthBandpass:
    """Butterworth IIR bandpass filter applied causally chunk-by-chunk, with
    optional IIR notch stops cascaded after the bandpass.

    On CUDA/MPS devices the filter runs entirely on the accelerator
    (``torch.jit.script`` loop, no CPU↔device roundtrip).  On CPU the
    original ``scipy.signal.sosfilt`` path is used.

    Filter state is maintained between :meth:`apply_chunk` calls so the
    result is numerically identical to processing the whole video at once.

    Args:
        fps: Frames per second of the video.
        freq_low: Lower cut-off frequency in Hz.
        freq_high: Upper cut-off frequency in Hz.
        order: Filter order (default 1 — matches reference MATLAB code).
        notch_freqs: Frequencies to notch out (Hz).  Each notch is a
            2nd-order IIR notch filter with Q = ``freq / notch_width``.
        notch_width: Bandwidth of each notch in Hz (default 1.0).
    """

    def __init__(
        self,
        fps: float,
        freq_low: float,
        freq_high: float,
        order: int = 1,
        notch_freqs: list[float] | None = None,
        notch_width: float = 1.0,
    ) -> None:
        self.fps = fps
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.order = order
        self.notch_freqs = notch_freqs or []
        self.notch_width = notch_width

        nyq = fps / 2.0
        low = freq_low / nyq
        high = min(freq_high / nyq, 1.0 - 1e-6)
        if low >= 1.0:
            raise ValueError(
                f"freq_low={freq_low} Hz is at or above the Nyquist frequency "
                f"({nyq} Hz for fps={fps}). Use a lower frequency or a higher-fps video."
            )
        if low >= high:
            raise ValueError(
                f"freq_high={freq_high} Hz is too close to or above the Nyquist frequency "
                f"({nyq} Hz for fps={fps}). The passband [{freq_low}, {freq_high}] Hz is "
                f"not representable. Use a lower freq_high or a higher-fps video."
            )
        self._sos = butter(order, [low, high], btype="bandpass", output="sos")

        # Cascade 2nd-order IIR notch sections
        for nf in self.notch_freqs:
            w0 = nf / nyq
            if w0 >= 1.0:
                raise ValueError(
                    f"Notch frequency {nf} Hz is at or above the Nyquist frequency "
                    f"({nyq} Hz for fps={fps})."
                )
            Q = max(nf / notch_width, 0.5)  # quality factor; clamp to avoid degenerate filter
            b_n, a_n = iirnotch(w0, Q)
            self._sos = np.vstack([self._sos, tf2sos(b_n, a_n)])

        # CPU state (scipy)
        self._zi: np.ndarray | None = None
        # GPU state: one (s1, s2) pair per SOS section
        self._zi_gpu: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        logger.debug(
            f"ButterworthBandpass order={order}: {freq_low}–{freq_high} Hz @ {fps} fps"
            + (f", notches={self.notch_freqs} ±{notch_width / 2} Hz" if self.notch_freqs else "")
        )

    # ------------------------------------------------------------------
    # Batch mode (all frames at once — no persistent state)
    # ------------------------------------------------------------------

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Filter *signal* along its first (time) dimension (batch mode).

        Args:
            signal: ``(T, ...)`` tensor.

        Returns:
            Filtered tensor with the same shape.
        """
        if _GPU_IIR_AVAILABLE and signal.device.type != "cpu":
            return self._apply_gpu(signal)
        return self._apply_cpu(signal)

    def _apply_cpu(self, signal: torch.Tensor) -> torch.Tensor:
        original_shape = signal.shape
        T = signal.shape[0]
        flat = np.from_dlpack(signal.detach().float().cpu().contiguous()).reshape(T, -1)  # (T, N)

        flat_t = np.ascontiguousarray(flat.T)  # (N, T)
        filtered_t = sosfilt(self._sos, flat_t, axis=-1)
        filtered_c = np.ascontiguousarray(filtered_t.T, dtype=np.float32)  # (T, N)
        result = (
            torch.from_dlpack(filtered_c)
            .reshape(original_shape)
            .to(device=signal.device, dtype=signal.dtype)
        )
        logger.debug(f"ButterworthBandpass (CPU batch): filtered {tuple(signal.shape)}")
        return result

    def _apply_gpu(self, signal: torch.Tensor) -> torch.Tensor:
        """GPU batch mode: run the JIT loop from zero initial state."""
        assert _sos_step_gpu is not None
        original_shape = signal.shape
        T = signal.shape[0]
        x = signal.float().reshape(T, -1)  # (T, N)
        N = x.shape[1]

        y = x
        for sec in range(self._sos.shape[0]):
            b0, b1, b2 = (
                float(self._sos[sec, 0]),
                float(self._sos[sec, 1]),
                float(self._sos[sec, 2]),
            )
            a1, a2 = float(self._sos[sec, 4]), float(self._sos[sec, 5])
            s1 = torch.zeros(N, device=signal.device, dtype=torch.float32)
            s2 = torch.zeros(N, device=signal.device, dtype=torch.float32)
            y, s1, s2 = _sos_step_gpu(y, b0, b1, b2, a1, a2, s1, s2)

        logger.debug(f"ButterworthBandpass (GPU batch): filtered {tuple(signal.shape)}")
        return y.reshape(original_shape).to(dtype=signal.dtype)

    # ------------------------------------------------------------------
    # Streaming / chunk mode (state carries across calls)
    # ------------------------------------------------------------------

    def apply_chunk(self, signal: torch.Tensor) -> torch.Tensor:
        """Filter a chunk of frames along the time dimension, updating state.

        Equivalent to calling :meth:`step` T times in sequence; the IIR state
        is updated so the next call picks up seamlessly.

        Uses the GPU JIT path on CUDA/MPS devices (no PCIe roundtrip).

        Args:
            signal: ``(T, ...)`` tensor where ``T`` is the chunk length.

        Returns:
            Filtered tensor with the same shape.
        """
        if _GPU_IIR_AVAILABLE and signal.device.type != "cpu":
            return self._apply_chunk_gpu(signal)
        return self._apply_chunk_cpu(signal)

    def _apply_chunk_cpu(self, signal: torch.Tensor) -> torch.Tensor:
        original_shape = signal.shape
        T = signal.shape[0]
        if self._zi is None:
            self.reset(signal.shape[1:])

        flat = np.from_dlpack(signal.detach().float().cpu().contiguous()).reshape(T, -1)  # (T, P)

        filtered, self._zi = sosfilt(self._sos, flat, axis=0, zi=self._zi)
        out_c = np.ascontiguousarray(filtered, dtype=np.float32)
        return (
            torch.from_dlpack(out_c)
            .reshape(original_shape)
            .to(device=signal.device, dtype=signal.dtype)
        )

    def _apply_chunk_gpu(self, signal: torch.Tensor) -> torch.Tensor:
        """GPU streaming mode: JIT loop with persistent state across chunks."""
        assert _sos_step_gpu is not None
        original_shape = signal.shape
        T = signal.shape[0]
        x = signal.float().reshape(T, -1)  # (T, N)
        N = x.shape[1]

        # Lazy-init GPU state (one (s1, s2) pair per SOS section)
        if self._zi_gpu is None:
            self._zi_gpu = [
                (
                    torch.zeros(N, device=signal.device, dtype=torch.float32),
                    torch.zeros(N, device=signal.device, dtype=torch.float32),
                )
                for _ in range(self._sos.shape[0])
            ]

        y = x
        new_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        for sec in range(self._sos.shape[0]):
            b0, b1, b2 = (
                float(self._sos[sec, 0]),
                float(self._sos[sec, 1]),
                float(self._sos[sec, 2]),
            )
            a1, a2 = float(self._sos[sec, 4]), float(self._sos[sec, 5])
            s1, s2 = self._zi_gpu[sec]
            y, s1, s2 = _sos_step_gpu(y, b0, b1, b2, a1, a2, s1, s2)
            new_states.append((s1, s2))

        self._zi_gpu = new_states
        return y.reshape(original_shape).to(dtype=signal.dtype)

    # ------------------------------------------------------------------
    # Streaming mode (one frame at a time) — CPU only
    # ------------------------------------------------------------------

    def reset(self, signal_shape: tuple[int, ...]) -> None:
        """Initialise CPU filter state for streaming on a signal of *signal_shape*.

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
        flat = np.from_dlpack(frame.detach().float().cpu().contiguous()).flatten()  # (N,)
        filtered_flat, self._zi = sosfilt(self._sos, flat[np.newaxis, :], axis=0, zi=self._zi)
        out_c = np.ascontiguousarray(filtered_flat[0].reshape(original_shape), dtype=np.float32)
        result = torch.from_dlpack(out_c).to(device=frame.device, dtype=frame.dtype)
        return result
