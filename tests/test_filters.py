"""Tests for temporal bandpass filters."""

import math
import torch
import pytest
from pyevm.filters.temporal import IdealBandpass, ButterworthBandpass


FPS = 30.0
T = 300  # frames


def _make_signal(freq_hz: float, shape: tuple[int, ...] = (T,)) -> torch.Tensor:
    """Pure sinusoid at *freq_hz* with *T* samples at *FPS*."""
    t = torch.linspace(0, (T - 1) / FPS, T)
    sig = torch.sin(2 * math.pi * freq_hz * t)
    # Broadcast into arbitrary spatial shape
    for _ in range(len(shape) - 1):
        sig = sig.unsqueeze(-1)
    return sig.expand(shape)


# ---------------------------------------------------------------------------
# IdealBandpass
# ---------------------------------------------------------------------------

class TestIdealBandpass:
    def test_passband_preserved(self):
        """Signal in the pass band should pass through nearly unchanged."""
        filt = IdealBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        sig = _make_signal(1.0)
        out = filt.apply(sig)
        # Correlation: out and sig should be highly correlated
        corr = (out * sig).sum() / (out.norm() * sig.norm() + 1e-9)
        assert corr.item() > 0.9, f"Passband correlation too low: {corr.item():.3f}"

    def test_stopband_suppressed(self):
        """Signal outside the pass band should be strongly attenuated."""
        filt = IdealBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        sig = _make_signal(10.0)  # well outside band
        out = filt.apply(sig)
        energy_ratio = out.norm() / (sig.norm() + 1e-9)
        assert energy_ratio.item() < 0.1, f"Stopband attenuation insufficient: {energy_ratio.item():.3f}"

    def test_output_shape_preserved(self):
        filt = IdealBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        sig = torch.rand(T, 3, 32, 32)
        out = filt.apply(sig)
        assert out.shape == sig.shape

    def test_output_dtype_preserved(self):
        filt = IdealBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        sig = torch.rand(T, 4, 8, 8, dtype=torch.float32)
        out = filt.apply(sig)
        assert out.dtype == sig.dtype

    def test_zero_signal(self):
        filt = IdealBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        sig = torch.zeros(T, 3, 16, 16)
        out = filt.apply(sig)
        assert torch.allclose(out, sig, atol=1e-6)

    def test_dc_rejected(self):
        """DC (0 Hz) should be rejected by the bandpass filter."""
        filt = IdealBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        sig = torch.ones(T)
        out = filt.apply(sig)
        assert out.abs().max().item() < 0.01


# ---------------------------------------------------------------------------
# ButterworthBandpass
# ---------------------------------------------------------------------------

class TestButterworthBandpass:
    def test_output_shape_preserved(self):
        filt = ButterworthBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        sig = torch.rand(T, 3, 32, 32)
        out = filt.apply(sig)
        assert out.shape == sig.shape

    def test_passband_preserved(self):
        filt = ButterworthBandpass(fps=FPS, freq_low=0.5, freq_high=2.0, order=4)
        sig = _make_signal(1.0)
        out = filt.apply(sig)
        # Use the second half to avoid transient
        half = T // 2
        corr = (out[half:] * sig[half:]).sum() / (out[half:].norm() * sig[half:].norm() + 1e-9)
        assert corr.item() > 0.85

    def test_stopband_suppressed(self):
        filt = ButterworthBandpass(fps=FPS, freq_low=0.5, freq_high=2.0, order=4)
        sig = _make_signal(12.0)
        out = filt.apply(sig)
        half = T // 2
        energy_ratio = out[half:].norm() / (sig[half:].norm() + 1e-9)
        assert energy_ratio.item() < 0.2

    def test_streaming_matches_batch(self):
        """Frame-by-frame streaming should approximate batch filtering."""
        filt_batch = ButterworthBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        filt_stream = ButterworthBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)

        sig_1d = torch.rand(T, 4)  # (T, N)
        batch_out = filt_batch.apply(sig_1d)

        filt_stream.reset((4,))
        stream_frames = []
        for t in range(T):
            out_t = filt_stream.step(sig_1d[t])
            stream_frames.append(out_t)
        stream_out = torch.stack(stream_frames, dim=0)

        # Streaming IIR == batch IIR — should be essentially identical
        assert torch.allclose(batch_out, stream_out, atol=1e-5), \
            f"Max diff: {(batch_out - stream_out).abs().max().item()}"

    def test_reset_clears_state(self):
        filt = ButterworthBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        filt.reset((8, 8))
        assert filt._zi is not None

    def test_output_finite(self):
        filt = ButterworthBandpass(fps=FPS, freq_low=0.5, freq_high=2.0)
        sig = torch.rand(T, 3, 8, 8)
        out = filt.apply(sig)
        assert torch.isfinite(out).all()
