"""Tests for the streaming (process_stream) pipeline.

All three magnifiers expose a ``process_stream()`` method that buffers frames
into chunks, processes them on the GPU, and yields individual output frames.
These tests verify correctness (shape, range, numerical consistency) and
robustness (chunk boundaries, edge cases) without requiring a real video file.
"""

from __future__ import annotations

import torch
import pytest

from pyevm.magnification.color import ColorMagnifier
from pyevm.magnification.motion import MotionMagnifier
from pyevm.magnification.phase import PhaseMagnifier

FPS = 30.0


def _stream(video: torch.Tensor):
    """Yield individual ``(C, H, W)`` frames from a ``(T, C, H, W)`` tensor."""
    for frame in video:
        yield frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect(gen) -> torch.Tensor:
    """Collect all frames from a generator into a ``(T, C, H, W)`` tensor."""
    return torch.stack(list(gen))


# ---------------------------------------------------------------------------
# ColorMagnifier streaming
# ---------------------------------------------------------------------------

class TestColorMagnifierStream:

    def test_output_shape(self, small_video):
        mag = ColorMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.shape == small_video.shape

    def test_output_range(self, small_video):
        mag = ColorMagnifier(alpha=50.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.min().item() >= 0.0, f"Min {out.min().item()} < 0"
        assert out.max().item() <= 1.0, f"Max {out.max().item()} > 1"

    def test_output_finite(self, small_video):
        mag = ColorMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_output_dtype(self, small_video):
        mag = ColorMagnifier(n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.dtype == small_video.dtype

    def test_chunk_size_one(self, small_video):
        """chunk_size=1: each frame processed individually — must not crash."""
        mag = ColorMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=1))
        assert out.shape == small_video.shape

    def test_chunk_larger_than_video(self, small_video):
        """chunk_size > n_frames: entire video in one chunk — must not crash."""
        mag = ColorMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        T = small_video.shape[0]
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=T * 2))
        assert out.shape == small_video.shape

    def test_matches_batch_butterworth(self, small_video):
        """Stream output must match batch process() when both use Butterworth."""
        kwargs = dict(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3,
                      filter_type="butterworth")
        batch_out = ColorMagnifier(**kwargs).process(small_video, FPS)
        stream_out = _collect(
            ColorMagnifier(**kwargs).process_stream(_stream(small_video), FPS, chunk_size=8)
        )
        assert torch.allclose(batch_out, stream_out, atol=1e-5), (
            f"Max diff: {(batch_out - stream_out).abs().max().item():.2e}"
        )

    def test_chunk_boundary_consistency(self, small_video):
        """Different chunk sizes must give the same output (IIR state is preserved)."""
        kwargs = dict(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out_8  = _collect(ColorMagnifier(**kwargs).process_stream(_stream(small_video), FPS, chunk_size=8))
        out_15 = _collect(ColorMagnifier(**kwargs).process_stream(_stream(small_video), FPS, chunk_size=15))
        assert torch.allclose(out_8, out_15, atol=1e-5), (
            f"Max diff between chunk_size=8 and chunk_size=15: "
            f"{(out_8 - out_15).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# MotionMagnifier streaming
# ---------------------------------------------------------------------------

class TestMotionMagnifierStream:

    def test_output_shape(self, small_video):
        mag = MotionMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.shape == small_video.shape

    def test_output_range(self, small_video):
        mag = MotionMagnifier(alpha=20.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_output_finite(self, small_video):
        mag = MotionMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert torch.isfinite(out).all()

    def test_output_dtype(self, small_video):
        mag = MotionMagnifier(n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.dtype == small_video.dtype

    def test_chunk_size_one(self, small_video):
        mag = MotionMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=1))
        assert out.shape == small_video.shape

    def test_chunk_larger_than_video(self, small_video):
        mag = MotionMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        T = small_video.shape[0]
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=T * 2))
        assert out.shape == small_video.shape

    def test_matches_batch_butterworth(self, small_video):
        """Stream output must match batch process() (both use Butterworth by default)."""
        kwargs = dict(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3,
                      filter_type="butterworth")
        batch_out  = MotionMagnifier(**kwargs).process(small_video, FPS)
        stream_out = _collect(
            MotionMagnifier(**kwargs).process_stream(_stream(small_video), FPS, chunk_size=8)
        )
        assert torch.allclose(batch_out, stream_out, atol=1e-5), (
            f"Max diff: {(batch_out - stream_out).abs().max().item():.2e}"
        )

    def test_chunk_boundary_consistency(self, small_video):
        """Different chunk sizes must give the same output."""
        kwargs = dict(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out_8  = _collect(MotionMagnifier(**kwargs).process_stream(_stream(small_video), FPS, chunk_size=8))
        out_15 = _collect(MotionMagnifier(**kwargs).process_stream(_stream(small_video), FPS, chunk_size=15))
        assert torch.allclose(out_8, out_15, atol=1e-5), (
            f"Max diff: {(out_8 - out_15).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# PhaseMagnifier streaming
# ---------------------------------------------------------------------------

class TestPhaseMagnifierStream:

    def test_output_shape(self, small_video):
        mag = PhaseMagnifier(factor=2.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.shape == small_video.shape

    def test_output_range(self, small_video):
        mag = PhaseMagnifier(factor=5.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_output_finite(self, small_video):
        mag = PhaseMagnifier(factor=2.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert torch.isfinite(out).all()

    def test_output_dtype(self, small_video):
        mag = PhaseMagnifier(n_scales=2, n_orientations=4)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=8))
        assert out.dtype == small_video.dtype

    def test_chunk_size_one(self, small_video):
        mag = PhaseMagnifier(factor=2.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=1))
        assert out.shape == small_video.shape

    def test_chunk_larger_than_video(self, small_video):
        mag = PhaseMagnifier(factor=2.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        T = small_video.shape[0]
        out = _collect(mag.process_stream(_stream(small_video), FPS, chunk_size=T * 2))
        assert out.shape == small_video.shape

    def test_chunk_boundary_consistency(self, small_video):
        """Different chunk sizes must give the same output (IIR state preserved)."""
        kwargs = dict(factor=2.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        out_8  = _collect(PhaseMagnifier(**kwargs).process_stream(_stream(small_video), FPS, chunk_size=8))
        out_15 = _collect(PhaseMagnifier(**kwargs).process_stream(_stream(small_video), FPS, chunk_size=15))
        assert torch.allclose(out_8, out_15, atol=1e-5), (
            f"Max diff: {(out_8 - out_15).abs().max().item():.2e}"
        )
