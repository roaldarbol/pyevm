"""Integration tests for magnification algorithms (no video file needed).

All tests use synthetic tensor data so they run offline without any media.
The tests validate:
  - output tensor shape matches input
  - output values are in [0, 1] (clamped)
  - processing is deterministic (same seed → same output)
  - both filter types work for each algorithm
"""

import torch

from pyevm.magnification.color import ColorMagnifier
from pyevm.magnification.motion import MotionMagnifier
from pyevm.magnification.phase import PhaseMagnifier

FPS = 30.0


# ---------------------------------------------------------------------------
# ColorMagnifier
# ---------------------------------------------------------------------------


class TestColorMagnifier:
    def test_output_shape(self, small_video):
        mag = ColorMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = mag.process(small_video, FPS)
        assert out.shape == small_video.shape

    def test_output_range(self, small_video):
        mag = ColorMagnifier(alpha=50.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = mag.process(small_video, FPS)
        assert out.min().item() >= 0.0, f"Min value {out.min().item()} < 0"
        assert out.max().item() <= 1.0, f"Max value {out.max().item()} > 1"

    def test_deterministic(self, cpu_device):
        torch.manual_seed(5)
        video = torch.rand(20, 3, 32, 32, device=cpu_device)
        mag = ColorMagnifier(alpha=20.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out1 = mag.process(video.clone(), FPS)
        out2 = mag.process(video.clone(), FPS)
        assert torch.allclose(out1, out2)

    def test_butterworth_filter(self, small_video):
        mag = ColorMagnifier(
            alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3, filter_type="butterworth"
        )
        out = mag.process(small_video, FPS)
        assert out.shape == small_video.shape
        assert out.min().item() >= 0.0 and out.max().item() <= 1.0

    def test_zero_alpha_unchanged(self, small_video):
        """With alpha=0 and chrom_attenuation=0, output should equal input."""
        mag = ColorMagnifier(
            alpha=0.0,
            freq_low=0.5,
            freq_high=3.0,
            n_levels=3,
            chrom_attenuation=0.0,
        )
        out = mag.process(small_video, FPS)
        assert torch.allclose(out, small_video.clamp(0, 1), atol=1e-4), (
            f"Max diff: {(out - small_video).abs().max().item()}"
        )

    def test_output_dtype(self, small_video):
        mag = ColorMagnifier(n_levels=3)
        out = mag.process(small_video, FPS)
        assert out.dtype == small_video.dtype


# ---------------------------------------------------------------------------
# MotionMagnifier
# ---------------------------------------------------------------------------


class TestMotionMagnifier:
    def test_output_shape(self, small_video):
        mag = MotionMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = mag.process(small_video, FPS)
        assert out.shape == small_video.shape

    def test_output_range(self, small_video):
        mag = MotionMagnifier(alpha=20.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out = mag.process(small_video, FPS)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_deterministic(self, cpu_device):
        torch.manual_seed(3)
        video = torch.rand(20, 3, 32, 32, device=cpu_device)
        mag = MotionMagnifier(alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3)
        out1 = mag.process(video.clone(), FPS)
        out2 = mag.process(video.clone(), FPS)
        assert torch.allclose(out1, out2)

    def test_ideal_filter(self, small_video):
        mag = MotionMagnifier(
            alpha=10.0, freq_low=0.5, freq_high=3.0, n_levels=3, filter_type="ideal"
        )
        out = mag.process(small_video, FPS)
        assert out.shape == small_video.shape

    def test_adaptive_alpha_clamps(self):
        """Fine levels are suppressed; coarser levels receive increasing amplification."""
        mag = MotionMagnifier(alpha=20.0, lambda_c=16.0, n_levels=6)
        # Finest level (lambda=2 px): alpha_max = 8*2/16 - 1 = 0 → clamped to 0
        assert mag._alpha_for_level(0) == 0.0
        # Mid level (lambda=16 px): alpha_max = 8*16/16 - 1 = 7 → alpha_eff = 7
        assert mag._alpha_for_level(3) == 7.0
        # Coarsest level (lambda=64 px): alpha_max = 31, clamped to alpha=20
        assert mag._alpha_for_level(5) == 20.0

    def test_output_dtype(self, small_video):
        mag = MotionMagnifier(n_levels=3)
        out = mag.process(small_video, FPS)
        assert out.dtype == small_video.dtype


# ---------------------------------------------------------------------------
# PhaseMagnifier
# ---------------------------------------------------------------------------


class TestPhaseMagnifier:
    def test_output_shape(self, small_video):
        mag = PhaseMagnifier(factor=2.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        out = mag.process(small_video, FPS)
        assert out.shape == small_video.shape

    def test_output_range(self, small_video):
        mag = PhaseMagnifier(factor=5.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        out = mag.process(small_video, FPS)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_output_finite(self, small_video):
        mag = PhaseMagnifier(factor=2.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4)
        out = mag.process(small_video, FPS)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_butterworth_filter(self, small_video):
        mag = PhaseMagnifier(
            factor=2.0,
            freq_low=0.5,
            freq_high=3.0,
            n_scales=2,
            n_orientations=4,
            filter_type="butterworth",
        )
        out = mag.process(small_video, FPS)
        assert out.shape == small_video.shape

    def test_sigma_zero_no_smoothing(self, small_video):
        """sigma=0 should disable spatial smoothing without errors."""
        mag = PhaseMagnifier(
            factor=2.0, freq_low=0.5, freq_high=3.0, n_scales=2, n_orientations=4, sigma=0.0
        )
        out = mag.process(small_video, FPS)
        assert out.shape == small_video.shape

    def test_output_dtype(self, small_video):
        mag = PhaseMagnifier(n_scales=2, n_orientations=4)
        out = mag.process(small_video, FPS)
        assert out.dtype == small_video.dtype
