"""Algorithm-correctness tests for phase-based EVM (Wadhwa et al. 2013).

These tests verify mathematical properties, not just I/O shapes.

1.  Amplitude-weighted spatial smoothing (Eq. 17)
    - Low-amplitude (noisy-phase) pixels must not corrupt adjacent
      high-amplitude pixels.
    - When amplitude is uniform the weighted blur equals a plain Gaussian.
    - Both 2-D (H,W) and batched (T,H,W) inputs are handled.

2.  Static video is unchanged
    - A video where every frame is identical must produce output ≈ input
      (Δφ = 0 → filtered phase = 0 → no amplification).

3.  In-band sinusoidal motion is amplified
    - A cosine fringe oscillating at a frequency inside the passband should
      show increased temporal variation after magnification.

4.  Out-of-band motion is NOT amplified
    - Motion whose temporal frequency is outside the passband should be
      rejected; output ≈ input.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from pyevm.magnification.phase import PhaseMagnifier


FPS = 30.0


# ---------------------------------------------------------------------------
# Helper: build a synthetic horizontal cosine fringe video
# ---------------------------------------------------------------------------

def _fringe_video(
    T: int,
    H: int,
    W: int,
    k_cycles: float,
    motion_amp: float,
    motion_freq: float,
    fps: float = FPS,
) -> torch.Tensor:
    """(T, 3, H, W) float32 cosine fringe that oscillates horizontally.

    All three colour channels are identical (grayscale fringe in [0, 1]).
    """
    k = 2 * math.pi * k_cycles / W
    t = torch.arange(T, dtype=torch.float32) / fps
    disp = motion_amp * torch.sin(2 * math.pi * motion_freq * t)  # (T,)
    x = torch.arange(W, dtype=torch.float32)
    frames = [
        (0.5 + 0.5 * torch.cos(k * (x - disp[i])))
        .unsqueeze(0).expand(3, H, W)
        for i in range(T)
    ]
    return torch.stack(frames)  # (T, 3, H, W)


# ---------------------------------------------------------------------------
# 1.  Amplitude-weighted spatial smoothing
# ---------------------------------------------------------------------------

class TestAmplitudeWeightedSmoothing:
    """_smooth_phase must implement Eq. 17 of Wadhwa et al. 2013."""

    def test_noisy_low_amplitude_region_does_not_corrupt_neighbours(self):
        """Phase noise in a low-amplitude half must not bleed into the clean half.

        Setup
        -----
        Left half:  amplitude = 1  (high), phase = 1.0 rad (clean constant)
        Right half: amplitude = 1e-6 (≈ 0), phase = random noise ∈ [−π, π]

        After amplitude-weighted smoothing the interior of the left half
        should remain ≈ 1.0 rad; the noisy right half carries negligible weight.
        """
        torch.manual_seed(0)
        H, W = 64, 64
        sigma = 2.0
        clean_val = 1.0
        margin = int(3 * sigma) + 1   # stay away from the left/right boundary

        amplitude = torch.zeros(H, W)
        amplitude[:, : W // 2] = 1.0
        amplitude[:, W // 2 :] = 1e-6

        phase = torch.zeros(H, W)
        phase[:, : W // 2] = clean_val
        phase[:, W // 2 :] = (torch.rand(H, W // 2) * 2 - 1) * math.pi

        mag = PhaseMagnifier(n_scales=2, n_orientations=4, sigma=sigma)
        smoothed = mag._smooth_phase(phase, amplitude)

        interior = smoothed[:, margin : W // 2 - margin]
        assert interior.numel() > 0, (
            "Interior region is empty — increase W or reduce sigma"
        )
        max_err = (interior - clean_val).abs().max().item()
        assert max_err < 0.05, (
            f"High-amplitude region corrupted by noisy neighbour: "
            f"max error = {max_err:.4f} (should be < 0.05)"
        )

    def test_plain_blur_would_fail_same_case(self):
        """Sanity check: unweighted Gaussian blur IS corrupted by the noisy half.

        This verifies that the test scenario is strong enough to distinguish
        the weighted and unweighted cases.
        """
        torch.manual_seed(0)
        H, W = 64, 64
        sigma = 2.0
        clean_val = 1.0
        margin = int(3 * sigma) + 1

        amplitude = torch.zeros(H, W)
        amplitude[:, : W // 2] = 1.0
        amplitude[:, W // 2 :] = 1e-6

        phase = torch.zeros(H, W)
        phase[:, : W // 2] = clean_val
        phase[:, W // 2 :] = (torch.rand(H, W // 2) * 2 - 1) * math.pi

        # Unweighted Gaussian blur (what the old code did)
        radius = int(3 * sigma)
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=torch.float32) - radius
        g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g1d /= g1d.sum()
        kernel = torch.outer(g1d, g1d).unsqueeze(0).unsqueeze(0)
        plain = F.conv2d(phase.unsqueeze(0).unsqueeze(0), kernel, padding=radius).squeeze()

        interior_plain = plain[:, margin : W // 2 - margin]
        max_err_plain = (interior_plain - clean_val).abs().max().item()

        assert max_err_plain > 0.05, (
            "Expected unweighted smoothing to show contamination near the "
            f"boundary, but max error was only {max_err_plain:.4f}. "
            "The test scenario may be too weak."
        )

    def test_uniform_amplitude_equals_plain_gaussian(self):
        """With uniform amplitude the weighted blur must equal a plain Gaussian
        in the image interior (where the full kernel fits within bounds).

        Near the edges, the weighted version normalises by the partial kernel
        sum that falls within the image (correct boundary handling) while the
        plain blur implicitly zero-pads; these differ by design.  We compare
        only the interior region where both agree.
        """
        torch.manual_seed(1)
        H, W = 32, 32
        sigma = 1.5
        radius = int(3 * sigma)

        phase = torch.randn(H, W)
        amplitude = torch.ones(H, W)

        mag = PhaseMagnifier(n_scales=2, n_orientations=4, sigma=sigma)
        weighted = mag._smooth_phase(phase, amplitude)

        size = 2 * radius + 1
        coords = torch.arange(size, dtype=torch.float32) - radius
        g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g1d /= g1d.sum()
        kernel = torch.outer(g1d, g1d).unsqueeze(0).unsqueeze(0)
        plain = F.conv2d(phase.unsqueeze(0).unsqueeze(0), kernel, padding=radius).squeeze()

        # Compare only interior pixels — full kernel fits, denominator = 1
        w_int = weighted[radius:-radius, radius:-radius]
        p_int = plain[radius:-radius, radius:-radius]
        assert w_int.numel() > 0, "Interior region empty — increase H/W or reduce sigma"
        assert torch.allclose(w_int, p_int, atol=1e-5), (
            "Amplitude-weighted smoothing with uniform amplitude should "
            "reduce to plain Gaussian blur in the interior"
        )

    def test_handles_2d_input(self):
        torch.manual_seed(2)
        H, W = 32, 32
        phase = torch.randn(H, W)
        amplitude = torch.rand(H, W)
        mag = PhaseMagnifier(n_scales=2, n_orientations=4, sigma=2.0)
        out = mag._smooth_phase(phase, amplitude)
        assert out.shape == (H, W)

    def test_handles_3d_batch_input(self):
        torch.manual_seed(3)
        T, H, W = 8, 32, 32
        phase = torch.randn(T, H, W)
        amplitude = torch.rand(T, H, W)
        mag = PhaseMagnifier(n_scales=2, n_orientations=4, sigma=2.0)
        out = mag._smooth_phase(phase, amplitude)
        assert out.shape == (T, H, W)

    def test_sigma_zero_returns_phase_unchanged(self):
        torch.manual_seed(4)
        H, W = 32, 32
        phase = torch.randn(H, W)
        amplitude = torch.rand(H, W)
        mag = PhaseMagnifier(n_scales=2, n_orientations=4, sigma=0.0)
        out = mag._smooth_phase(phase, amplitude)
        assert torch.equal(out, phase)


# ---------------------------------------------------------------------------
# 2.  Static video is unchanged
# ---------------------------------------------------------------------------

class TestStaticVideoUnchanged:
    """Δφ = 0 for a static scene → filtered phase = 0 → output = input."""

    def test_identical_frames_with_smoothing(self):
        torch.manual_seed(5)
        T = 30
        frame = torch.rand(3, 64, 64)
        video = frame.unsqueeze(0).expand(T, -1, -1, -1).clone()

        mag = PhaseMagnifier(
            factor=25.0, freq_low=0.5, freq_high=3.0,
            n_scales=3, n_orientations=4, sigma=3.0,
        )
        out = mag.process(video, fps=FPS)

        max_diff = (out - video.clamp(0, 1)).abs().max().item()
        assert max_diff < 1e-4, (
            f"Static video should be unchanged; max diff = {max_diff:.6f}"
        )

    def test_identical_frames_sigma_zero(self):
        torch.manual_seed(6)
        T = 30
        frame = torch.rand(3, 64, 64)
        video = frame.unsqueeze(0).expand(T, -1, -1, -1).clone()

        mag = PhaseMagnifier(
            factor=25.0, freq_low=0.5, freq_high=3.0,
            n_scales=3, n_orientations=4, sigma=0.0,
        )
        out = mag.process(video, fps=FPS)

        max_diff = (out - video.clamp(0, 1)).abs().max().item()
        assert max_diff < 1e-4, (
            f"Static video (sigma=0) should be unchanged; max diff = {max_diff:.6f}"
        )


# ---------------------------------------------------------------------------
# 3.  In-band motion is amplified
# ---------------------------------------------------------------------------

class TestKnownMotionMagnification:
    """Temporal variation should increase after magnifying in-band motion."""

    def test_in_band_fringe_motion_amplified(self):
        """Cosine fringe oscillating at 1 Hz (inside 0.5–2 Hz band) is amplified.

        We measure the temporal standard deviation across all pixels and verify
        that the output has larger variation than the input.
        """
        T = 60         # 2 s @ 30 fps
        factor = 4.0
        motion_freq = 1.0   # Hz — inside the [0.5, 2.0] Hz passband
        motion_amp = 0.4    # pixels

        video = _fringe_video(T, H=32, W=64, k_cycles=6.0,
                              motion_amp=motion_amp, motion_freq=motion_freq)

        mag = PhaseMagnifier(
            factor=factor, freq_low=0.5, freq_high=2.0,
            n_scales=2, n_orientations=4, sigma=0.0,  # no smoothing for clarity
        )
        out = mag.process(video, fps=FPS)

        in_std  = video.float().std(dim=0).mean().item()
        out_std = out.float().std(dim=0).mean().item()

        assert out_std > in_std, (
            f"In-band motion should be amplified: "
            f"output std {out_std:.4f} should exceed input std {in_std:.4f}"
        )

    def test_out_of_band_fringe_motion_not_amplified(self):
        """Cosine fringe oscillating at 10 Hz (outside 0.5–2 Hz band) is rejected.

        The temporal standard deviation of the output should be close to that
        of the input (the ideal filter passes no 10 Hz content).
        """
        T = 60
        factor = 10.0
        motion_freq = 10.0   # Hz — well outside the [0.5, 2.0] Hz passband
        motion_amp = 0.4

        video = _fringe_video(T, H=32, W=64, k_cycles=6.0,
                              motion_amp=motion_amp, motion_freq=motion_freq)

        mag = PhaseMagnifier(
            factor=factor, freq_low=0.5, freq_high=2.0,
            n_scales=2, n_orientations=4, sigma=0.0,
        )
        out = mag.process(video, fps=FPS)

        in_std  = video.float().std(dim=0).mean().item()
        out_std = out.float().std(dim=0).mean().item()

        # Output variation should be similar to input (not amplified)
        ratio = out_std / (in_std + 1e-8)
        assert ratio < 3.0, (
            f"Out-of-band motion should not be amplified: "
            f"output/input std ratio = {ratio:.2f} (should be < 3)"
        )

    def test_larger_factor_gives_more_amplification(self):
        """Higher factor → more temporal variation in the output."""
        T = 60
        motion_freq = 1.0

        video = _fringe_video(T, H=32, W=64, k_cycles=6.0,
                              motion_amp=0.4, motion_freq=motion_freq)

        def _run(factor: float) -> float:
            mag = PhaseMagnifier(
                factor=factor, freq_low=0.5, freq_high=2.0,
                n_scales=2, n_orientations=4, sigma=0.0,
            )
            return mag.process(video, fps=FPS).std(dim=0).mean().item()

        std_low  = _run(1.0)
        std_high = _run(8.0)

        assert std_high > std_low, (
            f"factor=8 output std {std_high:.4f} should exceed "
            f"factor=1 output std {std_low:.4f}"
        )
