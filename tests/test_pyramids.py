"""Tests for Gaussian, Laplacian, and steerable pyramid implementations."""

import torch

from pyevm.pyramids.gaussian import GaussianPyramid
from pyevm.pyramids.laplacian import LaplacianPyramid
from pyevm.pyramids.steerable import SteerablePyramid

# ---------------------------------------------------------------------------
# Gaussian pyramid
# ---------------------------------------------------------------------------


class TestGaussianPyramid:
    def test_level_count(self, small_frame):
        pyr = GaussianPyramid(n_levels=4)
        levels = pyr.build(small_frame)
        assert len(levels) == 4

    def test_spatial_halving(self, small_frame):
        pyr = GaussianPyramid(n_levels=4)
        levels = pyr.build(small_frame)
        _, _, H0, W0 = levels[0].shape
        _, _, H1, W1 = levels[1].shape
        assert H1 == H0 // 2
        assert W1 == W0 // 2

    def test_output_dtype(self, small_frame):
        pyr = GaussianPyramid(n_levels=3)
        levels = pyr.build(small_frame)
        for lev in levels:
            assert lev.dtype == torch.float32

    def test_collapse_shape(self, small_frame):
        pyr = GaussianPyramid(n_levels=4)
        levels = pyr.build(small_frame)
        result = pyr.collapse(levels)
        assert (
            result.shape == small_frame.unsqueeze(0).shape
            or result.shape[2:] == small_frame.shape[2:]
        )

    def test_accepts_3d_input(self, cpu_device):
        pyr = GaussianPyramid(n_levels=3)
        frame_3d = torch.rand(3, 64, 64, device=cpu_device)
        levels = pyr.build(frame_3d)
        assert len(levels) == 3

    def test_values_bounded(self, small_frame):
        """Gaussian blurring should not produce values outside [0, 1] for [0, 1] input."""
        pyr = GaussianPyramid(n_levels=4)
        levels = pyr.build(small_frame)
        for lev in levels:
            assert lev.min().item() >= -1e-4
            assert lev.max().item() <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# Laplacian pyramid
# ---------------------------------------------------------------------------


class TestLaplacianPyramid:
    def test_level_count(self, small_frame):
        pyr = LaplacianPyramid(n_levels=4)
        levels = pyr.build(small_frame)
        assert len(levels) == 4

    def test_reconstruction_fidelity(self, small_frame):
        """Collapsing the unmodified Laplacian pyramid should recover the original."""
        pyr = LaplacianPyramid(n_levels=4)
        levels = pyr.build(small_frame)
        reconstructed = pyr.collapse(levels)
        original = small_frame.unsqueeze(0) if small_frame.dim() == 3 else small_frame
        assert torch.allclose(reconstructed, original, atol=1e-4), (
            f"Max abs diff: {(reconstructed - original).abs().max().item()}"
        )

    def test_detail_levels_near_zero_mean(self, small_frame):
        """Laplacian detail bands should have near-zero mean (band-pass property)."""
        pyr = LaplacianPyramid(n_levels=4)
        levels = pyr.build(small_frame)
        for i, lev in enumerate(levels[:-1]):  # exclude residual
            mean = lev.mean().abs().item()
            assert mean < 0.1, f"Level {i} mean {mean:.4f} is unexpectedly large"

    def test_spatial_halving(self, small_frame):
        pyr = LaplacianPyramid(n_levels=4)
        levels = pyr.build(small_frame)
        _, _, H0, W0 = levels[0].shape
        _, _, H2, W2 = levels[2].shape
        assert H2 == H0 // 4
        assert W2 == W0 // 4

    def test_single_level_reconstruction(self, small_frame):
        """Even with n_levels=1 the pyramid round-trips correctly."""
        pyr = LaplacianPyramid(n_levels=1)
        levels = pyr.build(small_frame)
        reconstructed = pyr.collapse(levels)
        original = small_frame.unsqueeze(0) if small_frame.dim() == 3 else small_frame
        assert torch.allclose(reconstructed, original, atol=1e-4)


# ---------------------------------------------------------------------------
# Steerable pyramid
# ---------------------------------------------------------------------------


class TestSteerablePyramid:
    def test_build_returns_expected_keys(self, luma_frame):
        pyr = SteerablePyramid(n_scales=2, n_orientations=4)
        result = pyr.build(luma_frame)
        assert "highpass" in result
        assert "lowpass" in result
        assert "subbands" in result
        assert "sizes" in result

    def test_subband_count(self, luma_frame):
        n_scales, n_orient = 3, 4
        pyr = SteerablePyramid(n_scales=n_scales, n_orientations=n_orient)
        result = pyr.build(luma_frame)
        assert len(result["subbands"]) == n_scales
        for scale_bands in result["subbands"]:
            assert len(scale_bands) == n_orient

    def test_subbands_are_complex(self, luma_frame):
        pyr = SteerablePyramid(n_scales=2, n_orientations=4)
        result = pyr.build(luma_frame)
        for scale_bands in result["subbands"]:
            for band in scale_bands:
                assert band.is_complex(), "Sub-band coefficients must be complex"

    def test_reconstruction_fidelity(self, luma_frame):
        """Collapsing the unmodified pyramid should approximately recover the original."""
        pyr = SteerablePyramid(n_scales=3, n_orientations=6)
        pyramid = pyr.build(luma_frame)
        reconstructed = pyr.collapse(pyramid)
        assert reconstructed.shape == luma_frame.shape
        # The tight-frame design gives near-perfect reconstruction.
        # Allow a small margin for floating-point and DFT up/downsample approximation.
        max_err = (reconstructed - luma_frame).abs().max().item()
        assert max_err < 0.1, f"Reconstruction error too large: {max_err:.4f}"

    def test_highpass_shape(self, luma_frame):
        pyr = SteerablePyramid(n_scales=2, n_orientations=4)
        result = pyr.build(luma_frame)
        assert result["highpass"].shape == luma_frame.shape

    def test_lowpass_smaller(self, luma_frame):
        n_scales = 3
        pyr = SteerablePyramid(n_scales=n_scales, n_orientations=4)
        result = pyr.build(luma_frame)
        H, W = luma_frame.shape
        lp_h, lp_w = result["lowpass"].shape
        assert lp_h <= H // (2 ** (n_scales - 1))
        assert lp_w <= W // (2 ** (n_scales - 1))

    def test_phase_has_finite_values(self, luma_frame):
        pyr = SteerablePyramid(n_scales=2, n_orientations=4)
        result = pyr.build(luma_frame)
        for scale_bands in result["subbands"]:
            for band in scale_bands:
                phase = torch.angle(band)
                assert torch.isfinite(phase).all(), "Phase contains non-finite values"

    def test_reconstruction_fidelity_odd_height(self, cpu_device):
        """Round-trip must be accurate for inputs whose height becomes odd after downsampling.

        1080p video has height 1080 → 540 → 270 → 135 (odd) at scale 3.
        A wrong ``ph`` in ``_upsample_dft`` (off by 1) when going from 135→270
        creates a one-pixel phase error that manifests as a visible black bar.
        This test catches that regression.
        """
        torch.manual_seed(9)
        # Use height=270 so one downsample gives 135 (odd) and the next upsample is 135→270
        luma = torch.rand(270, 480, device=cpu_device)
        pyr = SteerablePyramid(n_scales=3, n_orientations=4, device=cpu_device)
        pyramid = pyr.build(luma)
        reconstructed = pyr.collapse(pyramid)
        max_err = (reconstructed - luma).abs().max().item()
        assert max_err < 0.05, (
            f"Reconstruction error {max_err:.4f} too large for odd-height input "
            f"(likely off-by-one in _upsample_dft)"
        )
