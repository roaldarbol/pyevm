# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-13

### Added

- `ColorMagnifier`: Eulerian colour magnification (Wu et al. 2012) with Gaussian pyramid, ideal and Butterworth bandpass filters, streaming support.
- `MotionMagnifier`: Eulerian motion magnification (Wu et al. 2012) with Laplacian pyramid.
- `PhaseMagnifier`: Phase-based motion magnification (Wadhwa et al. 2013) with analytic steerable pyramid, circular phase wrapping, large-motion attenuation (Fig. 11).
- Notch filter support in all three magnifiers (ideal FFT mask + IIR cascade) for removing light flicker (50/60 Hz) or other periodic artefacts.
- Nyquist validation with clear error messages when requested frequencies exceed `fps / 2`.
- GPU-accelerated video I/O via `torchcodec` (optional `fast-io` extra; Linux/Windows via pixi).
- CLI (`pyevm color / motion / phase / info / dashboard`) powered by Typer.
- Interactive Streamlit dashboard (`pyevm dashboard`).
- Automatic device selection: CUDA → MPS → CPU, overridable with `--device`.
- PyPI distribution name `pyeulervid`; import as `pyevm`; CLI as `pyevm`.

[Unreleased]: https://github.com/roaldarbol/pyevm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/roaldarbol/pyevm/releases/tag/v0.1.0
