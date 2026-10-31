# Installation

## pip

```bash
pip install pyeulervid
```

The Streamlit dashboard is an optional extra:

```bash
pip install "pyeulervid[app]"
```

Python 3.11 or later is recommended. On Python 3.10, some dependencies stay on
versions with known vulnerabilities because their fixes require 3.11: `filelock`
(used by PyTorch) and, with the `app` extra, Pillow.

## conda / pixi (conda-forge)

A conda-forge package is planned but not available yet. It will use the same name
as on PyPI, `pyeulervid`.

## GPU-accelerated video I/O

Faster video decoding uses `torchcodec`. It has no Windows pip wheels, so it is not
declared as a pip dependency. pip users on Linux/macOS can add it manually:

```bash
pip install pyeulervid torchcodec
```

The planned conda-forge package will include torchcodec on Windows, Linux, and
macOS. Until then, the
[pixi development environment](https://github.com/roaldarbol/pyevm/blob/main/CONTRIBUTING.md#development-setup)
provides it.

Without torchcodec, pyevm automatically falls back to the OpenCV decoder.

## Hardware

pyevm automatically selects the best available compute device:

1. **CUDA** — NVIDIA GPU
2. **MPS** — Apple Silicon GPU
3. **CPU** — fallback

Override with `--device cuda`, `--device mps`, or `--device cpu`.

## Requirements

- Python 3.10 or later
- PyTorch 2.2 or later
