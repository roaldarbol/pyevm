# Installation

## pip

```bash
pip install pyeulervid
```

## conda / pixi (conda-forge)

```bash
# conda
conda install -c conda-forge pyevm

# pixi
pixi add pyevm
```

## GPU-accelerated video I/O

Faster video decoding uses `torchcodec`, which ships with the **conda-forge**
package on Windows, Linux, and macOS — so `conda`/`pixi` installs include it
automatically:

```bash
conda install -c conda-forge pyevm   # torchcodec included
pixi add pyevm
```

torchcodec has no Windows pip wheels, so it is not declared as a pip dependency.
pip users on Linux/macOS can add it manually:

```bash
pip install pyeulervid torchcodec
```

Without torchcodec, pyevm automatically falls back to the OpenCV decoder.

## Hardware

pyevm automatically selects the best available compute device:

1. **CUDA** — NVIDIA GPU
2. **MPS** — Apple Silicon GPU
3. **CPU** — fallback

Override with `--device cuda`, `--device mps`, or `--device cpu`.

## Requirements

- Python 3.12 or later
- PyTorch 2.2 or later
