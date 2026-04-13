# Installation

## pip

```bash
pip install pyeulervid
```

## uv

```bash
uv add pyeulervid
```

## conda / pixi (conda-forge)

```bash
# conda
conda install -c conda-forge pyevm

# pixi
pixi add pyevm
```

## Optional: GPU-accelerated video I/O

`torchcodec` enables faster video decoding via FFmpeg. pip wheels are **Linux-only**; on Windows install via conda/pixi (conda-forge provides Windows CUDA builds):

```bash
# pip / uv (Linux only)
pip install "pyeulervid[fast-io]"
uv add "pyeulervid[fast-io]"

# conda-forge — works on Windows + Linux + macOS
conda install -c conda-forge pyevm  # torchcodec is included automatically
pixi add pyevm
```

## Hardware

pyevm automatically selects the best available compute device:

1. **CUDA** — NVIDIA GPU
2. **MPS** — Apple Silicon GPU
3. **CPU** — fallback

Override with `--device cuda`, `--device mps`, or `--device cpu`.

## Requirements

- Python 3.12 or later
- PyTorch 2.2 or later
