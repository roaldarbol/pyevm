"""Command-line interface for the EVM package.

Usage examples
--------------
.. code-block:: bash

    # Colour magnification (pulse detection)
    evm color input.mp4 output.mp4 --alpha 50 --freq-low 0.4 --freq-high 3.0

    # Motion magnification
    evm motion input.mp4 output.mp4 --alpha 20 --freq-low 0.4 --freq-high 3.0

    # Phase-based magnification
    evm phase input.mp4 output.mp4 --factor 10 --freq-low 0.4 --freq-high 3.0

    # Inspect device (shows which GPU/CPU will be used)
    evm info

Add ``--debug`` to any command for verbose logging.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger

app = typer.Typer(
    name="evm",
    help="Eulerian Video Magnification — colour, motion, and phase-based algorithms.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Shared options / helpers
# ---------------------------------------------------------------------------

_DebugOption = Annotated[
    bool,
    typer.Option("--debug", help="Enable verbose debug logging.", is_flag=True),
]

_DeviceOption = Annotated[
    Optional[str],
    typer.Option(
        "--device",
        help="Compute device: 'cuda', 'mps', or 'cpu'. Auto-detected when omitted.",
    ),
]


def _setup_logging(debug: bool) -> None:
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    if debug:
        logger.debug("Debug logging enabled")


def _get_device(device_str: str | None) -> "torch.device":  # noqa: F821
    import torch  # noqa: PLC0415
    from pyevm.device import get_device  # noqa: PLC0415
    if device_str is not None:
        return torch.device(device_str)
    return get_device()


def _load_video(input_path: Path, device: "torch.device", max_frames: int | None):  # noqa: F821
    from pyevm.io.video import VideoReader  # noqa: PLC0415
    reader = VideoReader(input_path, device=device, max_frames=max_frames)
    logger.info(f"Loading video: {input_path}")
    frames, fps = reader.read()
    meta = reader.metadata
    logger.info(
        f"  {meta['n_frames']} frames, {fps:.2f} fps, "
        f"{meta['width']}×{meta['height']} px"
    )
    return frames, fps


def _save_video(frames: "torch.Tensor", output_path: Path, fps: float) -> None:  # noqa: F821
    from pyevm.io.video import VideoWriter  # noqa: PLC0415
    writer = VideoWriter(output_path, fps=fps)
    logger.info(f"Saving result → {output_path}")
    writer.write(frames)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def color(
    input: Annotated[Path, typer.Argument(help="Input video file.")],
    output: Annotated[Path, typer.Argument(help="Output video file.")],
    alpha: Annotated[float, typer.Option(help="Luminance amplification factor.")] = 50.0,
    freq_low: Annotated[float, typer.Option("--freq-low", help="Lower bandpass frequency (Hz).")] = 0.4,
    freq_high: Annotated[float, typer.Option("--freq-high", help="Upper bandpass frequency (Hz).")] = 3.0,
    n_levels: Annotated[int, typer.Option(help="Gaussian pyramid levels.")] = 6,
    chrom_attenuation: Annotated[float, typer.Option("--chrom-attenuation", help="Chrominance attenuation (0–1).")] = 0.1,
    pyramid_level: Annotated[Optional[int], typer.Option("--pyramid-level", help="Pyramid level to filter.")] = None,
    filter_type: Annotated[str, typer.Option("--filter", help="'ideal' or 'butterworth'.")] = "ideal",
    max_frames: Annotated[Optional[int], typer.Option("--max-frames", help="Limit frames read.")] = None,
    device: _DeviceOption = None,
    debug: _DebugOption = False,
) -> None:
    """Colour-based EVM — amplifies subtle colour changes (e.g. pulse)."""
    _setup_logging(debug)
    from pyevm.magnification.color import ColorMagnifier  # noqa: PLC0415
    dev = _get_device(device)
    frames, fps = _load_video(input, dev, max_frames)
    magnifier = ColorMagnifier(
        alpha=alpha,
        freq_low=freq_low,
        freq_high=freq_high,
        n_levels=n_levels,
        chrom_attenuation=chrom_attenuation,
        pyramid_level=pyramid_level,
        filter_type=filter_type,
        device=dev,
    )
    result = magnifier.process(frames, fps)
    _save_video(result, output, fps)


@app.command()
def motion(
    input: Annotated[Path, typer.Argument(help="Input video file.")],
    output: Annotated[Path, typer.Argument(help="Output video file.")],
    alpha: Annotated[float, typer.Option(help="Nominal amplification factor.")] = 20.0,
    freq_low: Annotated[float, typer.Option("--freq-low", help="Lower bandpass frequency (Hz).")] = 0.4,
    freq_high: Annotated[float, typer.Option("--freq-high", help="Upper bandpass frequency (Hz).")] = 3.0,
    n_levels: Annotated[int, typer.Option(help="Laplacian pyramid levels.")] = 6,
    lambda_c: Annotated[float, typer.Option("--lambda-c", help="Spatial wavelength cutoff (px).")] = 16.0,
    filter_type: Annotated[str, typer.Option("--filter", help="'butterworth' or 'ideal'.")] = "butterworth",
    max_frames: Annotated[Optional[int], typer.Option("--max-frames", help="Limit frames read.")] = None,
    device: _DeviceOption = None,
    debug: _DebugOption = False,
) -> None:
    """Motion-based EVM — amplifies subtle physical motion (e.g. breathing, vibrations)."""
    _setup_logging(debug)
    from pyevm.magnification.motion import MotionMagnifier  # noqa: PLC0415
    dev = _get_device(device)
    frames, fps = _load_video(input, dev, max_frames)
    magnifier = MotionMagnifier(
        alpha=alpha,
        freq_low=freq_low,
        freq_high=freq_high,
        n_levels=n_levels,
        lambda_c=lambda_c,
        filter_type=filter_type,
        device=dev,
    )
    result = magnifier.process(frames, fps)
    _save_video(result, output, fps)


@app.command()
def phase(
    input: Annotated[Path, typer.Argument(help="Input video file.")],
    output: Annotated[Path, typer.Argument(help="Output video file.")],
    factor: Annotated[float, typer.Option(help="Phase amplification factor.")] = 10.0,
    freq_low: Annotated[float, typer.Option("--freq-low", help="Lower bandpass frequency (Hz).")] = 0.4,
    freq_high: Annotated[float, typer.Option("--freq-high", help="Upper bandpass frequency (Hz).")] = 3.0,
    n_scales: Annotated[int, typer.Option(help="Pyramid scales.")] = 4,
    n_orientations: Annotated[int, typer.Option(help="Orientation bands per scale.")] = 6,
    sigma: Annotated[float, typer.Option(help="Spatial phase smoothing (0 = off).")] = 3.0,
    filter_type: Annotated[str, typer.Option("--filter", help="'ideal' or 'butterworth'.")] = "ideal",
    max_frames: Annotated[Optional[int], typer.Option("--max-frames", help="Limit frames read.")] = None,
    device: _DeviceOption = None,
    debug: _DebugOption = False,
) -> None:
    """Phase-based EVM — artifact-free motion magnification (Wadhwa et al. 2013)."""
    _setup_logging(debug)
    from pyevm.magnification.phase import PhaseMagnifier  # noqa: PLC0415
    dev = _get_device(device)
    frames, fps = _load_video(input, dev, max_frames)
    magnifier = PhaseMagnifier(
        factor=factor,
        freq_low=freq_low,
        freq_high=freq_high,
        n_scales=n_scales,
        n_orientations=n_orientations,
        sigma=sigma,
        filter_type=filter_type,
        device=dev,
    )
    result = magnifier.process(frames, fps)
    _save_video(result, output, fps)


@app.command()
def info(
    device: _DeviceOption = None,
    debug: _DebugOption = False,
) -> None:
    """Show detected compute device and package information."""
    _setup_logging(debug)
    import torch  # noqa: PLC0415
    from pyevm.device import device_info, get_device  # noqa: PLC0415
    dev = _get_device(device)
    di = device_info(dev)
    typer.echo("EVM — Eulerian Video Magnification")
    typer.echo(f"  PyTorch version : {torch.__version__}")
    typer.echo(f"  Compute device  : {di['device']}")
    typer.echo(f"  Device name     : {di['name']}")
    if "vram_gb" in di:
        typer.echo(f"  VRAM            : {di['vram_gb']} GB")
    typer.echo(f"  CUDA available  : {torch.cuda.is_available()}")
    typer.echo(f"  MPS available   : {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    app()
