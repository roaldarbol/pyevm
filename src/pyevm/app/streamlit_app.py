"""Streamlit web application for Eulerian Video Magnification.

Launch with:
    pyevm dashboard
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from loguru import logger

# Configure log level before any pyevm imports so module-level loggers pick it up.
_log_level = os.environ.get("PYEVM_LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stderr, level=_log_level)

import streamlit as st  # noqa: E402
import torch  # noqa: E402

from pyevm.device import get_device  # noqa: E402
from pyevm.io.video import VideoReader, VideoWriter  # noqa: E402
from pyevm.magnification.color import ColorMagnifier  # noqa: E402
from pyevm.magnification.motion import MotionMagnifier  # noqa: E402
from pyevm.magnification.phase import PhaseMagnifier  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Eulerian Video Magnification",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def _detect_device() -> torch.device:
    return get_device()


def _bytes_to_temp_file(data: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        return Path(tmp.name)


def _read_video(path: Path, device: torch.device, max_frames: int | None):
    reader = VideoReader(path, device=device, max_frames=max_frames)
    frames, fps = reader.read()
    meta = reader.metadata
    return frames, fps, meta


def _write_video_to_bytes(frames: torch.Tensor, fps: float) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = Path(tmp.name)
    writer = VideoWriter(out_path, fps=fps)
    writer.write(frames)
    data = out_path.read_bytes()
    out_path.unlink(missing_ok=True)
    return data


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _sidebar_device(device: torch.device) -> None:
    st.sidebar.markdown("### Compute Device")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        st.sidebar.success(f"CUDA GPU: {name}")
    elif device.type == "mps":
        st.sidebar.success("Apple Silicon GPU (MPS)")
    else:
        st.sidebar.warning("CPU (no GPU detected)")


def _color_params() -> dict:
    st.sidebar.markdown("### Algorithm Parameters")
    return {
        "alpha": st.sidebar.slider("Alpha (amplification)", 5.0, 200.0, 50.0, step=5.0),
        "freq_low": st.sidebar.number_input(
            "Freq low (Hz)", 0.05, 1000.0, 0.4, step=0.05, format="%.2f"
        ),
        "freq_high": st.sidebar.number_input(
            "Freq high (Hz)", 0.1, 1000.0, 3.0, step=0.05, format="%.2f"
        ),
        "n_levels": st.sidebar.slider("Pyramid levels", 2, 8, 6),
        "chrom_attenuation": st.sidebar.slider("Chrominance attenuation", 0.0, 1.0, 0.1, step=0.05),
        "filter_type": st.sidebar.selectbox("Filter type", ["ideal", "butterworth"]),
    }


def _motion_params() -> dict:
    st.sidebar.markdown("### Algorithm Parameters")
    return {
        "alpha": st.sidebar.slider("Alpha (amplification)", 1.0, 100.0, 20.0, step=1.0),
        "freq_low": st.sidebar.number_input(
            "Freq low (Hz)", 0.05, 1000.0, 0.4, step=0.05, format="%.2f"
        ),
        "freq_high": st.sidebar.number_input(
            "Freq high (Hz)", 0.1, 1000.0, 3.0, step=0.05, format="%.2f"
        ),
        "n_levels": st.sidebar.slider("Pyramid levels", 2, 8, 6),
        "lambda_c": st.sidebar.slider("Lambda c (spatial cutoff, px)", 4.0, 64.0, 16.0, step=2.0),
        "filter_type": st.sidebar.selectbox("Filter type", ["butterworth", "ideal"]),
    }


def _phase_params() -> dict:
    import math as _math

    st.sidebar.markdown("### Algorithm Parameters")
    params: dict = {
        "factor": st.sidebar.slider("Factor (phase amplification)", 1.0, 100.0, 10.0, step=1.0),
        "freq_low": st.sidebar.number_input(
            "Freq low (Hz)", 0.05, 1000.0, 0.4, step=0.05, format="%.2f"
        ),
        "freq_high": st.sidebar.number_input(
            "Freq high (Hz)", 0.1, 1000.0, 3.0, step=0.05, format="%.2f"
        ),
        "n_scales": st.sidebar.slider("Pyramid scales", 2, 8, 6),
        "n_orientations": st.sidebar.slider("Orientations per scale", 2, 8, 8),
        "sigma": st.sidebar.slider("Phase smoothing sigma (0 = off)", 0.0, 10.0, 0.0, step=0.5),
        "filter_type": st.sidebar.selectbox("Filter type", ["ideal", "butterworth"]),
    }
    st.sidebar.markdown("### Large-Motion Attenuation")
    params["attenuate_motion"] = st.sidebar.checkbox(
        "Attenuate large motions (Fig. 11)",
        value=False,
        help="Wrap amplified phase changes larger than the threshold back to zero, "
        "suppressing camera shake or other large global motions while keeping "
        "subtle local vibrations.",
    )
    if params["attenuate_motion"]:
        params["attenuate_mag"] = st.sidebar.slider(
            "Attenuation threshold (rad)",
            0.1,
            _math.pi * 2,
            _math.pi,
            step=0.05,
            help="Amplified phase changes above this value are attenuated. "
            "π (≈ 3.14) is the default and largest unambiguous phase step.",
        )
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = _detect_device()

    # --- Header ---
    st.title("🔬 Eulerian Video Magnification")
    st.markdown(
        "Reveal invisible motion and colour changes in video using spatial "
        "decomposition and temporal bandpass filtering."
    )

    # --- Sidebar ---
    st.sidebar.title("Settings")
    _sidebar_device(device)

    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["Color (Wu 2012)", "Motion (Wu 2012)", "Phase (Wadhwa 2013)"],
    )

    max_frames = st.sidebar.number_input(
        "Max frames (0 = all)", min_value=0, max_value=10000, value=0, step=10
    )
    max_frames_val = int(max_frames) if max_frames > 0 else None

    if algorithm == "Color (Wu 2012)":
        params = _color_params()
    elif algorithm == "Motion (Wu 2012)":
        params = _motion_params()
    else:
        params = _phase_params()

    # --- Notch filter (shared across all algorithms) ---
    st.sidebar.markdown("### Notch Filters")
    notch_input = st.sidebar.text_input(
        "Notch frequencies (Hz)",
        value="",
        placeholder="e.g. 50, 60",
        help="Comma-separated list of frequencies to suppress. "
        "Useful for removing 50/60 Hz light flicker.",
    )
    notch_freqs: list[float] = []
    for tok in notch_input.split(","):
        tok = tok.strip()
        if tok:
            try:
                notch_freqs.append(float(tok))
            except ValueError:
                st.sidebar.warning(f"Ignoring invalid notch frequency: '{tok}'")
    if notch_freqs:
        notch_width = st.sidebar.slider(
            "Notch width (Hz)",
            0.1,
            5.0,
            1.0,
            step=0.1,
            help="Bandwidth of each notch. Wider removes more signal around the target frequency.",
        )
        params["notch_freqs"] = notch_freqs
        params["notch_width"] = notch_width

    # --- Upload ---
    st.markdown("## 1. Upload Video")
    uploaded = st.file_uploader(
        "Drag and drop or click to upload",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
    )

    if uploaded is None:
        st.info("Upload a video to get started.")
        return

    # Store bytes before reading (file_uploader cursor moves on .read())
    video_bytes_in = uploaded.read()
    suffix = Path(uploaded.name).suffix
    tmp_in = _bytes_to_temp_file(video_bytes_in, suffix)

    # --- Video info + preview ---
    st.markdown("## 2. Original Video")
    try:
        frames, fps, meta = _read_video(tmp_in, device, max_frames_val)
    except Exception as exc:
        st.error(f"Could not read video: {exc}")
        tmp_in.unlink(missing_ok=True)
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", meta["n_frames"])
    col2.metric("FPS", f"{fps:.2f}")
    col3.metric("Width", meta["width"])
    col4.metric("Height", meta["height"])

    st.video(video_bytes_in)

    # --- Process ---
    st.markdown("## 3. Process")
    if st.button("Run Magnification", type="primary", width="stretch"):
        with st.spinner("Processing… this may take a moment for long videos."):
            try:
                logger.info(f"Starting {algorithm} with params: {params}")

                if algorithm == "Color (Wu 2012)":
                    magnifier = ColorMagnifier(device=device, **params)
                elif algorithm == "Motion (Wu 2012)":
                    magnifier = MotionMagnifier(device=device, **params)
                else:
                    magnifier = PhaseMagnifier(device=device, **params)

                result = magnifier.process(frames, fps)
                logger.info("Magnification complete")

            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                logger.exception("Processing error")
                tmp_in.unlink(missing_ok=True)
                return

        st.success("Done!")

        # --- Results ---
        st.markdown("## 4. Results")

        with st.spinner("Encoding output video…"):
            video_bytes_out = _write_video_to_bytes(result, fps)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.video(video_bytes_in)
        with c2:
            st.markdown("**Magnified**")
            st.video(video_bytes_out)

        out_name = f"{Path(uploaded.name).stem}_magnified.mp4"
        st.download_button(
            label="⬇ Download magnified video",
            data=video_bytes_out,
            file_name=out_name,
            mime="video/mp4",
            width="stretch",
        )

    tmp_in.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
