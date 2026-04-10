"""Streamlit web application for Eulerian Video Magnification.

Launch with:
    streamlit run src/evm/app/streamlit_app.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import streamlit as st
import torch
from loguru import logger

from pyevm.device import get_device
from pyevm.io.video import VideoReader, VideoWriter
from pyevm.magnification.color import ColorMagnifier
from pyevm.magnification.motion import MotionMagnifier
from pyevm.magnification.phase import PhaseMagnifier

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
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
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
# UI
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
        "freq_low": st.sidebar.number_input("Freq low (Hz)", 0.05, 10.0, 0.4, step=0.05, format="%.2f"),
        "freq_high": st.sidebar.number_input("Freq high (Hz)", 0.1, 30.0, 3.0, step=0.1, format="%.1f"),
        "n_levels": st.sidebar.slider("Pyramid levels", 2, 8, 6),
        "chrom_attenuation": st.sidebar.slider("Chrominance attenuation", 0.0, 1.0, 0.1, step=0.05),
        "filter_type": st.sidebar.selectbox("Filter type", ["ideal", "butterworth"]),
    }


def _motion_params() -> dict:
    st.sidebar.markdown("### Algorithm Parameters")
    return {
        "alpha": st.sidebar.slider("Alpha (amplification)", 1.0, 100.0, 20.0, step=1.0),
        "freq_low": st.sidebar.number_input("Freq low (Hz)", 0.05, 10.0, 0.4, step=0.05, format="%.2f"),
        "freq_high": st.sidebar.number_input("Freq high (Hz)", 0.1, 30.0, 3.0, step=0.1, format="%.1f"),
        "n_levels": st.sidebar.slider("Pyramid levels", 2, 8, 6),
        "lambda_c": st.sidebar.slider("Lambda c (spatial cutoff, px)", 4.0, 64.0, 16.0, step=2.0),
        "filter_type": st.sidebar.selectbox("Filter type", ["butterworth", "ideal"]),
    }


def _phase_params() -> dict:
    st.sidebar.markdown("### Algorithm Parameters")
    return {
        "factor": st.sidebar.slider("Factor (phase amplification)", 1.0, 100.0, 10.0, step=1.0),
        "freq_low": st.sidebar.number_input("Freq low (Hz)", 0.05, 10.0, 0.4, step=0.05, format="%.2f"),
        "freq_high": st.sidebar.number_input("Freq high (Hz)", 0.1, 30.0, 3.0, step=0.1, format="%.1f"),
        "n_scales": st.sidebar.slider("Pyramid scales", 2, 8, 4),
        "n_orientations": st.sidebar.slider("Orientations per scale", 2, 8, 6),
        "sigma": st.sidebar.slider("Phase smoothing sigma (0 = off)", 0.0, 10.0, 3.0, step=0.5),
        "filter_type": st.sidebar.selectbox("Filter type", ["ideal", "butterworth"]),
    }


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

    # Write upload to temp file
    suffix = Path(uploaded.name).suffix
    tmp_in = _bytes_to_temp_file(uploaded.read(), suffix)

    # --- Video info ---
    st.markdown("## 2. Video Info")
    try:
        frames, fps, meta = _read_video(tmp_in, device, max_frames_val)
    except Exception as exc:
        st.error(f"Could not read video: {exc}")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", meta["n_frames"])
    col2.metric("FPS", f"{fps:.2f}")
    col3.metric("Width", meta["width"])
    col4.metric("Height", meta["height"])

    # Show first frame as preview
    first_frame = frames[0].permute(1, 2, 0).cpu().numpy()
    st.image(first_frame, caption="First frame (original)", use_container_width=True)

    # --- Process ---
    st.markdown("## 3. Process")
    if st.button("Run Magnification", type="primary", use_container_width=True):
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
                return

        st.success("Done!")

        # --- Results ---
        st.markdown("## 4. Results")

        # Side-by-side first frame comparison
        c1, c2 = st.columns(2)
        with c1:
            st.image(first_frame, caption="Original — first frame", use_container_width=True)
        with c2:
            result_frame = result[0].permute(1, 2, 0).cpu().float().numpy()
            st.image(result_frame, caption="Magnified — first frame", use_container_width=True)

        # Download button
        with st.spinner("Encoding output video…"):
            video_bytes = _write_video_to_bytes(result, fps)

        out_name = f"{Path(uploaded.name).stem}_magnified.mp4"
        st.download_button(
            label="⬇ Download magnified video",
            data=video_bytes,
            file_name=out_name,
            mime="video/mp4",
            use_container_width=True,
        )

    # Cleanup
    tmp_in.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
