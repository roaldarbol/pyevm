"""Video I/O — fast decoding via decord (preferred) or OpenCV fallback.

VideoReader
    Reads all frames into a ``(T, C, H, W)`` float32 tensor in ``[0, 1]``.
    Uses *decord* when available (significantly faster; GPU decode on CUDA),
    falling back to OpenCV.

VideoWriter
    Writes a ``(T, C, H, W)`` float32 tensor to a video file.
    Pipes raw frames to FFmpeg for hardware-accelerated encoding (H.264 via
    VideoToolbox on macOS, NVENC on CUDA systems).  Falls back to
    ``cv2.VideoWriter`` when FFmpeg is not found.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class VideoReader:
    """Read a video file into a tensor.

    Args:
        path: Path to the video file.
        device: Tensor device (CPU only when using decord CPU bridge; the
            returned tensor is always moved to *device* after reading).
        max_frames: Limit number of frames read (``None`` = all).
    """

    def __init__(
        self,
        path: str | Path,
        device: torch.device | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.device = device or torch.device("cpu")
        self.max_frames = max_frames
        self._meta: dict | None = None

    @property
    def metadata(self) -> dict:
        """Return ``{"fps": float, "n_frames": int, "height": int, "width": int}``."""
        if self._meta is None:
            self._meta = self._read_metadata()
        return self._meta

    def _read_metadata(self) -> dict:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return {"fps": fps, "n_frames": n_frames, "height": height, "width": width}

    def read(self) -> tuple[torch.Tensor, float]:
        """Read video frames.

        Returns:
            ``(frames, fps)`` where *frames* is ``(T, C, H, W)`` float32
            tensor in ``[0, 1]`` on *self.device*.
        """
        try:
            frames, fps = self._read_decord()
            logger.info(f"VideoReader: read {frames.shape[0]} frames via decord")
        except Exception as exc:
            logger.debug(f"decord unavailable ({exc}), falling back to OpenCV")
            frames, fps = self._read_opencv()
            logger.info(f"VideoReader: read {frames.shape[0]} frames via OpenCV")

        return frames.to(self.device), fps

    def _read_decord(self) -> tuple[torch.Tensor, float]:
        import decord  # noqa: PLC0415
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(self.path), ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        indices = list(range(min(len(vr), self.max_frames or len(vr))))
        # Returns (T, H, W, C) uint8
        frames_hwc = vr.get_batch(indices)
        frames = frames_hwc.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        return frames, fps

    def _read_opencv(self) -> tuple[torch.Tensor, float]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_list: list[np.ndarray] = []
        limit = self.max_frames or int(1e9)
        while len(frame_list) < limit:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads BGR → convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame_rgb)
        cap.release()

        if not frame_list:
            raise RuntimeError(f"No frames could be read from {self.path}")

        arr = np.stack(frame_list, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
        frames = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, C, H, W)
        return frames, fps


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class VideoWriter:
    """Write a tensor to a video file.

    Prefers piping frames through FFmpeg for better codec support and
    hardware-accelerated encoding.  Falls back to ``cv2.VideoWriter``.

    Args:
        path: Output file path (.mp4 recommended).
        fps: Frames per second.
        use_ffmpeg: Try FFmpeg first (default ``True``).
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        use_ffmpeg: bool = True,
    ) -> None:
        self.path = Path(path)
        self.fps = fps
        self.use_ffmpeg = use_ffmpeg

    def write(self, frames: torch.Tensor) -> None:
        """Write *frames* to disk.

        Args:
            frames: ``(T, C, H, W)`` float tensor in ``[0, 1]`` or uint8 in
                ``[0, 255]``.
        """
        frames = frames.cpu()
        if frames.is_floating_point():
            frames_u8 = (frames.clamp(0, 1) * 255).byte()
        else:
            frames_u8 = frames.byte()

        T, C, H, W = frames_u8.shape

        if self.use_ffmpeg and shutil.which("ffmpeg") is not None:
            self._write_ffmpeg(frames_u8, H, W)
        else:
            logger.debug("FFmpeg not found; falling back to OpenCV VideoWriter")
            self._write_opencv(frames_u8, H, W)

    def _write_ffmpeg(self, frames_u8: torch.Tensor, H: int, W: int) -> None:
        """Pipe raw RGB24 frames to FFmpeg."""
        logger.debug(f"VideoWriter: writing {frames_u8.shape[0]} frames via FFmpeg → {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Try hardware-accelerated encoder; fall back to libx264
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(self.path),
        ]
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for t in range(frames_u8.shape[0]):
                # (C, H, W) → (H, W, C) numpy bytes
                frame_np = frames_u8[t].permute(1, 2, 0).numpy()
                proc.stdin.write(frame_np.tobytes())  # type: ignore[union-attr]
            proc.stdin.close()  # type: ignore[union-attr]
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
        except Exception:
            proc.kill()
            raise

        logger.info(f"VideoWriter: saved {frames_u8.shape[0]} frames to {self.path}")

    def _write_opencv(self, frames_u8: torch.Tensor, H: int, W: int) -> None:
        """Write using cv2.VideoWriter (fallback)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (W, H))
        if not writer.isOpened():
            raise IOError(f"Cannot open VideoWriter for {self.path}")
        for t in range(frames_u8.shape[0]):
            frame_np = frames_u8[t].permute(1, 2, 0).numpy()  # (H, W, C) RGB
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()
        logger.info(f"VideoWriter(OpenCV): saved {frames_u8.shape[0]} frames to {self.path}")
