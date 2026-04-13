"""Video I/O — fast decoding via torchcodec (preferred) or OpenCV fallback.

VideoReader
    Reads all frames into a ``(T, C, H, W)`` float32 tensor in ``[0, 1]``,
    or streams them one at a time via :meth:`VideoReader.stream`.
    Uses *torchcodec* when available (GPU-accelerated decode, Linux pip /
    Windows conda), falling back to OpenCV.

VideoWriter
    Writes a ``(T, C, H, W)`` float32 tensor to a video file, or accepts a
    frame generator via :meth:`VideoWriter.write_stream`.
    Pipes raw frames to FFmpeg for hardware-accelerated encoding.  Falls back
    to ``cv2.VideoWriter`` when FFmpeg is not found.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Generator, Iterable
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

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
            raise OSError(f"Cannot open video: {self.path}")
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
            frames, fps = self._read_torchcodec()
            logger.info(f"VideoReader: read {frames.shape[0]} frames via torchcodec")
        except Exception as exc:
            logger.debug(f"torchcodec unavailable ({exc}), falling back to OpenCV")
            frames, fps = self._read_opencv()
            logger.info(f"VideoReader: read {frames.shape[0]} frames via OpenCV")

        return frames, fps

    def stream(self) -> Generator[torch.Tensor, None, None]:
        """Yield frames one at a time as ``(C, H, W)`` float32 tensors on *self.device*.

        Memory cost is constant — only one decoded frame lives in RAM at a
        time, regardless of video length.  Use this for large videos where
        :meth:`read` would exhaust available memory.

        Uses *torchcodec* when available (GPU-accelerated), falling back to
        OpenCV.
        """
        try:
            yield from self._stream_torchcodec()
            return
        except Exception as exc:
            logger.debug(f"torchcodec stream unavailable ({exc}), falling back to OpenCV")

        yield from self._stream_opencv()

    def _read_torchcodec(self) -> tuple[torch.Tensor, float]:
        from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

        meta = self.metadata
        fps = meta["fps"]
        limit = (
            min(self.max_frames, meta["n_frames"])
            if self.max_frames is not None
            else meta["n_frames"]
        )
        decoder = VideoDecoder(str(self.path), device=str(self.device))
        # Slice returns (T, C, H, W) uint8 tensor already on self.device
        frames = decoder[0:limit].float() * (1.0 / 255.0)
        return frames, fps

    def _stream_torchcodec(self) -> Generator[torch.Tensor, None, None]:
        from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

        meta = self.metadata
        limit = (
            min(self.max_frames, meta["n_frames"])
            if self.max_frames is not None
            else meta["n_frames"]
        )
        decoder = VideoDecoder(str(self.path), device=str(self.device))
        with tqdm(total=limit, desc="   Reading", unit="frame", position=2, leave=True) as bar:
            for i in range(limit):
                # decoder[i] → (C, H, W) uint8 tensor on self.device
                yield decoder[i].float() * (1.0 / 255.0)
                bar.update(1)

    def _stream_opencv(self) -> Generator[torch.Tensor, None, None]:
        meta = self.metadata
        limit = (
            min(self.max_frames, meta["n_frames"])
            if self.max_frames is not None
            else meta["n_frames"]
        )
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self.path}")
        count = 0
        try:
            with tqdm(total=limit, desc="   Reading", unit="frame", position=2, leave=True) as bar:
                while count < limit:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield (
                        torch.from_numpy(frame_rgb.astype(np.float32) * (1.0 / 255.0))
                        .permute(2, 0, 1)
                        .to(self.device)
                    )
                    count += 1
                    bar.update(1)
        finally:
            cap.release()

        if count == 0:
            raise RuntimeError(f"No frames could be read from {self.path}")

    def _read_opencv(self) -> tuple[torch.Tensor, float]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        limit = min(self.max_frames, total) if self.max_frames is not None else total

        # Pre-allocate the full output tensor so we never hold a separate list of
        # uint8 arrays AND a float32 copy at the same time.  Each frame is decoded
        # straight into its slot; peak extra RAM is just one frame's worth of CV
        # buffer (~6 MB at 1080p) instead of the entire video twice over.
        frames = torch.empty(limit, 3, height, width, dtype=torch.float32)
        count = 0
        with tqdm(total=limit, desc="Reading", unit="frame", leave=False) as bar:
            while count < limit:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[count] = torch.from_numpy(
                    frame_rgb.astype(np.float32) * (1.0 / 255.0)
                ).permute(2, 0, 1)
                count += 1
                bar.update(1)
        cap.release()

        if count == 0:
            raise RuntimeError(f"No frames could be read from {self.path}")

        return frames[:count], fps


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

    def write_stream(
        self,
        frames: Iterable[torch.Tensor],
        height: int,
        width: int,
        n_frames: int | None = None,
    ) -> None:
        """Write frames from a generator to disk without buffering the full video.

        Args:
            frames: Iterable of ``(C, H, W)`` float32 or uint8 tensors.
            height: Frame height in pixels (needed to open the encoder upfront).
            width: Frame width in pixels.
            n_frames: Total frame count, used only for the progress bar.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_ffmpeg and shutil.which("ffmpeg") is not None:
            self._stream_ffmpeg(frames, height, width, n_frames)
        else:
            logger.debug("FFmpeg not found; falling back to OpenCV VideoWriter")
            self._stream_opencv(frames, height, width, n_frames)

    def _stream_ffmpeg(
        self,
        frames: Iterable[torch.Tensor],
        H: int,
        W: int,
        n_frames: int | None,
    ) -> None:
        """Open an FFmpeg pipe and feed frames one at a time."""
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(self.path),
        ]
        logger.debug(f"VideoWriter stream: FFmpeg command: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        count = 0
        try:
            with tqdm(
                total=n_frames, desc="  Writing", unit="frame", position=0, leave=True
            ) as bar:
                for frame in frames:
                    frame_u8 = (frame.cpu().clamp(0, 1) * 255).byte()
                    frame_np = frame_u8.permute(1, 2, 0).numpy()
                    proc.stdin.write(frame_np.tobytes())  # type: ignore[union-attr]
                    count += 1
                    bar.update(1)
            proc.stdin.close()  # type: ignore[union-attr]
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
        except Exception:
            proc.kill()
            raise
        logger.info(f"VideoWriter: saved {count} frames to {self.path}")

    def _stream_opencv(
        self,
        frames: Iterable[torch.Tensor],
        H: int,
        W: int,
        n_frames: int | None,
    ) -> None:
        """Write frames one at a time via cv2.VideoWriter."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (W, H))
        if not writer.isOpened():
            raise OSError(f"Cannot open VideoWriter for {self.path}")
        count = 0
        with tqdm(total=n_frames, desc="  Writing", unit="frame", position=0, leave=True) as bar:
            for frame in frames:
                frame_u8 = (frame.cpu().clamp(0, 1) * 255).byte()
                frame_np = frame_u8.permute(1, 2, 0).numpy()
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                count += 1
                bar.update(1)
        writer.release()
        logger.info(f"VideoWriter(OpenCV): saved {count} frames to {self.path}")

    def _write_ffmpeg(self, frames_u8: torch.Tensor, H: int, W: int) -> None:
        """Pipe raw RGB24 frames to FFmpeg."""
        logger.debug(f"VideoWriter: writing {frames_u8.shape[0]} frames via FFmpeg → {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Try hardware-accelerated encoder; fall back to libx264
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(self.path),
        ]
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with tqdm(
                total=frames_u8.shape[0], desc="  Writing", unit="frame", position=0, leave=True
            ) as bar:
                for t in range(frames_u8.shape[0]):
                    frame_np = frames_u8[t].permute(1, 2, 0).numpy()
                    proc.stdin.write(frame_np.tobytes())  # type: ignore[union-attr]
                    bar.update(1)
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
            raise OSError(f"Cannot open VideoWriter for {self.path}")
        with tqdm(
            total=frames_u8.shape[0], desc="  Writing", unit="frame", position=0, leave=True
        ) as bar:
            for t in range(frames_u8.shape[0]):
                frame_np = frames_u8[t].permute(1, 2, 0).numpy()  # (H, W, C) RGB
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                bar.update(1)
        writer.release()
        logger.info(f"VideoWriter(OpenCV): saved {frames_u8.shape[0]} frames to {self.path}")
