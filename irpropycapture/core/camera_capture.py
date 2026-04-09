"""Camera capture utilities."""

from __future__ import annotations

import os
import platform
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


def _fourcc_code(tag: str) -> int:
    return cv2.VideoWriter.fourcc(*tag)


def _open_capture(camera_index: int, width: int, height: int, fps: int) -> tuple[cv2.VideoCapture | None, str]:
    """
    Open capture with the most reliable backend for the current platform.
    """
    backend = _preferred_backend()

    if platform.system() == "Darwin" and camera_index == 0:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"input_format;avfoundation|framerate;{fps}|video_size;{width}x{height}|pixel_format;yuyv422"
        )
        for source in ("USB-Kamera:none", "USB Camera:none", "0:none"):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FORMAT, -1.0)
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
                return cap, "ffmpeg-avfoundation-usb"
            cap.release()

    cap = cv2.VideoCapture(camera_index, backend)
    if cap.isOpened():
        return cap, "native"
    cap.release()
    return None, ""


def _configure_capture_for_raw(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    """Apply capture settings that maximize chance of receiving thermal raw frames."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    # Disable automatic RGB conversion when backend supports it.
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)


def _read_convertible_frame(
    cap: cv2.VideoCapture,
    frame_attempts: int,
) -> tuple[bool, str]:
    last_error = "No frame received from camera."
    for _ in range(frame_attempts):
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.03)
            continue
        try:
            _convert_capture_to_pipeline_frame(frame)
            return True, "ok"
        except ValueError as exc:
            last_error = str(exc)
            time.sleep(0.03)
            continue
    return False, last_error


def _preferred_backend() -> int:
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_MSMF
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def _pack_for_temperature_pipeline(raw_16: np.ndarray) -> np.ndarray:
    """Convert 16-bit sensor frame to the expected 384x512 uint8 layout."""
    if raw_16.ndim != 2:
        raise ValueError(f"Expected 2D 16-bit frame, got shape {raw_16.shape}.")
    if raw_16.dtype != np.uint16:
        raise ValueError(f"Expected uint16 frame, got {raw_16.dtype}.")

    if raw_16.shape[1] != 256:
        raise ValueError(f"Expected width 256, got {raw_16.shape[1]}.")

    if raw_16.shape[0] >= 384:
        sensor_rows = raw_16[192:384, :]
    elif raw_16.shape[0] >= 192:
        sensor_rows = raw_16[-192:, :]
    else:
        raise ValueError(f"Expected at least 192 rows, got {raw_16.shape[0]}.")

    packed = sensor_rows.view(np.uint8).reshape(192, 512).copy()
    output = np.zeros((384, 512), dtype=np.uint8)
    output[192:384, :] = packed
    return output


def _convert_capture_to_pipeline_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize common OpenCV capture outputs to pipeline-compatible raw bytes."""
    if frame.ndim == 2 and frame.shape[0] == 1 and frame.dtype == np.uint8:
        # OpenCV FFMPEG raw packet mode returns one line containing packed bytes.
        flat = frame.reshape(-1)
        expected_size = 256 * 384 * 2
        if flat.size < expected_size:
            raise ValueError(f"Raw packet too small: {flat.size} < {expected_size}.")
        return flat[:expected_size].reshape(384, 512).copy()

    if frame.ndim == 2:
        if frame.dtype == np.uint8 and frame.shape == (384, 512):
            return frame.copy()
        if frame.dtype == np.uint16:
            return _pack_for_temperature_pipeline(frame)
        if frame.dtype == np.uint8 and frame.shape[0] >= 384 and frame.shape[1] >= 512:
            return frame[:384, :512].copy()
        raise ValueError(f"Unsupported 2D frame format: shape={frame.shape}, dtype={frame.dtype}.")

    if frame.ndim == 3 and frame.shape[2] == 2:
        # Some backends expose YUYV/UYVY as 2-channel bytes; keep raw bytes.
        if frame.dtype != np.uint8:
            raise ValueError(f"Unsupported 2-channel dtype: {frame.dtype}.")
        h, w, _ = frame.shape
        if h < 384 or w < 256:
            raise ValueError(f"2-channel frame too small: {frame.shape}.")
        raw_bytes = frame[:384, :256, :].reshape(384, 512)
        return raw_bytes.copy()

    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        # Decoded RGB/BGR images do not preserve thermal sensor raw payload.
        raise ValueError(
            "Camera backend returned decoded color frames. "
            "Please configure a raw 16-bit / packed YUV mode for the thermal camera."
        )

    raise ValueError(f"Unsupported frame format: shape={frame.shape}, dtype={frame.dtype}.")


def _default_max_index() -> int:
    """Return a conservative scan limit per platform."""
    if platform.system() == "Darwin":
        # AVFoundation often exposes a small contiguous index range.
        return 3
    return 10


def list_opencv_camera_devices(
    max_index: int | None = None,
    width: int = 256,
    height: int = 384,
    fps: int = 25,
    required_name: str | None = None,
) -> list[tuple[int, str]]:
    if platform.system() == "Darwin":
        # On macOS we target the known thermal source explicitly.
        ok, _ = probe_opencv_source(camera_index=0, width=width, height=height, fps=fps, frame_attempts=12)
        if ok:
            return [(0, "USB-Kamera")]
        return []

    devices: list[tuple[int, str]] = []
    max_scan_index = _default_max_index() if max_index is None else max_index
    _ = required_name  # Kept for compatibility; OpenCV index probing is name-agnostic.
    for index in range(max_scan_index + 1):
        ok, _ = probe_opencv_source(
            camera_index=index,
            width=width,
            height=height,
            fps=fps,
            frame_attempts=12,
        )
        if not ok:
            continue
        name = f"Camera {index}"
        devices.append((index, name))
    return devices


def probe_opencv_source(
    camera_index: int,
    width: int = 256,
    height: int = 384,
    fps: int = 25,
    frame_attempts: int = 20,
) -> tuple[bool, str]:
    if platform.system() == "Darwin" and camera_index != 0:
        return False, "Only AVFoundation device index 0 (USB-Kamera) is supported in strict macOS mode."

    cap, mode = _open_capture(camera_index, width, height, fps)
    if cap is None:
        return False, f"Could not open camera index {camera_index}."

    # 1) Try default stream first.
    _configure_capture_for_raw(cap, width, height, fps)
    ok, err = _read_convertible_frame(cap, frame_attempts)
    if ok:
        cap.release()
        return True, mode or "ok"

    # 2) Try explicit thermal-related pixel formats.
    fourcc_candidates = ["Y16 ", "Y16", "YUY2", "UYVY", "YUYV"]
    last_error = err
    for fourcc_tag in fourcc_candidates:
        cap.set(cv2.CAP_PROP_FOURCC, float(_fourcc_code(fourcc_tag[:4].ljust(4))))
        ok, err = _read_convertible_frame(cap, frame_attempts)
        if ok:
            cap.release()
            return True, mode or "ok"
        last_error = f"{fourcc_tag.strip() or fourcc_tag}: {err}"
    cap.release()
    return False, last_error


class OpenCVCaptureWorker(QThread):
    frame_ready = Signal(object)
    error = Signal(str)
    camera_opened = Signal()

    def __init__(self, camera_index: int, width: int = 256, height: int = 384, fps: int = 25) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self._capture: cv2.VideoCapture | None = None
        self._capture_mode: str = ""

    def run(self) -> None:
        try:
            self._capture, self._capture_mode = _open_capture(self.camera_index, self.width, self.height, self.fps)
            if self._capture is None:
                self.error.emit(f"Could not open camera index {self.camera_index}.")
                return
            _configure_capture_for_raw(self._capture, self.width, self.height, self.fps)
            if self._capture_mode == "native":
                for fourcc_tag in ("Y16 ", "Y16", "YUY2", "UYVY", "YUYV"):
                    self._capture.set(cv2.CAP_PROP_FOURCC, float(_fourcc_code(fourcc_tag[:4].ljust(4))))
        except Exception as exc:
            self.error.emit(f"Camera start failed: {exc}")
            return

        if self._capture is None or not self._capture.isOpened():
            self.error.emit(f"Could not open camera index {self.camera_index}.")
            return

        self._running = True
        self.camera_opened.emit()
        while self._running:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            try:
                pipeline_frame = _convert_capture_to_pipeline_frame(frame)
            except ValueError as exc:
                self.error.emit(str(exc))
                break
            self.frame_ready.emit(pipeline_frame)
        self._terminate()

    def stop(self) -> None:
        self._running = False
        self.wait(1500)
        self._terminate()

    def _terminate(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
