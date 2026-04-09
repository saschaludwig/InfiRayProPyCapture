"""Camera capture utilities."""

from __future__ import annotations

import re
import subprocess
import time

import numpy as np
from PySide6.QtCore import QThread, Signal


def list_ffmpeg_avfoundation_devices() -> list[tuple[int, str]]:
    try:
        process = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = process.stderr + "\n" + process.stdout
    except Exception:
        return []

    devices: list[tuple[int, str]] = []
    in_video_section = False
    for line in output.splitlines():
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
        if match:
            devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def probe_ffmpeg_source(
    camera_index: int,
    width: int = 256,
    height: int = 384,
    fps: int = 25,
    timeout_seconds: int = 6,
) -> tuple[bool, str]:
    ok, pix_fmt, err = select_working_pix_fmt(camera_index, width, height, fps, timeout_seconds)
    if ok:
        return True, pix_fmt
    return False, err


def select_working_pix_fmt(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    timeout_seconds: int = 6,
) -> tuple[bool, str, str]:
    frame_size = width * height * 2
    pix_fmts = ["uyvy422", "yuyv422", "gray16be", "gray16le"]
    last_err = ""
    for pix_fmt in pix_fmts:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-framerate",
            str(fps),
            "-video_size",
            f"{width}x{height}",
            "-i",
            f"{camera_index}:none",
            "-frames:v",
            "1",
            "-pix_fmt",
            pix_fmt,
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            process = subprocess.run(cmd, capture_output=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            last_err = f"{pix_fmt}: timeout"
            continue
        except Exception as exc:
            last_err = f"{pix_fmt}: {exc}"
            continue

        if process.returncode != 0:
            stderr_text = process.stderr.decode("utf-8", errors="ignore").strip()
            last_err = f"{pix_fmt}: {stderr_text}"
            continue
        if len(process.stdout) < frame_size:
            last_err = f"{pix_fmt}: incomplete frame"
            continue

        return True, pix_fmt, ""
    return False, "", last_err or "No working pixel format."


class FfmpegCaptureWorker(QThread):
    frame_ready = Signal(object)
    error = Signal(str)
    camera_opened = Signal()

    def __init__(self, camera_index: int, width: int = 256, height: int = 384, fps: int = 25, pix_fmt: str = "uyvy422") -> None:
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt = pix_fmt
        self._running = False
        self._process: subprocess.Popen | None = None

    def run(self) -> None:
        frame_size = self.width * self.height * 2
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-f",
            "avfoundation",
            "-framerate",
            str(self.fps),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            f"{self.camera_index}:none",
            "-pix_fmt",
            self.pix_fmt,
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as exc:
            self.error.emit(f"FFmpeg start failed: {exc}")
            return

        if self._process.stdout is None:
            self.error.emit("FFmpeg stdout unavailable.")
            return

        self._running = True
        self.camera_opened.emit()
        pending = bytearray()
        while self._running:
            chunk = self._process.stdout.read(frame_size)
            if not chunk:
                time.sleep(0.01)
                continue
            pending.extend(chunk)
            if len(pending) < frame_size:
                continue
            frame_bytes = bytes(pending[:frame_size])
            del pending[:frame_size]
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(self.height, self.width * 2)
            self.frame_ready.emit(frame)
        self._terminate()

    def stop(self) -> None:
        self._running = False
        self.wait(1500)
        self._terminate()

    def _terminate(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            self._process = None
