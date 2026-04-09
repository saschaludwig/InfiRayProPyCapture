"""Video recording utility."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self.output_path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(self, output_path: Path, frame_size: tuple[int, int], fps: float = 25.0) -> bool:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if not writer.isOpened():
            return False
        self._writer = writer
        self.output_path = output_path
        return True

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame_bgr)

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
