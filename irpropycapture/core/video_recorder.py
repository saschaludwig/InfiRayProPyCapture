"""Video recording utility."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from irpropycapture.core.perf import PerfReporter


class VideoRecorder:
    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=6)
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._perf = PerfReporter("VideoRecorder")
        self.output_path: Path | None = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(self, output_path: Path, frame_size: tuple[int, int], fps: float = 25.0) -> bool:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if not writer.isOpened():
            return False
        with self._lock:
            self._writer = writer
            self.output_path = output_path
            self._stop_event.clear()
            self._queue = queue.Queue(maxsize=6)
            self._worker = threading.Thread(target=self._run_writer_loop, name="VideoRecorderWorker", daemon=True)
            self._worker.start()
        return True

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            return
        frame_copy = frame_bgr.copy()
        try:
            self._queue.put_nowait(frame_copy)
        except queue.Full:
            # Keep video recording responsive by dropping the oldest queued frame.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame_copy)
            except queue.Full:
                return

    def stop(self) -> None:
        writer: cv2.VideoWriter | None
        with self._lock:
            writer = self._writer
            worker = self._worker
            self._worker = None
            self._writer = None
            self._stop_event.set()
        if writer is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if worker is not None:
            worker.join(timeout=1.5)
        writer.release()

    def _run_writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break
            writer = self._writer
            if writer is None:
                break
            start = time.perf_counter()
            writer.write(frame)
            self._perf.observe("write_frame", time.perf_counter() - start)
