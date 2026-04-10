"""Background worker for thermal frame processing and chart rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, QWaitCondition, Signal

from irpropycapture.core.image_processor import apply_orientation, format_temperature, render_thermal_image
from irpropycapture.core.temperature_processor import HistogramPoint, TemperatureHistoryPoint, TemperatureProcessor


@dataclass
class ProcessingSettings:
    color_map_name: str
    manual_range_enabled: bool
    manual_min_temp: float
    manual_max_temp: float
    preview_interpolation: str
    orientation: str
    show_grid: bool
    show_min_max: bool
    grid_density: str
    unit: str
    preview_width: int
    preview_height: int
    histogram_width: int
    histogram_height: int
    history_width: int
    history_height: int


@dataclass
class ProcessingResult:
    preview_rgb: np.ndarray
    export_bgr: np.ndarray
    histogram_rgb: np.ndarray
    history_rgb: np.ndarray
    thermal_celsius: np.ndarray
    min_value: float
    max_value: float
    average: float
    center: float


class ProcessingWorker(QThread):
    """Consumes latest frame and emits ready-to-display artifacts."""

    processed = Signal(object)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._processor = TemperatureProcessor()
        self._mutex = QMutex()
        self._wait = QWaitCondition()
        self._latest_frame: np.ndarray | None = None
        self._latest_settings: ProcessingSettings | None = None
        self._running = False
        self._cached_history_bgr: np.ndarray | None = None
        self._cached_history_gen: int = -1
        self._cached_history_settings: tuple[str, int, int] = ("", 0, 0)

    def submit_frame(self, frame: np.ndarray, settings: ProcessingSettings) -> None:
        """Store only the latest frame/settings pair (latest-frame-wins)."""
        with QMutexLocker(self._mutex):
            self._latest_frame = frame.copy()
            self._latest_settings = settings
            self._wait.wakeOne()

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._running = False
            self._wait.wakeOne()
        self.wait(1500)

    def run(self) -> None:
        self._running = True
        while True:
            frame: np.ndarray | None = None
            settings: ProcessingSettings | None = None
            with QMutexLocker(self._mutex):
                while self._running and self._latest_frame is None:
                    self._wait.wait(self._mutex)
                if not self._running:
                    return
                frame = self._latest_frame
                settings = self._latest_settings
                self._latest_frame = None
                self._latest_settings = None
            if frame is None or settings is None:
                continue
            try:
                processed = self._process_frame(frame, settings)
            except Exception as exc:
                self.error.emit(f"Frame processing failed: {exc}")
                continue
            self.processed.emit(processed)

    def _process_frame(self, frame: np.ndarray, settings: ProcessingSettings) -> ProcessingResult:
        result = self._processor.get_temperatures(frame)
        thermal = result.temperatures.reshape(192, 256)
        rendered = render_thermal_image(
            thermal,
            color_map_name=settings.color_map_name,
            manual_range_enabled=settings.manual_range_enabled,
            manual_min_temp=settings.manual_min_temp,
            manual_max_temp=settings.manual_max_temp,
        )
        rendered = cv2.resize(rendered, (1024, 768), interpolation=cv2.INTER_NEAREST)
        rendered = apply_orientation(rendered, settings.orientation)

        oriented_thermal = None
        if settings.show_grid or settings.show_min_max:
            oriented_thermal = apply_orientation(thermal, settings.orientation)

        export_bgr = rendered.copy()
        if oriented_thermal is not None:
            if settings.show_grid:
                _draw_grid(export_bgr, oriented_thermal, settings.grid_density, settings.unit)
            if settings.show_min_max:
                _draw_min_max(export_bgr, oriented_thermal, settings.unit)

        preview_bgr = _resize_preview(rendered, settings.preview_width, settings.preview_height, settings.preview_interpolation)
        if oriented_thermal is not None:
            if settings.show_grid:
                _draw_grid(preview_bgr, oriented_thermal, settings.grid_density, settings.unit)
            if settings.show_min_max:
                _draw_min_max(preview_bgr, oriented_thermal, settings.unit)

        histogram_bgr = _build_histogram_image(
            result.histogram,
            settings.unit,
            settings.color_map_name,
            settings.histogram_width,
            settings.histogram_height,
        )

        history_key = (settings.unit, settings.history_width, settings.history_height)
        if (
            self._cached_history_bgr is not None
            and result.history_generation == self._cached_history_gen
            and history_key == self._cached_history_settings
        ):
            history_bgr = self._cached_history_bgr
        else:
            history_bgr = _build_history_image(result.temperature_history, settings.unit, settings.history_width, settings.history_height)
            self._cached_history_bgr = history_bgr
            self._cached_history_gen = result.history_generation
            self._cached_history_settings = history_key

        return ProcessingResult(
            preview_rgb=cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB),
            export_bgr=export_bgr,
            histogram_rgb=cv2.cvtColor(histogram_bgr, cv2.COLOR_BGR2RGB),
            history_rgb=cv2.cvtColor(history_bgr, cv2.COLOR_BGR2RGB),
            thermal_celsius=thermal,
            min_value=result.min_value,
            max_value=result.max_value,
            average=result.average,
            center=result.center,
        )


def _resize_preview(image_bgr: np.ndarray, width: int, height: int, interpolation_mode: str) -> np.ndarray:
    if width <= 0 or height <= 0:
        return image_bgr.copy()
    interpolation = cv2.INTER_NEAREST if interpolation_mode == "Fast" else cv2.INTER_LINEAR
    src_h, src_w = image_bgr.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    out_w = max(1, int(src_w * scale))
    out_h = max(1, int(src_h * scale))
    return cv2.resize(image_bgr, (out_w, out_h), interpolation=interpolation)


def _draw_grid(image_bgr: np.ndarray, thermal_celsius: np.ndarray, density: str, unit: str) -> None:
    if density == "Low":
        step_x, step_y = 64, 48
    elif density == "High":
        step_x, step_y = 24, 18
    else:
        step_x, step_y = 32, 24

    h, w = thermal_celsius.shape
    scale_x = image_bgr.shape[1] / float(w)
    scale_y = image_bgr.shape[0] / float(h)
    for y in range(step_y // 2, h, step_y):
        for x in range(step_x // 2, w, step_x):
            value = float(thermal_celsius[y, x])
            text = format_temperature(value, unit)
            px = int(x * scale_x)
            py = int(y * scale_y)
            cv2.putText(image_bgr, text, (px - 26, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image_bgr, text, (px - 26, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_min_max(image_bgr: np.ndarray, thermal_celsius: np.ndarray, unit: str) -> None:
    height, width = thermal_celsius.shape
    border = 2
    if width > border * 2 and height > border * 2:
        roi = thermal_celsius[border : height - border, border : width - border]
        min_index = int(np.argmin(roi))
        max_index = int(np.argmax(roi))
        roi_w = roi.shape[1]
        min_y, min_x = divmod(min_index, roi_w)
        max_y, max_x = divmod(max_index, roi_w)
        min_y += border
        min_x += border
        max_y += border
        max_x += border
    else:
        min_index = int(np.argmin(thermal_celsius))
        max_index = int(np.argmax(thermal_celsius))
        min_y, min_x = divmod(min_index, width)
        max_y, max_x = divmod(max_index, width)

    scale_x = image_bgr.shape[1] / float(width)
    scale_y = image_bgr.shape[0] / float(height)
    min_px = int(min_x * scale_x)
    min_py = int(min_y * scale_y)
    max_px = int(max_x * scale_x)
    max_py = int(max_y * scale_y)

    cv2.drawMarker(image_bgr, (min_px, min_py), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
    cv2.drawMarker(image_bgr, (max_px, max_py), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    min_label = f"Min {format_temperature(float(thermal_celsius[min_y, min_x]), unit)}"
    max_label = f"Max {format_temperature(float(thermal_celsius[max_y, max_x]), unit)}"
    min_org = _choose_label_origin(image_bgr, min_px, min_py, min_label)
    max_org = _choose_label_origin(image_bgr, max_px, max_py, max_label)
    _draw_text_with_outline(image_bgr, min_label, min_org, (255, 220, 220))
    _draw_text_with_outline(image_bgr, max_label, max_org, (220, 220, 255))


def _draw_text_with_outline(image_bgr: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _choose_label_origin(image_bgr: np.ndarray, anchor_x: int, anchor_y: int, text: str) -> tuple[int, int]:
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    margin = 6
    offsets = [(8, -8), (8, 14), (-text_width - 8, -8), (-text_width - 8, 14)]
    img_h, img_w = image_bgr.shape[:2]
    for dx, dy in offsets:
        x = anchor_x + dx
        y = anchor_y + dy
        left = x
        right = x + text_width
        top = y - text_height
        bottom = y + baseline
        if left >= margin and right <= img_w - margin and top >= margin and bottom <= img_h - margin:
            return x, y
    x = min(max(anchor_x + 8, margin), max(margin, img_w - margin - text_width))
    y = min(max(anchor_y - 8, margin + text_height), max(margin + text_height, img_h - margin - baseline))
    return x, y


def _build_histogram_image(
    histogram: list[HistogramPoint],
    unit: str,
    color_map_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    hist_width = max(width, 220)
    hist_height = max(height, 160)
    canvas = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    bar_x0 = 46
    bar_x1 = 86
    curve_x0 = 96
    curve_x1 = canvas.shape[1] - 1

    gradient = np.linspace(255, 0, canvas.shape[0], dtype=np.uint8).reshape(canvas.shape[0], 1)
    gradient = np.repeat(gradient, bar_x1 - bar_x0, axis=1)
    gradient_color = render_thermal_image(
        gradient.astype(np.float32),
        color_map_name=color_map_name,
        manual_range_enabled=False,
        manual_min_temp=0.0,
        manual_max_temp=1.0,
    )
    canvas[:, bar_x0:bar_x1, :] = gradient_color

    if histogram:
        top_temp = _convert_temp(max(point.x for point in histogram), unit)
        bottom_temp = _convert_temp(min(point.x for point in histogram), unit)
        cv2.putText(canvas, f"{top_temp:.1f}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{bottom_temp:.1f}",
            (2, canvas.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    if len(histogram) >= 2:
        x_values = np.array([float(point.y) for point in histogram], dtype=np.float32)
        y_values = np.array([_convert_temp(float(point.x), unit) for point in histogram], dtype=np.float32)
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 0.1)
        points = []
        for x_value, y_value in zip(x_values, y_values):
            px = int(curve_x0 + (x_value - x_min) / x_span * (curve_x1 - curve_x0))
            py = int((1.0 - (y_value - y_min) / y_span) * (canvas.shape[0] - 1))
            points.append((px, py))
        smooth = _smooth_polyline(points, iterations=2)
        cv2.polylines(canvas, [np.array(smooth, dtype=np.int32)], False, (255, 180, 0), 2, cv2.LINE_AA)
    return canvas


def _build_history_image(history: list[TemperatureHistoryPoint], unit: str, width: int, height: int) -> np.ndarray:
    canvas_width = max(width, 960)
    canvas_height = max(height, 180)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    left_margin = 56
    right_margin = 96
    top_margin = 8
    bottom_margin = 24
    plot_x0 = left_margin
    plot_x1 = max(plot_x0 + 10, canvas.shape[1] - right_margin)
    plot_y0 = top_margin
    plot_y1 = max(plot_y0 + 10, canvas.shape[0] - bottom_margin)

    cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (65, 65, 65), 1)
    if len(history) < 2:
        return canvas

    y_min = _convert_temp(min(point.min_value for point in history), unit)
    y_max = _convert_temp(max(point.max_value for point in history), unit)
    y_span = max(y_max - y_min, 0.1)
    t_min = history[0].timestamp
    t_max = history[-1].timestamp
    t_span = max(t_max - t_min, 1e-3)

    for i in range(1, 4):
        gy = int(plot_y0 + i * (plot_y1 - plot_y0) / 4.0)
        cv2.line(canvas, (plot_x0, gy), (plot_x1, gy), (55, 55, 55), 1, cv2.LINE_AA)

    y_tick_count = 5
    for i in range(y_tick_count):
        ratio = i / float(y_tick_count - 1)
        gy = int(plot_y0 + ratio * (plot_y1 - plot_y0))
        tick_value = y_max - ratio * (y_max - y_min)
        cv2.putText(canvas, f"{tick_value:.0f}", (8, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (185, 185, 185), 1, cv2.LINE_AA)
        cv2.line(canvas, (plot_x0 - 4, gy), (plot_x0, gy), (120, 120, 120), 1, cv2.LINE_AA)

    if t_span <= 10:
        tick_step = 1
    elif t_span <= 30:
        tick_step = 2
    elif t_span <= 60:
        tick_step = 5
    else:
        tick_step = 10

    first_tick = int(np.ceil(t_min / tick_step) * tick_step)
    tick_ts = float(first_tick)
    while tick_ts <= t_max:
        ratio = (tick_ts - t_min) / t_span
        gx = int(plot_x0 + ratio * (plot_x1 - plot_x0))
        cv2.line(canvas, (gx, plot_y0), (gx, plot_y1), (50, 50, 50), 1, cv2.LINE_AA)
        tick_text = datetime.fromtimestamp(tick_ts).strftime("%H:%M:%S")
        cv2.putText(canvas, tick_text, (gx - 26, canvas.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (175, 175, 175), 1, cv2.LINE_AA)
        tick_ts += tick_step

    _draw_history_line(canvas, history, lambda p: p.max_value, (0, 0, 255), t_min, t_span, y_min, y_span, unit, plot_x0, plot_x1, plot_y0, plot_y1)
    _draw_history_line(canvas, history, lambda p: p.min_value, (255, 0, 0), t_min, t_span, y_min, y_span, unit, plot_x0, plot_x1, plot_y0, plot_y1)
    _draw_history_line(canvas, history, lambda p: p.average, (0, 255, 0), t_min, t_span, y_min, y_span, unit, plot_x0, plot_x1, plot_y0, plot_y1)
    _draw_history_line(canvas, history, lambda p: p.center, (0, 165, 255), t_min, t_span, y_min, y_span, unit, plot_x0, plot_x1, plot_y0, plot_y1)

    legend_items = [("Max", (0, 0, 255)), ("Min", (255, 0, 0)), ("Ave", (0, 255, 0)), ("Center", (0, 165, 255))]
    lx = plot_x1 + 12
    ly = 36
    for label, color in legend_items:
        cv2.circle(canvas, (lx, ly), 4, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (lx + 10, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        ly += 24
    return canvas


def _draw_history_line(
    canvas: np.ndarray,
    history: list[TemperatureHistoryPoint],
    value_getter,
    color: tuple[int, int, int],
    t_min: float,
    t_span: float,
    y_min: float,
    y_span: float,
    unit: str,
    plot_x0: int,
    plot_x1: int,
    plot_y0: int,
    plot_y1: int,
) -> None:
    points = []
    for point in history:
        x = int(plot_x0 + (point.timestamp - t_min) / t_span * (plot_x1 - plot_x0))
        y_value = _convert_temp(float(value_getter(point)), unit)
        y = int(plot_y0 + (1.0 - (y_value - y_min) / y_span) * (plot_y1 - plot_y0))
        points.append((x, y))
    smooth = _smooth_polyline(points, iterations=2)
    cv2.polylines(canvas, [np.array(smooth, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def _convert_temp(temp_celsius: float, unit: str) -> float:
    if unit == "F":
        return temp_celsius * 9.0 / 5.0 + 32.0
    return temp_celsius


def _smooth_polyline(points: list[tuple[int, int]], iterations: int = 2) -> list[tuple[float, float]]:
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in points]
    output = [(float(x), float(y)) for x, y in points]
    for _ in range(max(iterations, 0)):
        if len(output) < 3:
            break
        refined: list[tuple[float, float]] = [output[0]]
        for i in range(len(output) - 1):
            p0 = output[i]
            p1 = output[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            refined.extend([q, r])
        refined.append(output[-1])
        output = refined
    return output
