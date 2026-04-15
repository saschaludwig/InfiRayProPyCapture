"""Background worker for thermal frame processing and chart rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, QWaitCondition, Signal

from irpropycapture.core.image_processor import (
    apply_orientation,
    format_temperature_overlay,
    render_thermal_image,
)
from irpropycapture.core.perf import PerfReporter
from irpropycapture.core.temperature_processor import HistogramPoint, TemperatureHistoryPoint, TemperatureProcessor

# Lower bounds for chart rendering; keep aligned with MainWindow widget minimums.
MIN_HISTOGRAM_RENDER_HEIGHT = 120
MIN_HISTORY_RENDER_HEIGHT = 120


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
    camera_temperature_range: int
    unit: str
    preview_width: int
    preview_height: int
    histogram_width: int
    histogram_height: int
    history_width: int
    history_height: int
    frame_sequence_id: int


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
    timings_ms: dict[str, float]
    frame_sequence_id: int


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
        self._cached_hist_gradient: np.ndarray | None = None
        self._cached_hist_gradient_key: tuple[str, int, int] = ("", 0, 0)
        self._perf = PerfReporter("ProcessingWorker")

    def submit_frame(self, frame: np.ndarray, settings: ProcessingSettings) -> None:
        """Store only the latest frame/settings pair (latest-frame-wins)."""
        with QMutexLocker(self._mutex):
            self._latest_frame = frame
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
        total_start = time.perf_counter()
        decode_start = time.perf_counter()
        smooth_for_high_range = settings.camera_temperature_range == 1
        result = self._processor.get_temperatures(frame, smooth_for_high_range=smooth_for_high_range)
        decode_elapsed = time.perf_counter() - decode_start
        thermal = result.temperatures.reshape(192, 256)
        measurement_thermal = result.measurement_temperatures.reshape(192, 256)

        render_start = time.perf_counter()
        rendered = render_thermal_image(
            thermal,
            color_map_name=settings.color_map_name,
            manual_range_enabled=settings.manual_range_enabled,
            manual_min_temp=settings.manual_min_temp,
            manual_max_temp=settings.manual_max_temp,
            auto_min_temp=result.min_value,
            auto_max_temp=result.max_value,
        )
        rendered = apply_orientation(rendered, settings.orientation)
        render_elapsed = time.perf_counter() - render_start

        oriented_thermal = None
        if settings.show_grid or settings.show_min_max:
            oriented_thermal = apply_orientation(measurement_thermal, settings.orientation)

        overlay_start = time.perf_counter()
        export_bgr = _resize_export(rendered)
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
        overlay_elapsed = time.perf_counter() - overlay_start

        histogram_start = time.perf_counter()
        if settings.manual_range_enabled:
            histogram_min = settings.manual_min_temp
            histogram_max = max(settings.manual_max_temp, settings.manual_min_temp + 0.1)
        else:
            histogram_min = result.min_value
            histogram_max = result.max_value
        histogram_points = TemperatureProcessor.compute_histogram(
            result.measurement_temperatures,
            histogram_min,
            histogram_max,
            bins=max(2, len(result.histogram)),
        )
        gradient_color = self._get_histogram_gradient(
            color_map_name=settings.color_map_name,
            height=max(settings.histogram_height, MIN_HISTOGRAM_RENDER_HEIGHT),
            width=40,
        )
        histogram_bgr = _build_histogram_image(
            histogram_points,
            settings.unit,
            settings.histogram_width,
            settings.histogram_height,
            gradient_color,
        )
        histogram_elapsed = time.perf_counter() - histogram_start

        history_start = time.perf_counter()
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
        history_elapsed = time.perf_counter() - history_start

        convert_start = time.perf_counter()
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        histogram_rgb = cv2.cvtColor(histogram_bgr, cv2.COLOR_BGR2RGB)
        history_rgb = cv2.cvtColor(history_bgr, cv2.COLOR_BGR2RGB)
        convert_elapsed = time.perf_counter() - convert_start
        total_elapsed = time.perf_counter() - total_start
        timings_ms = {
            "decode": decode_elapsed * 1000.0,
            "render": render_elapsed * 1000.0,
            "overlay_resize": overlay_elapsed * 1000.0,
            "histogram": histogram_elapsed * 1000.0,
            "history": history_elapsed * 1000.0,
            "rgb_convert": convert_elapsed * 1000.0,
            "total": total_elapsed * 1000.0,
        }
        for stage, value_ms in timings_ms.items():
            self._perf.observe(stage, value_ms / 1000.0)
        return ProcessingResult(
            preview_rgb=preview_rgb,
            export_bgr=export_bgr,
            histogram_rgb=histogram_rgb,
            history_rgb=history_rgb,
            thermal_celsius=thermal,
            min_value=result.min_value,
            max_value=result.max_value,
            average=result.average,
            center=result.center,
            timings_ms=timings_ms,
            frame_sequence_id=settings.frame_sequence_id,
        )

    def _get_histogram_gradient(self, color_map_name: str, height: int, width: int) -> np.ndarray:
        key = (color_map_name, height, width)
        if self._cached_hist_gradient is not None and key == self._cached_hist_gradient_key:
            return self._cached_hist_gradient
        gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
        gradient = np.repeat(gradient, width, axis=1)
        gradient_color = render_thermal_image(
            gradient.astype(np.float32),
            color_map_name=color_map_name,
            manual_range_enabled=False,
            manual_min_temp=0.0,
            manual_max_temp=1.0,
            auto_min_temp=0.0,
            auto_max_temp=255.0,
        )
        self._cached_hist_gradient = gradient_color
        self._cached_hist_gradient_key = key
        return gradient_color


def _resize_preview(image_bgr: np.ndarray, width: int, height: int, interpolation_mode: str) -> np.ndarray:
    if width <= 0 or height <= 0:
        return image_bgr.copy()
    interpolation = cv2.INTER_NEAREST if interpolation_mode == "Fast" else cv2.INTER_LINEAR
    src_h, src_w = image_bgr.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    out_w = max(1, int(src_w * scale))
    out_h = max(1, int(src_h * scale))
    return cv2.resize(image_bgr, (out_w, out_h), interpolation=interpolation)


def _resize_export(image_bgr: np.ndarray) -> np.ndarray:
    src_h, src_w = image_bgr.shape[:2]
    if src_h > src_w:
        target = (768, 1024)
    else:
        target = (1024, 768)
    return cv2.resize(image_bgr, target, interpolation=cv2.INTER_NEAREST)


def append_export_color_scale(
    image_bgr: np.ndarray,
    color_map_name: str,
    min_temp_c: float,
    max_temp_c: float,
    unit: str,
) -> np.ndarray:
    """Append a labeled color scale on the long edge of the export image."""
    image_height, image_width = image_bgr.shape[:2]
    if image_height <= 0 or image_width <= 0:
        return image_bgr

    def _format_temp_label(temp_celsius: float) -> str:
        if unit == "F":
            temp_value = temp_celsius * 9.0 / 5.0 + 32.0
            return f"{temp_value:.1f} F"
        return f"{temp_celsius:.1f} C"

    border_px = 2
    separator_color = (0, 0, 0)
    label_color = (220, 220, 220)
    label_outline = (0, 0, 0)

    if image_height > image_width:
        scale_width = max(42, min(68, image_width // 12))
        gradient = np.linspace(255, 0, image_height, dtype=np.uint8).reshape(image_height, 1)
        gradient = np.repeat(gradient, scale_width, axis=1)
        color_scale = render_thermal_image(
            gradient.astype(np.float32),
            color_map_name=color_map_name,
            manual_range_enabled=False,
            manual_min_temp=0.0,
            manual_max_temp=1.0,
            auto_min_temp=0.0,
            auto_max_temp=255.0,
        )
        panel_width = border_px + scale_width
        panel = np.zeros((image_height, panel_width, 3), dtype=np.uint8)
        panel[:, :border_px, :] = separator_color
        bar_x0 = border_px
        bar_x1 = border_px + scale_width
        panel[:, bar_x0:bar_x1, :] = color_scale

        top_label = _format_temp_label(max_temp_c)
        bottom_label = _format_temp_label(min_temp_c)
        text_x = bar_x0 + 4
        cv2.putText(panel, top_label, (text_x, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_outline, 2, cv2.LINE_AA)
        cv2.putText(panel, top_label, (text_x, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
        bottom_y = max(20, image_height - 8)
        cv2.putText(
            panel,
            bottom_label,
            (text_x, bottom_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            label_outline,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            bottom_label,
            (text_x, bottom_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            label_color,
            1,
            cv2.LINE_AA,
        )
        return np.concatenate((image_bgr, panel), axis=1)

    scale_height = max(34, min(58, image_height // 10))
    gradient = np.linspace(0, 255, image_width, dtype=np.uint8).reshape(1, image_width)
    gradient = np.repeat(gradient, scale_height, axis=0)
    color_scale = render_thermal_image(
        gradient.astype(np.float32),
        color_map_name=color_map_name,
        manual_range_enabled=False,
        manual_min_temp=0.0,
        manual_max_temp=1.0,
        auto_min_temp=0.0,
        auto_max_temp=255.0,
    )
    panel_height = border_px + scale_height
    panel = np.zeros((panel_height, image_width, 3), dtype=np.uint8)
    panel[:border_px, :, :] = separator_color
    bar_y0 = border_px
    bar_y1 = border_px + scale_height
    panel[bar_y0:bar_y1, :, :] = color_scale
    min_label = _format_temp_label(min_temp_c)
    max_label = _format_temp_label(max_temp_c)
    text_y = max(bar_y0 + 16, bar_y1 - 8)
    cv2.putText(panel, min_label, (6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_outline, 2, cv2.LINE_AA)
    cv2.putText(panel, min_label, (6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
    (max_text_width, _), _ = cv2.getTextSize(max_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    max_text_x = max(6, image_width - max_text_width - 6)
    cv2.putText(
        panel,
        max_label,
        (max_text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        label_outline,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        max_label,
        (max_text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        label_color,
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((image_bgr, panel), axis=0)


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
            text = format_temperature_overlay(value, unit)
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

    min_label = f"Min {format_temperature_overlay(float(thermal_celsius[min_y, min_x]), unit)}"
    max_label = f"Max {format_temperature_overlay(float(thermal_celsius[max_y, max_x]), unit)}"
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
    width: int,
    height: int,
    gradient_color: np.ndarray,
) -> np.ndarray:
    hist_width = max(width, 220)
    hist_height = max(height, MIN_HISTOGRAM_RENDER_HEIGHT)
    canvas = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    bar_x0 = 46
    bar_x1 = 86
    curve_x0 = 96
    curve_x1 = canvas.shape[1] - 1

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
    canvas_height = max(height, MIN_HISTORY_RENDER_HEIGHT)
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
    plot_width = max(32, plot_x1 - plot_x0)
    sampled_history = _downsample_history(history, max_points=max(200, min(600, plot_width // 2)))
    t_min = sampled_history[0].timestamp
    t_max = sampled_history[-1].timestamp
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

    _draw_history_line(canvas, sampled_history, lambda p: p.max_value, (0, 0, 255), t_min, t_span, y_min, y_span, unit, plot_x0, plot_x1, plot_y0, plot_y1)
    _draw_history_line(canvas, sampled_history, lambda p: p.min_value, (255, 0, 0), t_min, t_span, y_min, y_span, unit, plot_x0, plot_x1, plot_y0, plot_y1)
    _draw_history_line(canvas, sampled_history, lambda p: p.average, (0, 255, 0), t_min, t_span, y_min, y_span, unit, plot_x0, plot_x1, plot_y0, plot_y1)
    _draw_history_line(canvas, sampled_history, lambda p: p.center, (0, 165, 255), t_min, t_span, y_min, y_span, unit, plot_x0, plot_x1, plot_y0, plot_y1)

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
    if len(points) > 700:
        smooth = [(float(x), float(y)) for x, y in points]
    else:
        iterations = 1 if len(points) > 300 else 2
        smooth = _smooth_polyline(points, iterations=iterations)
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


def _downsample_history(history: list[TemperatureHistoryPoint], max_points: int) -> list[TemperatureHistoryPoint]:
    if len(history) <= max_points or max_points < 2:
        return history
    step = len(history) / float(max_points - 1)
    sampled: list[TemperatureHistoryPoint] = []
    index = 0.0
    while int(index) < len(history) - 1:
        sampled.append(history[int(index)])
        index += step
    sampled.append(history[-1])
    return sampled
