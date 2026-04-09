"""Main window with capture, analysis and export tools."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from irpropycapture.core.camera_capture import OpenCVCaptureWorker, list_opencv_camera_devices, probe_opencv_source
from irpropycapture.core.image_processor import (
    AVAILABLE_COLOR_MAPS,
    apply_orientation,
    format_temperature,
    render_thermal_image,
)
from irpropycapture.core.image_saver import save_png
from irpropycapture.core.state import AppState, load_state, save_state
from irpropycapture.core.temperature_processor import HistogramPoint, TemperatureHistoryPoint, TemperatureProcessor
from irpropycapture.core.video_recorder import VideoRecorder


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IrProPyCapture")
        self.resize(1300, 820)

        self.capture_worker: OpenCVCaptureWorker | None = None
        self.processor = TemperatureProcessor()
        self.recorder = VideoRecorder()
        self.available_camera_items: list[tuple[str, int, str, int, int, float]] = []
        self.state: AppState = load_state()
        self.last_render_bgr: np.ndarray | None = None
        self.last_temps_celsius: np.ndarray | None = None
        self.last_history_points: list[TemperatureHistoryPoint] = []
        self.last_histogram_points: list[HistogramPoint] = []

        self.preview = QLabel("No image")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.info = QLabel("No data")
        self.info.setWordWrap(True)
        self.histogram_label = QLabel()
        self.history_label = QLabel()
        self.histogram_label.setMinimumHeight(160)
        self.histogram_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.history_label.setMinimumHeight(180)
        # Keep horizontal size flexible even after large pixmaps were rendered.
        # Otherwise QLabel size hints can prevent the window from shrinking again.
        self.history_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.history_label.setMinimumWidth(0)
        self.history_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.camera_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh Cameras")
        self.start_button = QPushButton("Start Camera")
        self.snapshot_button = QPushButton("Save PNG")
        self.record_button = QPushButton("Start Recording")

        self.color_map_combo = QComboBox()
        self.color_map_combo.addItems(AVAILABLE_COLOR_MAPS)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["C", "F"])
        self.preview_interpolation_combo = QComboBox()
        self.preview_interpolation_combo.addItems(["Fast", "Smooth"])
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Normal", "Rotate Left", "Rotate Right", "Flip Horizontal", "Flip Vertical"])
        self.grid_checkbox = QCheckBox("Show temperature grid")
        self.min_max_checkbox = QCheckBox("Show min/max markers")
        self.grid_density_combo = QComboBox()
        self.grid_density_combo.addItems(["Low", "Medium", "High"])

        self.manual_range_checkbox = QCheckBox("Manual range")
        self.min_spin = QSpinBox()
        self.max_spin = QSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setRange(-50, 300)
            spin.setSingleStep(1)

        self._restore_state_to_controls()

        top = QHBoxLayout()
        top.addWidget(self.camera_combo, stretch=1)
        top.addWidget(self.refresh_button)
        top.addWidget(self.start_button)
        top.addWidget(self.snapshot_button)
        top.addWidget(self.record_button)

        controls_box = QGroupBox("Controls")
        controls_form = QFormLayout()
        controls_form.addRow("Color map", self.color_map_combo)
        controls_form.addRow("Temperature unit", self.unit_combo)
        controls_form.addRow("Preview interpolation", self.preview_interpolation_combo)
        controls_form.addRow("Orientation", self.orientation_combo)
        controls_form.addRow(self.grid_checkbox)
        controls_form.addRow(self.min_max_checkbox)
        controls_form.addRow("Grid density", self.grid_density_combo)
        controls_form.addRow(self.manual_range_checkbox)
        controls_form.addRow("Manual min", self.min_spin)
        controls_form.addRow("Manual max", self.max_spin)
        controls_box.setLayout(controls_form)

        charts_box = QGroupBox("Color scale + Histogram")
        charts_layout = QVBoxLayout()
        charts_layout.addWidget(self.histogram_label)
        charts_box.setLayout(charts_layout)

        side = QVBoxLayout()
        side.addWidget(charts_box, stretch=1)
        side.addWidget(controls_box)
        side.addWidget(self.info)

        body = QHBoxLayout()
        body.addWidget(self.preview, stretch=3)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMaximumWidth(340)
        body.addWidget(side_widget, stretch=1)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addLayout(body, stretch=1)
        root.addWidget(QLabel("Temperature history"))
        root.addWidget(self.history_label)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.refresh_button.clicked.connect(self.refresh_camera_list)
        self.start_button.clicked.connect(self.toggle_capture)
        self.snapshot_button.clicked.connect(self.save_snapshot)
        self.record_button.clicked.connect(self.toggle_recording)
        self.color_map_combo.currentTextChanged.connect(self._persist_state)
        self.unit_combo.currentTextChanged.connect(self._persist_state)
        self.preview_interpolation_combo.currentTextChanged.connect(self._persist_state)
        self.orientation_combo.currentTextChanged.connect(self._persist_state)
        self.grid_checkbox.toggled.connect(self._persist_state)
        self.min_max_checkbox.toggled.connect(self._persist_state)
        self.grid_density_combo.currentTextChanged.connect(self._persist_state)
        self.manual_range_checkbox.toggled.connect(self._persist_state)
        self.min_spin.valueChanged.connect(self._persist_state)
        self.max_spin.valueChanged.connect(self._persist_state)

        self.refresh_camera_list()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.capture_worker is not None:
            self.capture_worker.stop()
            self.capture_worker = None
        if self.recorder.is_recording:
            self.recorder.stop()
        self._persist_state()
        super().closeEvent(event)

    def _restore_state_to_controls(self) -> None:
        self.color_map_combo.setCurrentText(self.state.color_map)
        self.unit_combo.setCurrentText(self.state.temperature_format)
        self.preview_interpolation_combo.setCurrentText(self.state.preview_interpolation)
        self.grid_checkbox.setChecked(self.state.show_temperature_grid)
        self.min_max_checkbox.setChecked(self.state.show_min_max_markers)
        self.orientation_combo.setCurrentText(self.state.orientation)
        self.grid_density_combo.setCurrentText(self.state.grid_density)
        self.manual_range_checkbox.setChecked(self.state.manual_range_enabled)
        self.min_spin.setValue(int(self.state.manual_min_temp))
        self.max_spin.setValue(int(self.state.manual_max_temp))

    def _persist_state(self) -> None:
        self.state.color_map = self.color_map_combo.currentText()
        self.state.temperature_format = self.unit_combo.currentText()
        self.state.preview_interpolation = self.preview_interpolation_combo.currentText()
        self.state.show_temperature_grid = self.grid_checkbox.isChecked()
        self.state.show_min_max_markers = self.min_max_checkbox.isChecked()
        self.state.orientation = self.orientation_combo.currentText()
        self.state.grid_density = self.grid_density_combo.currentText()
        self.state.manual_range_enabled = self.manual_range_checkbox.isChecked()
        self.state.manual_min_temp = float(self.min_spin.value())
        self.state.manual_max_temp = float(self.max_spin.value())
        pos = self.camera_combo.currentIndex()
        if 0 <= pos < len(self.available_camera_items):
            _, idx, name, _, _, _ = self.available_camera_items[pos]
            self.state.camera_index = idx
            self.state.camera_name = name
        save_state(self.state)

    def refresh_camera_list(self) -> None:
        self.camera_combo.clear()
        self.available_camera_items.clear()
        devices = list_opencv_camera_devices(
            width=256,
            height=384,
            fps=25,
        )
        for idx, name in devices:
            label = f"Camera {idx}: {name} [256x384@25]"
            self.camera_combo.addItem(label)
            self.available_camera_items.append(("opencv", idx, name, 256, 384, 25.0))
        if not self.available_camera_items:
            self.camera_combo.addItem("No compatible thermal camera found")
        for pos, item in enumerate(self.available_camera_items):
            _, idx, name, _, _, _ = item
            if idx == self.state.camera_index or name == self.state.camera_name:
                self.camera_combo.setCurrentIndex(pos)
                break

    def selected_camera_item(self) -> tuple[str, int, str, int, int, float]:
        pos = self.camera_combo.currentIndex()
        if 0 <= pos < len(self.available_camera_items):
            return self.available_camera_items[pos]
        return ("opencv", 0, "Unknown", 256, 384, 25.0)

    def toggle_capture(self) -> None:
        if self.capture_worker is not None:
            self.capture_worker.stop()
            self.capture_worker = None
            self.start_button.setText("Start Camera")
            return

        if not self.available_camera_items:
            QMessageBox.information(
                self,
                "Camera",
                "No compatible thermal camera found.\nExpected mode: 256x384 raw-compatible stream.",
            )
            return

        _, index, _, width, height, fps = self.selected_camera_item()
        ok, probe_msg = probe_opencv_source(index, width, height, int(fps))
        if not ok:
            QMessageBox.critical(self, "Camera Error", probe_msg)
            return

        self.capture_worker = OpenCVCaptureWorker(index, width, height, int(fps))
        self.capture_worker.frame_ready.connect(self.on_frame_ready)
        self.capture_worker.error.connect(lambda msg: QMessageBox.critical(self, "Capture Error", msg))
        self.capture_worker.start()
        self.start_button.setText("Stop Camera")
        self._persist_state()

    def on_frame_ready(self, frame: np.ndarray) -> None:
        try:
            result = self.processor.get_temperatures(frame)
            thermal = result.temperatures.reshape(192, 256)
            self.last_temps_celsius = thermal

            rendered = render_thermal_image(
                thermal,
                color_map_name=self.color_map_combo.currentText(),
                manual_range_enabled=self.manual_range_checkbox.isChecked(),
                manual_min_temp=float(self.min_spin.value()),
                manual_max_temp=float(self.max_spin.value()),
            )
            rendered = cv2.resize(rendered, (1024, 768), interpolation=cv2.INTER_NEAREST)
            orientation = self.orientation_combo.currentText()
            rendered = apply_orientation(rendered, orientation)

            oriented_thermal = None
            if self.grid_checkbox.isChecked() or self.min_max_checkbox.isChecked():
                # Keep grid/min-max labels at fixed screen size by drawing them
                # after preview scaling in _set_preview_image.
                oriented_thermal = apply_orientation(thermal, orientation)

            # Build an export/record frame that includes temperature overlays.
            rendered_with_overlays = rendered.copy()
            if oriented_thermal is not None:
                if self.grid_checkbox.isChecked():
                    self._draw_grid_fixed_screen_size(rendered_with_overlays, oriented_thermal)
                if self.min_max_checkbox.isChecked():
                    self._draw_min_max_fixed_screen_size(rendered_with_overlays, oriented_thermal)

            self.last_render_bgr = rendered_with_overlays
            self._set_preview_image(
                rendered,
                overlay_thermal=oriented_thermal if self.grid_checkbox.isChecked() else None,
                overlay_min_max=oriented_thermal if self.min_max_checkbox.isChecked() else None,
            )
            self._update_stats(result.min_value, result.max_value, result.average, result.center)
            self._update_histogram(result.histogram)
            self._update_history(result.temperature_history)
            if self.recorder.is_recording:
                self.recorder.write_frame(rendered_with_overlays)
        except Exception as exc:
            self.preview.setText(f"Decode error: {exc}")

    def _set_preview_image(
        self,
        image_bgr: np.ndarray,
        overlay_thermal: np.ndarray | None = None,
        overlay_min_max: np.ndarray | None = None,
    ) -> None:
        display_bgr = image_bgr
        target_size = self.preview.size()
        if target_size.width() > 0 and target_size.height() > 0:
            interpolation = (
                cv2.INTER_NEAREST
                if self.preview_interpolation_combo.currentText() == "Fast"
                else cv2.INTER_LINEAR
            )
            src_h, src_w = image_bgr.shape[:2]
            scale = min(target_size.width() / float(src_w), target_size.height() / float(src_h))
            out_w = max(1, int(src_w * scale))
            out_h = max(1, int(src_h * scale))
            display_bgr = cv2.resize(image_bgr, (out_w, out_h), interpolation=interpolation)

        if overlay_thermal is not None:
            self._draw_grid_fixed_screen_size(display_bgr, overlay_thermal)
        if overlay_min_max is not None:
            self._draw_min_max_fixed_screen_size(display_bgr, overlay_min_max)

        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.preview.setPixmap(pixmap)

    def _draw_grid_fixed_screen_size(self, image_bgr: np.ndarray, thermal_celsius: np.ndarray) -> None:
        if self.grid_density_combo.currentText() == "Low":
            step_x, step_y = 64, 48
        elif self.grid_density_combo.currentText() == "High":
            step_x, step_y = 24, 18
        else:
            step_x, step_y = 32, 24

        h, w = thermal_celsius.shape
        scale_x = image_bgr.shape[1] / float(w)
        scale_y = image_bgr.shape[0] / float(h)
        font_scale = 0.38
        for y in range(step_y // 2, h, step_y):
            for x in range(step_x // 2, w, step_x):
                value = float(thermal_celsius[y, x])
                text = format_temperature(value, self.unit_combo.currentText())
                px = int(x * scale_x)
                py = int(y * scale_y)
                cv2.putText(image_bgr, text, (px - 26, py), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, text, (px - 26, py), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_min_max_fixed_screen_size(self, image_bgr: np.ndarray, thermal_celsius: np.ndarray) -> None:
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

        min_label = f"Min {format_temperature(float(thermal_celsius[min_y, min_x]), self.unit_combo.currentText())}"
        max_label = f"Max {format_temperature(float(thermal_celsius[max_y, max_x]), self.unit_combo.currentText())}"
        min_org = self._choose_label_origin(image_bgr, min_px, min_py, min_label)
        max_org = self._choose_label_origin(image_bgr, max_px, max_py, max_label)
        self._draw_text_with_outline(image_bgr, min_label, min_org, (255, 220, 220))
        self._draw_text_with_outline(image_bgr, max_label, max_org, (220, 220, 255))

    def _update_stats(self, min_c: float, max_c: float, avg_c: float, center_c: float) -> None:
        self.info.setText(
            f"Min {format_temperature(min_c, self.unit_combo.currentText())} | "
            f"Max {format_temperature(max_c, self.unit_combo.currentText())} | "
            f"Avg {format_temperature(avg_c, self.unit_combo.currentText())} | "
            f"Center {format_temperature(center_c, self.unit_combo.currentText())}"
        )

    def _update_histogram(self, histogram: list[HistogramPoint]) -> None:
        self.last_histogram_points = list(histogram)
        hist_width = max(self.histogram_label.width(), 220)
        hist_height = max(self.histogram_label.height(), 160)
        canvas = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        bar_x0 = 46
        bar_x1 = 86
        curve_x0 = 96
        curve_x1 = canvas.shape[1] - 1

        # Draw vertical color scale using currently selected colormap.
        gradient = np.linspace(255, 0, canvas.shape[0], dtype=np.uint8).reshape(canvas.shape[0], 1)
        gradient = np.repeat(gradient, bar_x1 - bar_x0, axis=1)
        gradient_color = render_thermal_image(
            gradient.astype(np.float32),
            color_map_name=self.color_map_combo.currentText(),
            manual_range_enabled=False,
            manual_min_temp=0.0,
            manual_max_temp=1.0,
        )
        canvas[:, bar_x0:bar_x1, :] = gradient_color

        if histogram:
            top_temp = self._convert_temp(max(point.x for point in histogram))
            bottom_temp = self._convert_temp(min(point.x for point in histogram))
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
            y_values = np.array([self._convert_temp(float(point.x)) for point in histogram], dtype=np.float32)
            x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
            y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
            x_span = max(x_max - x_min, 1.0)
            y_span = max(y_max - y_min, 0.1)
            points = []
            for x_value, y_value in zip(x_values, y_values):
                px = int(curve_x0 + (x_value - x_min) / x_span * (curve_x1 - curve_x0))
                py = int((1.0 - (y_value - y_min) / y_span) * (canvas.shape[0] - 1))
                points.append((px, py))
            smooth = self._smooth_polyline(points, iterations=2)
            cv2.polylines(canvas, [np.array(smooth, dtype=np.int32)], False, (255, 180, 0), 2, cv2.LINE_AA)
        self.histogram_label.setPixmap(self._pixmap_from_bgr(canvas))

    def _update_history(self, history: list[TemperatureHistoryPoint]) -> None:
        self.last_history_points = list(history)
        canvas_width = max(self.history_label.width(), 960)
        canvas = np.zeros((180, canvas_width, 3), dtype=np.uint8)
        left_margin = 56
        right_margin = 96
        top_margin = 8
        bottom_margin = 24
        plot_x0 = left_margin
        plot_x1 = max(plot_x0 + 10, canvas.shape[1] - right_margin)
        plot_y0 = top_margin
        plot_y1 = max(plot_y0 + 10, canvas.shape[0] - bottom_margin)

        cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (65, 65, 65), 1)
        if len(history) >= 2:
            y_min = self._convert_temp(min(point.min_value for point in history))
            y_max = self._convert_temp(max(point.max_value for point in history))
            y_span = max(y_max - y_min, 0.1)
            t_min = history[0].timestamp
            t_max = history[-1].timestamp
            t_span = max(t_max - t_min, 1e-3)

            # Horizontal and vertical grid for better temporal readability.
            for i in range(1, 4):
                gy = int(plot_y0 + i * (plot_y1 - plot_y0) / 4.0)
                cv2.line(canvas, (plot_x0, gy), (plot_x1, gy), (55, 55, 55), 1, cv2.LINE_AA)

            # Left-side temperature axis labels.
            y_tick_count = 5
            for i in range(y_tick_count):
                ratio = i / float(y_tick_count - 1)
                gy = int(plot_y0 + ratio * (plot_y1 - plot_y0))
                tick_value = y_max - ratio * (y_max - y_min)
                cv2.putText(
                    canvas,
                    f"{tick_value:.0f}",
                    (8, gy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (185, 185, 185),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(canvas, (plot_x0 - 4, gy), (plot_x0, gy), (120, 120, 120), 1, cv2.LINE_AA)

            # Swift Charts uses a real Date axis; emulate it with fixed-time ticks.
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
                cv2.putText(
                    canvas,
                    tick_text,
                    (gx - 26, canvas.shape[0] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (175, 175, 175),
                    1,
                    cv2.LINE_AA,
                )
                tick_ts += tick_step

            self._draw_history_line(
                canvas,
                history,
                lambda p: p.max_value,
                (0, 0, 255),
                t_min,
                t_span,
                y_min,
                y_span,
                plot_x0,
                plot_x1,
                plot_y0,
                plot_y1,
            )
            self._draw_history_line(
                canvas,
                history,
                lambda p: p.min_value,
                (255, 0, 0),
                t_min,
                t_span,
                y_min,
                y_span,
                plot_x0,
                plot_x1,
                plot_y0,
                plot_y1,
            )
            self._draw_history_line(
                canvas,
                history,
                lambda p: p.average,
                (0, 255, 0),
                t_min,
                t_span,
                y_min,
                y_span,
                plot_x0,
                plot_x1,
                plot_y0,
                plot_y1,
            )
            self._draw_history_line(
                canvas,
                history,
                lambda p: p.center,
                (0, 165, 255),
                t_min,
                t_span,
                y_min,
                y_span,
                plot_x0,
                plot_x1,
                plot_y0,
                plot_y1,
            )

            # Right-side legend matching Swift semantics.
            legend_items = [
                ("Max", (0, 0, 255)),
                ("Min", (255, 0, 0)),
                ("Ave", (0, 255, 0)),
                ("Center", (0, 165, 255)),
            ]
            lx = plot_x1 + 12
            ly = 36
            for label, color in legend_items:
                cv2.circle(canvas, (lx, ly), 4, color, -1, cv2.LINE_AA)
                cv2.putText(canvas, label, (lx + 10, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
                ly += 24
        self.history_label.setPixmap(self._pixmap_from_bgr(canvas))

    @staticmethod
    def _draw_text_with_outline(image_bgr: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
        cv2.putText(image_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _choose_label_origin(self, image_bgr: np.ndarray, anchor_x: int, anchor_y: int, text: str) -> tuple[int, int]:
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        margin = 6
        offsets = [
            (8, -8),   # top-right
            (8, 14),   # bottom-right
            (-text_width - 8, -8),  # top-left
            (-text_width - 8, 14),  # bottom-left
        ]
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

        # Fallback: clamp into image bounds.
        x = min(max(anchor_x + 8, margin), max(margin, img_w - margin - text_width))
        y = min(max(anchor_y - 8, margin + text_height), max(margin + text_height, img_h - margin - baseline))
        return x, y

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.last_render_bgr is not None:
            self._set_preview_image(self.last_render_bgr)
        if self.last_histogram_points:
            self._update_histogram(self.last_histogram_points)
        if self.last_history_points:
            self._update_history(self.last_history_points)

    def _draw_history_line(
        self,
        canvas: np.ndarray,
        history: list[TemperatureHistoryPoint],
        value_getter,
        color: tuple[int, int, int],
        t_min: float,
        t_span: float,
        y_min: float,
        y_span: float,
        plot_x0: int,
        plot_x1: int,
        plot_y0: int,
        plot_y1: int,
    ) -> None:
        points = []
        for point in history:
            x = int(plot_x0 + (point.timestamp - t_min) / t_span * (plot_x1 - plot_x0))
            y_value = self._convert_temp(float(value_getter(point)))
            y = int(plot_y0 + (1.0 - (y_value - y_min) / y_span) * (plot_y1 - plot_y0))
            points.append((x, y))
        smooth = self._smooth_polyline(points, iterations=2)
        cv2.polylines(canvas, [np.array(smooth, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    def _convert_temp(self, temp_celsius: float) -> float:
        if self.unit_combo.currentText() == "F":
            return temp_celsius * 9.0 / 5.0 + 32.0
        return temp_celsius

    @staticmethod
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

    @staticmethod
    def _pixmap_from_bgr(image_bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def save_snapshot(self) -> None:
        if self.last_render_bgr is None:
            QMessageBox.information(self, "Save PNG", "No frame available.")
            return
        default_name = datetime.now().strftime("thermal_%Y%m%d_%H%M%S.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", default_name, "PNG Image (*.png)")
        if not path:
            return
        ok = save_png(Path(path), self.last_render_bgr)
        if not ok:
            QMessageBox.critical(self, "Save PNG", "Failed to save image.")

    def toggle_recording(self) -> None:
        if self.recorder.is_recording:
            self.recorder.stop()
            self.record_button.setText("Start Recording")
            return
        if self.last_render_bgr is None:
            QMessageBox.information(self, "Record MP4", "Start the stream first.")
            return
        default_name = datetime.now().strftime("thermal_%Y%m%d_%H%M%S.mp4")
        path, _ = QFileDialog.getSaveFileName(self, "Save MP4", default_name, "MP4 Video (*.mp4)")
        if not path:
            return
        h, w, _ = self.last_render_bgr.shape
        if not self.recorder.start(Path(path), (w, h), fps=25.0):
            QMessageBox.critical(self, "Record MP4", "Failed to start recorder.")
            return
        self.record_button.setText("Stop Recording")
