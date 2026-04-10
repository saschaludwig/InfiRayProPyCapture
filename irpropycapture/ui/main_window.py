"""Main window with capture, analysis and export tools."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QFontDatabase, QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
from irpropycapture.core.frame_processing_worker import (
    MIN_HISTOGRAM_RENDER_HEIGHT,
    MIN_HISTORY_RENDER_HEIGHT,
    ProcessingResult,
    ProcessingSettings,
    ProcessingWorker,
)
from irpropycapture.core.image_processor import AVAILABLE_COLOR_MAPS, format_temperature_ui
from irpropycapture.core.perf import PerfReporter
from irpropycapture.core.image_saver import save_png
from irpropycapture.core.state import AppState, load_state, save_state
from irpropycapture.core.video_recorder import VideoRecorder

# Layout caps: histogram chrome + plot; temperature history strip under the main row.
MAX_CHARTS_BOX_HEIGHT_PX = 250
MAX_HISTORY_WIDGET_HEIGHT_PX = 200


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IrProPyCapture")
        self.state: AppState = load_state()

        self.capture_worker: OpenCVCaptureWorker | None = None
        self.processing_worker: ProcessingWorker | None = None
        self._gui_busy = False
        self.recorder = VideoRecorder()
        self.available_camera_items: list[tuple[str, int, str, int, int, float]] = []
        self.last_render_bgr: np.ndarray | None = None
        self.last_temps_celsius: np.ndarray | None = None
        self._last_preview_rgb: np.ndarray | None = None
        self._last_histogram_rgb: np.ndarray | None = None
        self._last_history_rgb: np.ndarray | None = None
        self._perf = PerfReporter("MainWindow")
        self._state_save_timer = QTimer(self)
        self._state_save_timer.setSingleShot(True)
        self._state_save_timer.timeout.connect(self._persist_state)

        self.preview = QLabel("No image")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 200)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.info = QLabel("No data")
        self.info.setWordWrap(True)
        self.histogram_label = QLabel()
        self.history_label = QLabel()
        self.histogram_label.setMinimumHeight(MIN_HISTOGRAM_RENDER_HEIGHT)
        self.histogram_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.history_label.setMinimumHeight(MIN_HISTORY_RENDER_HEIGHT)
        self.history_label.setMaximumHeight(MAX_HISTORY_WIDGET_HEIGHT_PX)
        # Horizontal: flexible so wide history pixmaps do not block shrinking the window.
        self.history_label.setMinimumWidth(0)
        # Vertical: share a little extra root height (capped); most growth stays on the preview row.
        self.history_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.MinimumExpanding,
        )
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
        controls_form.setVerticalSpacing(2)
        controls_form.setHorizontalSpacing(6)
        controls_form.setContentsMargins(8, 4, 8, 4)
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

        stats_box = QGroupBox("Temperature Stats")
        stats_layout = QGridLayout()
        stats_layout.setVerticalSpacing(0)
        stats_layout.setHorizontalSpacing(8)
        stats_layout.setContentsMargins(8, 2, 8, 4)
        self.min_value_label = QLabel("--")
        self.max_value_label = QLabel("--")
        self.avg_value_label = QLabel("--")
        self.center_value_label = QLabel("--")
        fixed_value_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        ui_point_size = self.font().pointSizeF()
        if ui_point_size > 0:
            fixed_value_font.setPointSizeF(ui_point_size)
        for value_label in (
            self.min_value_label,
            self.max_value_label,
            self.avg_value_label,
            self.center_value_label,
        ):
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setFont(fixed_value_font)
            value_label.setMinimumWidth(96)
        stats_layout.addWidget(QLabel("Min"), 0, 0)
        stats_layout.addWidget(self.min_value_label, 0, 1)
        stats_layout.addWidget(QLabel("Max"), 1, 0)
        stats_layout.addWidget(self.max_value_label, 1, 1)
        stats_layout.addWidget(QLabel("Avg"), 2, 0)
        stats_layout.addWidget(self.avg_value_label, 2, 1)
        stats_layout.addWidget(QLabel("Center"), 3, 0)
        stats_layout.addWidget(self.center_value_label, 3, 1)
        stats_box.setLayout(stats_layout)

        charts_box = QGroupBox("Color scale + Histogram")
        charts_box.setMaximumHeight(MAX_CHARTS_BOX_HEIGHT_PX)
        charts_layout = QVBoxLayout()
        charts_layout.setContentsMargins(4, 4, 4, 4)
        charts_layout.addWidget(self.histogram_label)
        charts_box.setLayout(charts_layout)

        # Stats directly under the histogram so Min/Max/Avg/Center stay visible on short displays.
        side = QVBoxLayout()
        side.setSpacing(4)
        # Grow the histogram area with the sidebar height, but never above MAX_CHARTS_BOX_HEIGHT_PX.
        side.addWidget(charts_box, stretch=1)
        side.addWidget(stats_box, stretch=0)
        side.addWidget(controls_box, stretch=0)
        side.addStretch(1)

        body = QHBoxLayout()
        body.addWidget(self.preview, stretch=3)
        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setMaximumWidth(340)
        body.addWidget(side_widget, stretch=1)

        history_title = QLabel("Temperature history")

        root = QVBoxLayout()
        root.setSpacing(4)
        root.addLayout(top)
        # Most vertical growth goes to the preview row; history gets a smaller share (max 200px).
        root.addLayout(body, stretch=5)
        root.addWidget(history_title, stretch=0)
        root.addWidget(self.history_label, stretch=1)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self.refresh_button.clicked.connect(self.refresh_camera_list)
        self.start_button.clicked.connect(self.toggle_capture)
        self.snapshot_button.clicked.connect(self.save_snapshot)
        self.record_button.clicked.connect(self.toggle_recording)
        self.color_map_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.unit_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.preview_interpolation_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.orientation_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.grid_checkbox.toggled.connect(self._schedule_state_persist)
        self.min_max_checkbox.toggled.connect(self._schedule_state_persist)
        self.grid_density_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.manual_range_checkbox.toggled.connect(self._schedule_state_persist)
        self.min_spin.valueChanged.connect(self._schedule_state_persist)
        self.max_spin.valueChanged.connect(self._schedule_state_persist)

        self.refresh_camera_list()

        self._apply_initial_window_geometry()

    def _apply_initial_window_geometry(self) -> None:
        """Resize and place the window from saved state if it fits the available desktop."""
        min_hint = self.minimumSizeHint()
        min_w = max(min_hint.width(), 320)
        min_h = max(min_hint.height(), 200)

        sw, sh = self.state.window_width, self.state.window_height
        sx, sy = self.state.window_x, self.state.window_y

        if sw <= 0 or sh <= 0:
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                self.resize(1260, 780)
                return
            ag = screen.availableGeometry()
            self.resize(min(1300, ag.width()), min(820, max(min_h, ag.height() - 48)))
            frame = self.frameGeometry()
            frame.moveCenter(ag.center())
            self.move(frame.topLeft())
            return

        screen = None
        if sx >= 0 and sy >= 0:
            screen = QGuiApplication.screenAt(QPoint(sx, sy))
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1260, 780)
            return

        ag = screen.availableGeometry()
        fw = max(min_w, min(sw, ag.width()))
        fh = max(min_h, min(sh, ag.height()))
        self.resize(fw, fh)

        if sx >= 0 and sy >= 0:
            self.move(sx, sy)
        else:
            frame = self.frameGeometry()
            frame.moveCenter(ag.center())
            self.move(frame.topLeft())

        self._ensure_window_fits_available_geometry()

    def _ensure_window_fits_available_geometry(self) -> None:
        """Nudge and shrink the window so it lies fully inside one screen's available rectangle."""
        screen = QGuiApplication.screenAt(self.frameGeometry().center()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        ag = screen.availableGeometry()
        frame = self.frameGeometry()

        fw = min(frame.width(), ag.width())
        fh = min(frame.height(), ag.height())
        fw = max(fw, self.minimumWidth())
        fh = max(fh, self.minimumHeight())
        if fw != frame.width() or fh != frame.height():
            self.resize(fw, fh)
            frame = self.frameGeometry()

        nx, ny = frame.x(), frame.y()
        if frame.right() > ag.right():
            nx = ag.right() - frame.width()
        if frame.bottom() > ag.bottom():
            ny = ag.bottom() - frame.height()
        if nx < ag.left():
            nx = ag.left()
        if ny < ag.top():
            ny = ag.top()
        if (nx, ny) != (frame.x(), frame.y()):
            self.move(nx, ny)

    def _persist_window_geometry_to_state(self) -> None:
        frame = self.frameGeometry()
        self.state.window_width = frame.width()
        self.state.window_height = frame.height()
        self.state.window_x = frame.x()
        self.state.window_y = frame.y()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_workers()
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
        self._persist_window_geometry_to_state()
        save_state(self.state)

    def _schedule_state_persist(self) -> None:
        self._state_save_timer.start(250)

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
            self._stop_workers()
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

        self.processing_worker = ProcessingWorker()
        self.processing_worker.processed.connect(self._on_processed_frame)
        self.processing_worker.error.connect(lambda msg: self.preview.setText(msg))
        self.processing_worker.start()

        self.capture_worker = OpenCVCaptureWorker(index, width, height, int(fps))
        self.capture_worker.frame_ready.connect(self.on_frame_ready)
        self.capture_worker.error.connect(lambda msg: QMessageBox.critical(self, "Capture Error", msg))
        self.capture_worker.start()
        self.start_button.setText("Stop Camera")
        self._persist_state()

    def on_frame_ready(self, frame: np.ndarray) -> None:
        if self.processing_worker is None:
            return
        settings = self._current_processing_settings()
        self.processing_worker.submit_frame(frame, settings)

    def _current_processing_settings(self) -> ProcessingSettings:
        target_size = self.preview.size()
        return ProcessingSettings(
            color_map_name=self.color_map_combo.currentText(),
            manual_range_enabled=self.manual_range_checkbox.isChecked(),
            manual_min_temp=float(self.min_spin.value()),
            manual_max_temp=float(self.max_spin.value()),
            preview_interpolation=self.preview_interpolation_combo.currentText(),
            orientation=self.orientation_combo.currentText(),
            show_grid=self.grid_checkbox.isChecked(),
            show_min_max=self.min_max_checkbox.isChecked(),
            grid_density=self.grid_density_combo.currentText(),
            unit=self.unit_combo.currentText(),
            preview_width=target_size.width(),
            preview_height=target_size.height(),
            histogram_width=max(self.histogram_label.width(), 220),
            histogram_height=max(self.histogram_label.height(), MIN_HISTOGRAM_RENDER_HEIGHT),
            history_width=max(self.history_label.width(), 960),
            history_height=max(
                min(self.history_label.height(), MAX_HISTORY_WIDGET_HEIGHT_PX),
                MIN_HISTORY_RENDER_HEIGHT,
            ),
        )

    def _on_processed_frame(self, result_obj: object) -> None:
        if self._gui_busy:
            return
        result = result_obj
        if not isinstance(result, ProcessingResult):
            return
        ui_start = time.perf_counter()
        self._gui_busy = True
        try:
            self.last_render_bgr = result.export_bgr
            self.last_temps_celsius = result.thermal_celsius
            self._last_preview_rgb = result.preview_rgb
            self._last_histogram_rgb = result.histogram_rgb
            self._last_history_rgb = result.history_rgb
            self.preview.setPixmap(self._pixmap_from_rgb(result.preview_rgb))
            self.histogram_label.setPixmap(self._pixmap_from_rgb(result.histogram_rgb))
            self.history_label.setPixmap(self._pixmap_from_rgb(result.history_rgb))
            self._update_stats(result.min_value, result.max_value, result.average, result.center)
            if self.recorder.is_recording:
                write_start = time.perf_counter()
                self.recorder.write_frame(result.export_bgr)
                self._perf.observe("record_enqueue", time.perf_counter() - write_start)
            self._perf.observe("pipeline_total", result.timings_ms.get("total", 0.0) / 1000.0)
        finally:
            self._gui_busy = False
            self._perf.observe("ui_present", time.perf_counter() - ui_start)

    def _stop_workers(self) -> None:
        if self.capture_worker is not None:
            self.capture_worker.stop()
            self.capture_worker = None
        if self.processing_worker is not None:
            self.processing_worker.stop()
            self.processing_worker = None

    def _update_stats(self, min_c: float, max_c: float, avg_c: float, center_c: float) -> None:
        unit = self.unit_combo.currentText()
        self.min_value_label.setText(format_temperature_ui(min_c, unit))
        self.max_value_label.setText(format_temperature_ui(max_c, unit))
        self.avg_value_label.setText(format_temperature_ui(avg_c, unit))
        self.center_value_label.setText(format_temperature_ui(center_c, unit))

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        self._schedule_state_persist()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._schedule_state_persist()
        if self._last_preview_rgb is not None:
            self.preview.setPixmap(self._pixmap_from_rgb(self._last_preview_rgb))
        if self._last_histogram_rgb is not None:
            self.histogram_label.setPixmap(self._pixmap_from_rgb(self._last_histogram_rgb))
        if self._last_history_rgb is not None:
            self.history_label.setPixmap(self._pixmap_from_rgb(self._last_history_rgb))

    @staticmethod
    def _pixmap_from_rgb(image_rgb: np.ndarray) -> QPixmap:
        h, w, _ = image_rgb.shape
        qimg = QImage(image_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def save_snapshot(self) -> None:
        if self.last_render_bgr is None:
            QMessageBox.information(self, "Save PNG", "No frame available.")
            return
        default_name = datetime.now().strftime("thermal_%Y%m%d_%H%M%S.png")
        initial_target = self._initial_save_target(self.state.last_image_save_dir, default_name)
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", str(initial_target), "PNG Image (*.png)")
        if not path:
            return
        selected_path = Path(path)
        self.state.last_image_save_dir = str(selected_path.parent)
        save_state(self.state)
        # Preserve the exact rendered orientation and aspect ratio for snapshots.
        ok = save_png(selected_path, self.last_render_bgr)
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
        initial_target = self._initial_save_target(self.state.last_recording_save_dir, default_name)
        path, _ = QFileDialog.getSaveFileName(self, "Save MP4", str(initial_target), "MP4 Video (*.mp4)")
        if not path:
            return
        selected_path = Path(path)
        self.state.last_recording_save_dir = str(selected_path.parent)
        save_state(self.state)
        h, w, _ = self.last_render_bgr.shape
        if not self.recorder.start(selected_path, (w, h), fps=25.0):
            QMessageBox.critical(self, "Record MP4", "Failed to start recorder.")
            return
        self.record_button.setText("Stop Recording")

    @staticmethod
    def _initial_save_target(last_directory: str, default_name: str) -> Path:
        if last_directory:
            directory = Path(last_directory)
            if directory.exists() and directory.is_dir():
                return directory / default_name
        return Path.home() / default_name
