"""Main window with capture, analysis and export tools."""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QFontDatabase, QGuiApplication, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from irpropycapture.core.camera_capture import OpenCVCaptureWorker, list_opencv_camera_devices, probe_opencv_source
from irpropycapture.core.camera_controls import (
    RANGE_MODE_HIGH,
    RANGE_MODE_LOW,
    TemperatureRange,
    apply_temperature_range,
    camera_control_startup_check,
)
from irpropycapture.core.frame_processing_worker import (
    MIN_HISTOGRAM_RENDER_HEIGHT,
    MIN_HISTORY_RENDER_HEIGHT,
    ProcessingResult,
    ProcessingSettings,
    ProcessingWorker,
    append_export_color_scale,
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
    camera_range_apply_finished = Signal(int, bool, str)

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
        self.camera_range_apply_finished.connect(self._on_camera_temperature_range_applied)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._camera_range_apply_in_progress = False
        self._pending_camera_temperature_range: int | None = None
        self._show_temperature_range_switch_overlay = False
        self._received_frame_count = 0
        self._wait_for_stream_resume_after_range_switch = False
        self._range_switch_completed_at: float | None = None
        self._processed_frames_since_range_switch = 0
        self._required_processed_frames_after_range_switch = 3
        self._range_switch_min_overlay_seconds_after_complete = 3.2
        self._last_auto_range_min_c: float | None = None
        self._last_auto_range_max_c: float | None = None

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
        self.snapshot_button = QPushButton("Save PNG (P)")
        self.record_button = QPushButton("Start Recording (R)")
        self.options_button = QToolButton()
        self.options_button.setText("⚙")
        self.options_button.setToolTip("Open additional settings")
        self.options_button.setAutoRaise(True)
        self.options_button.setStyleSheet("QToolButton { border: none; padding: 0px; margin: 0px; }")
        options_font = QFont(self.options_button.font())
        options_font.setBold(True)
        options_font.setPointSize(max(options_font.pointSize(), 24))
        self.options_button.setFont(options_font)
        self.options_button.setFixedHeight(self.record_button.sizeHint().height())

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
        self.manual_set_to_current_button = QPushButton("Set to current")
        self.camera_temperature_range_combo = QComboBox()
        self.camera_temperature_range_combo.addItem(TemperatureRange.LOW.value, RANGE_MODE_LOW)
        self.camera_temperature_range_combo.addItem(TemperatureRange.HIGH.value, RANGE_MODE_HIGH)
        self.min_spin = QSpinBox()
        self.max_spin = QSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setRange(-50, 600)
            spin.setSingleStep(1)

        self._restore_state_to_controls()

        top = QHBoxLayout()
        top.addWidget(self.camera_combo, stretch=1)
        top.addWidget(self.refresh_button)
        top.addWidget(self.start_button)
        top.addWidget(self.snapshot_button)
        top.addWidget(self.record_button)
        top.addWidget(self.options_button)

        controls_box = QGroupBox("Controls")
        controls_form = QFormLayout()
        controls_form.setVerticalSpacing(2)
        controls_form.setHorizontalSpacing(6)
        controls_form.setContentsMargins(8, 4, 8, 4)
        controls_form.addRow("Color map", self.color_map_combo)
        controls_form.addRow("Orientation", self.orientation_combo)
        controls_form.addRow(self.grid_checkbox)
        controls_form.addRow(self.min_max_checkbox)
        controls_form.addRow("Camera range", self.camera_temperature_range_combo)
        manual_range_row = QWidget()
        manual_range_layout = QHBoxLayout(manual_range_row)
        manual_range_layout.setContentsMargins(0, 0, 0, 0)
        manual_range_layout.setSpacing(6)
        manual_range_layout.addWidget(self.manual_range_checkbox)
        manual_range_layout.addWidget(self.manual_set_to_current_button)
        controls_form.addRow(manual_range_row)
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
        self._run_camera_control_startup_check()

        self.refresh_button.clicked.connect(self.refresh_camera_list)
        self.start_button.clicked.connect(self.toggle_capture)
        self.snapshot_button.clicked.connect(self.save_snapshot)
        self.record_button.clicked.connect(self.toggle_recording)
        self.options_button.clicked.connect(self._open_options_dialog)
        self.color_map_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.unit_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.preview_interpolation_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.orientation_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.grid_checkbox.toggled.connect(self._schedule_state_persist)
        self.min_max_checkbox.toggled.connect(self._schedule_state_persist)
        self.grid_density_combo.currentTextChanged.connect(self._schedule_state_persist)
        self.camera_temperature_range_combo.currentTextChanged.connect(self._on_camera_temperature_range_changed)
        self.manual_range_checkbox.toggled.connect(self._schedule_state_persist)
        self.manual_set_to_current_button.clicked.connect(self._set_manual_range_to_current)
        self.min_spin.valueChanged.connect(self._schedule_state_persist)
        self.max_spin.valueChanged.connect(self._schedule_state_persist)
        self._configure_hotkeys()

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
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._stop_workers()
        if self.recorder.is_recording:
            self.recorder.stop()
        self._persist_state()
        super().closeEvent(event)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.KeyPress and self.isActiveWindow():
            key = event.key()
            modifiers = event.modifiers()
            if modifiers in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier):
                if key == Qt.Key.Key_P:
                    self.save_snapshot_quick()
                    return True
                if key == Qt.Key.Key_R:
                    self.toggle_recording_quick()
                    return True
        return super().eventFilter(watched, event)

    def _restore_state_to_controls(self) -> None:
        self.color_map_combo.setCurrentText(self.state.color_map)
        self.unit_combo.setCurrentText(self.state.temperature_format)
        self.preview_interpolation_combo.setCurrentText(self.state.preview_interpolation)
        self.grid_checkbox.setChecked(self.state.show_temperature_grid)
        self.min_max_checkbox.setChecked(self.state.show_min_max_markers)
        self.orientation_combo.setCurrentText(self.state.orientation)
        self.grid_density_combo.setCurrentText(self.state.grid_density)
        target_range_mode = RANGE_MODE_HIGH if self.state.camera_temperature_range_mode else RANGE_MODE_LOW
        target_index = self.camera_temperature_range_combo.findData(target_range_mode)
        self.camera_temperature_range_combo.setCurrentIndex(target_index if target_index >= 0 else 0)
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
        selected_mode = self.camera_temperature_range_combo.currentData()
        self.state.camera_temperature_range_mode = int(selected_mode) if selected_mode is not None else RANGE_MODE_HIGH
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

    def _configure_hotkeys(self) -> None:
        self.save_png_action = QAction("Quick Save PNG", self)
        self.save_png_action.setShortcut(QKeySequence("P"))
        self.save_png_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.save_png_action.triggered.connect(self.save_snapshot_quick)
        self.addAction(self.save_png_action)

        self.record_action = QAction("Quick Toggle Recording", self)
        self.record_action.setShortcut(QKeySequence("R"))
        self.record_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.record_action.triggered.connect(self.toggle_recording_quick)
        self.addAction(self.record_action)

    def _open_options_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Additional Settings")
        layout = QVBoxLayout(dialog)

        form = QFormLayout()
        unit_combo = QComboBox(dialog)
        unit_combo.addItems(["C", "F"])
        unit_combo.setCurrentText(self.unit_combo.currentText())

        interpolation_combo = QComboBox(dialog)
        interpolation_combo.addItems(["Fast", "Smooth"])
        interpolation_combo.setCurrentText(self.preview_interpolation_combo.currentText())

        grid_density_combo = QComboBox(dialog)
        grid_density_combo.addItems(["Low", "Medium", "High"])
        grid_density_combo.setCurrentText(self.grid_density_combo.currentText())
        export_color_scale_checkbox = QCheckBox("Color scale in saved image/video", dialog)
        export_color_scale_checkbox.setChecked(self.state.export_include_color_scale)

        form.addRow("Temperature Unit", unit_combo)
        form.addRow("Preview interpolation", interpolation_combo)
        form.addRow("Grid density", grid_density_combo)
        form.addRow(export_color_scale_checkbox)
        layout.addLayout(form)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.unit_combo.setCurrentText(unit_combo.currentText())
        self.preview_interpolation_combo.setCurrentText(interpolation_combo.currentText())
        self.grid_density_combo.setCurrentText(grid_density_combo.currentText())
        self.state.export_include_color_scale = export_color_scale_checkbox.isChecked()
        self._schedule_state_persist()

    def _resolve_export_scale_temperatures(self, frame_min_c: float | None = None, frame_max_c: float | None = None) -> tuple[float, float]:
        if self.manual_range_checkbox.isChecked():
            low = float(self.min_spin.value())
            high = float(self.max_spin.value())
            return min(low, high), max(low, high)
        resolved_min = self._last_auto_range_min_c if frame_min_c is None else frame_min_c
        resolved_max = self._last_auto_range_max_c if frame_max_c is None else frame_max_c
        if resolved_min is None or resolved_max is None:
            return 0.0, 100.0
        return min(float(resolved_min), float(resolved_max)), max(float(resolved_min), float(resolved_max))

    def _export_frame_with_optional_color_scale(
        self,
        export_frame_bgr: np.ndarray,
        frame_min_c: float | None = None,
        frame_max_c: float | None = None,
    ) -> np.ndarray:
        if not self.state.export_include_color_scale:
            return export_frame_bgr
        min_temp_c, max_temp_c = self._resolve_export_scale_temperatures(frame_min_c=frame_min_c, frame_max_c=frame_max_c)
        return append_export_color_scale(
            export_frame_bgr,
            self.color_map_combo.currentText(),
            min_temp_c,
            max_temp_c,
            self.unit_combo.currentText(),
        )

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
        self.capture_worker.camera_opened.connect(self._apply_selected_camera_temperature_range)
        self.capture_worker.error.connect(lambda msg: QMessageBox.critical(self, "Capture Error", msg))
        self.capture_worker.start()
        self.start_button.setText("Stop Camera")
        self._persist_state()

    def _on_camera_temperature_range_changed(self, _value: str) -> None:
        self._schedule_state_persist()
        if self.capture_worker is None:
            return
        self._apply_selected_camera_temperature_range()

    def _run_camera_control_startup_check(self) -> None:
        ok, message = camera_control_startup_check()
        if ok:
            self.statusBar().showMessage(message, 3000)
            return
        self.camera_temperature_range_combo.setEnabled(False)
        QMessageBox.warning(self, "Camera Temperature Range", message)
        self.statusBar().showMessage(message)

    def _apply_selected_camera_temperature_range(self) -> None:
        selected_mode = self.camera_temperature_range_combo.currentData()
        selected_range_mode = int(selected_mode) if selected_mode is not None else RANGE_MODE_HIGH
        if self._camera_range_apply_in_progress:
            self._pending_camera_temperature_range = selected_range_mode
            return
        self._wait_for_stream_resume_after_range_switch = False
        self._range_switch_completed_at = None
        self._processed_frames_since_range_switch = 0
        self._set_temperature_range_switch_overlay(True)
        self._camera_range_apply_in_progress = True

        def _apply_range_worker() -> None:
            ok, message = apply_temperature_range(selected_range_mode)
            self.camera_range_apply_finished.emit(selected_range_mode, ok, message)

        threading.Thread(target=_apply_range_worker, daemon=True).start()

    def _on_camera_temperature_range_applied(self, applied_range: int, ok: bool, message: str) -> None:
        self._camera_range_apply_in_progress = False
        if not ok:
            QMessageBox.warning(self, "Camera Temperature Range", message)
            self._wait_for_stream_resume_after_range_switch = False
            self._range_switch_completed_at = None
            self._processed_frames_since_range_switch = 0
            self._set_temperature_range_switch_overlay(False)
        else:
            self.statusBar().showMessage(message, 3000)
        if self._pending_camera_temperature_range is None:
            if ok:
                self._wait_for_stream_resume_after_range_switch = True
                self._range_switch_completed_at = time.perf_counter()
                self._processed_frames_since_range_switch = 0
            return
        pending_range = self._pending_camera_temperature_range
        self._pending_camera_temperature_range = None
        if pending_range != applied_range:
            self._wait_for_stream_resume_after_range_switch = False
            self._range_switch_completed_at = None
            self._processed_frames_since_range_switch = 0
            self._apply_selected_camera_temperature_range()
            return
        if ok:
            self._wait_for_stream_resume_after_range_switch = True
            self._range_switch_completed_at = time.perf_counter()
            self._processed_frames_since_range_switch = 0

    def _set_temperature_range_switch_overlay(self, enabled: bool) -> None:
        self._show_temperature_range_switch_overlay = enabled
        if enabled:
            self._show_temperature_range_overlay_on_preview()
            return
        if self._last_preview_rgb is not None:
            self.preview.setPixmap(self._pixmap_from_rgb(self._last_preview_rgb))
        else:
            self.preview.setText("No image")

    def _show_temperature_range_overlay_on_preview(self) -> None:
        if self._last_preview_rgb is None:
            self.preview.setText("Switching temperature range...")
            return
        overlay_rgb = self._last_preview_rgb.copy()
        text = "Switching temperature range..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max((overlay_rgb.shape[1] - text_width) // 2, 8)
        y = max((overlay_rgb.shape[0] + text_height) // 2, text_height + 8)
        top = max(y - text_height - 12, 0)
        bottom = min(y + baseline + 12, overlay_rgb.shape[0])
        left = max(x - 12, 0)
        right = min(x + text_width + 12, overlay_rgb.shape[1])
        cv2.rectangle(overlay_rgb, (left, top), (right, bottom), (0, 0, 0), -1)
        cv2.putText(overlay_rgb, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        self.preview.setPixmap(self._pixmap_from_rgb(overlay_rgb))

    def on_frame_ready(self, frame: np.ndarray) -> None:
        self._received_frame_count += 1
        if self.processing_worker is None:
            return
        settings = self._current_processing_settings()
        settings.frame_sequence_id = self._received_frame_count
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
            camera_temperature_range=int(self.camera_temperature_range_combo.currentData() or RANGE_MODE_HIGH),
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
            frame_sequence_id=self._received_frame_count,
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
            self._last_auto_range_min_c = result.min_value
            self._last_auto_range_max_c = result.max_value
            self._last_preview_rgb = result.preview_rgb
            self._last_histogram_rgb = result.histogram_rgb
            self._last_history_rgb = result.history_rgb
            if self._show_temperature_range_switch_overlay and self._wait_for_stream_resume_after_range_switch:
                self._processed_frames_since_range_switch += 1
                if (
                    self._range_switch_completed_at is not None
                    and (time.perf_counter() - self._range_switch_completed_at)
                    >= self._range_switch_min_overlay_seconds_after_complete
                    and self._processed_frames_since_range_switch >= self._required_processed_frames_after_range_switch
                ):
                    self._wait_for_stream_resume_after_range_switch = False
                    self._range_switch_completed_at = None
                    self._processed_frames_since_range_switch = 0
                    self._set_temperature_range_switch_overlay(False)
            if self._show_temperature_range_switch_overlay:
                self._show_temperature_range_overlay_on_preview()
            else:
                self.preview.setPixmap(self._pixmap_from_rgb(result.preview_rgb))
            self.histogram_label.setPixmap(self._pixmap_from_rgb(result.histogram_rgb))
            self.history_label.setPixmap(self._pixmap_from_rgb(result.history_rgb))
            self._update_stats(result.min_value, result.max_value, result.average, result.center)
            if self.recorder.is_recording:
                write_start = time.perf_counter()
                export_frame_bgr = self._export_frame_with_optional_color_scale(
                    result.export_bgr,
                    frame_min_c=result.min_value,
                    frame_max_c=result.max_value,
                )
                self.recorder.write_frame(export_frame_bgr)
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

    def _set_manual_range_to_current(self) -> None:
        if self._last_auto_range_min_c is None or self._last_auto_range_max_c is None:
            self.statusBar().showMessage("No frame yet to set manual range.", 2500)
            return
        current_min = int(math.floor(self._last_auto_range_min_c))
        current_max = int(math.ceil(self._last_auto_range_max_c))
        if current_max <= current_min:
            current_max = current_min + 1
        self.min_spin.setValue(current_min)
        self.max_spin.setValue(current_max)
        self.manual_range_checkbox.setChecked(True)
        self.statusBar().showMessage(
            f"Manual range set to current frame: {current_min} .. {current_max} C",
            3000,
        )

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        self._schedule_state_persist()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._schedule_state_persist()
        if self._last_preview_rgb is not None:
            if self._show_temperature_range_switch_overlay:
                self._show_temperature_range_overlay_on_preview()
            else:
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
        self._save_snapshot(use_dialog=True)

    def save_snapshot_quick(self) -> None:
        self._save_snapshot(use_dialog=False)

    def _save_snapshot(self, use_dialog: bool) -> None:
        if self.last_render_bgr is None:
            QMessageBox.information(self, "Save PNG", "No frame available.")
            return
        default_name = datetime.now().strftime("thermal_%Y%m%d_%H%M%S.png")
        if use_dialog:
            initial_target = self._initial_save_target(default_name)
            path, _ = QFileDialog.getSaveFileName(self, "Save PNG", str(initial_target), "PNG Image (*.png)")
            if not path:
                return
            selected_path = Path(path)
        else:
            selected_path = self._build_auto_export_path(default_name)
        if not self._set_shared_export_directory(selected_path.parent):
            QMessageBox.critical(self, "Save PNG", "Failed to prepare export directory.")
            return
        # Preserve the exact rendered orientation and aspect ratio for snapshots.
        export_frame_bgr = self._export_frame_with_optional_color_scale(self.last_render_bgr)
        ok = save_png(selected_path, export_frame_bgr)
        if not ok:
            QMessageBox.critical(self, "Save PNG", "Failed to save image.")
            return
        if not use_dialog:
            self.statusBar().showMessage(f"Saved PNG: {selected_path.name}", 2500)

    def toggle_recording(self) -> None:
        self._toggle_recording(use_dialog=True)

    def toggle_recording_quick(self) -> None:
        self._toggle_recording(use_dialog=False)

    def _toggle_recording(self, use_dialog: bool) -> None:
        if self.recorder.is_recording:
            self.recorder.stop()
            self.record_button.setText("Start Recording (R)")
            self.statusBar().showMessage("Recording stopped.", 2000)
            return
        if self.last_render_bgr is None:
            QMessageBox.information(self, "Record MP4", "Start the stream first.")
            return
        default_name = datetime.now().strftime("thermal_%Y%m%d_%H%M%S.mp4")
        if use_dialog:
            initial_target = self._initial_save_target(default_name)
            path, _ = QFileDialog.getSaveFileName(self, "Save MP4", str(initial_target), "MP4 Video (*.mp4)")
            if not path:
                return
            selected_path = Path(path)
        else:
            selected_path = self._build_auto_export_path(default_name)
        if not self._set_shared_export_directory(selected_path.parent):
            QMessageBox.critical(self, "Record MP4", "Failed to prepare export directory.")
            return
        initial_export_frame = self._export_frame_with_optional_color_scale(self.last_render_bgr)
        h, w, _ = initial_export_frame.shape
        if not self.recorder.start(selected_path, (w, h), fps=25.0):
            QMessageBox.critical(self, "Record MP4", "Failed to start recorder.")
            return
        self.record_button.setText("Stop Recording (R)")
        if not use_dialog:
            self.statusBar().showMessage(f"Recording started: {selected_path.name}", 2500)

    def _initial_save_target(self, default_name: str) -> Path:
        return self._default_export_directory() / default_name

    def _default_export_directory(self) -> Path:
        if self.state.export_save_dir:
            directory = Path(self.state.export_save_dir)
            if directory.exists() and directory.is_dir():
                return directory
        return Path.home()

    def _set_shared_export_directory(self, directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        self.state.export_save_dir = str(directory)
        save_state(self.state)
        return True

    def _build_auto_export_path(self, default_name: str) -> Path:
        base_directory = self._default_export_directory()
        candidate = base_directory / default_name
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        for index in range(1, 1000):
            next_candidate = base_directory / f"{stem}_{index:03d}{suffix}"
            if not next_candidate.exists():
                return next_candidate
        return base_directory / f"{stem}_{int(time.time())}{suffix}"
