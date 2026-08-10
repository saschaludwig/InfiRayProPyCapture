"""Unit tests for camera backend policy and listing helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from irpropycapture.core import camera_capture
from irpropycapture.core.camera_capture import (
    _frame_attempts_for_mode,
    _fourcc_candidates_for_mode,
    _open_capture_windows,
    _read_convertible_frame,
    _try_set_capture_fourcc,
    list_opencv_camera_devices,
    probe_opencv_source,
)


class CameraCapturePolicyTests(unittest.TestCase):
    def test_fourcc_candidates_for_mode(self) -> None:
        self.assertEqual(
            _fourcc_candidates_for_mode("native-v4l2"),
            ["Y16 ", "YUYV", "YUY2", "UYVY"],
        )
        self.assertEqual(
            _fourcc_candidates_for_mode("native-msmf"),
            ["Y16 ", "YUY2", "UYVY", "YUYV"],
        )
        self.assertEqual(
            _fourcc_candidates_for_mode("native-dshow"),
            ["Y16 ", "YUY2", "UYVY", "YUYV"],
        )
        self.assertEqual(
            _fourcc_candidates_for_mode("ffmpeg-avfoundation-usb"),
            ["YUYV", "UYVY", "YUY2", "Y16 "],
        )

    def test_frame_attempts_for_mode(self) -> None:
        self.assertEqual(_frame_attempts_for_mode("native-msmf", 4), 4)
        self.assertEqual(_frame_attempts_for_mode("native-msmf", 8), 8)
        self.assertEqual(_frame_attempts_for_mode("native-msmf", 20), 24)
        self.assertEqual(_frame_attempts_for_mode("native-v4l2", 20), 28)
        self.assertEqual(_frame_attempts_for_mode("native-dshow", 20), 20)

    def test_default_max_index_by_platform(self) -> None:
        cases = (
            ("Darwin", 3),
            ("Windows", 4),
            ("Linux", 10),
            ("FreeBSD", 6),
        )
        for system_name, expected in cases:
            with self.subTest(system=system_name):
                with patch("irpropycapture.core.camera_capture.platform.system", return_value=system_name):
                    self.assertEqual(camera_capture._default_max_index(), expected)

    def test_preferred_backend_by_platform(self) -> None:
        cases = (
            ("Darwin", cv2.CAP_AVFOUNDATION),
            ("Windows", cv2.CAP_MSMF),
            ("Linux", cv2.CAP_V4L2),
            ("FreeBSD", cv2.CAP_ANY),
        )
        for system_name, expected in cases:
            with self.subTest(system=system_name):
                with patch("irpropycapture.core.camera_capture.platform.system", return_value=system_name):
                    self.assertEqual(camera_capture._preferred_backend(), expected)

    def test_open_capture_windows_prefers_msmf_by_default(self) -> None:
        calls: list[int] = []

        def fake_open(_index: int, backend: int, mode: str, open_timeout_ms: int | None = None):
            calls.append(backend)
            if backend == cv2.CAP_MSMF:
                return MagicMock(name="msmf-cap"), mode
            return None, ""

        with patch("irpropycapture.core.camera_capture._open_capture_native", side_effect=fake_open):
            cap, mode = _open_capture_windows(0, prefer_dshow=False)
        self.assertIsNotNone(cap)
        self.assertEqual(mode, "native-msmf")
        self.assertEqual(calls, [cv2.CAP_MSMF])

    def test_open_capture_windows_prefers_dshow_when_requested(self) -> None:
        calls: list[int] = []

        def fake_open(_index: int, backend: int, mode: str, open_timeout_ms: int | None = None):
            calls.append(backend)
            if backend == cv2.CAP_DSHOW:
                return MagicMock(name="dshow-cap"), mode
            return None, ""

        with patch("irpropycapture.core.camera_capture._open_capture_native", side_effect=fake_open):
            cap, mode = _open_capture_windows(0, prefer_dshow=True)
        self.assertIsNotNone(cap)
        self.assertEqual(mode, "native-dshow")
        self.assertEqual(calls, [cv2.CAP_DSHOW])

    def test_read_convertible_frame_fails_fast_on_color(self) -> None:
        cap = MagicMock()
        cap.read.return_value = (True, np.zeros((384, 256, 3), dtype=np.uint8))
        ok, message = _read_convertible_frame(cap, frame_attempts=8)
        self.assertFalse(ok)
        self.assertIn("decoded color frames", message)
        self.assertEqual(cap.read.call_count, 1)

    def test_read_convertible_frame_retries_then_succeeds(self) -> None:
        good = np.zeros((384, 512), dtype=np.uint8)
        cap = MagicMock()
        cap.read.side_effect = [
            (False, None),
            (True, good),
        ]
        with patch("irpropycapture.core.camera_capture.time.sleep", return_value=None):
            ok, message = _read_convertible_frame(cap, frame_attempts=4)
        self.assertTrue(ok)
        self.assertEqual(message, "ok")
        self.assertEqual(cap.read.call_count, 2)

    def test_try_set_capture_fourcc_handles_cv2_error(self) -> None:
        cap = MagicMock()
        cap.set.side_effect = cv2.error("msmf fourcc unsupported")
        self.assertFalse(_try_set_capture_fourcc(cap, "Y16 "))

        cap_ok = MagicMock()
        cap_ok.set.return_value = True
        self.assertTrue(_try_set_capture_fourcc(cap_ok, "YUY2"))

    def test_list_opencv_camera_devices_windows_classification(self) -> None:
        classifications = {
            0: "reject",
            1: "accept",
            2: "missing",
            3: "missing",
            4: "missing",
        }

        def classify(index: int) -> str:
            return classifications[index]

        with (
            patch("irpropycapture.core.camera_capture.platform.system", return_value="Windows"),
            patch(
                "irpropycapture.core.camera_capture._classify_windows_list_candidate",
                side_effect=classify,
            ),
            patch("irpropycapture.core.camera_capture.probe_opencv_source") as probe_mock,
        ):
            devices = list_opencv_camera_devices(max_index=4)

        self.assertEqual(devices, [(1, "Camera 1")])
        probe_mock.assert_not_called()

    def test_list_opencv_camera_devices_windows_probe_path(self) -> None:
        with (
            patch("irpropycapture.core.camera_capture.platform.system", return_value="Windows"),
            patch(
                "irpropycapture.core.camera_capture._classify_windows_list_candidate",
                side_effect=["probe", "reject", "missing", "missing", "missing"],
            ),
            patch(
                "irpropycapture.core.camera_capture.probe_opencv_source",
                return_value=(True, "native-msmf"),
            ) as probe_mock,
        ):
            devices = list_opencv_camera_devices(max_index=4)

        self.assertEqual(devices, [(0, "Camera 0")])
        probe_mock.assert_called_once()

    def test_probe_opencv_source_skips_fourcc_after_color(self) -> None:
        cap = MagicMock()
        cap.read.return_value = (True, np.zeros((384, 256, 3), dtype=np.uint8))

        with (
            patch("irpropycapture.core.camera_capture.platform.system", return_value="Windows"),
            patch(
                "irpropycapture.core.camera_capture._open_capture",
                return_value=(cap, "native-msmf"),
            ),
            patch("irpropycapture.core.camera_capture._configure_capture_for_raw"),
            patch("irpropycapture.core.camera_capture._try_set_capture_fourcc") as fourcc_mock,
        ):
            ok, message = probe_opencv_source(0, frame_attempts=4)

        self.assertFalse(ok)
        self.assertIn("decoded color frames", message)
        fourcc_mock.assert_not_called()
        cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
