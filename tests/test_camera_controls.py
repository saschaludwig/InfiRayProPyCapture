"""Unit tests for camera USB temperature-range controls."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import usb.core

from irpropycapture.core import camera_controls
from irpropycapture.core.camera_controls import (
    RANGE_MODE_HIGH,
    RANGE_MODE_LOW,
    _WINDOWS_FILTER_HINT,
    _get_backend,
    _windows_control_access_message,
    apply_temperature_range,
    camera_control_startup_check,
)


class CameraControlsTests(unittest.TestCase):
    def test_get_backend_prefers_libusb0_on_windows(self) -> None:
        backend0 = object()
        backend1 = object()
        with (
            patch("irpropycapture.core.camera_controls.platform.system", return_value="Windows"),
            patch("irpropycapture.core.camera_controls.usb.backend.libusb0.get_backend", return_value=backend0),
            patch("irpropycapture.core.camera_controls.usb.backend.libusb1.get_backend", return_value=backend1) as get1,
        ):
            self.assertIs(_get_backend(), backend0)
            get1.assert_not_called()

    def test_get_backend_falls_back_to_libusb1_on_windows(self) -> None:
        backend1 = object()
        with (
            patch("irpropycapture.core.camera_controls.platform.system", return_value="Windows"),
            patch("irpropycapture.core.camera_controls.usb.backend.libusb0.get_backend", return_value=None),
            patch("irpropycapture.core.camera_controls.usb.backend.libusb1.get_backend", return_value=backend1),
        ):
            self.assertIs(_get_backend(), backend1)

    def test_get_backend_prefers_libusb1_elsewhere(self) -> None:
        backend0 = object()
        backend1 = object()
        with (
            patch("irpropycapture.core.camera_controls.platform.system", return_value="Linux"),
            patch("irpropycapture.core.camera_controls.usb.backend.libusb0.get_backend", return_value=backend0) as get0,
            patch("irpropycapture.core.camera_controls.usb.backend.libusb1.get_backend", return_value=backend1),
        ):
            self.assertIs(_get_backend(), backend1)
            get0.assert_not_called()

    def test_windows_control_access_message_appends_hint(self) -> None:
        with patch("irpropycapture.core.camera_controls.platform.system", return_value="Windows"):
            message = _windows_control_access_message(Exception("Entity not found"))
            self.assertIn("Entity not found", message)
            self.assertIn(_WINDOWS_FILTER_HINT, message)

            ready = _windows_control_access_message(Exception("Das Gerät ist nicht bereit."))
            self.assertIn(_WINDOWS_FILTER_HINT, ready)

            other = _windows_control_access_message(Exception("timeout waiting"))
            self.assertEqual(other, "timeout waiting")

        with patch("irpropycapture.core.camera_controls.platform.system", return_value="Linux"):
            plain = _windows_control_access_message(Exception("Entity not found"))
            self.assertEqual(plain, "Entity not found")

    def test_apply_temperature_range_rejects_unsupported_mode(self) -> None:
        ok, message = apply_temperature_range(99)
        self.assertFalse(ok)
        self.assertIn("Unsupported camera temperature range mode", message)

    def test_apply_temperature_range_maps_gain_values(self) -> None:
        device = MagicMock()
        written: list[int] = []

        def fake_write(_device: object, gain_value: int, timeout_seconds: float = 5.0) -> None:
            written.append(gain_value)

        with (
            patch("irpropycapture.core.camera_controls._find_camera_device", return_value=device),
            patch("irpropycapture.core.camera_controls._write_gain_select", side_effect=fake_write),
        ):
            ok_low, msg_low = apply_temperature_range(RANGE_MODE_LOW)
            ok_high, msg_high = apply_temperature_range(RANGE_MODE_HIGH)

        self.assertTrue(ok_low)
        self.assertTrue(ok_high)
        self.assertIn("Low", msg_low)
        self.assertIn("High", msg_high)
        self.assertEqual(written, [1, 0])

    def test_apply_temperature_range_missing_device(self) -> None:
        with patch("irpropycapture.core.camera_controls._find_camera_device", return_value=None):
            ok, message = apply_temperature_range(RANGE_MODE_HIGH)
        self.assertFalse(ok)
        self.assertEqual(message, "Thermal camera control interface not found.")

    def test_apply_temperature_range_usb_error_includes_hint_on_windows(self) -> None:
        with (
            patch("irpropycapture.core.camera_controls.platform.system", return_value="Windows"),
            patch("irpropycapture.core.camera_controls._find_camera_device", return_value=MagicMock()),
            patch(
                "irpropycapture.core.camera_controls._write_gain_select",
                side_effect=usb.core.USBError("Entity not found"),
            ),
        ):
            ok, message = apply_temperature_range(RANGE_MODE_HIGH)
        self.assertFalse(ok)
        self.assertIn("Entity not found", message)
        self.assertIn(_WINDOWS_FILTER_HINT, message)

    def test_camera_control_startup_check_no_backend(self) -> None:
        with patch("irpropycapture.core.camera_controls._get_backend", return_value=None):
            ok, message = camera_control_startup_check()
        self.assertFalse(ok)
        self.assertIn("USB backend not available", message)

    def test_camera_control_startup_check_device_missing(self) -> None:
        with (
            patch("irpropycapture.core.camera_controls._get_backend", return_value=object()),
            patch("irpropycapture.core.camera_controls._find_camera_device", return_value=None),
        ):
            ok, message = camera_control_startup_check()
        self.assertFalse(ok)
        self.assertEqual(message, "Thermal camera control interface not found.")

    def test_camera_control_startup_check_windows_skips_probe(self) -> None:
        with (
            patch("irpropycapture.core.camera_controls.platform.system", return_value="Windows"),
            patch("irpropycapture.core.camera_controls._get_backend", return_value=object()),
            patch("irpropycapture.core.camera_controls._find_camera_device", return_value=MagicMock()),
            patch("irpropycapture.core.camera_controls._probe_control_access") as probe_mock,
        ):
            ok, message = camera_control_startup_check()
        self.assertTrue(ok)
        self.assertIn("Start the stream", message)
        probe_mock.assert_not_called()

    def test_camera_control_startup_check_non_windows_probe_error(self) -> None:
        with (
            patch("irpropycapture.core.camera_controls.platform.system", return_value="Linux"),
            patch("irpropycapture.core.camera_controls._get_backend", return_value=object()),
            patch("irpropycapture.core.camera_controls._find_camera_device", return_value=MagicMock()),
            patch(
                "irpropycapture.core.camera_controls._probe_control_access",
                side_effect=usb.core.USBError("Access denied"),
            ),
        ):
            ok, message = camera_control_startup_check()
        self.assertFalse(ok)
        self.assertIn("Access denied", message)


if __name__ == "__main__":
    unittest.main()
