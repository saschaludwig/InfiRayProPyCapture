"""Helpers for camera-side runtime controls."""

from __future__ import annotations

import enum
import struct
import time

import usb.backend.libusb1
import usb.core

class TemperatureRange(str, enum.Enum):
    """Supported camera gain ranges exposed in the GUI."""

    LOW = "Low range (-20 to 180°C)"
    HIGH = "High range (-20 to 600°C)"

RANGE_MODE_LOW = 0
RANGE_MODE_HIGH = 1


_CAMERA_VENDOR_ID = 0x0BDA
_CAMERA_PRODUCT_ID = 0x5830
_CMD_PROP_TPD_PARAMS = 0x8514
_CMD_SET = 0x4000
_TPD_PROP_GAIN_SEL = 5


def camera_control_startup_check() -> tuple[bool, str]:
    """Validate whether USB backend and camera control interface are available."""
    backend = usb.backend.libusb1.get_backend()
    if backend is None:
        return (
            False,
            "USB backend not available. Install libusb (https://github.com/pyusb/pyusb).",
        )
    try:
        camera_device = usb.core.find(idVendor=_CAMERA_VENDOR_ID, idProduct=_CAMERA_PRODUCT_ID)
    except usb.core.NoBackendError:
        return (
            False,
            "pyusb has no backend available. Install a libusb backend on your system (e.g. libusb).",
        )
    if camera_device is None:
        return False, "Thermal camera control interface not found."
    return True, "Camera USB control interface is available."


def _check_camera_ready(camera_device: usb.core.Device) -> bool:
    status = camera_device.ctrl_transfer(0xC1, 0x44, 0x78, 0x200, 1)
    if status[0] & 1 == 0 and status[0] & 2 == 0:
        return True
    if status[0] & 0xFC != 0:
        raise RuntimeError(f"Camera command status error: {status[0]:#X}")
    return False


def _wait_until_camera_ready(camera_device: usb.core.Device, timeout_seconds: float = 5.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _check_camera_ready(camera_device):
            return True
        time.sleep(0.001)
    return False


def _write_gain_select(camera_device: usb.core.Device, gain_value: int, timeout_seconds: float) -> None:
    command = _CMD_PROP_TPD_PARAMS | _CMD_SET
    first_packet = struct.pack("<H", command) + struct.pack(">HI", _TPD_PROP_GAIN_SEL, gain_value)
    second_packet = struct.pack(">II", 0, 0)
    camera_device.ctrl_transfer(0x41, 0x45, 0x78, 0x9D00, first_packet)
    camera_device.ctrl_transfer(0x41, 0x45, 0x78, 0x1D08, second_packet)
    if not _wait_until_camera_ready(camera_device, timeout_seconds=timeout_seconds):
        raise TimeoutError("Camera did not become ready after gain command.")


def apply_temperature_range(selected_range_mode: int, timeout_seconds: float = 5.0) -> tuple[bool, str]:
    """Apply low/high gain temperature range via direct USB control transfers."""
    if selected_range_mode == RANGE_MODE_LOW:
        range_mode = TemperatureRange.LOW
    elif selected_range_mode == RANGE_MODE_HIGH:
        range_mode = TemperatureRange.HIGH
    else:
        return False, f"Unsupported camera temperature range mode: {selected_range_mode!r}"

    # Empirical behavior on current hardware/firmware shows the effective range
    # mapping is inverted compared to some external references:
    # - "Low range" should cap around ~180C
    # - "High range" should allow measurements well above 180C
    gain_value = 1 if range_mode is TemperatureRange.LOW else 0

    try:
        camera_device = usb.core.find(idVendor=_CAMERA_VENDOR_ID, idProduct=_CAMERA_PRODUCT_ID)
        if camera_device is None:
            return False, "Thermal camera control interface not found."
        _write_gain_select(camera_device, gain_value, timeout_seconds=timeout_seconds)
    except usb.core.NoBackendError:
        return (
            False,
            "pyusb has no backend available. Install a libusb backend on your system (e.g. libusb).",
        )
    except usb.core.USBError as exc:
        return False, f"USB error while applying camera temperature range: {exc}"
    except Exception as exc:
        return False, f"Failed to apply camera temperature range: {exc}"

    return True, f"Camera temperature range set to '{range_mode.value}'."
