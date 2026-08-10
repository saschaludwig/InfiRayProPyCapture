"""Unit tests for camera frame conversion helpers."""

from __future__ import annotations

import unittest

import numpy as np

from irpropycapture.core.camera_capture import (
    _convert_capture_to_pipeline_frame,
    _is_definite_incompatible_format,
    _looks_like_thermal_frame_geometry,
    _pack_for_temperature_pipeline,
)
from irpropycapture.core.temperature_processor import TemperatureProcessor


def _build_pipeline_frame_from_temps(thermal_celsius: np.ndarray) -> np.ndarray:
    encoded = np.clip((thermal_celsius + 273.2) * 64.0, 0.0, 65535.0).astype(np.uint16)
    high = ((encoded >> 8) & 0xFF).astype(np.uint8)
    low = (encoded & 0xFF).astype(np.uint8)
    packed = np.empty((192, 512), dtype=np.uint8)
    packed[:, 0::2] = high
    packed[:, 1::2] = low
    frame = np.zeros((384, 512), dtype=np.uint8)
    frame[192:384, :] = packed
    return frame


class CameraCaptureFrameTests(unittest.TestCase):
    def test_convert_uint8_pipeline_frame(self) -> None:
        frame = np.arange(384 * 512, dtype=np.uint8).reshape(384, 512)
        converted = _convert_capture_to_pipeline_frame(frame)
        self.assertEqual(converted.shape, (384, 512))
        self.assertEqual(converted.dtype, np.uint8)
        np.testing.assert_array_equal(converted, frame)

    def test_convert_raw_packet_line(self) -> None:
        payload = np.arange(256 * 384 * 2, dtype=np.uint8)
        frame = payload.reshape(1, -1)
        converted = _convert_capture_to_pipeline_frame(frame)
        self.assertEqual(converted.shape, (384, 512))
        np.testing.assert_array_equal(converted.reshape(-1), payload)

    def test_convert_uint16_192x256(self) -> None:
        raw = np.arange(192 * 256, dtype=np.uint16).reshape(192, 256)
        converted = _convert_capture_to_pipeline_frame(raw)
        self.assertEqual(converted.shape, (384, 512))
        self.assertEqual(converted.dtype, np.uint8)
        np.testing.assert_array_equal(converted[:192], 0)
        expected = raw.view(np.uint8).reshape(192, 512)
        np.testing.assert_array_equal(converted[192:384], expected)

    def test_convert_uint16_384x256_uses_bottom_half(self) -> None:
        raw = np.zeros((384, 256), dtype=np.uint16)
        raw[192:384, :] = np.arange(192 * 256, dtype=np.uint16).reshape(192, 256)
        converted = _convert_capture_to_pipeline_frame(raw)
        expected = raw[192:384, :].view(np.uint8).reshape(192, 512)
        np.testing.assert_array_equal(converted[192:384], expected)

    def test_convert_yuyv_two_channel(self) -> None:
        frame = np.arange(384 * 256 * 2, dtype=np.uint8).reshape(384, 256, 2)
        converted = _convert_capture_to_pipeline_frame(frame)
        self.assertEqual(converted.shape, (384, 512))
        np.testing.assert_array_equal(converted, frame.reshape(384, 512))

    def test_convert_rejects_decoded_color_frames(self) -> None:
        frame = np.zeros((384, 256, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "decoded color frames"):
            _convert_capture_to_pipeline_frame(frame)

    def test_convert_rejects_unsupported_2d(self) -> None:
        frame = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Unsupported 2D frame format"):
            _convert_capture_to_pipeline_frame(frame)

    def test_pack_validates_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 2D"):
            _pack_for_temperature_pipeline(np.zeros((10,), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "Expected uint16"):
            _pack_for_temperature_pipeline(np.zeros((192, 256), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Expected width 256"):
            _pack_for_temperature_pipeline(np.zeros((192, 128), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "Expected at least 192 rows"):
            _pack_for_temperature_pipeline(np.zeros((100, 256), dtype=np.uint16))

    def test_is_definite_incompatible_format(self) -> None:
        cases = (
            ("Camera backend returned decoded color frames.", True),
            ("Unsupported frame format: shape=(1, 2)", True),
            ("Unsupported 2D frame format: shape=(10, 10)", True),
            ("No frame received from camera.", False),
            ("Raw packet too small: 10 < 20", False),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(_is_definite_incompatible_format(message), expected)

    def test_looks_like_thermal_frame_geometry(self) -> None:
        cases = (
            (np.zeros((384, 256), dtype=np.uint16), True),
            (np.zeros((192, 256), dtype=np.uint16), True),
            (np.zeros((1, 196608), dtype=np.uint8), True),
            (np.zeros((384, 256, 3), dtype=np.uint8), True),
            (np.zeros((480, 640, 3), dtype=np.uint8), False),
            (np.zeros((100, 100), dtype=np.uint8), False),
        )
        for frame, expected in cases:
            with self.subTest(shape=frame.shape):
                self.assertEqual(_looks_like_thermal_frame_geometry(frame), expected)

    def test_pipeline_frame_roundtrip_temperature_decode(self) -> None:
        y_grid, x_grid = np.mgrid[0:192, 0:256]
        thermal = 22.0 + (x_grid / 255.0) * 10.0 + (y_grid / 191.0) * 3.0
        frame = _build_pipeline_frame_from_temps(thermal.astype(np.float32))
        converted = _convert_capture_to_pipeline_frame(frame)
        result = TemperatureProcessor().get_temperatures(converted)
        decoded = result.temperatures.reshape(192, 256)
        self.assertAlmostEqual(float(np.min(decoded)), float(np.min(thermal)), delta=0.6)
        self.assertAlmostEqual(float(np.max(decoded)), float(np.max(thermal)), delta=0.6)


if __name__ == "__main__":
    unittest.main()
