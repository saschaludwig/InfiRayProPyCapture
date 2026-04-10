"""Unit tests for temperature decoding."""

from __future__ import annotations

import unittest

import numpy as np

from irpropycapture.core.temperature_processor import TemperatureProcessor


def _build_frame_from_temps(thermal_celsius: np.ndarray) -> np.ndarray:
    encoded = np.clip((thermal_celsius + 273.2) * 64.0, 0.0, 65535.0).astype(np.uint16)
    high = ((encoded >> 8) & 0xFF).astype(np.uint8)
    low = (encoded & 0xFF).astype(np.uint8)
    packed = np.empty((192, 512), dtype=np.uint8)
    packed[:, 0::2] = high
    packed[:, 1::2] = low
    frame = np.zeros((384, 512), dtype=np.uint8)
    frame[192:384, :] = packed
    return frame


class TemperatureProcessorTests(unittest.TestCase):
    def test_decodes_big_endian_payload(self) -> None:
        y_grid, x_grid = np.mgrid[0:192, 0:256]
        thermal = 22.0 + (x_grid / 255.0) * 10.0 + (y_grid / 191.0) * 3.0
        frame = _build_frame_from_temps(thermal.astype(np.float32))

        processor = TemperatureProcessor()
        result = processor.get_temperatures(frame)
        decoded = result.temperatures.reshape(192, 256)

        self.assertEqual(decoded.shape, (192, 256))
        self.assertAlmostEqual(float(np.min(decoded)), float(np.min(thermal)), delta=0.6)
        self.assertAlmostEqual(float(np.max(decoded)), float(np.max(thermal)), delta=0.6)
        self.assertGreater(len(result.histogram), 0)

    def test_histogram_is_empty_for_flat_signal(self) -> None:
        values = np.ones((192 * 256,), dtype=np.float32) * 32.0
        histogram = TemperatureProcessor.compute_histogram(values, 32.0, 32.0, bins=50)
        self.assertEqual(histogram, [])


if __name__ == "__main__":
    unittest.main()
