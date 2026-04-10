"""Unit tests for thermal image rendering helpers."""

from __future__ import annotations

import unittest

import numpy as np

from irpropycapture.core.image_processor import render_thermal_image


class ImageProcessorTests(unittest.TestCase):
    def test_render_thermal_image_returns_bgr_image(self) -> None:
        thermal = np.linspace(20.0, 40.0, 192 * 256, dtype=np.float32).reshape(192, 256)
        image = render_thermal_image(
            thermal,
            color_map_name="Turbo",
            manual_range_enabled=False,
            manual_min_temp=0.0,
            manual_max_temp=1.0,
        )
        self.assertEqual(image.shape, (192, 256, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_precomputed_auto_range_matches_direct_range(self) -> None:
        thermal = np.linspace(10.0, 45.0, 64, dtype=np.float32).reshape(8, 8)
        direct = render_thermal_image(
            thermal,
            color_map_name="Inferno",
            manual_range_enabled=False,
            manual_min_temp=0.0,
            manual_max_temp=1.0,
        )
        precomputed = render_thermal_image(
            thermal,
            color_map_name="Inferno",
            manual_range_enabled=False,
            manual_min_temp=0.0,
            manual_max_temp=1.0,
            auto_min_temp=10.0,
            auto_max_temp=45.0,
        )
        np.testing.assert_array_equal(direct, precomputed)


if __name__ == "__main__":
    unittest.main()
