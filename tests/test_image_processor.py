"""Unit tests for thermal image rendering helpers."""

from __future__ import annotations

import unittest

import numpy as np

from irpropycapture.core.frame_processing_worker import append_export_color_scale
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

    def test_append_export_color_scale_adds_scale_on_right_for_portrait(self) -> None:
        image = np.full((1024, 768, 3), 32, dtype=np.uint8)
        result = append_export_color_scale(image, "Turbo", min_temp_c=20.0, max_temp_c=40.0, unit="C")
        self.assertEqual(result.shape[0], image.shape[0])
        self.assertGreater(result.shape[1], image.shape[1])
        np.testing.assert_array_equal(result[:, : image.shape[1], :], image)

    def test_append_export_color_scale_adds_scale_on_bottom_for_landscape(self) -> None:
        image = np.full((768, 1024, 3), 64, dtype=np.uint8)
        result = append_export_color_scale(image, "Inferno", min_temp_c=10.0, max_temp_c=50.0, unit="F")
        self.assertEqual(result.shape[1], image.shape[1])
        self.assertGreater(result.shape[0], image.shape[0])
        np.testing.assert_array_equal(result[: image.shape[0], :, :], image)


if __name__ == "__main__":
    unittest.main()
