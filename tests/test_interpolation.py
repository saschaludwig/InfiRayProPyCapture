"""Unit tests for preview/export interpolation helpers."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from irpropycapture.core.dnn_upscale import (
    dnn_superres_available,
    reset_dnn_upscaler_for_tests,
    resolve_espcn_model_path,
    upscale_bgr_sharp,
)
from irpropycapture.core.frame_processing_worker import (
    _interpolation_flag,
    _resize_export,
    _resize_preview,
    normalize_preview_interpolation,
)


class InterpolationTests(unittest.TestCase):
    def test_normalize_preview_interpolation_canonical_and_legacy(self) -> None:
        self.assertEqual(normalize_preview_interpolation("Nearest"), "Nearest")
        self.assertEqual(normalize_preview_interpolation("Linear"), "Linear")
        self.assertEqual(normalize_preview_interpolation("Cubic"), "Cubic")
        self.assertEqual(normalize_preview_interpolation("Lanczos"), "Lanczos")
        self.assertEqual(normalize_preview_interpolation("Sharp"), "Sharp")
        self.assertEqual(normalize_preview_interpolation("Fast"), "Nearest")
        self.assertEqual(normalize_preview_interpolation("Smooth"), "Cubic")
        self.assertEqual(normalize_preview_interpolation("unknown"), "Cubic")

    def test_interpolation_flags(self) -> None:
        self.assertEqual(_interpolation_flag("Nearest"), cv2.INTER_NEAREST)
        self.assertEqual(_interpolation_flag("Linear"), cv2.INTER_LINEAR)
        self.assertEqual(_interpolation_flag("Cubic"), cv2.INTER_CUBIC)
        self.assertEqual(_interpolation_flag("Lanczos"), cv2.INTER_LANCZOS4)
        self.assertEqual(_interpolation_flag("Sharp"), cv2.INTER_CUBIC)
        self.assertEqual(_interpolation_flag("Fast"), cv2.INTER_NEAREST)
        self.assertEqual(_interpolation_flag("Smooth"), cv2.INTER_CUBIC)

    def test_resize_preview_respects_mode(self) -> None:
        image = np.zeros((192, 256, 3), dtype=np.uint8)
        image[0, 0] = (0, 0, 255)
        nearest = _resize_preview(image, 512, 384, "Nearest")
        cubic = _resize_preview(image, 512, 384, "Cubic")
        self.assertEqual(nearest.shape, (384, 512, 3))
        self.assertEqual(cubic.shape, (384, 512, 3))
        # Nearest keeps hard block edges; cubic spreads the seed pixel.
        self.assertTrue(np.count_nonzero(cubic[..., 2]) > np.count_nonzero(nearest[..., 2]))

    def test_resize_export_uses_cubic_target(self) -> None:
        image = np.zeros((192, 256, 3), dtype=np.uint8)
        exported = _resize_export(image)
        self.assertEqual(exported.shape, (768, 1024, 3))

        portrait = np.zeros((256, 192, 3), dtype=np.uint8)
        exported_portrait = _resize_export(portrait)
        self.assertEqual(exported_portrait.shape, (1024, 768, 3))


class DnnUpscaleTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_dnn_upscaler_for_tests()

    def tearDown(self) -> None:
        reset_dnn_upscaler_for_tests()

    def test_model_file_is_packaged(self) -> None:
        self.assertTrue(resolve_espcn_model_path().is_file())

    def test_upscale_bgr_sharp_quadruples_resolution(self) -> None:
        image = np.zeros((192, 256, 3), dtype=np.uint8)
        image[20:40, 30:50] = (10, 80, 200)
        out = upscale_bgr_sharp(image)
        self.assertEqual(out.shape, (768, 1024, 3))
        self.assertEqual(out.dtype, np.uint8)

    @unittest.skipUnless(dnn_superres_available(), "opencv-contrib dnn_superres required")
    def test_dnn_path_runs_when_contrib_available(self) -> None:
        image = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        out = upscale_bgr_sharp(image)
        self.assertEqual(out.shape, (192, 256, 3))

    @unittest.skipUnless(dnn_superres_available(), "opencv-contrib dnn_superres required")
    def test_sharp_differs_from_lanczos(self) -> None:
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[:, :64] = (30, 40, 200)
        image[:, 64:] = (200, 180, 20)
        sharp = upscale_bgr_sharp(image)
        lanczos = cv2.resize(image, (512, 384), interpolation=cv2.INTER_LANCZOS4)
        diff = np.mean(np.abs(sharp.astype(np.int16) - lanczos.astype(np.int16)))
        self.assertGreater(diff, 1.0)


if __name__ == "__main__":
    unittest.main()
