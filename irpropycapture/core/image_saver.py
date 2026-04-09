"""Image save utility."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_png(path: Path, image_bgr: np.ndarray) -> bool:
    return bool(cv2.imwrite(str(path), image_bgr))
