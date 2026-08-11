"""ESPCN x4 upscaling via OpenCV dnn_superres for Sharp preview/export mode.

Model: ESPCN_x4.pb from https://github.com/fannymonori/TF-ESPCN (OpenCV superres zoo).
"""

from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np

_MODEL_FILENAME = "ESPCN_x4.pb"
_MODEL_NAME = "espcn"
_MODEL_SCALE = 4
# Unsharp after SR so Sharp is perceptibly crisper than Lanczos on IR colormaps.
_UNSHARP_AMOUNT = 0.85
_UNSHARP_SIGMA = 1.0

_lock = threading.Lock()
_sr_impl = None
_load_error: str | None = None


def resolve_espcn_model_path() -> Path:
    """Return the packaged ESPCN model path."""
    return Path(__file__).resolve().parent.parent / "resources" / "models" / _MODEL_FILENAME


# Backward-compatible alias used by older tests/imports.
def resolve_espcn_x2_model_path() -> Path:
    return resolve_espcn_model_path()


def dnn_superres_available() -> bool:
    """True when opencv-contrib dnn_superres can be imported."""
    try:
        from cv2 import dnn_superres  # noqa: F401

        return True
    except Exception:
        return False


def _ensure_model_loaded() -> bool:
    """Lazy-load the ESPCN model once. Returns False on permanent failure."""
    global _sr_impl, _load_error
    if _sr_impl is not None:
        return True
    if _load_error is not None:
        return False

    with _lock:
        if _sr_impl is not None:
            return True
        if _load_error is not None:
            return False
        try:
            from cv2 import dnn_superres

            model_path = resolve_espcn_model_path()
            if not model_path.is_file():
                raise FileNotFoundError(f"ESPCN model not found: {model_path}")
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(model_path))
            sr.setModel(_MODEL_NAME, _MODEL_SCALE)
            _sr_impl = sr
            return True
        except Exception as exc:
            _load_error = str(exc)
            return False


def _apply_unsharp(image_bgr: np.ndarray) -> np.ndarray:
    """Boost local contrast slightly after neural upscaling."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), _UNSHARP_SIGMA)
    return cv2.addWeighted(image_bgr, 1.0 + _UNSHARP_AMOUNT, blurred, -_UNSHARP_AMOUNT, 0)


def upscale_bgr_sharp(image_bgr: np.ndarray) -> np.ndarray:
    """Upscale a BGR image by 4x with ESPCN (+ unsharp), falling back to Lanczos."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("upscale_bgr_sharp expects a BGR image with shape (H, W, 3)")

    src_h, src_w = image_bgr.shape[:2]
    target_size = (src_w * _MODEL_SCALE, src_h * _MODEL_SCALE)

    if _ensure_model_loaded() and _sr_impl is not None:
        upscaled = _sr_impl.upsample(image_bgr)
    else:
        upscaled = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)

    return _apply_unsharp(upscaled)


def upscale_bgr_x2(image_bgr: np.ndarray) -> np.ndarray:
    """Compatibility wrapper: Sharp path is x4 now."""
    return upscale_bgr_sharp(image_bgr)


def reset_dnn_upscaler_for_tests() -> None:
    """Clear cached model state (tests only)."""
    global _sr_impl, _load_error
    with _lock:
        _sr_impl = None
        _load_error = None
