"""Thermal image rendering helpers."""

from __future__ import annotations

import cv2
import numpy as np


AVAILABLE_COLOR_MAPS = [
    "Turbo",
    "Inferno",
    "Jet",
    "Hot",
    "Black-Hot",
    "White-Hot",
    "Rainbow",
    "Rainbow-High-Contrast",
    "Parula",
    "Viridis",
    "Plasma",
    "Coolwarm",
    "Magma",
    "Twilight",
    "Autumn",
    "Spring",
    "Winter",
    "HSV",
    "Cubehelix",
    "Cividis",
    "Bone",
]


_CUSTOM_CONTROL_POINTS = {
    "Rainbow": [
        (0.5, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
    "Cubehelix": [
        (0.0, 0.0, 0.0),
        (0.2196, 0.1039, 0.3637),
        (0.3637, 0.37, 0.5847),
        (0.4769, 0.6008, 0.6823),
        (0.6456, 0.7474, 0.6137),
        (0.8645, 0.724, 0.578),
        (1.0, 1.0, 1.0),
    ],
    "Black-Hot": [
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    ],
    "White-Hot": [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ],
    "Rainbow-High-Contrast": [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 0.0, 0.5),
        (0.0, 1.0, 1.0),
        (0.0, 0.5, 0.0),
        (1.0, 1.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ],
}


def _build_colormap_lut(control_points: list[tuple[float, float, float]]) -> np.ndarray:
    anchors = np.array(control_points, dtype=np.float32)
    x_src = np.linspace(0.0, 1.0, anchors.shape[0], dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    r = np.interp(x_dst, x_src, anchors[:, 0])
    g = np.interp(x_dst, x_src, anchors[:, 1])
    b = np.interp(x_dst, x_src, anchors[:, 2])
    rgb = np.stack((r, g, b), axis=1)
    bgr = np.ascontiguousarray(rgb[:, ::-1] * 255.0, dtype=np.uint8)
    return bgr.reshape(256, 1, 3)


COLOR_MAP_TO_CV2 = {
    "Jet": "COLORMAP_JET",
    "Inferno": "COLORMAP_INFERNO",
    "Turbo": "COLORMAP_TURBO",
    "Magma": "COLORMAP_MAGMA",
    "Plasma": "COLORMAP_PLASMA",
    "Bone": "COLORMAP_BONE",
    "Hot": "COLORMAP_HOT",
    "Parula": "COLORMAP_PARULA",
    "Viridis": "COLORMAP_VIRIDIS",
    "Coolwarm": "COLORMAP_COOL",
    "Twilight": "COLORMAP_TWILIGHT",
    "Autumn": "COLORMAP_AUTUMN",
    "Spring": "COLORMAP_SPRING",
    "Winter": "COLORMAP_WINTER",
    "HSV": "COLORMAP_HSV",
    "Cividis": "COLORMAP_CIVIDIS",
}

CUSTOM_COLOR_MAP_LUTS = {
    name: _build_colormap_lut(points) for name, points in _CUSTOM_CONTROL_POINTS.items()
}


def _resolve_cv2_colormap(color_map_name: str) -> int | None:
    cv2_attr_name = COLOR_MAP_TO_CV2.get(color_map_name)
    if cv2_attr_name is None:
        return None
    return getattr(cv2, cv2_attr_name, None)


def format_temperature(value_celsius: float, temp_unit: str) -> str:
    if temp_unit == "F":
        value = value_celsius * 9.0 / 5.0 + 32.0
        return f"{value:.1f} F"
    return f"{value_celsius:.1f} C"


def render_thermal_image(
    thermal_celsius: np.ndarray,
    color_map_name: str,
    manual_range_enabled: bool,
    manual_min_temp: float,
    manual_max_temp: float,
) -> np.ndarray:
    if manual_range_enabled:
        t_min = manual_min_temp
        t_max = max(manual_max_temp, manual_min_temp + 0.1)
    else:
        t_min = float(np.min(thermal_celsius))
        t_max = float(np.max(thermal_celsius))
        if t_max - t_min < 0.1:
            t_max = t_min + 0.1

    normalized = np.clip((thermal_celsius - t_min) / (t_max - t_min), 0.0, 1.0)
    gray = (normalized * 255.0).astype(np.uint8)
    color_id = _resolve_cv2_colormap(color_map_name)
    if color_id is not None:
        return cv2.applyColorMap(gray, color_id)
    lut = CUSTOM_COLOR_MAP_LUTS.get(color_map_name)
    if lut is not None:
        return cv2.applyColorMap(gray, lut)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def draw_temperature_grid(image_bgr: np.ndarray, thermal_celsius: np.ndarray, density: str, temp_unit: str) -> None:
    if density == "Low":
        step_x, step_y = 64, 48
    elif density == "High":
        step_x, step_y = 24, 18
    else:
        step_x, step_y = 32, 24

    h, w = thermal_celsius.shape
    scale_x = image_bgr.shape[1] / float(w)
    scale_y = image_bgr.shape[0] / float(h)
    for y in range(step_y // 2, h, step_y):
        for x in range(step_x // 2, w, step_x):
            value = float(thermal_celsius[y, x])
            text = format_temperature(value, temp_unit)
            px = int(x * scale_x)
            py = int(y * scale_y)
            cv2.putText(image_bgr, text, (px - 26, py), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def apply_orientation(image_bgr: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "Rotate Left":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == "Rotate Right":
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "Flip Horizontal":
        return cv2.flip(image_bgr, 1)
    if orientation == "Flip Vertical":
        return cv2.flip(image_bgr, 0)
    return image_bgr

