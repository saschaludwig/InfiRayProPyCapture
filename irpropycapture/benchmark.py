"""Synthetic benchmark for the thermal decode/render pipeline."""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from irpropycapture.core.image_processor import render_thermal_image
from irpropycapture.core.temperature_processor import TemperatureProcessor


def _build_synthetic_frame(frame_index: int) -> np.ndarray:
    y_grid, x_grid = np.mgrid[0:192, 0:256]
    base = 26.0 + 7.0 * np.sin((x_grid + frame_index) / 24.0) + 3.0 * np.cos((y_grid + frame_index) / 28.0)
    encoded = np.clip((base + 273.2) * 64.0, 0, 65535).astype(np.uint16)
    high = ((encoded >> 8) & 0xFF).astype(np.uint8)
    low = (encoded & 0xFF).astype(np.uint8)
    packed = np.empty((192, 512), dtype=np.uint8)
    packed[:, 0::2] = high
    packed[:, 1::2] = low
    frame = np.zeros((384, 512), dtype=np.uint8)
    frame[192:384, :] = packed
    return frame


def _p95(values_ms: list[float]) -> float:
    if not values_ms:
        return 0.0
    sorted_values = sorted(values_ms)
    index = int(0.95 * (len(sorted_values) - 1))
    return sorted_values[index]


def run_benchmark(frames: int, color_map_name: str) -> None:
    processor = TemperatureProcessor()
    decode_ms: list[float] = []
    render_ms: list[float] = []
    total_ms: list[float] = []
    for frame_index in range(frames):
        frame = _build_synthetic_frame(frame_index)
        total_start = time.perf_counter()
        decode_start = time.perf_counter()
        result = processor.get_temperatures(frame)
        decode_ms.append((time.perf_counter() - decode_start) * 1000.0)

        thermal = result.temperatures.reshape(192, 256)
        render_start = time.perf_counter()
        bgr = render_thermal_image(
            thermal,
            color_map_name=color_map_name,
            manual_range_enabled=False,
            manual_min_temp=0.0,
            manual_max_temp=1.0,
            auto_min_temp=result.min_value,
            auto_max_temp=result.max_value,
        )
        _ = cv2.resize(bgr, (1024, 768), interpolation=cv2.INTER_NEAREST)
        render_ms.append((time.perf_counter() - render_start) * 1000.0)
        total_ms.append((time.perf_counter() - total_start) * 1000.0)

    print(f"Frames: {frames}")
    print(
        "Decode  avg={:.2f}ms p95={:.2f}ms max={:.2f}ms".format(
            statistics.fmean(decode_ms), _p95(decode_ms), max(decode_ms)
        )
    )
    print(
        "Render  avg={:.2f}ms p95={:.2f}ms max={:.2f}ms".format(
            statistics.fmean(render_ms), _p95(render_ms), max(render_ms)
        )
    )
    print(
        "Total   avg={:.2f}ms p95={:.2f}ms max={:.2f}ms fps~{:.1f}".format(
            statistics.fmean(total_ms), _p95(total_ms), max(total_ms), 1000.0 / max(statistics.fmean(total_ms), 1e-6)
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic thermal pipeline benchmark.")
    parser.add_argument("--frames", type=int, default=300, help="Number of synthetic frames.")
    parser.add_argument("--colormap", type=str, default="Turbo", help="Colormap name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(frames=max(20, args.frames), color_map_name=args.colormap)
