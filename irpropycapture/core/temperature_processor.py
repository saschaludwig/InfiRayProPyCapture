"""Temperature decoding."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class HistogramPoint:
    x: float
    y: int


@dataclass
class TemperatureHistoryPoint:
    timestamp: float
    min_value: float
    max_value: float
    average: float
    center: float


@dataclass
class TemperatureResult:
    temperatures: np.ndarray
    min_value: float
    max_value: float
    average: float
    center: float
    histogram: list[HistogramPoint]
    temperature_history: list[TemperatureHistoryPoint]
    history_generation: int


class TemperatureProcessor:
    def __init__(self) -> None:
        self.width = 256
        self.height = 192
        self.temperature_history: deque[TemperatureHistoryPoint] = deque()
        self.history_update_interval = 0.1
        self.max_history_seconds = 60.0
        self._history_generation: int = 0
        self._last_history_snapshot: list[TemperatureHistoryPoint] = []

    def get_temperatures(self, frame_bgr: np.ndarray, start_row: int = 192) -> TemperatureResult:
        raw = frame_bgr
        if raw.shape[0] < self.height or raw.shape[1] < self.width * 2:
            raise ValueError(f"Unexpected frame shape: {raw.shape}")

        candidates = self._build_candidates(raw)
        best = max(candidates, key=lambda item: item["score"])
        temperatures = best["temps"]

        min_v = float(np.min(temperatures))
        max_v = float(np.max(temperatures))
        avg_v = float(np.mean(temperatures))
        center_i = self.width * (self.height // 2) + (self.width // 2)
        center_v = float(temperatures[center_i])
        histogram = self.compute_histogram(temperatures, min_v, max_v, bins=50)
        gen_before = self._history_generation
        if min_v > -20.0:
            self.update_temperature_history(min_v, max_v, avg_v, center_v)
        if self._history_generation != gen_before:
            self._last_history_snapshot = list(self.temperature_history)
        return TemperatureResult(
            temperatures=temperatures,
            min_value=min_v,
            max_value=max_v,
            average=avg_v,
            center=center_v,
            histogram=histogram,
            temperature_history=self._last_history_snapshot,
            history_generation=self._history_generation,
        )

    def _build_candidates(self, raw: np.ndarray) -> list[dict]:
        rows_192 = raw[192 : 192 + self.height, : self.width * 2]
        candidates: list[dict] = []
        packed = rows_192.reshape(self.height, self.width, 2)
        be = (packed[:, :, 0].astype(np.uint16) << 8) | packed[:, :, 1].astype(np.uint16)
        le = (packed[:, :, 1].astype(np.uint16) << 8) | packed[:, :, 0].astype(np.uint16)
        candidates.append(self._candidate_entry("row192_be", be))
        candidates.append(self._candidate_entry("row192_le", le))
        return candidates

    def _candidate_entry(self, name: str, values: np.ndarray) -> dict:
        temps = (values.astype(np.float32).reshape(-1) / 64.0) - 273.2
        min_v = float(np.min(temps))
        max_v = float(np.max(temps))
        avg_v = float(np.mean(temps))
        score = self._score(min_v, max_v, avg_v)
        return {"name": name, "temps": temps, "min": min_v, "max": max_v, "avg": avg_v, "score": score}

    @staticmethod
    def _score(min_v: float, max_v: float, avg_v: float) -> float:
        spread = max_v - min_v
        score = 0.0
        if -20.0 <= min_v <= 60.0:
            score += 3.0
        if -5.0 <= avg_v <= 90.0:
            score += 4.0
        if 10.0 <= max_v <= 160.0:
            score += 3.0
        if 1.0 <= spread <= 80.0:
            score += 3.0
        # Hard penalties for obvious garbage decodes.
        if max_v > 250.0:
            score -= 12.0
        if min_v <= -120.0:
            score -= 12.0
        return score

    @staticmethod
    def compute_histogram(values: np.ndarray, min_value: float, max_value: float, bins: int) -> list[HistogramPoint]:
        temp_range = max_value - min_value
        if bins < 2 or temp_range <= 0.0:
            return []
        bin_width = temp_range / float(bins - 1)
        bin_indexes = np.floor((values - min_value) / bin_width).astype(np.int32)
        bin_indexes = np.clip(bin_indexes, 0, bins - 1)
        counts = np.bincount(bin_indexes, minlength=bins)
        return [HistogramPoint(x=float(i) * bin_width + min_value, y=int(counts[i])) for i in range(bins)]

    def update_temperature_history(self, min_value: float, max_value: float, average: float, center: float) -> None:
        now = time.time()
        if self.temperature_history and (now - self.temperature_history[-1].timestamp) < self.history_update_interval:
            return
        self.temperature_history.append(
            TemperatureHistoryPoint(
                timestamp=now,
                min_value=min_value,
                max_value=max_value,
                average=average,
                center=center,
            )
        )
        while len(self.temperature_history) >= 2 and (self.temperature_history[-1].timestamp - self.temperature_history[0].timestamp) > self.max_history_seconds:
            self.temperature_history.popleft()
        self._history_generation += 1
