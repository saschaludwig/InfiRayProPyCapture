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
    measurement_temperatures: np.ndarray
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
        self._smoothed_measurement_temperatures: np.ndarray | None = None
        self._measurement_smoothing_was_enabled = False

    def get_temperatures(
        self,
        frame_bgr: np.ndarray,
        start_row: int = 192,
        smooth_for_high_range: bool = False,
    ) -> TemperatureResult:
        _ = start_row
        raw = frame_bgr
        if raw.shape[0] < self.height or raw.shape[1] < self.width * 2:
            raise ValueError(f"Unexpected frame shape: {raw.shape}")

        temperatures = self._decode_best_candidate(raw)
        measurement_temperatures = self._smooth_measurement_temperatures(temperatures, enabled=smooth_for_high_range)
        min_v = float(np.min(measurement_temperatures))
        max_v = float(np.max(measurement_temperatures))
        avg_v = float(np.mean(measurement_temperatures))
        center_i = self.width * (self.height // 2) + (self.width // 2)
        center_v = float(measurement_temperatures[center_i])
        histogram = self.compute_histogram(measurement_temperatures, min_v, max_v, bins=50)
        gen_before = self._history_generation
        if min_v > -20.0:
            self.update_temperature_history(min_v, max_v, avg_v, center_v)
        if self._history_generation != gen_before:
            self._last_history_snapshot = list(self.temperature_history)
        return TemperatureResult(
            temperatures=temperatures,
            measurement_temperatures=measurement_temperatures,
            min_value=min_v,
            max_value=max_v,
            average=avg_v,
            center=center_v,
            histogram=histogram,
            temperature_history=self._last_history_snapshot,
            history_generation=self._history_generation,
        )

    def _smooth_measurement_temperatures(self, temperatures: np.ndarray, enabled: bool) -> np.ndarray:
        """Smooth only the measurement matrix while keeping render matrix untouched."""
        if not enabled:
            self._smoothed_measurement_temperatures = None
            self._measurement_smoothing_was_enabled = False
            return temperatures

        alpha = 0.20
        if self._smoothed_measurement_temperatures is None or not self._measurement_smoothing_was_enabled:
            self._smoothed_measurement_temperatures = temperatures.copy()
            self._measurement_smoothing_was_enabled = True
            return self._smoothed_measurement_temperatures.copy()

        self._smoothed_measurement_temperatures = (
            (1.0 - alpha) * self._smoothed_measurement_temperatures + alpha * temperatures
        ).astype(np.float32, copy=False)
        self._measurement_smoothing_was_enabled = True
        return self._smoothed_measurement_temperatures.copy()

    def _decode_best_candidate(self, raw: np.ndarray) -> np.ndarray:
        rows_192 = raw[192 : 192 + self.height, : self.width * 2]
        packed = rows_192.reshape(self.height, self.width, 2)
        be = (packed[:, :, 0].astype(np.uint16) << 8) | packed[:, :, 1].astype(np.uint16)
        le = (packed[:, :, 1].astype(np.uint16) << 8) | packed[:, :, 0].astype(np.uint16)
        be_score = self._sampled_score(be)
        le_score = self._sampled_score(le)
        # When scores are close, decode both fully to avoid accuracy regressions.
        if abs(be_score - le_score) < 1.5:
            be_temps = self._decode_values_to_celsius(be)
            le_temps = self._decode_values_to_celsius(le)
            be_full_score = self._score(float(np.min(be_temps)), float(np.max(be_temps)), float(np.mean(be_temps)))
            le_full_score = self._score(float(np.min(le_temps)), float(np.max(le_temps)), float(np.mean(le_temps)))
            return be_temps if be_full_score >= le_full_score else le_temps
        if be_score > le_score:
            return self._decode_values_to_celsius(be)
        return self._decode_values_to_celsius(le)

    def _sampled_score(self, values: np.ndarray) -> float:
        sampled = values[::4, ::4]
        temps = self._decode_values_to_celsius(sampled)
        return self._score(float(np.min(temps)), float(np.max(temps)), float(np.mean(temps)))

    @staticmethod
    def _decode_values_to_celsius(values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32).reshape(-1) / 64.0) - 273.2

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
