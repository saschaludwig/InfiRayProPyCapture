"""Lightweight performance instrumentation helpers."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field


def is_perf_enabled() -> bool:
    """Return True when runtime performance reporting is enabled."""
    return os.environ.get("IRPRO_PY_CAPTURE_PERF", "0").strip() in {"1", "true", "True", "yes", "on"}


@dataclass
class _StageStats:
    """Running stats for one named stage."""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    recent_ms: deque[float] = field(default_factory=lambda: deque(maxlen=240))


class PerfReporter:
    """Collect durations per stage and periodically log summaries."""

    def __init__(self, name: str, interval_seconds: float = 2.0, enabled: bool | None = None) -> None:
        self._name = name
        self._interval_seconds = max(interval_seconds, 0.5)
        self._enabled = is_perf_enabled() if enabled is None else enabled
        self._stages: dict[str, _StageStats] = {}
        self._last_report_at = time.perf_counter()
        self._logger = logging.getLogger("irpropycapture.perf")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def observe(self, stage: str, duration_seconds: float) -> None:
        """Record one duration sample and report periodically."""
        if not self._enabled:
            return
        duration_ms = max(0.0, duration_seconds * 1000.0)
        stats = self._stages.setdefault(stage, _StageStats())
        stats.count += 1
        stats.total_ms += duration_ms
        stats.max_ms = max(stats.max_ms, duration_ms)
        stats.recent_ms.append(duration_ms)
        self._report_if_due()

    def measure(self, stage: str) -> "_PerfMeasure":
        """Create a context manager that reports stage duration."""
        return _PerfMeasure(self, stage)

    def _report_if_due(self) -> None:
        now = time.perf_counter()
        if (now - self._last_report_at) < self._interval_seconds:
            return
        self._last_report_at = now
        if not self._stages:
            return
        parts: list[str] = []
        for stage in sorted(self._stages):
            stats = self._stages[stage]
            avg_ms = stats.total_ms / float(stats.count)
            p95_ms = _percentile_95(stats.recent_ms)
            parts.append(
                f"{stage}: avg={avg_ms:.2f}ms p95={p95_ms:.2f}ms max={stats.max_ms:.2f}ms n={stats.count}"
            )
        self._logger.info("%s | %s", self._name, " | ".join(parts))


class _PerfMeasure:
    def __init__(self, reporter: PerfReporter, stage: str) -> None:
        self._reporter = reporter
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> "_PerfMeasure":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        elapsed = time.perf_counter() - self._start
        self._reporter.observe(self._stage, elapsed)
        return False


def _percentile_95(values: deque[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(0.95 * (len(sorted_values) - 1))
    return sorted_values[index]
