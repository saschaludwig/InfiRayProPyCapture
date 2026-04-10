"""Application state persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


STATE_PATH = Path.home() / ".irpropycapture_state.json"


@dataclass
class AppState:
    color_map: str = "Turbo"
    orientation: str = "Normal"
    show_temperature_grid: bool = False
    show_min_max_markers: bool = True
    grid_density: str = "Medium"
    temperature_format: str = "C"
    preview_interpolation: str = "Smooth"
    manual_range_enabled: bool = False
    manual_min_temp: float = 20.0
    manual_max_temp: float = 40.0
    camera_index: int = 0
    camera_name: str = ""
    last_image_save_dir: str = ""
    last_recording_save_dir: str = ""
    # Last main window frame geometry (client area + frame). Zero size = use defaults on startup.
    window_width: int = 0
    window_height: int = 0
    # Top-left of the window frame; negative = position was never saved, center on primary screen.
    window_x: int = -1
    window_y: int = -1


def load_state() -> AppState:
    if not STATE_PATH.exists():
        return AppState()
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        valid_keys = {field.name for field in fields(AppState)}
        filtered_payload = {key: value for key, value in payload.items() if key in valid_keys}
        return AppState(**filtered_payload)
    except Exception:
        return AppState()


def save_state(state: AppState) -> None:
    try:
        STATE_PATH.write_text(json.dumps(asdict(state), ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        # Keep state persistence best-effort and never crash UI.
        pass
