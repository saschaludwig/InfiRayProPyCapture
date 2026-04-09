# IrProPyCapture

Python port of the thermal camera viewer with a PySide6 desktop UI.

## Features

- Live thermal stream from USB camera via OpenCV (`cv2.VideoCapture`)
- Colormap rendering (`Turbo`, `Inferno`, `Jet`, `Hot`, `Rainbow`, `Parula`, `Viridis`, `Plasma`, `Coolwarm`, `Magma`, `Twilight`, `Autumn`, `Spring`, `Winter`, `HSV`, `Cubehelix`, `Cividis`, `Bone`)
- Temperature stats: Min, Max, Average, Center
- Optional temperature grid overlay
- Optional min/max markers with smart labels
- Preview scaling with aspect ratio lock and selectable interpolation (`Fast`/`Smooth`)
- Histogram with active colormap scale
- Bottom temperature history chart (Max/Min/Ave/Center) with time axis
- Manual temperature range (optional)
- Orientation controls (rotate/flip)
- PNG snapshot export
- MP4 recording export
- Persistent UI state (camera selection, colormap, controls)

## Requirements

- macOS, Linux, or Windows (OpenCV camera backend)
- Python 3.11+ (3.14 also works)

Python dependencies are listed in `requirements.txt`.

## Setup

From the `IrProPyCapture` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
source IrProPyCapture/.venv/bin/activate
python -m IrProPyCapture.irpropycapture.main
```

Or from inside `IrProPyCapture`:

```bash
source .venv/bin/activate
python -m irpropycapture.main
```

## Usage

1. Click **Refresh Cameras**.
2. Select the desired thermal camera.
3. Click **Start Camera**.
4. Adjust colormap, orientation, grid, manual range, and marker options.
5. Use **Save PNG** for snapshots and **Start/Stop Recording** for MP4.

## Acknowledgements

This Python port is based on the awesome original Swift/macOS project by **atomic14**.
Many thanks to the original author for the groundwork and implementation details.

Original repository:
- [https://github.com/atomic14/InfiRayCapture](https://github.com/atomic14/InfiRayCapture)

## Notes

- On first run, the OS may request camera permission.
- UI state is stored in `~/.irpropycapture_state.json`.
- If no image appears, verify camera access permissions and that the camera is not used by another application.

## Platform Notes

- macOS:
  - Current thermal-camera path prefers AVFoundation via OpenCV FFMPEG input source (`USB-Kamera:none`) for the known USB device.
  - If the thermal stream is not found, verify camera permissions in System Settings and close other apps using the device.

- Linux:
  - Capture uses OpenCV V4L2 backend and probes camera indices for raw-compatible formats (`Y16`/`YUYV`/`UYVY`).
  - Ensure your user can access `/dev/video*` devices (udev/group permissions).

- Windows:
  - Capture uses OpenCV MSMF backend first, then DirectShow fallback for broader USB camera compatibility.
  - If no stream opens, verify camera privacy settings and close software that may lock the camera.

