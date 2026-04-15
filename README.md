# InfiRay P2 Pro Python Capture

Python port of the thermal camera viewer by **atomic14** with a PySide6 desktop UI.

## Features

- Live thermal stream from USB camera via OpenCV (`cv2.VideoCapture`)
- Colormap rendering: `Turbo`, `Inferno`, `Jet`, `Hot`, `Black-Hot`, `White-Hot`, `Rainbow`, `Rainbow-High-Contrast`, `Parula`, `Viridis`, `Plasma`, `Coolwarm`, `Magma`, `Twilight`, `Autumn`, `Spring`, `Winter`, `HSV`, `Cubehelix`, `Cividis`, `Bone`
- Temperature stats: Min, Max, Average, Center
- Optional temperature grid overlay
- Optional min/max markers with smart labels
- Preview scaling with aspect ratio lock and selectable interpolation (`Fast`/`Smooth`)
- Histogram with active colormap scale
- 1 minute temperature history chart (Max/Min/Ave/Center)
- Manual temperature range (optional)
- Camera temperature range switch (`Low (-20~150°C)` / `High (100~600°C)`)
- High-range measurement smoothing for more stable values (Min/Max/Average/Center, grid, min/max markers)
- Orientation controls (rotate/flip)
- PNG snapshot export
- MP4 recording export
- Shared export directory for PNG and MP4 (persisted)
- Keyboard shortcuts: `p` (save PNG instantly), `r` (start/stop recording instantly)
- Persistent UI state (camera selection, colormap, controls, window size)

## Screenshot ##
<img width="1187" height="1342" alt="screenshot" src="https://github.com/user-attachments/assets/374d06d2-63d7-4c20-9f1c-9e964ec39821" />

## Requirements

- macOS, Linux, or Windows (OpenCV camera backend)
- Python 3.11+

For camera temperature range switching, `pyusb` also requires a working `libusb` backend on the system.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and test tooling, use `pip install -e ".[dev]"`.

### USB backend setup (for temperature range switching)

- macOS:
  - `brew install libusb`
- Linux:
  - install `libusb-1.0` via your package manager (distribution-specific package name)
  - ensure user/device permissions allow USB control access
- Windows:
  - install a compatible USB backend/driver stack (https://github.com/libusb/libusb/wiki/Windows#user-content-How_to_use_libusb_on_Windows)

## Run

From the repository root:

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
   - These buttons keep the save dialog and update the shared export directory.
6. Use keyboard shortcuts for quick actions (without save dialog):
   - `p` saves a PNG immediately in the shared export directory.
   - `r` starts/stops MP4 recording immediately in the shared export directory.

## Acknowledgements

This Python port is based on the awesome original Swift/macOS project by **atomic14**.
https://github.com/atomic14/InfiRayCapture
Many thanks to the original author for the groundwork and implementation details.
Range selection was made possible thanks to hints from the project by **LeoDJ**
https://github.com/LeoDJ/P2Pro-Viewer


## Notes

- On first run, the OS may request camera permission.
- UI state is stored in `~/.irpropycapture_state.json`.
- PNG and MP4 share one persisted export directory.
- If no image appears, verify camera access permissions and that the camera is not used by another application.
- Optional performance logs can be enabled with `IRPRO_PY_CAPTURE_PERF=1`.

## Platform Notes

- macOS:
  - Current thermal-camera path prefers AVFoundation via OpenCV FFMPEG input source (`USB-Kamera:none`) for the known USB device.
  - If the thermal stream is not found, verify camera permissions in System Settings and close other apps using the device.

- Linux:
  - Capture uses OpenCV V4L2 backend and probes camera indices for raw-compatible formats (`Y16`/`YUYV`/`UYVY`).
  - Ensure your user can access `/dev/video*` devices (udev/group permissions).
  - For temperature range switching, also ensure USB access permissions for the thermal device (libusb path).

- Windows:
  - Capture uses OpenCV MSMF backend first, then DirectShow fallback for broader USB camera compatibility.
  - If no stream opens, verify camera privacy settings and close software that may lock the camera.
  - Temperature range switching depends on a working `pyusb` backend and a compatible USB driver binding.

