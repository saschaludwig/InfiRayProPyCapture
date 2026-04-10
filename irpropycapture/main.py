"""Application entry point."""

from __future__ import annotations

import logging
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from irpropycapture.core.perf import is_perf_enabled
from irpropycapture.ui.main_window import MainWindow


def main() -> int:
    if is_perf_enabled():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)

    # Let Python handle SIGINT so Ctrl+C works even inside the Qt event loop.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Qt needs to yield to Python periodically to process the signal.
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
