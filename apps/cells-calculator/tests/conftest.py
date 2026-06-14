"""Shared pytest configuration for the Cells Calculator test suite."""
import os

# Render Qt off-screen so GUI tests need no display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Keep ultralytics/YOLO config inside the repo cache; quiet TensorFlow.
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), ".cache", "ultralytics"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# On Windows TensorFlow's native runtime must initialize before torch / PyQt5
# or its DLL load fails (see main._preload_tensorflow_if_needed). Import it
# first here, when available, so StarDist tests don't hit that load-order crash.
try:  # pragma: no cover - environment dependent
    import tensorflow  # noqa: F401
except Exception:
    pass

import pytest


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for GUI-touching tests."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
