"""GUI-layer tests (run off-screen via the qapp fixture)."""
from pathlib import Path

import pandas as pd


def test_compute_grid_dims_bounds(qapp):
    from ui.MainWindow import (
        _compute_grid_dims,
        DETECTION_GRID_MAX_COLS,
        DETECTION_GRID_MAX_ROWS,
    )
    assert _compute_grid_dims(100, 100, 0) == (1, 1)
    cols, rows = _compute_grid_dims(1000, 1000, 100000)
    assert 1 <= cols <= DETECTION_GRID_MAX_COLS
    assert 1 <= rows <= DETECTION_GRID_MAX_ROWS
    # wide image -> more columns than rows
    wcols, wrows = _compute_grid_dims(4000, 500, 400)
    assert wcols >= wrows


def test_load_image_sample(qapp):
    from ui.MainWindow import MainWindow
    win = MainWindow()
    assert win.load_image(str(Path(__file__).resolve().parents[1] / "data" / "TYPE_13_10.jpg")) is True
    assert win.viewer.has_image()
    w, h, ch = win._image_dims
    assert w > 0 and h > 0 and ch == 3


def test_show_original_toggle(qapp):
    """'Show original' must stay toggled and swap original vs prediction correctly.

    Regression: stateChanged(int) compared against the Qt.CheckState enum was
    always False under PySide6, so checking the box showed the masks and
    silently unchecked itself.
    """
    import numpy as np
    from ui.MainWindow import MainWindow
    win = MainWindow()
    original = np.zeros((8, 8, 3), np.uint8); original[..., 0] = 10   # red marker
    prediction = np.zeros((8, 8, 3), np.uint8); prediction[..., 1] = 200  # green marker
    win.original_image = original
    win.prediction_image = prediction

    win._set_current_image(False)  # state after Calculate: show prediction/masks
    assert win.cbShowOriginal.isChecked() is False

    win.cbShowOriginal.setChecked(True)          # user asks for the original
    assert win.cbShowOriginal.isChecked() is True  # stays checked (no self-uncheck)
    shown = win.viewer.get_image()
    assert shown is not None and shown[..., 0].max() == 10 and shown[..., 1].max() == 0

    win.cbShowOriginal.setChecked(False)         # back to masks
    shown = win.viewer.get_image()
    assert shown is not None and shown[..., 1].max() == 200


def _win_with_detections(qapp, df):
    from ui.MainWindow import MainWindow
    win = MainWindow()
    win._image_dims = (512, 512, 3)
    win._image_name = "t.png"
    win.inference_duration = 1.0

    class _M:
        model_name = "YOLO"
    win.current_model = _M()
    win.detections = df
    return win


def test_show_detection_stats_includes_micrometers(qapp):
    df = pd.DataFrame({"diameter": [0.03], "area": [0.001], "volume": [1e-5]})
    win = _win_with_detections(qapp, df)
    win.spnUmPerMm.setValue(0.325)
    win.show_detection_stats()
    txt = win._info_panel._text_edit.toPlainText()
    assert "µm/px" in txt and "µm²" in txt and "µm³" in txt
    assert "Objects detected: 1" in txt


def test_show_detection_stats_empty_guard(qapp):
    win = _win_with_detections(qapp, pd.DataFrame({"diameter": [], "area": [], "volume": []}))
    win.show_detection_stats()  # must not raise / produce NaN
    txt = win._info_panel._text_edit.toPlainText()
    assert "Objects detected: 0" in txt
    assert "nan" not in txt.lower()


def test_show_detection_stats_no_micrometers_when_scale_zero(qapp):
    df = pd.DataFrame({"diameter": [0.03], "area": [0.001], "volume": [1e-5]})
    win = _win_with_detections(qapp, df)
    win.spnUmPerMm.setValue(0.0)
    win.show_detection_stats()
    txt = win._info_panel._text_edit.toPlainText()
    assert "µm/px" not in txt
