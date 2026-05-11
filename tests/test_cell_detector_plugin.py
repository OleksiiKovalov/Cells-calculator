"""Regression tests for CellDetectorPlugin size-range calibration."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from UI.right_layout.plugins.CellDetectorPlugin import CellDetectorPlugin
from UI.app_globals import get_global, set_global


class _RangeSliderStub:
    def __init__(self):
        self.calls = []

    def change_default(self, min_size, max_size):
        self.calls.append((min_size, max_size))


class _SignalStub:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _ReusableModelStub:
    def __init__(self, model_name):
        self.model_name = model_name
        self.path = "already-loaded-model"
        self.cell_counter = SimpleNamespace(original_image_path=None)
        self.calls = []

    def calculate(self, **kwargs):
        self.calls.append(kwargs)
        return {"Cells": 1, "Nuclei": -100, "%": -100}


def _build_plugin():
    plugin = CellDetectorPlugin.__new__(CellDetectorPlugin)
    plugin.range_slider = _RangeSliderStub()
    plugin.default_object_size = {"min_size": 0.0, "max_size": 1.0}
    return plugin


def test_set_size_prefers_segmentation_area_when_available():
    plugin = _build_plugin()
    detections = pd.DataFrame(
        {
            "box": [
                np.array([0.0, 0.0, 120.0, 120.0]),
                np.array([0.0, 0.0, 300.0, 300.0]),
            ],
            "area": [0.04, 0.09],
            "mask": [np.zeros((4, 2)), np.zeros((4, 2))],
        }
    )

    plugin.set_size(detections)

    assert plugin.range_slider.calls[-1] == (0.04, 0.09)


def test_set_size_falls_back_to_normalized_box_area_for_series():
    plugin = _build_plugin()
    boxes = pd.Series(
        [
            np.array([0.0, 0.0, 0.2, 0.3]),
            np.array([0.0, 0.0, 0.4, 0.5]),
        ]
    )

    plugin.set_size(boxes)

    assert plugin.range_slider.calls[-1] == (0.06, 0.2)


def test_call_inference_reuses_loaded_model_with_same_name(monkeypatch):
    plugin = CellDetectorPlugin.__new__(CellDetectorPlugin)
    plugin.model = _ReusableModelStub("Detector")
    plugin.models = {
        "Detector": {
            "path": "would-reload-before-fix",
            "object_size": {},
            "model_type": "cellcounter",
        }
    }
    plugin.lsm_path = "sample.lsm"
    plugin.parametrs = {"Cell": 3, "Nuclei": 2}
    plugin.plugin_signal = _SignalStub()
    plugin.draw_bounding = 0
    monkeypatch.setattr(
        "UI.right_layout.plugins.CellDetectorPlugin.Model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Model should not be reloaded")
        ),
    )

    result = plugin.call_inference("Detector")

    assert result == {"Cells": 1, "Nuclei": -100, "%": -100}
    assert plugin.model.calls == [
        {"img_path": "sample.lsm", "cell_channel": 3, "nuclei_channel": 2}
    ]
    assert plugin.model.cell_counter.original_image_path == "sample.lsm"


def test_render_filtered_result_keeps_full_detection_cache(monkeypatch):
    plugin = CellDetectorPlugin.__new__(CellDetectorPlugin)
    plugin.object_size = {"color_map": "tab20", "alpha": 0.75}
    full_detections = pd.DataFrame(
        {
            "class_id": [0, 0],
            "confidence": [0.9, 0.8],
            "box": [np.array([1, 1, 3, 3]), np.array([5, 5, 2, 2])],
            "scale": [1, 1],
        }
    )
    filtered_detections = full_detections.iloc[:1].copy()
    cell_counter = SimpleNamespace(
        original_image=np.zeros((12, 12, 3), dtype=np.uint8),
        detections=full_detections,
        prediction_image=None,
    )
    model = SimpleNamespace(cell_counter=cell_counter)

    monkeypatch.setattr(
        "UI.right_layout.plugins.CellDetectorPlugin.render_detector_predictions",
        lambda image, detections, filename: image,
    )
    set_global("detections", full_detections)

    plugin.render_model_result(
        model,
        {"Cells": filtered_detections},
        update_global_detections=False,
    )

    assert get_global("detections") is full_detections
    assert cell_counter.prediction_image is not None
