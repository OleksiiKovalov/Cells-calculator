"""Regression tests for CellDetectorPlugin size-range calibration."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

import UI.right_layout.plugins.CellDetectorPlugin as plugin_module
from UI.right_layout.plugins.CellDetectorPlugin import CellDetectorPlugin
from UI.app_globals import get_global, set_global
from model.PredictionResult import PredictionResult


class _RangeSliderStub:
    def __init__(self):
        self.calls = []

    def change_default(self, min_size, max_size):
        self.calls.append((min_size, max_size))


class _SignalStub:
    """Mock of PyQt5 Signal for testing without GUI."""
    
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        """Record emitted signal arguments."""
        self.calls.append(args)
    
    def connect(self, handler):
        """Mock connect to accept signal handlers."""
        pass


class _ReusableModelStub:
    def __init__(self, model_name):
        self.model_name = model_name
        self.path = "already-loaded-model"
        self.cell_counter = SimpleNamespace(original_image_path=None)
        self.calls = []

    def calculate(self, **kwargs):
        self.calls.append(kwargs)
        return {"Cells": 1, "Nuclei": -100, "%": -100}


class _ArtifactModelStub:
    def __init__(self, model_name):
        self.model_name = model_name
        self.path = "already-loaded-model"
        self.calls = []
        self.original_image = np.full((7, 8, 3), 30, dtype=np.uint8)
        self.inference_image = np.full((9, 10, 3), 90, dtype=np.uint8)
        self.detections = pd.DataFrame(
            {
                "class_id": [0],
                "confidence": [0.9],
                "box": [np.array([1, 1, 2, 2])],
                "scale": [1],
            }
        )
        self.result = PredictionResult(
            cells=self.detections,
            original_image=self.original_image,
            inference_image=self.inference_image,
        )
        self.cell_counter = SimpleNamespace(
            original_image_path=None,
            original_image=None,
            inference_image=None,
            detections=self.detections,
            prediction_image=None,
        )

    def calculate(self, **kwargs):
        self.calls.append(kwargs)
        return {"Cells": self.result, "Nuclei": -100, "%": -100}


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


def test_set_size_uses_original_image_area_for_detector_boxes():
    plugin = _build_plugin()
    plugin.model = SimpleNamespace(
        cell_counter=SimpleNamespace(
            original_image=np.zeros((2048, 2048, 3), dtype=np.uint8)
        )
    )
    detections = pd.DataFrame(
        {
            "box": [np.array([0.0, 0.0, 1024.0, 1024.0])],
            "scale": [1.0],
        }
    )

    plugin.set_size(detections)

    assert plugin.range_slider.calls[-1] == (0.25, 0.25)


def test_range_slider_filter_uses_display_image_area(monkeypatch):
    plugin = CellDetectorPlugin.__new__(CellDetectorPlugin)
    plugin.object_size = {"color_map": "tab20", "alpha": 0.75}
    plugin.model = None
    setattr(plugin, "draw_bounding_box", lambda: None)
    detections = pd.DataFrame(
        {
            "class_id": [0],
            "confidence": [0.9],
            "box": [np.array([0.0, 0.0, 1024.0, 1024.0])],
            "scale": [1.0],
        }
    )
    captured = {}

    def fake_render_detector_predictions(image, filtered, filename):
        captured["filtered"] = filtered
        return image

    monkeypatch.setattr(
        plugin_module,
        "render_detector_predictions",
        fake_render_detector_predictions,
    )
    set_global("detections", detections)
    set_global("image_display_base", np.zeros((2048, 2048, 3), dtype=np.uint8))
    set_global("image_inference", None)

    plugin.on_range_slider_changed(0.0, 0.25)

    assert captured["filtered"].shape[0] == 1


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
    plugin.plugin_signal = _SignalStub()  # type: ignore
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


def test_publish_inference_image_from_prediction_result(tmp_path):
    from UI.prediction_rendering import publish_inference_image

    inference_image = np.full((8, 9, 3), 120, dtype=np.uint8)
    detections = pd.DataFrame({"box": [np.array([1, 1, 2, 2])]})
    result = PredictionResult(cells=detections, inference_image=inference_image)
    model = SimpleNamespace(cell_counter=SimpleNamespace())
    output_path = tmp_path / "inference.png"

    set_global("image_inference", None)
    published = publish_inference_image(
        model,
        {"Cells": result},
        filename=str(output_path),
    )

    assert published is inference_image
    assert output_path.exists()
    assert np.array_equal(get_global("image_inference"), inference_image)
    assert np.array_equal(model.cell_counter.inference_image, inference_image)


def test_call_inference_publishes_returned_inference_image(monkeypatch, tmp_path):
    from UI.prediction_rendering import publish_inference_image

    plugin = CellDetectorPlugin.__new__(CellDetectorPlugin)
    plugin.model = _ArtifactModelStub("Detector")
    plugin.models = {
        "Detector": {
            "path": "already-loaded-model",
            "object_size": {},
            "model_type": "cellcounter",
        }
    }
    plugin.lsm_path = "sample.png"
    plugin.parametrs = {"Cell": 3, "Nuclei": 2}
    plugin.plugin_signal = _SignalStub()
    plugin.draw_bounding = 0
    output_path = tmp_path / "inference.png"

    monkeypatch.setattr(
        plugin_module,
        "publish_inference_image",
        lambda model, result: publish_inference_image(
            model,
            result,
            filename=str(output_path),
        ),
    )
    monkeypatch.setattr(
        CellDetectorPlugin, 
        'render_model_result',
        lambda self, model, result, *args, **kwargs: None
    )
    set_global("image_inference", None)

    result = plugin.call_inference("Detector")

    assert result["Cells"] is plugin.model.result
    assert output_path.exists()
    assert np.array_equal(
        get_global("image_inference"),
        plugin.model.inference_image,
    )
    assert np.array_equal(
        plugin.model.cell_counter.inference_image,
        plugin.model.inference_image,
    )


def test_render_model_result_uses_prediction_result_for_segmentation(monkeypatch):
    plugin = CellDetectorPlugin.__new__(CellDetectorPlugin)
    plugin.object_size = {"color_map": "tab20", "alpha": 0.75}
    original_image = np.full((12, 13, 3), 40, dtype=np.uint8)
    inference_image = np.full((16, 17, 3), 80, dtype=np.uint8)
    detections = pd.DataFrame(
        {
            "id_label": [1],
            "confidence": [0.9],
            "area": [0.1],
            "mask": [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            ],
        }
    )
    result = PredictionResult(
        cells=detections,
        original_image=original_image,
        inference_image=inference_image,
    )
    cell_counter = SimpleNamespace(
        original_image=None,
        inference_image=None,
        detections=detections,
        prediction_image=None,
    )
    model = SimpleNamespace(cell_counter=cell_counter)
    captured = {}

    def fake_plot_predictions_with_alignment(
        original,
        inference,
        pred_masks,
        filename,
        colormap,
        alpha,
        color_ids,
        mask_coordinate_space,
    ):
        captured["original"] = original
        captured["inference"] = inference
        captured["pred_masks"] = pred_masks
        captured["color_ids"] = color_ids
        return original

    monkeypatch.setattr(
        plugin_module,
        "plot_predictions_with_alignment",
        fake_plot_predictions_with_alignment,
    )
    set_global("image_display_base", None)
    set_global("image_inference", None)

    rendered = plugin.render_model_result(model, {"Cells": result})

    assert np.array_equal(captured["original"], original_image)
    assert np.array_equal(captured["inference"], inference_image)
    assert len(captured["pred_masks"]) == 1
    assert np.array_equal(captured["pred_masks"][0], detections["mask"].iloc[0])
    assert captured["color_ids"] == [1]
    assert cell_counter.prediction_image is rendered


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
