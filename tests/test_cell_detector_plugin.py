"""Regression tests for CellDetectorPlugin size-range calibration."""

import numpy as np
import pandas as pd

from UI.right_layout.plugins.CellDetectorPlugin import CellDetectorPlugin


class _RangeSliderStub:
    def __init__(self):
        self.calls = []

    def change_default(self, min_size, max_size):
        self.calls.append((min_size, max_size))


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
