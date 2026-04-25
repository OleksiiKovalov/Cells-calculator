"""Tests for generic per-model dead-cell metrics."""

import cv2
import numpy as np
import pandas as pd

from model.Model import Model, calculate_standard
from model.utils import calculate_alive_percentage


class StubCellCounter:
    """Minimal stub that mimics detector-style output."""

    def count_cells(self, img_path):
        return pd.DataFrame({"box": [np.array([0, 0, 1, 1]) for _ in range(4)]})


class StubNucleiCounter:
    """Minimal stub for deterministic nuclei counts."""

    def __init__(self):
        self.calls = 0
        self.threshold = 100
        self.eps = 2
        self.min_samples = 5

    def countNuclei(self, img_channel):
        self.calls += 1
        return 1


def test_calculate_alive_percentage_returns_sentinel_when_no_cells():
    assert calculate_alive_percentage(0, 3) == -100


def test_calculate_standard_fills_nuclei_and_alive_for_regular_images(tmp_path):
    image_path = tmp_path / "stained.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 255
    assert cv2.imwrite(str(image_path), image)

    result = calculate_standard(
        StubCellCounter(),
        str(image_path),
        nuclei_count=1,
        nuclei_channel=1
    )

    assert result["Nuclei"] == 1
    assert result["Cells"].shape[0] == 4
    assert result["%"] == 75.0


def test_get_nuclei_count_uses_cache_for_same_image(tmp_path):
    image_path = tmp_path / "stained.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 255
    assert cv2.imwrite(str(image_path), image)

    model = Model.__new__(Model)
    model.nuclei_counter = StubNucleiCounter()
    Model._nuclei_cache.clear()

    first = model.get_nuclei_count(str(image_path), nuclei_channel=1)
    second = model.get_nuclei_count(str(image_path), nuclei_channel=1)

    assert first == 1
    assert second == 1
    assert model.nuclei_counter.calls == 1
