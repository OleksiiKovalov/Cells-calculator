"""Tests for generic per-model dead-cell metrics."""

import cv2
import numpy as np
import pandas as pd
import tiffile

import model.Model as model_module
from model import BaseModel as base_model_module
from model.CellCounter import CellCounter
from model.Model import Model, calculate_standard
from model.utils import (
    calculate_alive_percentage,
    extract_nuclei_channel,
    read_lsm_img,
)


class StubCellCounter:
    """Minimal stub that mimics detector-style output."""

    inference_duration = 0

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


class StubBaseModel(base_model_module.BaseModel):
    """Small BaseModel subclass for exercising count_cells."""

    def count_x20(self, input_image, filename):
        return pd.DataFrame({"box": [np.array([0, 0, 1, 1])]})


class EmptyDnn:
    """DNN stub that returns no detector rows."""

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return np.zeros((1, 5, 1), dtype=np.float32)


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
        nuclei_count=1
    )

    assert result["Nuclei"] == 1
    assert result["Cells"].shape[0] == 4
    assert result["%"] == 75.0


def test_count_cells_allows_input_that_is_already_temp_path(tmp_path, monkeypatch):
    temp_image_path = tmp_path / "cell_tmp_img.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(temp_image_path), image)

    monkeypatch.setattr(base_model_module, "IMAGE_FILE_NAME_TMP", str(temp_image_path))
    model = StubBaseModel.__new__(StubBaseModel)
    model.object_size = {"scale": 20}

    result = model.count_cells(str(temp_image_path))

    assert result.shape[0] == 1


def test_cellcounter_returns_empty_dataframe_when_nms_has_no_boxes(tmp_path):
    image_path = tmp_path / "cells.png"
    output_path = tmp_path / "detections.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    counter = CellCounter.__new__(CellCounter)
    counter.detections = None
    counter.model = EmptyDnn()
    counter.object_size = {"signal": lambda *args, **kwargs: None}
    counter.out_dir = tmp_path

    result = counter.count_x20(str(image_path), filename=str(output_path))

    assert list(result.columns) == ["class_id", "class_name", "confidence", "box", "scale"]
    assert result.empty
    assert output_path.exists()


def test_calculate_standard_omits_nuclei_metrics_when_not_provided(tmp_path):
    image_path = tmp_path / "cells.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    result = calculate_standard(StubCellCounter(), str(image_path))

    assert result["Nuclei"] == -100
    assert result["Cells"].shape[0] == 4
    assert result["%"] == -100


def test_get_nuclei_count_does_not_cache_for_same_image(tmp_path):
    image_path = tmp_path / "stained.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 255
    assert cv2.imwrite(str(image_path), image)

    model = Model.__new__(Model)
    model.nuclei_counter = StubNucleiCounter()

    first = model.get_nuclei_count(str(image_path), nuclei_channel=1)
    second = model.get_nuclei_count(str(image_path), nuclei_channel=1)

    assert first == 1
    assert second == 1
    assert model.nuclei_counter.calls == 2


def test_calculate_skips_nuclei_count_for_segmenter_model(tmp_path):
    image_path = tmp_path / "stained.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 255
    assert cv2.imwrite(str(image_path), image)

    model = Model.__new__(Model)
    model.model_type = "yolo"
    model.cell_counter = StubCellCounter()
    model.nuclei_counter = StubNucleiCounter()

    result = model.calculate(str(image_path), nuclei_channel=1)

    assert result["Nuclei"] == -100
    assert result["%"] == -100
    assert model.nuclei_counter.calls == 0


def test_calculate_lsm_skips_nuclei_count_for_segmenter_model(tmp_path, monkeypatch):
    image_path = tmp_path / "stained.lsm"
    image_path.write_bytes(b"placeholder")

    def fake_calculate_lsm(
        cell_counter,
        nuclei_counter,
        img_path,
        cell_channel=0,
        nuclei_channel=1,
        nuclei_count=None
    ):
        return {
            "Nuclei": nuclei_count,
            "Cells": cell_counter.count_cells(img_path),
            "%": calculate_alive_percentage(1, nuclei_count),
        }

    monkeypatch.setattr(model_module, "calculate_lsm", fake_calculate_lsm)

    model = Model.__new__(Model)
    model.model_type = "instanseg"
    model.cell_counter = StubCellCounter()
    model.nuclei_counter = StubNucleiCounter()

    result = model.calculate(str(image_path), nuclei_channel=1)

    assert result["Nuclei"] == -100
    assert result["%"] == -100
    assert model.nuclei_counter.calls == 0


def test_read_lsm_img_handles_single_channel_2d_series(tmp_path):
    image_path = tmp_path / "single_channel.lsm"
    image = np.zeros((16, 24), dtype=np.uint8)
    tiffile.imwrite(str(image_path), image)

    result = read_lsm_img(str(image_path))

    assert result.shape == (16, 24, 3)
    assert extract_nuclei_channel(str(image_path), nuclei_channel=1) is None


def test_read_lsm_img_uses_series_shape_for_multichannel_lsm(tmp_path):
    image_path = tmp_path / "two_channel.lsm"
    image = np.zeros((2, 20, 30), dtype=np.uint8)
    image[1, :, :] = 255
    tiffile.imwrite(str(image_path), image, metadata={"axes": "CYX"})

    result = read_lsm_img(str(image_path), nuclei_channel=1)
    nuclei = extract_nuclei_channel(str(image_path), nuclei_channel=1)

    assert result.shape == (20, 30, 3)
    assert nuclei.shape == (20, 30)
    assert nuclei.max() == 255
