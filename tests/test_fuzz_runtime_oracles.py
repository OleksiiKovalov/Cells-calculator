import numpy as np
import pandas as pd
import pytest

from scripts import fuzz_runtime


def _segmentation_case():
    return {
        "model_data": {"model_type": "yolo"},
        "model_name": "YOLO test",
    }


def _result_with(mask, box=None):
    cells = pd.DataFrame(
        {
            "id_label": [1],
            "box": [np.array([0.1, 0.1, 0.2, 0.2]) if box is None else box],
            "mask": [mask],
            "confidence": [0.9],
            "diameter": [0.1],
            "area": [0.01],
            "volume": [0.001],
        }
    )
    return {"Cells": cells, "Nuclei": -100, "%": -100}


def test_strict_oracle_accepts_valid_segmentation_result():
    result = _result_with(
        np.array(
            [
                [0.1, 0.1],
                [0.2, 0.1],
                [0.2, 0.2],
            ]
        )
    )

    fuzz_runtime.validate_model_result(
        _segmentation_case(),
        result,
        oracle_level="strict",
        image_size=(64, 64),
    )


def test_strict_oracle_rejects_degenerate_mask():
    result = _result_with(np.array([[0.1, 0.1], [0.2, 0.2]]))

    with pytest.raises(AssertionError, match="Degenerate mask"):
        fuzz_runtime.validate_model_result(
            _segmentation_case(),
            result,
            oracle_level="strict",
            image_size=(64, 64),
        )


def test_strict_oracle_rejects_negative_box_extent():
    result = _result_with(
        np.array(
            [
                [0.1, 0.1],
                [0.2, 0.1],
                [0.2, 0.2],
            ]
        ),
        box=np.array([0.1, 0.1, -0.2, 0.2]),
    )

    with pytest.raises(AssertionError, match="Negative box extent"):
        fuzz_runtime.validate_model_result(
            _segmentation_case(),
            result,
            oracle_level="strict",
            image_size=(64, 64),
        )


def test_counts_fingerprint_ignores_numeric_summary():
    first = _result_with(
        np.array(
            [
                [0.1, 0.1],
                [0.2, 0.1],
                [0.2, 0.2],
            ]
        )
    )
    second = _result_with(
        np.array(
            [
                [0.1, 0.1],
                [0.2, 0.1],
                [0.2, 0.2],
            ]
        )
    )
    second["Cells"].loc[0, "confidence"] = 0.7

    assert (
        fuzz_runtime.result_fingerprint(first, "counts")
        == fuzz_runtime.result_fingerprint(second, "counts")
    )
    assert (
        fuzz_runtime.result_fingerprint(first, "full")
        != fuzz_runtime.result_fingerprint(second, "full")
    )
