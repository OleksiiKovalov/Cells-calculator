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


def test_strict_oracle_allows_degenerate_mask_as_rendering_edge():
    result = _result_with(np.array([[0.1, 0.1], [0.2, 0.2]]))

    fuzz_runtime.validate_model_result(
        _segmentation_case(),
        result,
        oracle_level="strict",
        image_size=(64, 64),
    )


def test_paranoid_oracle_rejects_degenerate_mask():
    result = _result_with(np.array([[0.1, 0.1], [0.2, 0.2]]))

    with pytest.raises(AssertionError, match="Degenerate mask"):
        fuzz_runtime.validate_model_result(
            _segmentation_case(),
            result,
            oracle_level="paranoid",
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


def test_normalize_loaded_array_preserves_singleton_spatial_axis_with_axes():
    image = np.zeros((3, 64, 1), dtype=np.uint8)

    normalized = fuzz_runtime.normalize_loaded_array(image, axes="CYX")

    assert normalized.shape == (64, 1, 3)


def test_normalize_loaded_array_flattens_non_spatial_axes_as_channels():
    image = np.zeros((2, 3, 5, 7), dtype=np.uint8)

    normalized = fuzz_runtime.normalize_loaded_array(image, axes="TCYX")

    assert normalized.shape == (5, 7, 6)


def test_write_minimizer_image_preserves_non_lsm_extension(tmp_path):
    image = np.zeros((8, 9, 3), dtype=np.uint8)

    image_path, image_meta = fuzz_runtime.write_minimizer_image(tmp_path, image, "tif")

    assert image_path.name == "input.tif"
    assert image_meta["extension"] == "tif"


def test_image_minimization_candidates_accept_one_dimensional_image():
    image = np.zeros((7,), dtype=np.uint8)

    candidates = fuzz_runtime.image_minimization_candidates(image)

    assert candidates
    assert all(candidate.ndim >= 2 for _name, candidate in candidates)
