"""Golden-image regression tests for nuclei counting and app model outputs."""

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Union

# Keep libraries that write settings on import inside the repo-local cache.
ULTRALYTICS_CONFIG_DIR = Path.cwd() / ".cache" / "ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pytest

from model.Model import Model
from model.NucleiCounter import NucleiCounter
from model.PredictionResult import unwrap_prediction_cells
import model.utils as model_utils


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "golden" / "image_regression_baseline.json"
MODEL_CONFIG_PATH = ROOT / "modelconfig.json"
KNOWN_MODELS = {
    "cellcounter": "model.CellCounter.CellCounter",
    "cellpose": "model.CellposeSegmenter.CellposeSegmenter",
    "yolo": "model.YOLOSegmenter.YoloSegmenter",
    "instanseg": "model.InstanSegSegmenter.InstansegSegmenter",
    "stardist": "model.StardistSegmenter.StardistSegmenter",
}
OBJECT_SIZE = {
    "min_size": 100,
    "max_size": 0.0,
    "signal": lambda *args, **kwargs: None,
    "round_parametr_slider": 10**6,
    "round_parametr_value_input": 10**4,
    "color_map": "viridis",
    "color_map_list": [],
    "line_width": 100.0,
    "scale": 20,
    "um_per_px": 0.325,
}
PIXEL_VALUE_ABS_TOLERANCE = 2
PIXEL_SUM_ABS_TOLERANCE = 10
PIXEL_SUM_REL_TOLERANCE = 0.002
OBJECT_COUNT_ABS_TOLERANCE = 1
OBJECT_COUNT_REL_TOLERANCE = 0.03
MODEL_COUNT_ABS_TOLERANCE = 5
MODEL_COUNT_REL_TOLERANCE = 0.08
MODEL_FLOAT_TOLERANCES = {
    "confidence_sum": (5.0, 0.10),
    "confidence_mean": (0.05, 0.10),
    "area_sum": (0.03, 0.15),
    "area_mean": (0.0005, 0.20),
    "diameter_mean": (0.005, 0.15),
}
MetricValue = Union[float, int]


def _load_golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _load_enabled_model_config():
    models = json.loads(
        MODEL_CONFIG_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=OrderedDict,
    )
    return OrderedDict(
        (name, data)
        for name, data in models.items()
        if "enabled" not in data or str(data.get("enabled", "true")).lower() == "true"
    )


def _model_cases():
    golden_models = _load_golden()["models"]
    cases = []
    for model_name, model_data in golden_models.items():
        for image_path, expected in model_data["images"].items():
            cases.append((model_name, model_data, image_path, expected))
    return cases


def _fingerprint(image):
    if image is None:
        return None
    array = np.ascontiguousarray(image)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": int(array.min()) if array.size else None,
        "max": int(array.max()) if array.size else None,
        "sum": int(array.sum()) if array.size else 0,
    }


def _allowed_delta(expected, abs_tolerance, rel_tolerance=0.0):
    return max(abs_tolerance, abs(float(expected)) * rel_tolerance)


def _assert_close(actual, expected, label, abs_tolerance, rel_tolerance=0.0):
    delta = _allowed_delta(expected, abs_tolerance, rel_tolerance)
    assert abs(float(actual) - float(expected)) <= delta, (
        f"{label}: expected {expected} +/- {delta}, got {actual}"
    )


def _assert_image_stats_close(actual, expected, label):
    assert actual is not None, f"{label}: expected image stats, got None"
    assert actual["shape"] == expected["shape"], (
        f"{label}.shape: expected {expected['shape']}, got {actual['shape']}"
    )
    assert actual["dtype"] == expected["dtype"], (
        f"{label}.dtype: expected {expected['dtype']}, got {actual['dtype']}"
    )
    for key in ("min", "max"):
        _assert_close(
            actual[key],
            expected[key],
            f"{label}.{key}",
            abs_tolerance=PIXEL_VALUE_ABS_TOLERANCE,
        )
    _assert_close(
        actual["sum"],
        expected["sum"],
        f"{label}.sum",
        abs_tolerance=PIXEL_SUM_ABS_TOLERANCE,
        rel_tolerance=PIXEL_SUM_REL_TOLERANCE,
    )


def _assert_count_close(actual, expected, label):
    _assert_close(
        actual,
        expected,
        label,
        abs_tolerance=OBJECT_COUNT_ABS_TOLERANCE,
        rel_tolerance=OBJECT_COUNT_REL_TOLERANCE,
    )


def _get_registered_model(model_type):
    return {"model_class": KNOWN_MODELS[model_type]}


def _summarize_model_output(result) -> dict[str, MetricValue]:
    cells = unwrap_prediction_cells(result["Cells"])
    if not hasattr(cells, "shape"):
        return {"count": int(cells)}

    summary: dict[str, MetricValue] = {"count": int(cells.shape[0])}
    if "confidence" in cells:
        confidence = cells["confidence"].astype(float)
        summary["confidence_sum"] = round(float(confidence.sum()), 6)
        summary["confidence_mean"] = (
            round(float(confidence.mean()), 6) if len(confidence) else 0.0
        )
    if "area" in cells:
        areas = cells["area"].astype(float)
        summary["area_sum"] = round(float(areas.sum()), 6)
        summary["area_mean"] = round(float(areas.mean()), 6) if len(areas) else 0.0
    if "diameter" in cells:
        diameters = cells["diameter"].astype(float)
        summary["diameter_mean"] = (
            round(float(diameters.mean()), 6) if len(diameters) else 0.0
        )
    return summary


def test_model_baseline_covers_enabled_app_models():
    enabled_models = set(_load_enabled_model_config())
    golden_models = set(_load_golden()["models"])

    assert golden_models == enabled_models, (
        "Model golden baseline must match enabled models from modelconfig.json. "
        f"Missing: {sorted(enabled_models - golden_models)}; "
        f"extra: {sorted(golden_models - enabled_models)}"
    )


def test_nuclei_counter_matches_committed_image_baseline():
    golden = _load_golden()["nuclei"]
    counter = NucleiCounter()

    for image_path, expected in golden.items():
        channel = model_utils.extract_nuclei_channel(
            str(ROOT / image_path),
            nuclei_channel=1,
        )
        actual_count = counter.countNuclei(channel) if channel is not None else None

        _assert_image_stats_close(
            _fingerprint(channel),
            expected["channel"],
            f"nuclei[{image_path}].channel",
        )
        _assert_count_close(
            actual_count,
            expected["count"],
            f"nuclei[{image_path}].count",
        )


@pytest.mark.parametrize(
    "model_name,model_data,image_path,expected",
    _model_cases(),
)
def test_enabled_app_model_matches_golden_baseline(
    model_name,
    model_data,
    image_path,
    expected,
):
    enabled_config = _load_enabled_model_config()
    config = enabled_config[model_name]
    model_path = ROOT / config["path"]
    if str(config["path"]).startswith("trainedmodels") and not model_path.exists():
        raise AssertionError(
            f"{model_name} weights are missing at {model_path}. "
            "Run python scripts/download_models.py before pytest."
        )

    logger = logging.getLogger("image_golden_regression")
    logger.addHandler(logging.NullHandler())
    model = Model(
        logger=logger,
        get_registered_model=_get_registered_model,
        path=config["path"],
        object_size=OBJECT_SIZE.copy(),
        model_type=config["model_type"],
        model_data=config,
        model_name=model_name,
    )
    model.cell_counter.original_image_path = str(ROOT / image_path)

    result = model.calculate(str(ROOT / image_path))
    actual = _summarize_model_output(result)
    label = f"model[{model_name}][{image_path}]"

    _assert_close(
        actual["count"],
        expected["count"],
        f"{label}.count",
        MODEL_COUNT_ABS_TOLERANCE,
        MODEL_COUNT_REL_TOLERANCE,
    )
    for metric, expected_value in expected.items():
        if metric == "count":
            continue
        assert metric in actual, f"{label}.{metric}: missing in actual result {actual}"
        abs_tolerance, rel_tolerance = MODEL_FLOAT_TOLERANCES[metric]
        _assert_close(
            actual[metric],
            expected_value,
            f"{label}.{metric}",
            abs_tolerance,
            rel_tolerance,
        )
