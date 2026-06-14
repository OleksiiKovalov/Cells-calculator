"""Smoke test: every enabled model runs through the NEW Model.inference seam
without crashing and returns a detections DataFrame with the canonical columns.

Models whose local weights are missing, whose backend isn't installed, or whose
built-in weights can't be downloaded are skipped (not failed).
"""
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = Path(__file__).parent / "data" / "TYPE_13_10.jpg"
CANON_COLS = {"id_label", "box", "mask", "confidence", "diameter", "area", "volume"}

KNOWN_MODELS = {
    "yolo": "model.YOLOSegmenter.YoloSegmenter",
    "instanseg": "model.InstanSegSegmenter.InstansegSegmenter",
    "cellpose": "model.CellposeSegmenter.CellposeSegmenter",
    "stardist": "model.StardistSegmenter.StardistSegmenter",
}
BACKEND_DEP = {
    "yolo": "ultralytics",
    "instanseg": "instanseg",
    "cellpose": "cellpose",
    "stardist": "stardist",
}
_DOWNLOAD_HINTS = ("download", "url", "connection", "http", "timed out",
                   "no such file", "not found", "certificate")


def _enabled_models():
    cfg = json.loads((ROOT / "modelconfig.json").read_text(encoding="utf-8"),
                     object_pairs_hook=OrderedDict)
    return [(name, data) for name, data in cfg.items()
            if str(data.get("enabled", "true")).lower() == "true"]


_CASES = _enabled_models()


@pytest.mark.parametrize("name,data", _CASES, ids=[c[0] for c in _CASES])
def test_model_runs_without_crashing(name, data):
    model_type = data["model_type"]
    dep = BACKEND_DEP.get(model_type)
    if dep:
        pytest.importorskip(dep)
    path = data["path"]
    if str(path).startswith("trainedmodels") and not (ROOT / path).exists():
        pytest.skip(f"weights missing: {path}")

    from ui.app_globals import register_model
    from model.Model import Model
    from model.utils import read_img

    register_model(model_type, KNOWN_MODELS[model_type], False)
    img = read_img(str(SAMPLE_IMAGE))
    # Downscale for a fast smoke pass.
    scale = 512 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    try:
        model = Model(path=path, model_type=model_type, model_data=data, model_name=name)
        det = model.inference(img)
    except Exception as exc:  # built-in weights download / offline -> skip
        if any(h in str(exc).lower() for h in _DOWNLOAD_HINTS):
            pytest.skip(f"{name}: backend/weights unavailable ({exc})")
        raise

    assert det is not None, f"{name} returned None"
    assert CANON_COLS.issubset(set(det.columns)), f"{name} columns: {list(det.columns)}"
