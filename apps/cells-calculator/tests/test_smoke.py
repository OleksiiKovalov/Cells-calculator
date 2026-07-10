"""Smoke test: every enabled model runs through the NEW Model.inference seam
without crashing and returns a detections DataFrame with the canonical columns.

Models whose local weights are missing, whose backend isn't installed, or whose
built-in weights can't be downloaded are skipped (not failed).
"""
from pathlib import Path

import cv2
import pytest

from tests._models import BACKEND_DEP, DOWNLOAD_HINTS, KNOWN_MODELS, ROOT, enabled_models

SAMPLE_IMAGE = Path(__file__).parent / "data" / "TYPE_13_10.jpg"
CANON_COLS = {"id_label", "box", "mask", "confidence", "diameter", "area", "volume"}

_CASES = list(enabled_models().items())


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
        if any(h in str(exc).lower() for h in DOWNLOAD_HINTS):
            pytest.skip(f"{name}: backend/weights unavailable ({exc})")
        raise

    assert det is not None, f"{name} returned None"
    assert CANON_COLS.issubset(set(det.columns)), f"{name} columns: {list(det.columns)}"
