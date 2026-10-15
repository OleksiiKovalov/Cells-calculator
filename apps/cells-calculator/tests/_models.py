"""Shared model-registry helpers for the test suite.

Centralizes the small tables and the enabled-models reader that several test
modules (smoke, runtime fuzz, golden regressions) and the fuzzing runner used to
each redeclare.
"""
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "modelconfig.json"

# model_type -> dotted segmenter class path (for register_model).
KNOWN_MODELS = {
    "yolo": "model.YOLOSegmenter.YoloSegmenter",
    "instanseg": "model.InstanSegSegmenter.InstansegSegmenter",
    "cellpose": "model.CellposeSegmenter.CellposeSegmenter",
    "stardist": "model.StardistSegmenter.StardistSegmenter",
}

# model_type -> importable backend package (for pytest.importorskip).
BACKEND_DEP = {
    "yolo": "ultralytics",
    "instanseg": "instanseg",
    "cellpose": "cellpose",
    "stardist": "stardist",
}

# Error-message substrings that mean "backend/weights unavailable" — such cases
# are skipped rather than failed.
DOWNLOAD_HINTS = ("download", "url", "connection", "http", "timed out",
                  "no such file", "not found", "certificate")


def enabled_models():
    """Return an OrderedDict of {name: config} for models not disabled in modelconfig.json."""
    cfg = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"),
                     object_pairs_hook=OrderedDict)
    return OrderedDict(
        (name, data) for name, data in cfg.items()
        if str(data.get("enabled", "true")).lower() == "true"
    )
