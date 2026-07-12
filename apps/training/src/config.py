"""Configuration for Training Studio.

The app reads ``config.json`` from the application root (the folder that
contains ``src/``).  It makes the training, evaluation and export defaults
configurable without touching code.

Precedence (lowest to highest):
    built-in DEFAULTS  <  config.json  <  explicit CLI flags / GUI fields

Every consumer (the step scripts in ``src/`` and ``gui.py``) loads the config
through :func:`load_config` and then overrides individual values with whatever
the user typed.

This mirrors the sibling ``morphology/src/config.py`` deep-merge scheme.
"""

import copy
import json
from pathlib import Path

# The app root is the parent of this ``src/`` directory.
APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = APP_ROOT / "config.json"

# Built-in fallback used when config.json is missing or partial.  Keys here also
# document the full, expected shape of the config file.
DEFAULTS = {
    # Dataset preparation/conversion lives in the sibling Dataset Viewer app;
    # this app only trains on a prepared .pth. Hence no preprocess/dataset config.
    # --- Training (instanseg_training, cell 23) -----------------------------
    "train": {
        "model_folder": "my_first_instanseg",
        "source_dataset": "my_dataset",
        "target_segmentation": "C",     # C = Cells, N = Nuclei
        "requested_pixel_size": 0.5,
        "num_epochs": 500,
        "max_no_improvement": 5,        # early stopping
        "hotstart_training": 5,
        "resume_weights": "",           # optional model_weights_best.pth to resume from
    },
    # --- Evaluation (test.py, cell 28) --------------------------------------
    "evaluate": {
        "set": "Test",                  # Train | Validation | Test
        "target_segmentation": "C",
        "save_images": True,
        "cpu_and_ram": True,
    },
    # --- Export (export_to_torchscript, cell 26) ----------------------------
    "export": {
        "version": "1",                 # last folder name in the model path
        "show_example": False,
    },
}


def _deep_merge(base, override):
    """Recursively merge ``override`` onto a copy of ``base`` (dicts only)."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    """Return the merged config dict (DEFAULTS overlaid with ``config.json``).

    ``path`` may point at an alternative config file; when omitted the app-root
    ``config.json`` is used.  A missing file simply yields the built-in defaults.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.is_file():
        try:
            user_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: could not read config {cfg_path} ({exc}); using defaults.")
            user_cfg = {}
    else:
        user_cfg = {}
    return _deep_merge(DEFAULTS, user_cfg)
