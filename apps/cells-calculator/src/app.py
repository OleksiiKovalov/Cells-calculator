"""PySide6 image viewer application.

Features:
    - Main window with menu, toolbar (buttons bar), and status bar.
    - Central image viewer supporting zoom in / zoom out / pan / scroll / fit / reset.
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

from ui.app_globals import FILENAME_MODEL_CONFIG, register_model


def _preload_tensorflow_if_needed():
    """Import TensorFlow before anything else when a StarDist model is enabled.

    On Windows, TensorFlow's native runtime must initialize before the other
    native libraries (PySide6/Qt, torch) or its DLL load fails. Those are pulled
    in by the PySide6 / MainWindow imports below, so this MUST run first — and only
    when StarDist is actually enabled, to avoid loading TensorFlow for users who
    don't need it.
    """
    try:
        with open(FILENAME_MODEL_CONFIG, 'r') as f:
            cfg = json.load(f)
        needs_tf = any(
            v.get('model_type') == 'stardist'
            and str(v.get('enabled', 'true')).lower() == 'true'
            for v in cfg.values()
        )
        if needs_tf:
            import tensorflow  # noqa: F401  (imported for its load-order side effect)
    except Exception:
        pass


_preload_tensorflow_if_needed()

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.MainWindow import MainWindow
from ui.splashscreen import close_splash, init_splash


def load_models_on_startup():
    """Read modelconfig.json, register enabled models, and return them.

    Returns:
        OrderedDict: The enabled models keyed by name, in config order.
    """
    known_models = {
        "cellcounter": 'model.CellCounter.CellCounter',
        "cellpose": 'model.CellposeSegmenter.CellposeSegmenter',
        "yolo": 'model.YOLOSegmenter.YoloSegmenter',
        "instanseg": 'model.InstanSegSegmenter.InstansegSegmenter',
        "stardist": 'model.StardistSegmenter.StardistSegmenter'
    }

    with open(FILENAME_MODEL_CONFIG, 'r') as f:
        models = json.load(f, object_pairs_hook=OrderedDict)
        models = OrderedDict((k, v) for k, v in models.items() if 'enabled' not in v or v.get('enabled','true').lower() == 'true')

    #register all models from config to make them available in UI/processing    
    for model_name, model_data in models.items():
        model_type = model_data.get('model_type')
        register_model(model_type, known_models.get(model_type), model_data.get('preload', False))
    return models

def main():
    """Build the QApplication, show the splash, create the MainWindow and run the app."""
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "resources" / "Cells-calculator-v3-icon2.png")))
    init_splash()
    loaded_models = load_models_on_startup()
    import ui.imports  # Ensure all UI imports are done before MainWindow

    win = MainWindow()
    win._register_segmenters(loaded_models)
    win.showMaximized()

    close_splash()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
