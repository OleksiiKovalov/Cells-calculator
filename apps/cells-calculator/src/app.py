"""PySide6 image viewer application.

Features:
    - Main window with menu, toolbar (buttons bar), and status bar.
    - Central image viewer supporting zoom in / zoom out / pan / scroll / fit / reset.
"""

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

from ui.app_globals import FILENAME_MODEL_CONFIG, register_model

# Model names resolved/downloaded by the backend library itself rather than
# read from a local file, keyed by model_type. These are exempt from the
# missing-file check.
BUILTIN_MODEL_NAMES = {
    "cellpose": {"cyto", "nuclei", "cyto2", "cyto3"},
    "instanseg": {"brightfield_nuclei", "fluorescence_nuclei_and_cells"},
    "stardist": {"2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"},
}


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


def find_missing_model_files(loaded_models):
    """Return a list of (name, path) for configured models whose local files are missing.

    Models whose `path` names a backend built-in (downloaded on first use, see
    BUILTIN_MODEL_NAMES) are skipped, since they have no local file to check.
    """
    missing = []
    for name, model_data in loaded_models.items():
        model_type = model_data.get('model_type')
        path = model_data.get('path')
        if not path:
            continue
        if path in BUILTIN_MODEL_NAMES.get(model_type, ()):
            continue
        if model_type == 'stardist':
            found = os.path.isdir(path) and os.path.isfile(os.path.join(path, 'weights_best.h5'))
        else:
            found = os.path.isfile(path)
        if not found:
            missing.append((name, path))
    return missing


def show_missing_models_dialog(missing):
    """Warn the user which configured models are missing their local files."""
    from PySide6.QtWidgets import QMessageBox

    lines = "\n".join(f"- {name}: {path}" for name, path in missing)
    QMessageBox.warning(
        None,
        "Missing Model Files",
        "The following configured model(s) are missing their files:\n\n"
        f"{lines}\n\n"
        "Please download and install the missing models according to the "
        "documentation (see trainedmodels/models.readme or README.md), "
        "then restart the application."
    )


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

    missing_models = find_missing_model_files(loaded_models)
    if missing_models:
        show_missing_models_dialog(missing_models)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
