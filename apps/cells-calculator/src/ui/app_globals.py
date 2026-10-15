"""
Application-wide globals for Cells Calculator.

Holds the few values shared across components: the model-config path, the
detection-overlay cache filename, and the registry of available models.
"""

FILENAME_MODEL_CONFIG = "modelconfig.json"

# Where plot_predictions writes the rendered detection overlay.
IMAGE_FILE_NAME_DETECTION = ".cache/cell_tmp_img_with_detections.png"


# Registry of models discovered from the config file, keyed by model_type.
_registered_models: dict = {}


def register_model(model_type: str, model_class, preload=False):
    """Register a model class (dotted path) by its model_type."""
    _registered_models[model_type] = {
        'model_type': model_type,
        'model_class': model_class,
        'preload': preload,
    }


def get_registered_model(name: str):
    """Retrieve a registered model entry by model_type, or None."""
    return _registered_models.get(name, None)
