"""
In this module the general Model class is defined which is used to calculate:
- cells only, if given microimage in JPG / PNG / TIF / BMP format;
- cells, nuclei and %, if given microimage in LSM format.
"""

# Standard library imports
import importlib
import os


# Local application imports
from ui.app_globals import get_registered_model
from model.utils import show_error_message

class Model:
    """
    Thin factory/facade over the concrete segmenter selected by ``model_type``.

    To use, create an instance and call ``instance.inference(image)``; it
    resolves the registered segmenter class for ``model_type`` (see
    ``modelconfig.json`` / ``ui.app_globals``), instantiates it, and delegates
    inference to it. ``inference`` returns the segmenter's detections DataFrame.
    """
    def __init__(
        self,
        path=os.path.join('trainedmodels', 'yolov8m-det.onnx'),
        model_type="",
        model_data=None,
        model_name=None
    ):
        """
        Initialize the general cell analysis model.

        Resolves and instantiates the segmenter registered for ``model_type``.

        Args:
            path (str): Path to the segmenter's weights. Defaults to a YOLO model.
            model_type (str): Type of detection model (registered in config). Required.
            model_data (dict): Model-specific configuration parameters. Optional.
            model_name (str): Human-readable model name for reporting. Optional.
        """
        self.path = path
        self.init_counter(path, model_type, model_data)
        self.model_name = model_name
        self.cell_counter.model_name = model_name

    def init_counter(self, path, model_type, model_data=None):
        """
        Resolve and instantiate the segmenter class registered for ``model_type``.

        The class is looked up dynamically from the model registry, imported,
        and stored on ``self.cell_counter``.
        """
        cell_counter_class_name = get_registered_model(model_type).get('model_class') if get_registered_model(model_type) else None
        if cell_counter_class_name is None:
            show_error_message("Model Initialization Error", f"Unknown model type '{model_type}' given as input. Please check configuration file and build settings.")
            raise ValueError(f"Unknown model type '{model_type}' given as input. Please check configuration file and build settings.")
        module_name, class_name = cell_counter_class_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cell_counter_class = getattr(module, class_name)
        self.cell_counter = cell_counter_class(path, model_data=model_data)

    def inference(self, image):
        """Run inference on the given image via the cell counter and return its detections."""
        return self.cell_counter.inference(image)
