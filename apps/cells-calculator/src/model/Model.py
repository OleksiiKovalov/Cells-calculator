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
    The class for object of general model.
    To use the class instances, define a new instance and use
    'instance.calculate(img_path)' command, where img_path is
    the path to .lsm image of proper quality.

    Input params are:
    - path: the path param for CellCounter;
    - threshold param for NucleiCounter;
    - eps param for NucleiCounter;
    - min_samples param for NucleiCounter.

    Output values are returned as dictionary and represent:
    - 'Nuclei': the number of nuclei detected;
    - 'Cells': the number of cells detected;
    - '%': the target percentage value obtained.
    """
    def __init__(
        self, 
        path=os.path.join('trainedmodels', 'yolov8m-det.onnx'),
        threshold=100, eps=5, min_samples=10,
        model_type = "",
        model_data = None,
        model_name = None
    ):
        """
        Initialize the general cell analysis model.

        Sets up the cell detection pipeline. Supports various detection
        models specified by model_type.

        Args:
            path (str): Path to pre-trained cell detection model. Defaults to YOLO model.
            threshold (int): Binarization threshold for nuclei detection. Defaults to 100.
            eps (int): DBSCAN eps parameter for nuclei clustering. Defaults to 5.
            min_samples (int): DBSCAN min_samples for nuclei. Defaults to 10.
            model_type (str): Type of detection model (registered in config). Required.
            model_data (dict): Model-specific configuration parameters. Optional.
            model_name (str): Human-readable model name for reporting. Optional.
        """
        self.path = path
        self.init_counter(path, model_type,model_data)
        self.inference_duration = 0
        self.model_name = model_name       
        self.cell_counter.model_name = model_name
        
    def init_counter(self, path,  model_type,model_data = None):
        """
        Helper constructor method for initializing cell counter param.
        Depending on the model file name, either CellCounter or Segmenter
        class is being called for initialization.
        """
        cell_counter_class_name = get_registered_model(model_type).get('model_class') if get_registered_model(model_type) else None
        if cell_counter_class_name is None:
            show_error_message("Model Initialization Error", f"Unknown model type '{model_type}' given as input. Please check configuration file and build settings.")
            raise ValueError(f"Unknown model type '{model_type}' given as input. Please check confoguration file and build settings.")
        module_name, class_name = cell_counter_class_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cell_counter_class = getattr(module, class_name)
        self.cell_counter =cell_counter_class(path, model_data = model_data)   

    def inference(self, image):
        """Run inference on the given image via the cell counter and return its detections."""
        return  self.cell_counter.inference(image)
