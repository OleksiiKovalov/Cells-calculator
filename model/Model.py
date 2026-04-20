"""
In this module the general Model class is defined which is used to calculate:
- cells only, if given microimage in JPG / PNG / TIF / BMP format;
- cells, nuclei and %, if given microimage in LSM format.
"""

# Standard library imports
import importlib
import os


# Local application imports
from UI.app_globals import get_registered_model
from model.NucleiCounter import NucleiCounter
from model.utils import is_image_valid, calculate_lsm, show_error_message

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
        object_size = { 'min_size' : 0, 'max_size' : 1, "scale": 20},
        model_type = "",
        model_data = None,
        model_name = None
    ):
        """
        Initialize the general cell analysis model.
        
        Sets up both cell and nuclei detection pipelines. Supports various detection
        models specified by model_type.
        
        Args:
            path (str): Path to pre-trained cell detection model. Defaults to YOLO model.
            threshold (int): Binarization threshold for nuclei detection. Defaults to 100.
            eps (int): DBSCAN eps parameter for nuclei clustering. Defaults to 5.
            min_samples (int): DBSCAN min_samples for nuclei. Defaults to 10.
            object_size (dict): Configuration dict with keys like 'scale', 'signal'. Optional.
            model_type (str): Type of detection model (registered in config). Required.
            model_data (dict): Model-specific configuration parameters. Optional.
            model_name (str): Human-readable model name for reporting. Optional.
        """
        self.nuclei_counter = NucleiCounter(
            threshold=threshold,
            eps=eps,
            min_samples=min_samples
        )
        self.path = path
        self.init_counter(path, object_size,model_type,model_data)
        self.inference_duration = 0
        self.model_name = model_name       
        self.cell_counter.model_name = model_name
        
    def init_counter(self, path, object_size, model_type,model_data = None):
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
        self.cell_counter =cell_counter_class(path, object_size = object_size,model_data = model_data)   

    def calculate(self, img_path, cell_channel=0, nuclei_channel=1):
        """
        Calculates the resulting target values.
        Input params are:
        - img_path: path to lsm/jpg/png/tif/bmp image;
        - cell_channel: channel with cells. Default to 0;
        - nuclei_channel: channel with stained nuclei. Default to 1.

        Returns the result as a dictionary with the following fields:
        - Nuclei: count for stained nuclei detected (given lsm image only);
        - Cells: count for all the cells detected;
        - %: the target percentage for alive cells (given lsm image only).
        """
        if img_path.endswith('lsm'):
            self.inference_duration = -1
            return calculate_lsm(self.cell_counter, self.nuclei_counter,
                  img_path, cell_channel, nuclei_channel)
        elif is_image_valid(img_path):
            result =calculate_standard(self.cell_counter, img_path)
            self.inference_duration = self.cell_counter.inference_duration
            return result

def calculate_standard(cell_counter, img_path : str):
    """
    Analyze standard image format (JPG/PNG/TIF/BMP) for cells only.
    
    Runs cell detection on single-channel or RGB images without nuclei information.
    
    Args:
        cell_counter (BaseModel): Instantiated cell detection model
        img_path (str): Path to image file (JPG, PNG, TIF, or BMP)
    
    Returns:
        dict: Analysis results with keys:
            - 'Nuclei': -100 (NaN encoding - not applicable for standard images)
            - 'Cells': int - count of detected cells
            - '%': -100 (NaN encoding - not applicable for standard images)
    """
    cell_count = cell_counter.count_cells(img_path)
    return {'Nuclei': -100, 'Cells': cell_count, '%': -100}