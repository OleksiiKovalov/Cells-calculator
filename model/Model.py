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
from model.utils import (
    calculate_alive_percentage,
    calculate_lsm,
    count_detected_objects,
    extract_nuclei_channel,
    is_image_valid,
    show_error_message
)

DETECTOR_MODEL_TYPE = "cellcounter"
NO_NUCLEI_METRIC = -100

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
        self.nuclei_counter = NucleiCounter(
            threshold=threshold,
            eps=eps,
            min_samples=min_samples
        )
        self.path = path
        self.model_type = model_type
        self.init_counter(path, object_size,model_type,model_data)
        self.inference_duration = 0
        self.model_name = model_name       
        self.cell_counter.model_name = model_name

    def get_nuclei_count(self, img_path, nuclei_channel=1):
        """Counts stained nuclei for detector-based cell counting."""
        nuclei_channel_img = extract_nuclei_channel(img_path, nuclei_channel=nuclei_channel)
        return 0 if nuclei_channel_img is None else self.nuclei_counter.countNuclei(nuclei_channel_img)

    def _uses_detector_metrics(self):
        """Returns True when nuclei and alive metrics are meaningful for this model."""
        return self.model_type == DETECTOR_MODEL_TYPE
        
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
        - Nuclei: count for stained nuclei detected;
        - Cells: count for all the cells detected;
        - %: the target percentage for alive cells.
        """
        include_nuclei = self._uses_detector_metrics()
        if img_path.endswith('lsm'):
            self.inference_duration = -1
            nuclei_count = (
                self.get_nuclei_count(img_path, nuclei_channel=nuclei_channel)
                if include_nuclei
                else NO_NUCLEI_METRIC
            )
            return calculate_lsm(self.cell_counter, self.nuclei_counter,
                  img_path, cell_channel, nuclei_channel, nuclei_count=nuclei_count)
        elif is_image_valid(img_path):
            result = calculate_standard(
                self.cell_counter,
                img_path,
                nuclei_count=NO_NUCLEI_METRIC
            )
            self.inference_duration = self.cell_counter.inference_duration
            return result

def calculate_standard(cell_counter, img_path: str, nuclei_count=NO_NUCLEI_METRIC):
    """
    Calculates cells, nuclei and alive percentage on a given standard image.
    Input params are:
    - cell_counter: CellCounter class instance;
    - img_path: path to lsm/jpg/png/tif/bmp image.
    - nuclei_count: precomputed count for stained nuclei detected.

    Returns the result as a dictionary with the following fields:
    - Nuclei: count for stained nuclei detected;
    - Cells: count for all the cells detected;
    - %: the target percentage for alive cells.
    """
    cell_count = cell_counter.count_cells(img_path)
    if nuclei_count == NO_NUCLEI_METRIC:
        percentage = NO_NUCLEI_METRIC
    else:
        percentage = calculate_alive_percentage(
            count_detected_objects(cell_count),
            nuclei_count
        )
    return {'Nuclei': nuclei_count, 'Cells': cell_count, '%': percentage}
