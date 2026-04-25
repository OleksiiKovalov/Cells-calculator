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
    _nuclei_cache = {}

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
        self.init_counter(path, object_size,model_type,model_data)
        self.inference_duration = 0
        self.model_name = model_name       
        self.cell_counter.model_name = model_name

    def _get_nuclei_cache_key(self, img_path, nuclei_channel):
        """Builds a stable cache key for dead-cell counting."""
        return (
            os.path.abspath(img_path),
            os.path.getmtime(img_path),
            nuclei_channel,
            self.nuclei_counter.threshold,
            self.nuclei_counter.eps,
            self.nuclei_counter.min_samples,
        )

    def get_nuclei_count(self, img_path, nuclei_channel=1):
        """Returns cached nuclei count for the same image/channel when available."""
        cache_key = self._get_nuclei_cache_key(img_path, nuclei_channel)
        cached_value = self._nuclei_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        nuclei_channel_img = extract_nuclei_channel(img_path, nuclei_channel=nuclei_channel)
        nuclei_count = 0 if nuclei_channel_img is None else self.nuclei_counter.countNuclei(nuclei_channel_img)

        if len(self._nuclei_cache) >= 32:
            self._nuclei_cache.clear()
        self._nuclei_cache[cache_key] = nuclei_count
        return nuclei_count
        
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
        if img_path.endswith('lsm'):
            self.inference_duration = -1
            nuclei_count = self.get_nuclei_count(img_path, nuclei_channel=nuclei_channel)
            return calculate_lsm(self.cell_counter, self.nuclei_counter,
                  img_path, cell_channel, nuclei_channel, nuclei_count=nuclei_count)
        elif is_image_valid(img_path):
            result = calculate_standard(
                self.cell_counter,
                img_path,
                nuclei_count=self.get_nuclei_count(img_path, nuclei_channel=nuclei_channel),
                nuclei_channel=nuclei_channel
            )
            self.inference_duration = self.cell_counter.inference_duration
            return result

def calculate_standard(cell_counter, img_path: str, nuclei_count, nuclei_channel=1):
    """
    Calculates cells, nuclei and alive percentage on a given standard image.
    Input params are:
    - cell_counter: CellCounter class instance;
    - img_path: path to lsm/jpg/png/tif/bmp image.
    - nuclei_count: precomputed count for stained nuclei detected.
    - nuclei_channel: channel with stained nuclei. Default to 1.

    Returns the result as a dictionary with the following fields:
    - Nuclei: count for stained nuclei detected;
    - Cells: count for all the cells detected;
    - %: the target percentage for alive cells.
    """
    cell_count = cell_counter.count_cells(img_path)
    percentage = calculate_alive_percentage(
        count_detected_objects(cell_count),
        nuclei_count
    )
    return {'Nuclei': nuclei_count, 'Cells': cell_count, '%': percentage}
