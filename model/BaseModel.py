"""
Here we define for baseline class of all segmenting or detecting models used in the application.
We define both the structure and the main functionality utils.
"""
import os
from pathlib import Path
import shutil

from model.sahi.auto_model import AutoDetectionModel

OUT_DIR = Path("cellprocesser_output")

class BaseModel():
    """
    Base class for general YOLO instance models.
    Implements the neccessary high-level functional utils for using the model.
    """
    def __init__(self, path_to_model: str, object_size):
        """
        Model constructor. Slightly differs for detectors and segmenters.

        Input:
        - path_to_model: str - path to .pt YOLO model file;
        - object_size: UI util param 
        """
        self.init_models(path_to_model)
        self.path_to_model = path_to_model
        self.object_size = object_size
        self.original_image = None
        self.detections = None
        self.out_dir = OUT_DIR
        os.makedirs(OUT_DIR, exist_ok=True)

    def init_models(self, path_to_model: str):
        """
        Helper function for initialization of actual YOLO model instances.

        Input:
        - path_to_model: str - path to .pt YOLO model file.
        """
        self.init_x10_model(path_to_model)
        self.init_x20_model(path_to_model)

    def init_x10_model(self, path_to_model: str):
        """
        Helper function for initialization of actual YOLO model instance for x10 images processing.

        Input:
        - path_to_model: str - path to .pt YOLO model file.
        """
        self.model_x10 = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=path_to_model,
            confidence_threshold=0.005,
            device="cpu", # or 'cuda:0'
        )

    def init_x20_model(self, path_to_model: str):
        """
        Helper function for initialization of actual YOLO model instance for x20 images processing.

        Input:
        - path_to_model: str - path to .pt YOLO model file.
        """
        raise NotImplementedError

    def count_cells(self, img_path):
        """
        By calling this method, the model class instance calculates cells on a given image.
        This method fully relies on the self.count() method.
        The input param is the path to RGB image of cells.
        The output param is optimized count of cells.
        """
        dst = os.path.join('.cache', 'cell_tmp_img.png')
        shutil.copy2(img_path, dst)
        detections = self.count(dst)
        if detections is None:
            return 0
        return detections

    def count(self, input_image, scale: int = 20,
              filename=".cache/cell_tmp_img_with_detections.png"):
        """General method for processing microimages of cells."""

        scale = self.object_size["scale"]
        assert scale in [10, 20], f"Scale must be either 10 or 20, instead received scale {scale}"
        if scale == 20:
            return self.count_x20(input_image, filename=filename)
        else:
            return self.count_x10(input_image, filename=filename)

    def count_x10(self, input_image, filename):
        """Method for processing images of x10 scale by applying sliding window approach."""
        raise NotImplementedError

    def count_x20(self, input_image, filename):
        """Method for processing images of x20 scale using single-time inference, as usual."""
        raise NotImplementedError

    def clear_cached_detections(self):
        """Resets cached detections of needed."""
        self.detections = None
