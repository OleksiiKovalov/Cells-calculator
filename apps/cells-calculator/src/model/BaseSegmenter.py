"""
Here we define for baseline class of all segmenting or detecting models used in the application.
We define both the structure and the main functionality utils.
"""

# Standard library imports
import time
from collections import OrderedDict

# Third-party imports
import numpy as np
import pandas as pd
import torch


class BaseSegmenter:
    """
    Base class for general YOLO instance models.

    Implements the necessary high-level functional utils for using the model.
    """
    detections: pd.DataFrame | None
    image_preprocess_settings_default: object

    def __init__(self, path_to_model: str,  model_data=None):
        """
        Model constructor.

        Args:
            path_to_model (str): Path to the model weights file.
            model_data (dict): Model-specific configuration parameters. Optional.
        """
        self.model_name = "<not specified>"
        self.model_data = model_data
        self.image_preprocess_settings_default = OrderedDict()
        self.use_gpu = False
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.use_gpu = self.device.type == "cuda"

        self.path_to_model = path_to_model
        self.init_model(path_to_model)
        self.inference_duration = 0.0

    def init_model(self, path_to_model: str):
        """Helper function for initialization of actual YOLO model instance for images processing."""
        pass

    def inference(self, input_image: np.ndarray) -> pd.DataFrame | None:
        """Run inference on an image, measuring duration, and return the detections DataFrame."""
        start_time = time.time()
        result = self.call_inference(input_image)
        self.inference_duration = time.time() - start_time
        return result

    def call_inference(self, input_image: np.ndarray) -> pd.DataFrame | None:
        """Method for processing images of x20 scale using single-time inference, as usual."""
        raise NotImplementedError
