"""
Here we define for baseline class of all segmenting or detecting models used in the application.
We define both the structure and the main functionality utils.
"""

# Standard library imports
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
import torch

# Local application imports
from UI.app_globals import IMAGE_FILE_NAME_TMP
from model.PredictionResult import PredictionResult, count_prediction_cells


OUT_DIR = Path("cellprocesser_output")


class BaseModel:
    """
    Base class for general YOLO instance models.

    Implements the necessary high-level functional utils for using the model.
    """
    original_image: np.ndarray | None
    inference_image: np.ndarray | None
    detections: pd.DataFrame | None
    image_preprocess_settings_default: OrderedDict

    def __init__(self, path_to_model: str, object_size, model_data=None):
        """
        Model constructor. Slightly differs for detectors and segmenters.

        Args:
            path_to_model (str): Path to .pt YOLO model file
            object_size (dict): UI configuration object with callbacks and settings
            model_data (dict): Model-specific configuration parameters. Optional.
        """
        self.original_image_path: str
        self.model_name = "<not specified>"
        self.model_data = model_data
        self.image_preprocess_settings_default = OrderedDict()
        self.use_gpu = False
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.use_gpu = self.device.type == "cuda"

        self.init_models(path_to_model)
        self.path_to_model = path_to_model
        self.object_size = object_size
        self.original_image = None
        self.inference_image = None
        self._last_original_image = None
        self._last_inference_image = None
        self.prediction_image = None
        self.detections = None
        self.out_dir = OUT_DIR
        os.makedirs(OUT_DIR, exist_ok=True)
        self.inference_duration = 0.0
        self.detectionCount = -1
        

    def init_models(self, path_to_model: str):
        """
        Helper function for initialization of actual YOLO model instances.    
        """
        self.init_x10_model(path_to_model)
        self.init_x20_model(path_to_model)

    def init_x10_model(self, path_to_model: str):
        """
        Helper function for initialization of actual YOLO model instance for x10 images processing.
        """
        pass

    def init_x20_model(self, path_to_model: str):
        """
        Helper function for initialization of actual YOLO model instance for x20 images processing.
        """
        pass

    def count_cells(self, img_path):
        """
        Calculate cells on a given image.
        
        This method fully relies on the self.count() method.
        
        Args:
            img_path (str): Path to RGB image of cells
            
        Returns:
            int: Optimized count of cells detected
        """
        dst = IMAGE_FILE_NAME_TMP
        if os.path.abspath(img_path) != os.path.abspath(dst):
            shutil.copy2(img_path, dst)
        detections = self.count(dst)
        if detections is None:
            return 0
        return detections

    def _prediction_result(
        self,
        cells,
        *,
        original_image=None,
        inference_image=None,
    ):
        """Build an explicit raw prediction result without UI side effects."""
        original = (
            original_image.copy()
            if hasattr(original_image, "copy")
            else original_image
        )
        inference = (
            inference_image.copy()
            if hasattr(inference_image, "copy")
            else inference_image
        )
        self._last_original_image = original
        self._last_inference_image = inference
        return PredictionResult(
            cells=cells,
            original_image=original,
            inference_image=inference,
        )

    def count(self, input_image, scale: int = 20):
        """
        Process microimages of cells with specified magnification scale.
        
        Automatically switches between x10 and x20 processing pipelines based on the
        scale parameter. Measures inference duration and updates detection count.
        
        Args:
            input_image (str): Path to the image file to process
            scale (int): Magnification scale, must be 10 or 20. Defaults to 20.

        Returns:
            pd.DataFrame | None: DataFrame with detection results or None if processing fails.
                            Columns: class_id, class_name, confidence, box, scale
        
        Raises:
            AssertionError: If scale is not 10 or 20.
        """
        scale = self.object_size["scale"]
        assert scale in [10, 20], f"Scale must be either 10 or 20, instead received scale {scale}"
        self.detectionCount = -1
        start_time = time.time()
        result = None
        if scale == 20:
            result =  self.count_x20(input_image)
        else:
            result =  self.count_x10(input_image)
        end_time = time.time()
        self.inference_duration = end_time - start_time
        self.detectionCount = count_prediction_cells(result)

        return result

    def count_x10(self, input_image):
        """
        Process images of x10 scale by applying sliding window approach.
        
        Args:
            input_image (str): Path to input image

        Returns:
            pd.DataFrame: Detection results
            
        Raises:
            NotImplementedError: Always raised (must be implemented by subclasses)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.count_x10 is not implemented for {input_image}"
        )

    def count_x20(self, input_image):
        """
        Process images of x20 scale using single-time inference.
        
        Args:
            input_image (str): Path to input image

        Returns:
            pd.DataFrame: Detection results
            
        Raises:
            NotImplementedError: Always raised (must be implemented by subclasses)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.count_x20 is not implemented for {input_image}"
        )

    def clear_cached_detections(self):
        """
        Reset cached detections when needed.
        
        Clears stored detection results to force re-processing on next inference.
        
        Returns:
            None
        """
        self.detections = None
       
