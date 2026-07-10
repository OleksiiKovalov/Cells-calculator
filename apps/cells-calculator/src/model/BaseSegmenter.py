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

# Local application imports
from model.utils import (
    IDENTITY_TRANSFORM,
    compose_transforms,
    invert_transform_points,
    process_loaded_image,
)


class BaseSegmenter:
    """
    Base class for general YOLO instance models.

    Implements the necessary high-level functional utils for using the model.
    """
    detections: pd.DataFrame | None
    image_preprocess_settings_default: object
    _original_shape: tuple[int, int] | None
    _inference_transform: dict
    _inference_shape: tuple[int, int] | None

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

        # Geometry recorded during preprocessing so detections can be mapped back
        # onto the original image (see preprocess / to_original_norm).
        self._original_shape = None
        self._inference_transform = dict(IDENTITY_TRANSFORM)
        self._inference_shape = None

        self.path_to_model = path_to_model
        self.init_model(path_to_model)
        self.inference_duration = 0.0

    def init_model(self, path_to_model: str):
        """Helper function for initialization of actual YOLO model instance for images processing."""
        pass

    # =====================================================================
    # Preprocessing geometry — shared by all segmenters
    # =====================================================================
    # A model rarely runs on the raw image: it is resized, padded, tiled, etc.
    # Detections come back in that preprocessed space, so every segmenter needs
    # to map them back onto the original image. Rather than reimplement that per
    # model, run preprocessing through ``preprocess`` (which records the exact
    # scale + pad applied) and convert detection polygons with ``to_original_norm``.

    def preprocess(self, input_image: np.ndarray, settings) -> np.ndarray:
        """Apply configured preprocessing, recording the geometry for mask-mapping.

        Runs ``process_loaded_image(settings)`` and stores the original image
        shape plus the net scale+pad transform the preprocessing applied, so
        :meth:`to_original_norm` can place detections correctly regardless of
        any resize/pad. Returns the preprocessed image.
        """
        self._original_shape = (int(input_image.shape[0]), int(input_image.shape[1]))
        processed, transform = process_loaded_image(
            input_image, settings, return_transform=True
        )
        self._inference_transform = transform
        self._inference_shape = (int(processed.shape[0]), int(processed.shape[1]))
        return np.asarray(processed)

    def add_geometry_step(self, new_image: np.ndarray, transform: dict) -> np.ndarray:
        """Record an extra resize/pad applied *after* :meth:`preprocess`.

        For geometry a model applies on top of the configured preprocessing —
        e.g. padding an image up to the model's minimum inference window. Pass
        the resulting image and the transform that produced it; returns the
        image for convenient chaining.
        """
        self._inference_transform = compose_transforms(
            self._inference_transform, transform
        )
        self._inference_shape = (int(new_image.shape[0]), int(new_image.shape[1]))
        return new_image

    def to_original_norm(self, pts_xy, src_shape) -> np.ndarray:
        """Map detection polygon points to normalized [0, 1] on the original image.

        Undoes all preprocessing geometry recorded by :meth:`preprocess` /
        :meth:`add_geometry_step`, so a mask found in the model's output space
        lands correctly on the original image.

        Args:
            pts_xy: (N, 2) x/y points in the model output's pixel space.
            src_shape: (height, width) of that output space.

        Returns:
            (N, 2) float32 array of x/y coordinates normalized to [0, 1] against
            the original image.
        """
        transform = getattr(self, "_inference_transform", None) or dict(IDENTITY_TRANSFORM)
        src_h, src_w = src_shape[:2]
        fed_h, fed_w = getattr(self, "_inference_shape", None) or (src_h, src_w)
        orig_h, orig_w = getattr(self, "_original_shape", None) or (src_h, src_w)

        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2).copy()
        # model output space -> fed inference space (folds any model rescaling)
        if src_w and src_h:
            pts[:, 0] *= fed_w / src_w
            pts[:, 1] *= fed_h / src_h
        # fed inference space -> original pixels
        pts = invert_transform_points(pts, transform)
        # original pixels -> normalized [0, 1]
        pts[:, 0] /= orig_w
        pts[:, 1] /= orig_h
        return pts

    def inference(self, input_image: np.ndarray) -> pd.DataFrame | None:
        """Run inference on an image, measuring duration, and return the detections DataFrame."""
        start_time = time.time()
        result = self.call_inference(input_image)
        self.inference_duration = time.time() - start_time
        return result

    def call_inference(self, input_image: np.ndarray) -> pd.DataFrame | None:
        """Method for processing images of x20 scale using single-time inference, as usual."""
        raise NotImplementedError
