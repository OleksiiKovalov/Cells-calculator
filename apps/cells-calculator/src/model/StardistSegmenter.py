"""StarDist-based cell segmentation.

Wraps a StarDist2D model behind the application's BaseSegmenter seam:
``call_inference(image: np.ndarray) -> pd.DataFrame`` with the standard
detection columns. Requires TensorFlow (StarDist's backend).
"""

# Standard library imports
import json
import os
import traceback
from collections import OrderedDict
from typing import Any, Dict, List

# Third-party imports
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.measure import regionprops
from stardist.models import model2d, StarDist2D

# Local application imports
from ui.errorhandling import app_logger
from model.BaseSegmenter import BaseSegmenter
from model.utils import plot_mask


class StardistSegmenter(BaseSegmenter):
    """Instance segmentation using the StarDist2D star-convex polygon model."""

    def init_model(self, path_to_model: str):
        """Load a built-in or custom StarDist model for x20 segmentation."""
        self.is_custom_model = False
        app_logger().warning(
            f"Stardist: Num GPUs Available: "
            f"{len(tf.config.list_physical_devices('GPU'))}"
        )
        if path_to_model in ("2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"):
            self.is_custom_model = False
            self.model = StarDist2D.from_pretrained(path_to_model)
            self.image_preprocess_settings_default = json.loads(
                '[{"gray2rgb":""},{"normalize":"1,99.8"}]',
                object_pairs_hook=OrderedDict,
            )
        else:
            self.is_custom_model = True
            path = os.path.dirname(path_to_model)
            name = os.path.basename(path_to_model)
            self.model = StarDist2D(None, name=name, basedir=path)
            self.image_preprocess_settings_default = json.loads(
                '[{"rgb2gray":""},{"normalize":"1,99.8"}]',
                object_pairs_hook=OrderedDict,
            )

    def call_inference(self, input_image, min_score=0.05):
        """Segment objects in *input_image* (an RGB ndarray) with StarDist."""
        image = input_image
        image_preprocess_settings = (
            self.model_data["image_preprocess"]
            if self.model_data and "image_preprocess" in self.model_data
            else self.image_preprocess_settings_default
        )
        # Preprocess through the base class so any resize/pad geometry is
        # recorded; stardist_results_to_pandas maps masks back via to_original_norm.
        img_inference = self.preprocess(image, image_preprocess_settings)
        try:
            # Clip distances before the C++ NMS to avoid a crash on extreme
            # values (e.g. > 1e22); restore the original afterwards.
            original_nms = model2d.non_maximum_suppression_sparse

            def safe_nms_sparse(dist, prob, points, *args, **kwargs):
                """Clip distances before NMS to avoid a crash on extreme values."""
                dist = np.clip(dist, a_min=None, a_max=10000.0)
                return original_nms(dist, prob, points, *args, **kwargs)

            model2d.non_maximum_suppression_sparse = safe_nms_sparse
            try:
                labels, details = self.model.predict_instances(img_inference)
            finally:
                model2d.non_maximum_suppression_sparse = original_nms

            detections = self.stardist_results_to_pandas(
                labels,
                scores=details["prob"],
            )
            return detections[detections["confidence"] >= min_score]
        except Exception as e:
            traceback.print_exc()
            app_logger().exception(e)
            raise RuntimeError(f"Error during StarDist inference: {e}")

    def stardist_results_to_pandas(
        self,
        instances,
        scores=None,
        labels=None,
    ) -> pd.DataFrame:
        """Convert a StarDist label map to the standard detections DataFrame.

        Contours are taken in the label map's (inference) space and mapped back
        onto the original image via ``self.to_original_norm`` (base class),
        which undoes any resize/pad recorded during ``preprocess`` — so masks
        align on the original image, matching the other segmenters.
        """
        data: Dict[str, List[Any]] = {
            "id_label": [],
            "box": [],
            "mask": [],
            "confidence": [],
            "diameter": [],
            "area": [],
            "volume": [],
        }
        props = regionprops(instances)

        src_shape = instances.shape[:2]
        original_h, original_w = getattr(self, "_original_shape", None) or src_shape

        for i, prop in enumerate(props):
            binary_mask = (instances == prop.label).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            pts = None
            if contours:
                contour = contours[0]
                if contour.ndim >= 2 and contour.shape[0] >= 3:
                    polygon = contour.astype(np.int32)
                    if polygon.ndim == 3 and polygon.shape[1] == 1:
                        pts = polygon
                    elif polygon.ndim == 2:
                        pts = polygon.reshape((-1, 1, 2))
            if pts is None:
                continue

            # Map the contour from label-map space to [0, 1] on the original image.
            norm_mask = self.to_original_norm(pts.reshape(-1, 2), src_shape=src_shape)
            # Box from the mapped contour's bounds (normalized, original space).
            x_min, y_min = norm_mask.min(axis=0)
            x_max, y_max = norm_mask.max(axis=0)
            box = [x_min, y_min, x_max - x_min, y_max - y_min]

            confidence = scores[i] if scores is not None and i < len(scores) else None

            data["id_label"].append(prop.label)
            data["box"].append(box)
            data["mask"].append(norm_mask)
            data["confidence"].append(confidence)

            _, morphology = plot_mask(norm_mask, image_size=(original_h, original_w))
            data["diameter"].append(morphology["diameter"])
            data["area"].append(morphology["area"])
            data["volume"].append(morphology["volume"])

        return pd.DataFrame(data)
