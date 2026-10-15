"""Cellpose-based cell segmentation.

Wraps a Cellpose model behind the application's BaseSegmenter seam:
``call_inference(image: np.ndarray) -> pd.DataFrame`` with the standard
detection columns (id_label, box, mask, confidence, diameter, area, volume).
"""

# Standard library imports
import json
import os
from collections import OrderedDict
from typing import Optional

# Third-party imports
import cv2
import numpy as np
import pandas as pd
from cellpose import models as cp_models
from scipy.ndimage import find_objects

# Local application imports
from ui.errorhandling import app_logger
from model.BaseSegmenter import BaseSegmenter


class CellposeSegmenter(BaseSegmenter):
    """Cell segmentation using the Cellpose deep-learning model."""

    def init_model(self, path_to_model: str):
        """Load a custom or built-in Cellpose model for x20 segmentation."""
        self.cellpose_diam = None
        if self.model_data and self.model_data.get("diameter"):
            try:
                self.cellpose_diam = float(self.model_data["diameter"])
            except (TypeError, ValueError):
                self.cellpose_diam = None

        self.image_preprocess_settings_default = json.loads(
            '[{"gray2rgb":""}]', object_pairs_hook=OrderedDict
        )

        if path_to_model and os.path.exists(path_to_model):
            print(f"Initializing Cellpose with model: {path_to_model}")
            self.model = cp_models.CellposeModel(
                gpu=self.use_gpu, pretrained_model=path_to_model
            )
        elif path_to_model in ("cyto", "nuclei", "cyto2", "cyto3"):
            print(f"Initializing Cellpose with standard model: {path_to_model}")
            self.model = cp_models.CellposeModel(
                gpu=self.use_gpu, model_type=path_to_model
            )
        else:
            default_model = "cyto"
            if path_to_model:
                print(
                    f"Warning: path/name '{path_to_model}' is not valid for "
                    f"Cellpose. Using '{default_model}'."
                )
            else:
                print(
                    f"Warning: no Cellpose model specified. "
                    f"Using '{default_model}'."
                )
            self.model = cp_models.CellposeModel(
                model_type=default_model, gpu=self.use_gpu
            )
            app_logger().warning(f"CellposeSegmenter: GPU used: {self.use_gpu}")

    def call_inference(self, input_image, min_score=0.05):
        """Segment cells in *input_image* (an RGB ndarray) with Cellpose."""
        image = input_image
        image_preprocess_settings = (
            self.model_data["image_preprocess"]
            if self.model_data and "image_preprocess" in self.model_data
            else self.image_preprocess_settings_default
        )
        # Preprocess through the base class so any resize/pad geometry is
        # recorded; cellpose_results_to_pandas maps masks back via to_original_norm.
        img_inference = self.preprocess(image, image_preprocess_settings)
        channels_to_use = [0, 0]
        try:
            masks, flows, styles = self.model.eval(
                img_inference, diameter=self.cellpose_diam, channels=channels_to_use
            )
            cellprob = flows[2]  # cell probability map
            detections = self.cellpose_results_to_pandas(
                masks,
                cellprob_map=cellprob,
            )
            return detections[detections["confidence"] >= min_score]
        except Exception as e:
            raise RuntimeError(f"Error during Cellpose inference: {e}") from e

    def cellpose_results_to_pandas(
        self,
        masks: np.ndarray,
        cellprob_map: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Convert a Cellpose label map to the standard detections DataFrame.

        Each positive integer in *masks* is one object (0 = background). Boxes
        and contours are found in the label map's (inference) space and mapped
        back onto the original image via ``self.to_original_norm`` (base class),
        which undoes any resize/pad recorded during ``preprocess`` — matching
        the other segmenters' convention.
        """
        src_shape = masks.shape[:2]
        original_shape = getattr(self, "_original_shape", None) or src_shape

        data = self._new_detection_frame()

        unique_object_ids = np.unique(masks)
        unique_object_ids = unique_object_ids[unique_object_ids != 0]

        for object_id in unique_object_ids:
            current_bin_mask = (masks == object_id)
            if int(np.sum(current_bin_mask)) == 0:
                continue

            # Bounding box from the object's slice (area > 0 guarantees a slice),
            # mapped to [0, 1] on the original image.
            y_slice, x_slice = find_objects(current_bin_mask)[0]
            (bx0, by0), (bx1, by1) = self.to_original_norm(
                [[x_slice.start, y_slice.start], [x_slice.stop, y_slice.stop]],
                src_shape=src_shape,
            )
            box = [bx0, by0, bx1 - bx0, by1 - by0]

            # Mask contour, mapped to [0, 1] on the original image.
            contours, _ = cv2.findContours(
                current_bin_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            normalized_contour = np.empty((0, 2), dtype=np.float32)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                normalized_contour = self.to_original_norm(
                    contour.squeeze(axis=1), src_shape=src_shape
                )

            # Confidence (mean cell probability inside the mask).
            if cellprob_map is not None:
                if cellprob_map.shape == current_bin_mask.shape:
                    object_pixels = cellprob_map[current_bin_mask]
                    confidence = float(np.mean(object_pixels)) if object_pixels.size > 0 else np.nan
                else:
                    confidence = np.nan
                    print(
                        f"Warning: cellprob_map shape {cellprob_map.shape} "
                        f"mismatches mask shape {current_bin_mask.shape} "
                        f"for object_id {object_id}. Confidence set to NaN."
                    )
            else:
                confidence = 1.0

            self._append_detection(
                data,
                id_label=int(object_id),
                norm_mask=normalized_contour,
                confidence=confidence,
                original_shape=original_shape,
                box=box,
            )

        return pd.DataFrame(data)
