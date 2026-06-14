"""Cellpose-based cell segmentation.

Wraps a Cellpose model behind the application's BaseSegmenter seam:
``call_inference(image: np.ndarray) -> pd.DataFrame`` with the standard
detection columns (id_label, box, mask, confidence, diameter, area, volume).
"""

# Standard library imports
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import cv2
import numpy as np
import pandas as pd
from cellpose import models as cp_models
from scipy.ndimage import find_objects

# Local application imports
from ui.errorhandling import app_logger
from model.BaseSegmenter import BaseSegmenter
from model.utils import plot_mask, process_loaded_image


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
        img_inference = process_loaded_image(
            image=image, settings=image_preprocess_settings
        )
        channels_to_use = [0, 0]
        try:
            masks, flows, styles = self.model.eval(
                img_inference, diameter=self.cellpose_diam, channels=channels_to_use
            )
            cellprob = flows[2]  # cell probability map
            detections = self.cellpose_results_to_pandas(
                masks,
                cellprob_map=cellprob,
                image_shape_for_norm=image.shape[:2],
            )
            return detections[detections["confidence"] >= min_score]
        except Exception as e:
            raise RuntimeError(f"Error during Cellpose inference: {e}")

    def cellpose_results_to_pandas(
        self,
        masks: np.ndarray,
        cellprob_map: Optional[np.ndarray] = None,
        image_shape_for_norm: Optional[Tuple[int, int]] = None,
        store_bin_mask: bool = False,
    ) -> pd.DataFrame:
        """Convert a Cellpose label map to the standard detections DataFrame.

        Each positive integer in *masks* is one object (0 = background). Boxes
        and contours are normalized to [0, 1]; morphology is computed from the
        normalized contour to match the other segmenters' convention.
        """
        if image_shape_for_norm is None:
            img_height, img_width = masks.shape[:2]
        else:
            img_height, img_width = image_shape_for_norm

        data: Dict[str, List[Any]] = {
            "id_label": [],
            "box": [],
            "mask": [],
            "confidence": [],
            "diameter": [],
            "area": [],
            "volume": [],
        }
        if store_bin_mask:
            data["bin_mask"] = []

        unique_object_ids = np.unique(masks)
        unique_object_ids = unique_object_ids[unique_object_ids != 0]
        if len(unique_object_ids) == 0:
            return pd.DataFrame(data)

        for object_id in unique_object_ids:
            current_bin_mask = (masks == object_id)
            area = int(np.sum(current_bin_mask))
            if area == 0:
                continue

            data["id_label"].append(int(object_id))
            if store_bin_mask:
                data["bin_mask"].append(current_bin_mask)

            # Bounding box (find_objects returns one slice-tuple per object).
            slices = find_objects(current_bin_mask)
            if not slices or slices[0] is None:
                data["box"].append([np.nan, np.nan, np.nan, np.nan])
                data["mask"].append([])
                data["confidence"].append(np.nan if cellprob_map is not None else 1.0)
                data["id_label"].pop()
                if store_bin_mask:
                    data["bin_mask"].pop()
                continue

            y_slice, x_slice = slices[0]
            y_min, y_max = y_slice.start, y_slice.stop
            x_min, x_max = x_slice.start, x_slice.stop
            data["box"].append([
                x_min / img_width,
                y_min / img_height,
                (x_max - x_min) / img_width,
                (y_max - y_min) / img_height,
            ])

            # Mask contour (normalized polygon).
            contours, _ = cv2.findContours(
                current_bin_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            normalized_contour_list = []
            if contours:
                contour = max(contours, key=cv2.contourArea)
                squeezed_contour = contour.squeeze(axis=1)
                normalized_contour_points = squeezed_contour.astype(np.float32) / \
                    np.array([img_width, img_height], dtype=np.float32)
                normalized_contour_list = normalized_contour_points.tolist()
            data["mask"].append(normalized_contour_list)

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
            data["confidence"].append(confidence)

            _, morphology = plot_mask(
                np.array(normalized_contour_list), image_size=image_shape_for_norm
            )
            data["diameter"].append(morphology["diameter"])
            data["area"].append(morphology["area"])
            data["volume"].append(morphology["volume"])

        return pd.DataFrame(data)
