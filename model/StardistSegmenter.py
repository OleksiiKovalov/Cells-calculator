import sys
import numpy as np

from model.BaseModel import BaseModel
from model.utils import *

import pandas as pd
import cv2  # OpenCV for findContours
from typing import  List, Dict, Any # For type hinting
from stardist.data import test_image_he_2d
from csbdeep.utils import normalize            

class StardistSegmenter(BaseModel):
    def __init__(self, path_to_model: str, object_size):
        super().__init__(path_to_model, object_size)
        self.cellpose_diam = 0
    
    def init_x20_model(self, path_to_model: str):
        from stardist.models import StarDist2D
        self.model = StarDist2D.from_pretrained(path_to_model)

    def init_x10_model(self, path_to_model):
        pass

    def count_x20(self, input_image, plot = True, colormap="tab20", tracking=False,
              filename=".cache/cell_tmp_img_with_detections.png", min_score=0.05,
              alpha=0.75, store_bin_mask=False, **kwargs):
        image = self.load_image(input_image)
        img_rgb = self.image_preprocess(image)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_image = img_rgb
        
        img_normalized = normalize(img_rgb, 1, 98.8, axis=(0, 1)) # Нормалізуємо інтенсивності
        img_clipped = np.clip(img_normalized, 0, 1)
        img_normalized = img_clipped
        
        try:
            labels, details = self.model.predict_instances(img_normalized,axes = "YXC", n_tiles=None)
            
            self.detections = self.stardist_results_to_pandas(labels, scores=details["prob"])
            
            detections = self.detections[self.detections['confidence'] >= min_score]
            if tracking is False:
                self.object_size['signal']("set_size", self.detections['box'].copy())
            original_image = self.original_image.copy()
            if tracking is False:
                filtered_detections = filter_detections(detections,
                                                        min_size = self.object_size['min_size'],
                                                        max_size= self.object_size['max_size'])
            else:
                filtered_detections = detections

            if plot is True:
                plot_predictions(original_image, filtered_detections['mask'].tolist(),
                                filename=filename, colormap=colormap, alpha=alpha)
            return filtered_detections
        except Exception as e:
            raise RuntimeError(f"Error when inferrecing StardistSegmenter: {e}")
        

    def count_x10(self, input_image: str, colormap="tab20",
              filename=".cache/cell_tmp_img_with_detections.png", min_score=0.01,
              alpha=0.75, **kwargs):
        raise NotImplementedError
    
    def image_preprocess(self,image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        return img_rgb

    def load_image(self, image_path):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Unable to load image {image_path}")
        if len(img_bgr.shape) == 2: img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4: img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        return img_bgr
    
    def stardist_results_to_pandas(self,instances, scores=None, labels=None) -> pd.DataFrame:
        data: Dict[str, List[Any]] = {
            "id_label": [],
            "box": [],
            "mask": [],
            "confidence": [],
            "diameter": [],
            "area": [],
            "volume": []
        }
        from skimage.measure import regionprops
        props = regionprops(instances)

        for i, prop in enumerate(props):
            # Extract bounding box (min_row, min_col, max_row, max_col)
            minr, minc, maxr, maxc = prop.bbox
            box = [minc, minr, maxc, maxr]  # Convert to [x_min, y_min, x_max, y_max]

            # Create binary mask for the object
            binary_mask = (instances == prop.label).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                if contour.ndim >= 2 and contour.shape[0] >= 3:
                 # Convert to int32 required by fillPoly
                    polygon_points_fillpoly = contour.astype(np.int32)
                 # Ensure shape is (N, 1, 2) - findContours usually returns this already
                    if polygon_points_fillpoly.ndim == 3 and polygon_points_fillpoly.shape[1] == 1:
                        pts = polygon_points_fillpoly
                 # Handle cases where findContours might return slightly different shapes sometimes
                    elif polygon_points_fillpoly.ndim == 2:
                        pts = polygon_points_fillpoly.reshape((-1, 1, 2))


            # Confidence (if provided)
            confidence = scores[i] if scores is not None and i < len(scores) else None

            # Label (if provided)
            id_label = prop.label #labels[i] if labels is not None and i < len(labels) else 0  # default: 0

            # Area
            area = prop.area

            # Diameter (equivalent diameter of a circle)
            diameter = prop.equivalent_diameter

            # Volume – 0.0 for 2D
            volume = 0.0

            # Append to data
            data["id_label"].append(id_label)
            data["box"].append(box)
            data["mask"].append(pts)
            data["confidence"].append(confidence)
            data["diameter"].append(diameter)
            data["area"].append(area)
            data["volume"].append(volume)

        return pd.DataFrame(data)