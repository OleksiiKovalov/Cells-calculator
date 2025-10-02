# Standard library imports
import json
import os
import traceback
from typing import List, Dict, Any

# Third-party imports
import cv2  # OpenCV for findContours
import numpy as np
import pandas as pd
import tensorflow as tf
from csbdeep.utils import normalize
from skimage.io import imread
from skimage.measure import regionprops
from skimage.transform import resize
from stardist.models import StarDist2D

# Local application imports
from UI.errorhandling import app_logger
from model.BaseModel import BaseModel
from model.utils import *
from model.utils import safeimagesave, safe_image_read
from UI.app_globals import IMAGE_FILE_NAME_DETECTION, IMAGE_FILE_NAME_GRID, IMAGE_FILE_NAME_INGFERENCE, IMAGE_FILE_NAME_INSTANCES, IMAGE_FILE_NAME_TMP



class StardistSegmenter(BaseModel):
    def __init__(self, path_to_model: str, object_size,model_data = None):
        self.is_custom_model = False
        super().__init__(path_to_model, object_size,model_data)
    
    def init_x20_model(self, path_to_model: str):
        app_logger().warning(f"Stardist: Num GPUs Available:{len(tf.config.list_physical_devices('GPU'))}")        
        if(path_to_model in ("2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018")):
            self.is_custom_model = False
            self.model = StarDist2D.from_pretrained(path_to_model)
            self.image_preprocess_settings_default = json.loads("[{\"gray2rgb\":\"\"} , {\"normalize\":\"1,99.8\"}]", object_pairs_hook=OrderedDict)
        else:
            self.is_custom_model = True
            path =os.path.dirname(path_to_model)
            name =os.path.basename(path_to_model)
            self.model = StarDist2D(None, name=name, basedir=path)
            self.image_preprocess_settings_default = json.loads("[{\"rgb2gray\":\"\"} , {\"normalize\":\"1,99.8\"}]", object_pairs_hook=OrderedDict)

    def init_x10_model(self, path_to_model):
        pass

    def count_x20(self, input_image, plot = True, colormap="tab20", tracking=False,
              filename=IMAGE_FILE_NAME_DETECTION, min_score=0.05,
              alpha=0.75, store_bin_mask=False, **kwargs):
        image = imread(input_image)
        image_preprocess_settings = self.model_data["image_preprocess"] if "image_preprocess" in self.model_data else self.image_preprocess_settings_default
        img_inference = process_loaded_image(image=image, settings=image_preprocess_settings)
        safeimagesave(img_inference, IMAGE_FILE_NAME_INGFERENCE)
       
        self.original_image = safegray2rgb(image)
        try:
            labels, details = None, None
            labels, details = self.model.predict_instances(img_inference)
            self.detections = self.stardist_results_to_pandas(labels, scores=details["prob"], original_shape = image.shape[:2], inference_shape = image.shape[:2])
            detections = self.detections[self.detections['confidence'] >= min_score]
            if tracking is False:
                self.object_size['signal']("set_size", self.detections['box'].copy())
                self.detections[['id_label', 'confidence', 'diameter', 'area',
                                 'volume']].to_csv(self.out_dir / f"{os.path.basename(self.original_image_path)}_{self.model_name}_cell_data.csv",
                                                   sep=';', index=False)
            original_image = self.original_image.copy()
            # if tracking is False:
            #     filtered_detections = filter_detections(detections,
            #                                             min_size = self.object_size['min_size'],
            #                                             max_size= self.object_size['max_size'])
            # else:
            #     filtered_detections = detections

            filtered_detections = detections
            self.prediction_image = None
            if plot is True:
                h, w = img_inference.shape[:2]
                o_h, o_w = original_image.shape[:2]
                #if image was scaled during preprocessing - scale the original image to show. it is a wrong way
                #todo: redo it in the correct way - we need to scale box/mask, not image
                if h!=o_h or w!=o_w:
                    original_image = resize_and_pad_cv (original_image, w, h)
                self.prediction_image = plot_predictions(original_image, filtered_detections['mask'].tolist(),
                                filename=filename, colormap=colormap, alpha=self.object_size.get("alpha", 0.75))
            return filtered_detections
        except Exception as e:
            traceback.print_exc()
            app_logger().exception(e)
            raise RuntimeError(f"Error when inferrecing StardistSegmenter: {e}")
        

    def count_x10(self, input_image: str, colormap="tab20",
              filename=IMAGE_FILE_NAME_DETECTION, min_score=0.01,
              alpha=0.75, **kwargs):
        raise NotImplementedError
    
    def image_preprocess(self,image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        return img_rgb

    def load_image(self, image_path):
        img_bgr = safe_image_read(image_path, color_mode='color')
        if img_bgr is None:
            raise RuntimeError(f"Unable to load image {image_path}")
        if len(img_bgr.shape) == 2: 
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4: 
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        return img_bgr
    
    def stardist_results_to_pandas(self,instances, scores=None, labels=None, original_shape=None, inference_shape=None) -> pd.DataFrame:
        data: Dict[str, List[Any]] = {
            "id_label": [],
            "box": [],
            "mask": [],
            "confidence": [],
            "diameter": [],
            "area": [],
            "volume": []
        }
        props = regionprops(instances)
        safeimagesave(instances, IMAGE_FILE_NAME_INSTANCES)

        for i, prop in enumerate(props):
            # Extract bounding box (min_row, min_col, max_row, max_col)
            minr, minc, maxr, maxc = prop.bbox
            box = [minc, minr, maxc, maxr]  # Convert to [x_min, y_min, x_max, y_max]

            # Create binary mask for the object
            binary_mask = (instances == prop.label).astype(np.uint8)
            #we need to resize shape
            if original_shape[0] != inference_shape[0] or original_shape[1]!=inference_shape[1]:
                binary_mask = resize(binary_mask, output_shape=original_shape, order=0, preserve_range=True, anti_aliasing=False).astype(binary_mask.dtype)
            
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
            
            bin_mask, morphology = plot_mask(np.array(pts), image_size=instances.shape)
            data['diameter'].append(morphology['diameter'])
            data['area'].append(morphology['area'])
            data['volume'].append(morphology['volume'])

        return pd.DataFrame(data)
