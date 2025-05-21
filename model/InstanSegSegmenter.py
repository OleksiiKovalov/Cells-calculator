import os
import sys
import numpy as np

from model.BaseModel import BaseModel
from model.utils import *
import torch
import torchvision
import instanseg
from instanseg import InstanSeg
from skimage.color import rgb2gray

import pandas as pd
import cv2  # OpenCV for findContours
from scipy.ndimage import find_objects  # For efficient bounding box calculation
from typing import Optional, List, Tuple, Dict, Any # For type hinting

class InstansegSegmenter(BaseModel):
    def __init__(self, path_to_model: str, object_size):
        super().__init__(path_to_model, object_size)
   
    def init_x20_model(self, path_to_model: str):
        if path_to_model and os.path.exists(path_to_model):
            print(f"Ініціалізація InstanSeg з моделлю: {path_to_model}")
            model_module = torch.jit.load(path_to_model)
            self.model = InstanSeg(model_module, verbosity=1)
        elif path_to_model in ['brightfield_nuclei', 'fluorescence_nuclei_and_cells']:
            print(f"Ініціалізація InstanSeg зі стандартною моделлю: {path_to_model}")
            self.model = InstanSeg(path_to_model, verbosity=1)
        else:
            default_model = 'fluorescence_nuclei_and_cells'
            if path_to_model:
                print(f"Попередження: Шлях/назва '{path_to_model}' не валідні для InstanSeg. Використовується '{default_model}'.")
            else:
                print(f"Попередження: Не вказано модель InstanSeg. Використовується '{default_model}'.")
            self.model = InstanSeg(default_model, verbosity=1)

    def init_x10_model(self, path_to_model):
        pass

    def count_x20(self, input_image, plot = True, colormap="tab20", tracking=False,
              filename=".cache/cell_tmp_img_with_detections.png", min_score=0.05,
              alpha=0.75, store_bin_mask=False, **kwargs):
        image = self.load_image(input_image)
        img_rgb = self.image_preprocess(image)
        self.original_image = img_rgb
        try:
            
            image_array, pixel_size = self.model.read_image(input_image)
            #labeled_output = self.model.eval_medium_image(image = image_array, return_image_tensor=False, target= "cells")
            labeled_output = self.model.eval_medium_image(image = img_rgb, return_image_tensor=False, target= "cells")
            
            # display = self.model.display(image_array, labeled_output)
            # from instanseg.utils.utils import show_images
            # show_images(image_array,display, colorbar=False, titles = ["Original Image", "Image with segmentation"])            
            
#            labeled_output = self.model.eval(image = input_image, save_output = False, save_overlay=False)
            self.detections = self.instanseg_results_to_pandas(labeled_output)
            
            detections = self.detections[self.detections['confidence'] >= min_score]
            if tracking is False:
                self.object_size['signal']("set_size", self.detections['box'].copy())
            original_image = self.original_image.copy()
            
            #todo restore tracking feature
            # if tracking is False:
            #     filtered_detections = filter_detections(detections,
            #                                             min_size = self.object_size['min_size'],
            #                                             max_size= self.object_size['max_size'])
            # else:
            #     filtered_detections = detections

            filtered_detections = detections

            if plot is True:
                plot_predictions(original_image, filtered_detections['mask'].tolist(), filename=filename, colormap=colormap, alpha=alpha)
            return filtered_detections
        except Exception as e:
            raise RuntimeError(f"Error when inferrecing InstanSeg: {e}")
        
    def plot_bounding_boxes(results, image, filename,colormap):
        hex_colors = hex_to_bgr(colormap_to_hex(colormap))
        overlay = image.copy()
        boxes = results['boxes']
        for i, box in enumerate(boxes):
            color = hex_colors[i % len(hex_colors)]
            draw_bounding_box(overlay, color, 1, )
            
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.imwrite(filename, image)
        return
    def count_x10(self, input_image: str, colormap="tab20",
              filename=".cache/cell_tmp_img_with_detections.png", min_score=0.01,
              alpha=0.75, **kwargs):
        raise NotImplementedError
    
    def image_preprocess(self,image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       
        if img_rgb.ndim == 3 and img_rgb.shape[-1] == 3: # Check if likely RGB
            img_prepared = img_rgb
        else:
            img_prepared = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        return img_prepared

    def load_image(self, image_path):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Unable to load image {image_path}")
        if len(img_bgr.shape) == 2: img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4: img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        return img_bgr
    
    def instanseg_results_to_pandas(self, labeled_output) -> pd.DataFrame:
        from instanseg.utils.utils import labels_to_features
        instanseg_objects = labels_to_features(labeled_output[:,0,:].numpy())
        data = {
            "id_label": [],
            "box": [],
            "mask": [],
            "confidence": [],
            "diameter": [],
            "area": [],
            "volume": []
        }

        features = instanseg_objects['features']
        from shapely.geometry import shape
        minx, miny, maxx, maxy = None, None, None, None
        for i, feature in enumerate(features):
            geom = shape(feature['geometry'])  # Convert to shapely geometry
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            if minx is None:
                minx, miny, maxx, maxy = bounds
            else:
                minx = min(minx, bounds[0])
                miny = min(miny, bounds[1])
                maxx = max(maxx, bounds[2])
                maxy = max(maxy, bounds[3])
                
            p_mask = feature['geometry']['coordinates'][0]
            data['id_label'].append(i)
            box = [minx, miny, maxx, maxy] 
            data['box'].append(box)
            data['mask'].append(p_mask)
            #todo restore confidence
            data['confidence'].append(1 #outputs.boxes.conf[i].cpu().detach().numpy()
                                    )
            #todo restore morphology
            morphology = None
            data['diameter'].append(None #morphology['diameter']
                                    )
            data['area'].append(None #morphology['area']
                                )
            data['volume'].append(None #morphology['volume']
                                    )
        return pd.DataFrame(data)
           
