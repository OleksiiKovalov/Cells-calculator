# Standard library imports
import inspect
import json
import os
from collections import OrderedDict

# Third-party imports
import cv2  # OpenCV for findContours
import numpy as np
import pandas as pd
import torch
from instanseg import InstanSeg
from instanseg.utils.utils import labels_to_features
from shapely.geometry import shape
from skimage.io import imread

# Local application imports
from UI.app_globals import set_global
from UI.errorhandling import app_logger
from model.BaseModel import BaseModel
from model.utils import *
from model.utils import safeimagesave, safe_image_read, safe_image_write
from UI.app_globals import IMAGE_FILE_NAME_DETECTION, IMAGE_FILE_NAME_INGFERENCE


class InstansegSegmenter(BaseModel):
    def __init__(self, path_to_model: str, object_size,model_data = None):
        super().__init__(path_to_model, object_size,model_data)
   
    def init_x20_model(self, path_to_model: str):
        self.image_preprocess_settings_default = json.loads("[{\"gray2rgb\":\"\"}]", object_pairs_hook=OrderedDict)
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            app_logger().warning(f"InstansegSegmenter: Device used:{device}")        
            self.model = self.model.to(device)
            
    def init_x10_model(self, path_to_model):
        pass

    def count_x20(self, input_image, plot = True, colormap="tab20", tracking=False,
              filename=IMAGE_FILE_NAME_DETECTION, min_score=0.05,
              alpha=0.75, store_bin_mask=False,x10=False, **kwargs):
        image = imread(input_image)

        if x10:
            config_node = self.model_data["x10"] if "x10"  in self.model_data else None
            app_logger().info("Using x10 configuration for InstanSeg inference.")
        else:
            config_node = self.model_data["x20"] if "x20"  in self.model_data else None
            app_logger().info("Using x20 configuration for InstanSeg inference.")

        if config_node is not None:
            app_logger().info(f"InstanSeg config found")
            image_preprocess_settings = config_node["image_preprocess"] if "image_preprocess" in config_node else self.image_preprocess_settings_default
            pixel_size = config_node["pixel_size"] if "pixel_size" in config_node else None
            tile_size = config_node["tile_size"] if "tile_size" in config_node else "512"
            if isinstance(tile_size, str) and tile_size.endswith('%'):
                tile_size = int(int(tile_size[:-1]) * max(image.shape[:2]) / 100)
                if tile_size < 210:
                    tile_size = 210
                app_logger().info(f"Calculated tile_size for InstanSeg inference: {tile_size}")
            tile_size = int(tile_size)
            method_name  = self.model_data["inference_method_name"] if "inference_method_name" in self.model_data else "eval_medium_image"
        else:
            app_logger().info(f"InstanSeg config not found, using defaults")
            image_preprocess_settings = self.image_preprocess_settings_default
            pixel_size = None
            tile_size = 512
            method_name = "eval_medium_image"

        img_inference = process_loaded_image(image=image, settings=image_preprocess_settings)
        safeimagesave(img_inference, IMAGE_FILE_NAME_INGFERENCE)
        self.original_image = safegray2rgb(image)

        try:
            method = getattr(self.model, method_name, None)
            if not method:
                raise AttributeError(f"Method '{method_name}' not found on model")
            
            # Check if method accepts tile_size parameter
            sig = inspect.signature(method)
            has_tile_size = 'tile_size' in sig.parameters
            
            # Prepare base arguments
            kwargs = {
                'image': img_inference,
                'return_image_tensor': False,
                'target': 'cells',
                'pixel_size': pixel_size
            }
            
            # Add tile_size only if method supports it and x10 is True
            if has_tile_size and x10:
                kwargs['tile_size'] = tile_size
            
            labeled_output = method(**kwargs)

            self.detections = self.instanseg_results_to_pandas(labeled_output)
            detections = self.detections[self.detections['confidence'] >= min_score]
            if tracking is False:
                self.object_size['signal']("set_size", self.detections['box'].copy())
                self.detections[['id_label', 'confidence', 'diameter', 'area',
                                 'volume']].to_csv(self.out_dir / f"{os.path.basename(self.original_image_path)}_{self.model_name}_cell_data.csv",
                                                   sep=';', index=False)
            original_image = self.original_image.copy()
            #todo restore tracking feature
            # if tracking is False:
            #     filtered_detections = filter_detections(detections,
            #                                             min_size = self.object_size['min_size'],
            #                                             max_size= self.object_size['max_size'])
            # else:
            #     filtered_detections = detections

            filtered_detections = detections

            set_global('detections', detections)
            set_global('image_inference', img_inference)
            set_global('image_original', original_image)
            set_global('image_detections', None)
            self.prediction_image = None

            if plot is True:
                h, w = img_inference.shape[:2]
                o_h, o_w = original_image.shape[:2]
                #if image was scaled during preprocessing - scale the original image to show. it is a wrong way
                #todo: redo it in the correct way - we need to scale box/mask, not image
                if h!=o_h or w!=o_w:
                    original_image = resize_and_pad_cv (original_image, w, h)
                self.prediction_image = plot_predictions(original_image, filtered_detections['mask'].tolist(), filename=filename, colormap=colormap
                                                         , alpha=self.object_size.get("alpha", 0.75))

            return filtered_detections
        except Exception as e:
            raise RuntimeError(f"Error when inferrecing InstanSeg: {e}")

    
    def count_x10(self, input_image: str, colormap="tab20",
              filename=IMAGE_FILE_NAME_DETECTION, min_score=0.01,
              alpha=0.75, **kwargs):
        return self.count_x20(input_image, plot = True, colormap=colormap, tracking=False,
              filename=filename, min_score=min_score,       alpha=alpha, store_bin_mask=False,x10=True, **kwargs)
    
    def instanseg_results_to_pandas(self, labeled_output) -> pd.DataFrame:
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
            bin_mask, morphology = plot_mask(np.array(p_mask), image_size=(labeled_output.shape[2],labeled_output.shape[3]))
            data['diameter'].append(morphology['diameter'])
            data['area'].append(morphology['area'])
            data['volume'].append(morphology['volume'])
            
        return pd.DataFrame(data)
