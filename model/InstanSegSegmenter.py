"""
InstanSeg-based cell/nuclei segmentation utilities.

This module defines the `InstansegSegmenter` class, which wraps the
InstanSeg model for instance segmentation of cells and nuclei in microscopy
images. It provides model initialization, configurable preprocessing,
inference methods for x20/x10 magnification, and conversion of labeled
segmentation outputs into a standardized pandas DataFrame.

Key responsibilities:
- initialize InstanSeg with a custom TorchScript model or built-in model
- load and preprocess microscopy images for inference
- run tiled or untiled inference depending on configuration
- generate normalized detection results with morphology and bounding box data
"""

# Standard library imports
import inspect
import json
import os
from collections import OrderedDict
from logging import Logger
from typing import Any

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
from model.BaseModel import BaseModel
from model.utils import (
    filter_segmentation_detections,
    plot_mask, 
    process_loaded_image,
    resize_and_pad_cv, 
    safegray2rgb, 
)


INSTANSEG_DEFAULT_OVERLAP = 200
INSTANSEG_MIN_WINDOW_SIZE = INSTANSEG_DEFAULT_OVERLAP + 1
INSTANSEG_MAX_PADDING_FLOOR = 512


class InstansegSegmenter(BaseModel):

    _logger: Logger

    """
    Cell/nuclei segmentation using InstanSeg deep learning model.
    
    Provides interface to InstanSeg for instance segmentation of cells and nuclei
    in microscopy images. Supports multiple inference methods with configurable
    tiling strategies for large images.
    
    Attributes:
        model (InstanSeg): The InstanSeg model instance
        image_preprocess_settings_default (OrderedDict): Default preprocessing settings
    """
    def __init__(self, path_to_model: str, object_size, logger: Logger, model_data=None):
        """
        Initialize InstanSeg segmenter.
        """
        super().__init__(path_to_model, object_size, model_data)
        self._logger = logger

    def init_x20_model(self, path_to_model: str):
        """
        Initialize InstanSeg model for x20 magnification segmentation.
        
        Loads custom TorchScript model or built-in model with GPU support.
        
        Args:
            path_to_model (str): Path to TorchScript model file, or name of built-in model
                                ('brightfield_nuclei', 'fluorescence_nuclei_and_cells')
            
        Note:
            Falls back to 'fluorescence_nuclei_and_cells' if path invalid.
            Automatically uses GPU if available.
        """
        self.image_preprocess_settings_default = json.loads(
            '[{"gray2rgb":""}]', object_pairs_hook=OrderedDict
        )

        if path_to_model and os.path.exists(path_to_model):
            print(f"Ініціалізація InstanSeg з моделлю: {path_to_model}")
            model_module = torch.jit.load(path_to_model)
            self.model = InstanSeg(model_module, verbosity=1)
        elif path_to_model in [
            'brightfield_nuclei',
            'fluorescence_nuclei_and_cells',
        ]:
            print(f"Ініціалізація InstanSeg зі стандартною моделлю: {path_to_model}")
            self.model = InstanSeg(path_to_model, verbosity=1)
        else:
            default_model = 'fluorescence_nuclei_and_cells'
            if path_to_model:
                print(
                    f"Попередження: Шлях/назва '{path_to_model}' не валідні для InstanSeg. "
                    f"Використовується '{default_model}'."
                )
            else:
                print(
                    f"Попередження: Не вказано модель InstanSeg. "
                    f"Використовується '{default_model}'."
                )
            self.model = InstanSeg(default_model, verbosity=1)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._logger.warning(f"InstansegSegmenter: Device used: {device}")
            self.model = self.model.to(device)

    def init_x10_model(self, path_to_model):
        """
        Initialize InstanSeg model for x10 magnification.
        
        Currently not implemented.
        """
        pass

    def _ensure_eval_window_size(self, image, method_name, tile_size):
        """Pad very narrow inputs so InstanSeg's fixed overlap is valid."""
        if method_name != 'eval_medium_image':
            return image

        height, width = image.shape[:2]
        padding_floor = max(
            min(int(tile_size), INSTANSEG_MAX_PADDING_FLOOR),
            INSTANSEG_MIN_WINDOW_SIZE,
        )
        target_height = max(height, padding_floor)
        target_width = max(width, padding_floor)

        if target_height == height and target_width == width:
            return image

        self._logger.info(
            "Padding InstanSeg inference image from "
            f"{width}x{height} to {target_width}x{target_height} "
            "to keep overlap smaller than the inference window."
        )
        return resize_and_pad_cv(image, target_width, target_height)

    def count_x20(
        self,
        input_image,
        tracking=False,
        min_score=0.05,
        x10=False,
    ):
        """
        Segment cells/nuclei using InstanSeg at specified magnification.
        
        Loads image, applies preprocessing, runs InstanSeg inference with optional
        tiling for large images, and converts results to structured format.
        
        Args:
            input_image (str): Path to input microscopy image
            tracking (bool): Whether in tracking mode. Defaults to False.
            min_score (float): Minimum confidence threshold (0-1). Defaults to 0.05.
            x10 (bool): Use x10 configuration if True, else x10. Defaults to False.
        
        Returns:
            pd.DataFrame: Instance segmentation results with columns:
                - id_label: Unique object identifier
                - box: Bounding box coordinates
                - mask: Polygon contour points
                - confidence: Detection confidence score
                - diameter, area, volume: Morphological properties
                
        Raises:
            RuntimeError: If InstanSeg inference fails
        """
        image = imread(input_image)

        if x10:
            config_node = self.model_data['x10'] if 'x10' in self.model_data else None
            self._logger.info('Using x10 configuration for InstanSeg inference.')
        else:
            config_node = self.model_data['x20'] if 'x20' in self.model_data else None
            self._logger.info('Using x20 configuration for InstanSeg inference.')

        if config_node is not None:
            self._logger.info('InstanSeg config found')
            image_preprocess_settings = (
                config_node['image_preprocess']
                if 'image_preprocess' in config_node
                else self.image_preprocess_settings_default
            )
            pixel_size = config_node['pixel_size'] if 'pixel_size' in config_node else None
            tile_size = config_node['tile_size'] if 'tile_size' in config_node else '512'
            if isinstance(tile_size, str) and tile_size.endswith('%'):
                tile_size = int(int(tile_size[:-1]) * max(image.shape[:2]) / 100)
                if tile_size < 210:
                    tile_size = 210
                self._logger.info(
                    f'Calculated tile_size for InstanSeg inference: {tile_size}'
                )
            tile_size = int(tile_size)
            method_name = (
                self.model_data['inference_method_name']
                if 'inference_method_name' in self.model_data
                else 'eval_medium_image'
            )
        else:
            self._logger.info('InstanSeg config not found, using defaults')
            image_preprocess_settings = self.image_preprocess_settings_default
            pixel_size = None
            tile_size = 512
            method_name = 'eval_medium_image'

        img_inference = process_loaded_image(
            image=image,
            settings=image_preprocess_settings
        )
        img_inference = self._ensure_eval_window_size(
            img_inference,
            method_name,
            tile_size,
        )
        self.original_image = safegray2rgb(image)

        try:
            method = getattr(self.model, method_name, None)
            if not method:
                raise AttributeError(f"Method '{method_name}' not found on model")
            
            # Check if method accepts tile_size parameter
            sig = inspect.signature(method)
            has_tile_size = 'tile_size' in sig.parameters
            
            # Prepare base arguments
            inference_kwargs = {
                'image': img_inference,
                'return_image_tensor': False,
                'target': 'cells',
                'pixel_size': pixel_size
            }
            
            # Add tile_size only if method supports it and x10 is True
            if has_tile_size and x10:
                inference_kwargs['tile_size'] = tile_size
            
            labeled_output = method(**inference_kwargs)

            self.detections = self.instanseg_results_to_pandas(labeled_output)
            detections = self.detections[self.detections['confidence'] >= min_score]
            if tracking is False:
                self.object_size['signal']('set_size', self.detections.copy())
                self.detections[
                    ['id_label', 'confidence', 'diameter', 'area', 'volume']
                ].to_csv(
                    self.out_dir / 
                    f'{os.path.basename(self.original_image_path)}_'
                    f'{self.model_name}_cell_data.csv',
                    sep=';',
                    index=False,
                )

            #todo restore tracking feature
            if tracking is False:
                signal_fn = self.object_size.get('signal')
                if callable(signal_fn):
                    signal_fn("set_size", self.detections.copy())
                
                raw_min_size = self.object_size.get('min_size', 0.0)
                raw_max_size = self.object_size.get('max_size', 1.0)
                size_metric = self.object_size.get('size_metric', 'area')

                raw_min_size = float(raw_min_size)
                raw_max_size = float(raw_max_size)

                # Convert percent-like UI values for normalized metrics
                if size_metric in ('area', 'diameter', 'volume'):
                    if raw_min_size > 1.0:
                        raw_min_size = raw_min_size / 100.0
                    if raw_max_size > 1.0:
                        raw_max_size = raw_max_size / 100.0

                min_size = min(raw_min_size, raw_max_size)
                max_size = max(raw_min_size, raw_max_size)

                print(
                    f"Instanseg detections before size filtering: {len(detections)} "
                    f"Raw object_size values: min={self.object_size.get('min_size')}, "
                    f"max={self.object_size.get('max_size')}, metric={self.object_size.get('size_metric', 'area')}"
                )

                filtered_detections = filter_segmentation_detections(detections,
                                                        min_size=min_size,
                                                        max_size=max_size,
                                                        size_metric=size_metric)

                print(
                    f"InstanSeg detections after size filter: {len(filtered_detections)} "
                    f"(min_size={min_size}, max_size={max_size}, metric={size_metric})"
                )

                filtered_detections[['id_label', 'confidence', 'diameter', 'area', 'volume']].to_csv(
                    self.out_dir / f"{os.path.basename(self.original_image_path)}_{self.model_name}_cell_data.csv",
                    sep=';',
                    index=False
                )
            else:
                filtered_detections = detections.copy()

            #filtered_detections = detections

            self.prediction_image = None

            return self._prediction_result(
                filtered_detections,
                original_image=self.original_image,
                inference_image=img_inference,
            )
        except Exception as e:
            raise RuntimeError(f"Error when inferrecing InstanSeg: {e}")

    def count_x10(self, input_image: str, min_score=0.01):
        """
        Segment cells/nuclei using InstanSeg at x10 magnification with tiling.
        
        Delegates to count_x20 with x10=True for tiled inference on large images.
        
        Args:
            input_image (str): Path to input microscopy image
            min_score (float): Minimum confidence threshold (0-1). Defaults to 0.01.
        
        Returns:
            pd.DataFrame: Instance segmentation results (same format as count_x20)
        """
        return self.count_x20(
            input_image,
            tracking=False,
            min_score=min_score,
            x10=True,
        )
    def _extract_polygon_from_geometry(self, geometry):
        """
        Extract the primary polygon and coordinate array from geometry.

        Args:
            geometry: GeoJSON-like geometry object or shapely-compatible geometry.

        Returns:
            tuple: (Polygon, np.ndarray) where the polygon is the primary polygon
            geometry and the ndarray contains the exterior coordinates.
            Returns (None, None) when geometry is empty, invalid, non-polygonal,
            or does not contain a valid polygon with at least 3 points.
        """
        geom = shape(geometry)

        if geom.is_empty:
            return None, None

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty:
            return None, None

        if geom.geom_type == "Polygon":
            poly = geom
        elif geom.geom_type == "MultiPolygon":
            if len(geom.geoms) == 0:
                return None, None
            poly = max(geom.geoms, key=lambda g: g.area)
        else:
            return None, None

        coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)
        if coords.shape[0] < 3:
            return None, None

        return poly, coords

    def instanseg_results_to_pandas(self, labeled_output) -> pd.DataFrame:
        """
        Convert InstanSeg labeled output to standardized DataFrame format.
        
        Processes InstanSeg instance segmentation output (as labeled tensor) into
        a pandas DataFrame with normalized coordinates and morphological features.
        
        Args:
            labeled_output (torch.Tensor): InstanSeg output tensor of shape
                (1, 1, height, width) containing instance labels as integers.
                Feature information is encoded in tensor metadata.
        
        Returns:
            pd.DataFrame: Standardized detection DataFrame with columns:
                - id_label: Unique instance identifier (0-indexed)
                - box: [x_min, y_min, x_max, y_max] bounding box
                - mask: List of polygon contour points
                - confidence: Detection confidence (currently set to 1.0)
                - diameter, area, volume: Morphological measurements
                
        Note:
            - All coordinates are normalized to [0, 1] range
            - Morphology calculated assuming spherical objects
            - Bounds computed as union of all feature bounds
        """
        if torch.is_tensor(labeled_output):
            label_map_np = labeled_output[:, 0, :].detach().cpu().numpy()
        else:
            label_map_np = labeled_output[:, 0, :]
        h, w = labeled_output.shape[-2], labeled_output.shape[-1]
        instanseg_objects = labels_to_features(label_map_np)
        data: dict[str, list[Any]] = {
            'id_label': [],
            'box': [],
            'mask': [],
            'confidence': [],
            'diameter': [],
            'area': [],
            'volume': [],
        }

        features = instanseg_objects['features']
        minx, miny, maxx, maxy = None, None, None, None
        for i, feature in enumerate(features):
            geom = shape(feature['geometry'])  # Convert to shapely geometry
            poly, p_mask = self._extract_polygon_from_geometry(geom)
            if poly is None:
                continue
            bounds = poly.bounds  # (minx, miny, maxx, maxy)
            minx, miny, maxx, maxy = bounds
            box = np.array([
                minx / w,
                miny / h,
                (maxx - minx) / w,
                (maxy - miny) / h
            ], dtype=np.float32)
            norm_mask = p_mask / np.array([w, h], dtype=np.float32)

            data['id_label'].append(i)
            data['box'].append(box)
            data['mask'].append(p_mask)
            #todo restore confidence
            data['confidence'].append(
                1 #outputs.boxes.conf[i].cpu().detach().numpy()
            )
            bin_mask, morphology = plot_mask(norm_mask, image_size=(h, w))
            data['diameter'].append(morphology['diameter'])
            data['area'].append(morphology['area'])
            data['volume'].append(morphology['volume'])

        return pd.DataFrame(data)
