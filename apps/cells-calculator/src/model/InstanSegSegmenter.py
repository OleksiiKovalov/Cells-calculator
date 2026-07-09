# Standard library imports
import inspect
import json
import os
from collections import OrderedDict
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
import torch
from instanseg import InstanSeg
from instanseg.utils.utils import labels_to_features
from shapely.geometry import shape

# Local application imports
from ui.errorhandling import app_logger
from model.BaseSegmenter import BaseSegmenter
from model.utils import (
    plot_mask,
    process_loaded_image,
    resize_and_pad_cv,
)

# InstanSeg's eval_medium_image uses a fixed overlap window; inputs narrower than
# that window crash. Pad small images up to a safe floor before inference.
INSTANSEG_DEFAULT_OVERLAP = 200
INSTANSEG_MIN_WINDOW_SIZE = INSTANSEG_DEFAULT_OVERLAP + 1
INSTANSEG_MAX_PADDING_FLOOR = 512


class InstansegSegmenter(BaseSegmenter):
    """
    Cell/nuclei segmentation using the InstanSeg deep learning model.

    Provides an interface to InstanSeg for instance segmentation of cells and
    nuclei in microscopy images, with configurable tiling for large images.

    Attributes:
        model (InstanSeg): The InstanSeg model instance.
        image_preprocess_settings_default (OrderedDict): Default preprocessing settings.
    """
    def init_model(self, path_to_model: str):
        """
        Initialize the InstanSeg model for x20 segmentation.

        Loads a custom TorchScript model or a built-in model with GPU support.

        Args:
            path_to_model (str): Path to a TorchScript model file, or the name of a
                built-in model ('brightfield_nuclei', 'fluorescence_nuclei_and_cells').

        Note:
            Falls back to 'fluorescence_nuclei_and_cells' if the path is invalid.
            Automatically uses GPU if available.
        """
        self.image_preprocess_settings_default = json.loads(
            '[{"gray2rgb":""}]', object_pairs_hook=OrderedDict
        )

        device = self.device.type

        if path_to_model and os.path.exists(path_to_model):
            print(f"Initializing InstanSeg with model: {path_to_model}")
            model_module = torch.jit.load(path_to_model, map_location=self.device)
            self.model = InstanSeg(model_module, device=device, verbosity=1)
        elif path_to_model in [
            'brightfield_nuclei',
            'fluorescence_nuclei_and_cells',
        ]:
            print(f"Initializing InstanSeg with standard model: {path_to_model}")
            self.model = InstanSeg(path_to_model, device=device, verbosity=1)
        else:
            default_model = 'fluorescence_nuclei_and_cells'
            if path_to_model:
                print(
                    f"Warning: path/name '{path_to_model}' is not valid for InstanSeg. "
                    f"Using '{default_model}'."
                )
            else:
                print(
                    f"Warning: no InstanSeg model specified. "
                    f"Using '{default_model}'."
                )
            self.model = InstanSeg(default_model, device=device, verbosity=1)

        app_logger().warning(f"InstansegSegmenter: Device used: {self.device}")

    def call_inference(
        self,
        input_image,
        **kwargs
    ):
        """
        Segment cells/nuclei with InstanSeg at x20 magnification.

        Applies preprocessing, runs InstanSeg inference with optional tiling for
        large images, and converts the results to the standard DataFrame format.

        Args:
            input_image: Input microscopy image (array) to segment.
            **kwargs: Additional inference configuration.

        Returns:
            pd.DataFrame: Instance segmentation results with the standard detection columns.

        Raises:
            RuntimeError: If InstanSeg inference fails.
        """
        config_node = self.model_data['x20'] if 'x20' in self.model_data else None
        app_logger().info('Using x20 configuration for InstanSeg inference.')
        if config_node is not None:
            app_logger().info('InstanSeg config found')
            image_preprocess_settings = (
                config_node['image_preprocess']
                if 'image_preprocess' in config_node
                else self.image_preprocess_settings_default
            )
            pixel_size = config_node['pixel_size'] if 'pixel_size' in config_node else None
            tile_size = config_node['tile_size'] if 'tile_size' in config_node else '512'
            if isinstance(tile_size, str) and tile_size.endswith('%'):
                tile_size = int(int(tile_size[:-1]) * max(input_image.shape[:2]) / 100)
                if tile_size < 210:
                    tile_size = 210
                app_logger().info(
                    f'Calculated tile_size for InstanSeg inference: {tile_size}'
                )
            tile_size = int(tile_size)
            method_name = (
                self.model_data['inference_method_name']
                if 'inference_method_name' in self.model_data
                else 'eval_medium_image'
            )
        else:
            app_logger().info('InstanSeg config not found, using defaults')
            image_preprocess_settings = self.image_preprocess_settings_default
            pixel_size = None
            tile_size = 512
            method_name = 'eval_medium_image'

        img_inference = process_loaded_image(
            image=input_image,
            settings=image_preprocess_settings
        )
        img_inference = self._ensure_eval_window_size(
            img_inference, method_name, tile_size
        )

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
            labeled_output = method(**kwargs)
            return self.instanseg_results_to_pandas(labeled_output)
        except Exception as e:
            raise RuntimeError(f"Error when inferrecing InstanSeg: {e}")

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

        app_logger().info(
            "Padding InstanSeg inference image from "
            f"{width}x{height} to {target_width}x{target_height} "
            "to keep overlap smaller than the inference window."
        )
        return resize_and_pad_cv(image, target_width, target_height)

    def _extract_polygon_from_geometry(self, geometry):
        """
        Extract the primary polygon and its coordinate array from a geometry.

        Args:
            geometry: GeoJSON-like or shapely-compatible geometry object.

        Returns:
            tuple: (Polygon, np.ndarray) of the primary polygon and its exterior
            coordinates. Returns (None, None) when the geometry is empty, invalid,
            non-polygonal, or has fewer than 3 points.
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
        Convert InstanSeg labeled output to the standardized DataFrame format.

        Args:
            labeled_output (torch.Tensor): InstanSeg output tensor of shape
                (1, 1, height, width) containing instance labels as integers.

        Returns:
            pd.DataFrame: Detection DataFrame with the standard columns
            (id_label, box, mask, confidence, diameter, area, volume).

        Note:
            - All coordinates are normalized to the [0, 1] range.
            - Morphology is calculated assuming spherical objects.
            - Confidence is currently set to 1.
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
            data['mask'].append(norm_mask)
            #todo restore confidence
            data['confidence'].append(
                1 #outputs.boxes.conf[i].cpu().detach().numpy()
            )
            bin_mask, morphology = plot_mask(norm_mask, image_size=(h, w))
            data['diameter'].append(morphology['diameter'])
            data['area'].append(morphology['area'])
            data['volume'].append(morphology['volume'])

        return pd.DataFrame(data)
