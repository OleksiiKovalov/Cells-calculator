"""
StarDist-based cell segmentation utilities.

This module provides a wrapper around the StarDist 2D model for cell
segmentation in microscopy images. It handles model initialization,
image preprocessing, inference, and conversion of predictions into
a standardized pandas DataFrame format with morphological metrics.

Key responsibilities:
- initialize StarDist2D model with custom or default configuration
- load and preprocess microscopy images for inference
- run StarDist segmentation and extract polygonal predictions
- compute morphological properties from segmentation masks
"""

# Standard library imports
import json
import os
import traceback
from typing import List, Dict, Any
from collections import OrderedDict

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
from model.utils import (
    plot_predictions,
    plot_mask,
    process_loaded_image,
    resize_and_pad_cv, 
    safegray2rgb, 
    safe_image_write, 
    safe_image_read, 
    plot_predictions_with_alignment
)
from UI.app_globals import (
    IMAGE_FILE_NAME_DETECTION, 
    IMAGE_FILE_NAME_GRID, 
    IMAGE_FILE_NAME_INGFERENCE, 
    IMAGE_FILE_NAME_INSTANCES, 
    IMAGE_FILE_NAME_TMP
)



class StardistSegmenter(BaseModel):
    """
    Cell/nuclei segmentation using StarDist deep learning model.
    
    StarDist performs instance segmentation via star-convex polygons. Supports
    both pre-trained models and custom fine-tuned models with configurable
    preprocessing pipelines.
    
    Attributes:
        model (StarDist2D): The StarDist model instance
        is_custom_model (bool): Whether model is custom-trained vs. pre-trained
        image_preprocess_settings_default (OrderedDict): Default preprocessing config
    """
    def __init__(self, path_to_model: str, object_size, model_data=None):
        """
        Initialize StarDist segmenter.
        """
        self.is_custom_model = False
        super().__init__(path_to_model, object_size,model_data)
    
    def init_x20_model(self, path_to_model: str):
        """
        Load StarDist model for x20 magnification segmentation.
        
        Args:
            path_to_model (str): Path to custom model directory or name of built-in model
                                ('2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018')
            
        Note:
            Pre-trained models use 'gray2rgb' preprocessing.
            Custom models use 'rgb2gray' preprocessing.
        """
        app_logger().warning(
            f"Stardist: Num GPUs Available:{len(tf.config.list_physical_devices('GPU'))}"
        )        
        if (path_to_model in ("2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018")):
            self.is_custom_model = False
            self.model = StarDist2D.from_pretrained(path_to_model)
            self.image_preprocess_settings_default = json.loads(
                "[{\"gray2rgb\":\"\"} , {\"normalize\":\"1,99.8\"}]",
                object_pairs_hook=OrderedDict
            )
        else:
            self.is_custom_model = True
            path =os.path.dirname(path_to_model)
            name =os.path.basename(path_to_model)
            self.model = StarDist2D(None, name=name, basedir=path)
            self.image_preprocess_settings_default = json.loads(
                "[{\"rgb2gray\":\"\"} , {\"normalize\":\"1,99.8\"}]",
                object_pairs_hook=OrderedDict
            )

    def count_x20(
        self,
        input_image,
        plot=True,
        colormap="tab20",
        tracking=False,
        filename=IMAGE_FILE_NAME_DETECTION,
        min_score=0.05,
        alpha=0.75,
        store_bin_mask=False,
        **kwargs
    ):
        """
        Segment objects using StarDist at x20 magnification.
        
        Args:
            input_image (str): Path to input microscopy image
            plot (bool): Whether to generate visualization. Defaults to True.
            colormap (str): Colormap for visualization. Defaults to 'tab20'.
            tracking (bool): Whether in tracking mode. Defaults to False.
            filename (str): Output visualization path. Defaults to IMAGE_FILE_NAME_DETECTION.
            min_score (float): Minimum confidence score threshold. Defaults to 0.05.
            alpha (float): Mask transparency (0-1). Defaults to 0.75.
            store_bin_mask (bool): Whether to store binary masks. Defaults to False.
            **kwargs: Additional arguments for compatibility
        
        Returns:
            pd.DataFrame: Instance segmentation results with columns:
                id_label, box, mask, confidence, diameter, area, volume
                
        Raises:
            RuntimeError: If StarDist inference fails
        """
        image = imread(input_image)
        image_preprocess_settings = self.model_data["image_preprocess"] if "image_preprocess" in self.model_data else self.image_preprocess_settings_default
        img_inference = process_loaded_image(image=image, settings=image_preprocess_settings)
        safe_image_write(img_inference, IMAGE_FILE_NAME_INGFERENCE, preserve_dtype=False)
       
        self.original_image = safegray2rgb(image)
        try:
            labels, details = None, None
            labels, details = self.model.predict_instances(img_inference)
            self.detections = self.stardist_results_to_pandas(
                labels,
                scores=details["prob"],
                original_shape=image.shape[:2],
                inference_shape=image.shape[:2]
            )
            detections = self.detections[self.detections['confidence'] >= min_score]
            if tracking is False:
                self.object_size['signal']("set_size", self.detections.copy())
                self.detections[
                    ['id_label', 'confidence', 'diameter', 'area', 'volume']
                ].to_csv(
                    self.out_dir / f"{os.path.basename(self.original_image_path)}_{self.model_name}_cell_data.csv",
                    sep=';',
                    index=False
                )
            original_image = self.original_image.copy()

            filtered_detections = detections
            self.prediction_image = None
            if plot:
                self.prediction_image = plot_predictions_with_alignment(
                    original_image,
                    img_inference,
                    filtered_detections["mask"].tolist(),
                    filename=filename,
                    colormap=colormap,
                    alpha=self.object_size.get("alpha", 0.75),
                )
            return filtered_detections
        
        except Exception as e:
            traceback.print_exc()
            app_logger().exception(e)
            raise RuntimeError(f"Error when inferrecing StardistSegmenter: {e}")
        

    def count_x10(
        self,
        input_image: str,
        filename=IMAGE_FILE_NAME_DETECTION,
        colormap="tab20",
        min_score=0.01,
        alpha=0.75,
        **kwargs
    ):
        """
        Segment objects using StarDist at x10 magnification.
        
        Not implemented.
        
        Args:
            input_image (str): Path to input image
            filename (str): Output path (unused)
            colormap (str): Colormap (unused)
            min_score (float): Minimum score (unused)
            alpha (float): Transparency (unused)
            **kwargs: Additional arguments (unused)
        
        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError
    
    def image_preprocess(self,image):
        """
        Preprocess image for StarDist inference.
        
        Converts BGR to RGB color space.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Image in RGB format
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        return img_rgb

    def load_image(self, image_path):
        """
        Load and prepare image for processing.
        
        Reads image, converts to BGR format, handles different channel counts.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Image in BGR format
            
        Raises:
            RuntimeError: If image cannot be loaded
        """
        img_bgr = safe_image_read(image_path, color_mode='color')
        if img_bgr is None:
            raise RuntimeError(f"Unable to load image {image_path}")
        if len(img_bgr.shape) == 2: 
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4: 
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        return img_bgr
    
    def stardist_results_to_pandas(
        self,
        instances,
        scores=None,
        labels=None,
        original_shape=None,
        inference_shape=None
    ) -> pd.DataFrame:
        """
        Convert StarDist instance segmentation output to standardized DataFrame.
        
        Processes StarDist labeled instances into normalized coordinates with
        morphological features. Handles resizing if inference dimensions differ
        from original image dimensions.
        
        Args:
            instances (np.ndarray): 2D integer array with unique labels per instance
            scores (np.ndarray): Confidence scores for each instance. Optional.
            labels (np.ndarray): Alternative label array (currently unused). Optional.
            original_shape (tuple): (height, width) of original image. Used for normalization.
            inference_shape (tuple): (height, width) of image used for inference.
        
        Returns:
            pd.DataFrame: Standardized detections with columns:
                - id_label: Instance identifier
                - box: [x_min, y_min, x_max, y_max] bounding box
                - mask: Polygon contour points (cv2 format)
                - confidence: Instance confidence score
                - diameter, area, volume: Morphological properties
        """
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
        safe_image_write(instances, IMAGE_FILE_NAME_INSTANCES, preserve_dtype=False)

        for i, prop in enumerate(props):
            # Extract bounding box (min_row, min_col, max_row, max_col)
            minr, minc, maxr, maxc = prop.bbox
            box = [minc, minr, maxc, maxr]  # Convert to [x_min, y_min, x_max, y_max]

            # Create binary mask for the object
            binary_mask = (instances == prop.label).astype(np.uint8)
            #we need to resize shape
            if original_shape[0] != inference_shape[0] or original_shape[1]!=inference_shape[1]:
                binary_mask = resize(
                    binary_mask,
                    output_shape=original_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(binary_mask.dtype)
            
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
