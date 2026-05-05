# Standard library imports
import json
import os
from typing import Optional, List, Tuple, Dict, Any
from collections import OrderedDict

# Third-party imports
import cv2  # OpenCV for findContours
import numpy as np
import pandas as pd
from cellpose import models as cp_models  # For Cellpose
from scipy.ndimage import find_objects  # Efficient bounding box calculation
from skimage.color import rgb2gray
from skimage.io import imread

# Local application imports
from UI.app_globals import (
    IMAGE_FILE_NAME_DETECTION,
    IMAGE_FILE_NAME_GRID,
    IMAGE_FILE_NAME_INGFERENCE,
    IMAGE_FILE_NAME_TMP,
)
from UI.errorhandling import app_logger
from model.BaseModel import BaseModel
from model.utils import (
    plot_mask,
    plot_predictions,
    plot_predictions_with_alignment,
    process_loaded_image,
    resize_and_pad_cv,
    safe_image_read,
    safegray2rgb,
    safe_image_write,
)


class CellposeSegmenter(BaseModel):
    """
    Cell segmentation using the Cellpose deep learning model.
    
    This class wraps the Cellpose segmentation model for detecting and segmenting
    cells in microscopy images. Supports both pre-trained and custom models with
    flexible preprocessing pipelines.
    
    Attributes:
        model (cellpose.models.CellposeModel): The Cellpose model instance
        cellpose_diam (float | None): Expected cell diameter for inference tuning
        image_preprocess_settings_default (OrderedDict): Default preprocessing config
    """
    def __init__(self, path_to_model: str, object_size, model_data=None):
        """
        Initialize Cellpose segmenter.
        """
        super().__init__(path_to_model, object_size, model_data)
        self.cellpose_diam = None
    
    def init_x20_model(self, path_to_model: str):
        """
        Initialize Cellpose model for x20 magnification segmentation.
        """
        self.image_preprocess_settings_default = json.loads(
            '[{"gray2rgb":""}]', object_pairs_hook=OrderedDict
        )
        if path_to_model and os.path.exists(path_to_model):
            print(f"Ініціалізація Cellpose з моделлю: {path_to_model}")
            self.model = cp_models.CellposeModel(
                gpu=self.use_gpu, pretrained_model=path_to_model
            )
        elif path_to_model in ["cyto", "nuclei", "cyto2", "cyto3"]:
            print(f"Ініціалізація Cellpose зі стандартною моделлю: {path_to_model}")
            self.model = cp_models.CellposeModel(
                gpu=self.use_gpu, model_type=path_to_model
            )
        else:
            default_model = "cyto"
            if path_to_model:
                print(
                    f"Попередження: Шлях/назва '{path_to_model}' не валідні "
                    f"для Cellpose. Використовується '{default_model}'."
                )
            else:
                print(
                    f"Попередження: Не вказано модель Cellpose. "
                    f"Використовується '{default_model}'."
                )
            self.model = cp_models.CellposeModel(
                model_type=default_model, gpu=self.use_gpu
            )
            app_logger().warning(f"CellposeSegmenter: GPU Used: {self.use_gpu}")

    def init_x10_model(self, path_to_model):
        """
        Initialize Cellpose model for x10 magnification.
        
        Currently not implemented.
        """
        pass

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
        **kwargs,
    ):
        """
        Segment cells in image using Cellpose at x20 magnification.
        
        Loads image, applies preprocessing, runs Cellpose segmentation, converts
        results to structured format, and optionally visualizes segmentations.
        
        Args:
            input_image (str): Path to input microscopy image
            plot (bool): Whether to generate visualization image. Defaults to True.
            colormap (str): Matplotlib colormap for mask visualization. Defaults to 'tab20'.
            tracking (bool): Whether in tracking mode (affects CSV output). Defaults to False.
            filename (str): Output path for visualization. Defaults to IMAGE_FILE_NAME_DETECTION.
            min_score (float): Minimum confidence score (0-1) to keep detections. Defaults to 0.05.
            alpha (float): Transparency for mask overlay (0-1). Defaults to 0.75.
            store_bin_mask (bool): Whether to store binary masks in DataFrame. Defaults to False.
            **kwargs: Additional arguments (unused, for API compatibility)
        
        Returns:
            pd.DataFrame: Segmentation results with columns:
                - id_label: Unique cell identifier
                - box: [x_min_norm, y_min_norm, width_norm, height_norm] (normalized)
                - mask: List of contour points (normalized coordinates)
                - confidence: Cell probability from model
                - diameter: Equivalent cell diameter (normalized)
                - area: Cell area (normalized to image area)
                - volume: Cell volume assuming spherical shape
                
        Raises:
            RuntimeError: If Cellpose inference fails
        """
        image = imread(input_image)
        image_preprocess_settings = (
            self.model_data["image_preprocess"]
            if "image_preprocess" in self.model_data
            else self.image_preprocess_settings_default
        )
        img_inference = process_loaded_image(
            image=image, settings=image_preprocess_settings
        )
        safe_image_write(img_inference, IMAGE_FILE_NAME_INGFERENCE, preserve_dtype=False)
        self.original_image = safegray2rgb(image)
        channels_to_use = [0, 0]  # Adapt!
        try:
            masks, flows, styles = self.model.eval(
                img_inference, diameter=self.cellpose_diam, channels=channels_to_use
            )
            print(f"Cellpose знайшов {np.max(masks)} об'єктів.")
            cellprob = flows[2] # Cell probability map
            self.detections = self.cellpose_results_to_pandas(
                masks,
                cellprob_map=cellprob,
                image_shape_for_norm=image.shape[:2], # Or masks.shape[:2] if appropriate
                store_bin_mask=False # Set to True if you need the binary masks in the DataFrame
            )
            detections = self.detections[self.detections["confidence"] >= min_score]
            if tracking is False:
                self.object_size["signal"](
                    "set_size", self.detections.copy()
                )
                self.detections[
                    ["id_label", "confidence", "diameter", "area", "volume"]
                ].to_csv(
                    self.out_dir
                    / f"{os.path.basename(self.original_image_path)}_"
                    f"{self.model_name}_cell_data.csv",
                    sep=";",
                    index=False,
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
            raise RuntimeError(f"Помилка інференсу Cellpose: {e}")
        

    def count_x10(
        self,
        input_image: str,
        filename=IMAGE_FILE_NAME_DETECTION,
        colormap="tab20",
        min_score=0.01,
        alpha=0.75,
        **kwargs,
    ):
        """
        Segment cells using Cellpose at x10 magnification.
        
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

    def image_preprocess(self, image):
        """
        Preprocess image for Cellpose inference.
        
        Converts RGB images to grayscale if needed.
        
        Args:
            image (np.ndarray): Input image array
            
        Returns:
            np.ndarray: Preprocessed image (grayscale)
        """
        if image.ndim == 3 and image.shape[-1] == 3:  # Check if likely RGB
            print("Converting RGB image to grayscale...")
            img_gray = rgb2gray(image)
            # Ensure it's float type, rescale if needed (rgb2gray outputs 0-1 float)
            # img_prepared = img_gray.astype(np.float32) # Optional explicit cast
            img_prepared = img_gray
        else:
            # Assume already grayscale or needs different handling
            print("Image already grayscale...")
            img_prepared = image
        return img_prepared

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
        img_bgr = safe_image_read(image_path, color_mode="color")
        if img_bgr is None:
            raise RuntimeError(
                f"Помилка: Не вдалося завантажити зображення {image_path}"
            )
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        return img_bgr
    
    def cellpose_results_to_pandas(
        self,
        masks: np.ndarray,
        cellprob_map: Optional[np.ndarray] = None,
        image_shape_for_norm: Optional[Tuple[int, int]] = None,
        store_bin_mask: bool = False
    ) -> pd.DataFrame:
        """Converts Cellpose segmentation results (masks) to a pandas DataFrame.

        This format is designed to be similar to what might be expected from
        an object detection model's output when processed into a structured format.

        Args:
            masks (np.ndarray): The 2D integer array output from Cellpose, where
                each unique positive integer represents a segmented object
                (0 is typically background).
            cellprob_map (Optional[np.ndarray]): The cell probability map from
                Cellpose (often from `flows[2]` if using `model.eval()`). Used
                to estimate a 'confidence' score. Should have the same H, W
                dimensions as `masks`.
            image_shape_for_norm (Optional[Tuple[int, int]]): The (height, width)
                of the original image, used for normalizing coordinates. If
                None, the shape of the `masks` array is used.
            store_bin_mask (bool): If True, stores the boolean binary mask
                (boolean array) for each object in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a detected
                object. Columns include:
                - id_label: The unique integer ID of the object from the mask.
                - box: Normalized bounding box
                  [x_min_norm, y_min_norm, width_norm, height_norm].
                - mask: List of normalized contour points
                  [[x1n, y1n], [x2n, y2n], ...].
                - confidence: Average cell probability within the mask, or 1.0 if
                  no map.
                - diameter: Equivalent diameter of a circle with the same area as
                  the mask.
                - area: Area of the mask in pixels.
                - volume: For 2D masks, this is the same as area.
                - bin_mask (optional): The boolean binary mask for the object.
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
            "volume": []
        }
        if store_bin_mask:
            data["bin_mask"] = []

        unique_object_ids = np.unique(masks)
        # Filter out background ID (usually 0)
        unique_object_ids = unique_object_ids[unique_object_ids != 0]

        if len(unique_object_ids) == 0: # Handle cases with no objects detected
            return pd.DataFrame(data)

        for object_id in unique_object_ids:
            current_bin_mask = (masks == object_id)
            
            # Area and Volume (for 2D, volume is treated as area)
            area = int(np.sum(current_bin_mask)) # Pixel count
            
            if area == 0: # Should not happen if object_id comes from unique valid masks
                continue

            data["id_label"].append(int(object_id))
            if store_bin_mask:
                data["bin_mask"].append(current_bin_mask)

            # Bounding Box calculation
            # find_objects on a binary mask of a single object returns a list with one item (tuple of slices)
            slices = find_objects(current_bin_mask)
            if not slices or slices[0] is None:
                # This case is highly unlikely if area > 0. Fill with NaNs for robustness.
                data['box'].append([np.nan, np.nan, np.nan, np.nan])
                data['mask'].append([])
                data['confidence'].append(np.nan if cellprob_map is not None else 1.0)
                # Remove the last added id_label, area, volume if we skip
                for key in ["id_label"]: data[key].pop()
                if store_bin_mask: data["bin_mask"].pop()
                continue

            y_slice, x_slice = slices[0]
            y_min, y_max = y_slice.start, y_slice.stop # y_max is exclusive (slice.stop)
            x_min, x_max = x_slice.start, x_slice.stop # x_max is exclusive

            box_width_px = x_max - x_min
            box_height_px = y_max - y_min
            
            # Normalized box: [x_min_norm, y_min_norm, width_norm, height_norm]
            x_min_norm = x_min / img_width
            y_min_norm = y_min / img_height
            width_norm = box_width_px / img_width
            height_norm = box_height_px / img_height
            data['box'].append([x_min_norm, y_min_norm, width_norm, height_norm])

            # Mask Contour (normalized polygon)
            contours, _ = cv2.findContours(current_bin_mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL,    # Retrieves only the extreme outer contours.
                                        cv2.CHAIN_APPROX_SIMPLE) # Compresses horizontal, vertical, and diagonal segments.

            normalized_contour_list = []
            if contours:
                # For a single, non-empty binary mask, we expect one major contour.
                # If multiple (e.g. mask has holes and RETR_TREE was used, or disjoint parts),
                # pick the largest by area. With RETR_EXTERNAL, this is simpler.
                contour = max(contours, key=cv2.contourArea)
                
                # Squeeze to remove redundant dimension: (N, 1, 2) -> (N, 2)
                squeezed_contour = contour.squeeze(axis=1)
                
                # Normalize contour points (x, y)
                normalized_contour_points = squeezed_contour.astype(np.float32) / \
                                            np.array([img_width, img_height], dtype=np.float32)
                normalized_contour_list = normalized_contour_points.tolist()
            data['mask'].append(normalized_contour_list)
            
            # Confidence (from cell probability map)
            if cellprob_map is not None:
                if cellprob_map.shape == current_bin_mask.shape: # Ensure alignment
                    object_pixels_in_cellprob = cellprob_map[current_bin_mask]
                    if object_pixels_in_cellprob.size > 0:
                        confidence = float(np.mean(object_pixels_in_cellprob))
                    else: # Should not happen if area > 0
                        confidence = np.nan 
                else:
                    # Shapes mismatch, cannot reliably calculate confidence for this mask object
                    confidence = np.nan 
                    print(f"Warning: cellprob_map shape {cellprob_map.shape} "
                        f"mismatches mask shape {current_bin_mask.shape} "
                        f"for object_id {object_id}. Confidence set to NaN."
                    )
            else:
                confidence = 1.0 # Default if no cell probability map is provided
                
            data['confidence'].append(confidence)
           
            bin_mask, morphology = plot_mask(np.array(normalized_contour_list), image_size=image_shape_for_norm)
            
            data['diameter'].append(morphology['diameter'])
            data['area'].append(morphology['area'])
            data['volume'].append(morphology['volume'])
            
        return pd.DataFrame(data)    
