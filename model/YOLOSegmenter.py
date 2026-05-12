"""
Here we define the general class for segmentation models used within the application.
These are the models for:
- segmenting L929 cellular monolayer;
- segmenting spherical MSCs;
- segmenting spheroids.
"""

# Third-party imports
import numpy as np
import pandas as pd
from ultralytics import YOLO

from model.BaseModel import BaseModel
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from model.utils import results_to_pandas, sahi_to_pandas

class YoloSegmenter(BaseModel):
    """
    Instance segmentation using YOLO11 model.
    
    Provides interface to YOLOv8/YOLO11 for instance segmentation with support
    for both single-image and tiled inference approaches. Includes:
    - Full-image inference for x20 magnification
    - Sliced inference with SAHI for x10 magnification
    
    Attributes:
        model (YOLO): YOLOv8/YOLO11 segmentation model
        model_x10 (AutoDetectionModel): SAHI-wrapped model for tiled inference
    """
    DETECTION_COLUMNS = [
        "id_label",
        "box",
        "mask",
        "confidence",
        "diameter",
        "area",
        "volume",
    ]

    def _get_sahi_predictions(self, input_image):
        """Run SAHI prediction with the tuned x10 inference settings."""
        return get_sliced_prediction(
            input_image,
            self.model_x10,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=.1,
            overlap_width_ratio=.1,
            perform_standard_pred=False,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            verbose=0,
        ).to_coco_predictions()

    def init_x20_model(self, path_to_model: str):
        """
        Load YOLO model for x20 magnification full-image inference.
        
        Args:
            path_to_model (str): Path to YOLO .pt model weights file
        """
        self.model = YOLO(path_to_model, task="segment")

    def init_x10_model(self, path_to_model):
        """
        Load YOLO model for x10 magnification tiled inference with SAHI.
        
        Uses SAHI (Sliced Aided Hyper Inference) for processing large x10 images
        by dividing into overlapping tiles and stitching predictions.
        
        Args:
            path_to_model (str): Path to YOLO .pt model weights file
            
        Note:
            Confidence threshold: 0.3 (aligned with x20 inference)
            Device: CPU (configurable in code)
        """
        self.model_x10 = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=path_to_model,
            confidence_threshold=0.3,
            device="cpu",  # or 'cuda:0'
        )

    def count_x20(
        self,
        input_image,
        tracking=False,
        min_score=0.05,
        store_bin_mask=False,
        **kwargs
    ):
        """
        This function performs inference on a given image using a pre-trained model.
        The general pipeline can be described through the following steps:
        1. Load model, load image;
        2. Perform forward propagation and get results: bboxes, masks, confs;
        3. Structure the output;
        4. Save output in RAM cache as pandas DataFrame for further possible recalculations;
        5. Return raw detections plus original/inference image artifacts.

        Args:
            input_image: path to input image
            tracking: whether tracking is enabled
            min_score: minimum confidence score
            store_bin_mask: whether to store binary mask
            **kwargs: additional configurations for model inference: conf, iou etc.

        Returns:
            list of dictionaries containing detections information.
        """
        # Every time detect from fresh
        self.detections = None
        
        outputs = self.model(
            input_image,
            conf=0.3,
            iou=0.6,
            max_det=2000,
            retina_masks=True,
            **kwargs
        )[0]
        self.original_image = outputs.orig_img
        inference_image = self.original_image.copy()
        self.h, self.w = outputs.orig_img.shape[0], outputs.orig_img.shape[1]
        if outputs.masks is None:
            self.detections = pd.DataFrame(columns=self.DETECTION_COLUMNS)
            self.object_size['signal']("set_size", self.detections['box'].copy())
            self.detections[['id_label', 'confidence', 'diameter', 'area', 'volume']].to_csv(
                self.out_dir / "cell_data.csv",
                sep=';',
                index=False
            )
            self.prediction_image = None
            return self._prediction_result(
                self.detections,
                original_image=self.original_image,
                inference_image=inference_image,
            )
        self.detections = results_to_pandas(outputs, store_bin_mask)
        self.detections['box'] = self.detections['box'].apply(
            lambda b: b * np.array([self.w, self.h, self.w, self.h])
        )

        if tracking is False:
            # Keep slider calibration aligned with morphology-based filtering.
            self.object_size['signal']("set_size", self.detections.copy())
            self.detections[['id_label', 'confidence', 'diameter', 'area', 'volume']].to_csv(
                self.out_dir / "cell_data.csv",
                sep=';',
                index=False
            )

        detections = self.detections[self.detections['confidence'] >= min_score]
        if tracking is False:
            self.object_size['signal']("set_size", self.detections.copy())
        filtered_detections = detections

        self.prediction_image = None
        return self._prediction_result(
            filtered_detections,
            original_image=self.original_image,
            inference_image=inference_image,
        )

    def count_x10(self, input_image: str, min_score=0.3):
        """
        Segment image using YOLO with SAHI tiling at x10 magnification.
        
        Divides large images into overlapping tiles, runs detection on each tile,
        and merges predictions. Caches detections for reuse within same image.
        
        Args:
            input_image (str): Path to input image
            min_score (float): Minimum confidence filter. Defaults to 0.3.
        
        Returns:
            pd.DataFrame: Segmentation results (same format as count_x20)
            
        Note:
            Tile configuration: 512x512 with 10% overlap
            Results are cached and reused if same image processed multiple times
        """
        if self.detections is None or self.original_image is None:
            self.original_image = read_image(input_image)
            outputs = self._get_sahi_predictions(input_image)
            self.h, self.w = self.original_image.shape[0], self.original_image.shape[1]
            self.detections = sahi_to_pandas(outputs, self.h, self.w)
            self.object_size['signal']("set_size", self.detections.copy())

        detections = self.detections[self.detections['confidence'] >= min_score]

        filtered_detections = detections
        inference_image = self.original_image.copy()
        
        self.prediction_image = None
        return self._prediction_result(
            filtered_detections,
            original_image=self.original_image,
            inference_image=inference_image,
        )
