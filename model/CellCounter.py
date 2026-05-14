"""
In this module the CellCounter class is defined which is used
to calculate cells on a given contrast microimage.
"""

# Third-party imports
import cv2
import numpy as np
import pandas as pd

from model.BaseModel import BaseModel
# Local application imports
from model.utils import safe_image_read


CLASSES = ["Cell"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class CellCounter(BaseModel):
    """
    The class for object which performs cell counting.

    This is a part of the general model for obtaining target percentage.
    The objects of this class are not to be used explicitly - they function
    inside of the general Model class defined further.

    Input param is the path to pre-trained object detection model
    which calculates cells by detecting them.

    Output value is the number of cells detected.
    """
    # def __init__(self, path, object_size):
    #     super().__init__(path, object_size)

    def init_x20_model(self, path_to_model: str):
        """
        Initialize ONNX model for x20 magnification inference.
        
        Loads an ONNX model from the specified path for cell detection
        at 20x magnification.
        
        Args:
            path_to_model (str): Path to the ONNX model file.
            
        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        self.model = cv2.dnn.readNetFromONNX(path_to_model)

    def init_x10_model(self, path_to_model):
        """
        Initialize model for x10 magnification.
        
        Currently not implemented.
        """
        self.model_x10 = None

    def count_x10(self, input_image):
        """
        Count cells at x10 magnification.
        
        Delegates to count_x20 method.
        
        Args:
            input_image (str): Path to input image

        Returns:
            pd.DataFrame: Detection results
        """
        return self.count_x20(input_image)


    def count_x20(self, input_image):
        """
        Perform inference on x20 magnification image using ONNX model.
        
        ---WARNING---: This function is deprecated and no longer used. The application now uses
        an ultralytics-based inference pipeline for improved performance and simplicity.
        
        Loads image, preprocesses for model input, runs ONNX inference, applies NMS,
        and saves CSV data for downstream UI/reporting use.
        
        Args:
            input_image (str): Path to input microscopy image

        Returns:
            pd.DataFrame: Detection results with columns:
                - class_id, class_name, confidence
                - box: [x, y, width, height]
                - scale: normalization factor
        
        Note:
            - Images are padded to square before inference (512x512)
            - NMS thresholds: score=0.25, iou=0.6
            - Confidence threshold: 0.2
        """
        # Read the input image
        inference_image = getattr(self, "_last_inference_image", None)
        if self.detections is None:
            original_image: np.ndarray = safe_image_read(
                input_image, color_mode="color"
            )
            if original_image is None:
                raise ValueError(f"Could not read image: {input_image}")
            self.original_image = original_image.copy()
            height, width, _ = original_image.shape

            # Prepare a square image for inference
            length = max((height, width))
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = original_image
            inference_image = image

            # Calculate scale factor
            scale = length / 512

            # Preprocess the image and prepare blob for model
            blob = cv2.dnn.blobFromImage(
                image, scalefactor=1 / 255, size=(512, 512), swapRB=True
            )
            self.model.setInput(blob)

            # Perform inference
            outputs = self.model.forward()
            # outputs = model(input_image, imgsz=512, conf=0.2, iou=0.6)
            # print("Outputs done")
            # return []

            # Prepare output array
            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []

            # Iterate through output to collect bounding boxes, confidence scores,
            # and class IDs
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                if maxScore >= 0.2:  # originally >= .25
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2],
                        outputs[0][i][3],
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            # Apply NMS (Non-maximum suppression)
            result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.6)  # score, nms thresholds
            result_boxes = np.array(result_boxes).flatten()

            detections = []

            # Iterate through NMS results to draw bounding boxes and labels
            for index in result_boxes:
                box = boxes[index]
                detection = {
                    "class_id": class_ids[index],
                    "class_name": CLASSES[class_ids[index]],
                    "confidence": scores[index],
                    "box": np.array(box),
                    "scale": scale,
                }
                detections.append(detection)

            # Perform square-based filtering of bboxes. Keep the expected
            # columns even when no objects pass the detector/NMS thresholds.
            detections = pd.DataFrame(
                detections,
                columns=["class_id", "class_name", "confidence", "box", "scale"],
            )
            self.detections = detections
            csv_data = self.detections.copy()
            csv_data["width"] = csv_data["box"].apply(
                lambda b: b[2] / length if b is not None else None
            )
            csv_data["height"] = csv_data["box"].apply(
                lambda b: b[3] / length if b is not None else None
            )
            csv_data["bbox_area"] = (
                csv_data["width"] * csv_data["height"] / length**2
            )
            csv_data[
                ["confidence", "width", "height", "bbox_area"]
            ].to_csv(
                self.out_dir / "cell_data.csv", sep=";", index=False
            )
            self.scale = scale
            # Change object_size for detection
            self.object_size["signal"]("set_size", detections["box"].copy())

        detections = self.detections
        self.object_size["signal"]("set_size", detections["box"].copy())
        # TODO: in this codeline, calculate max and min squares of obtained bboxes
        # and automatically set them as lower and upper bounds for the filtering
        # sliders if the sliders currently have default values (0 and 10) set up.
        # Otherwise do not re-set up them.
        # TODO: in this codeline, add initialization of min_size and max_size params
        # where their values are read from the boundary sliders. Scale them to be
        # in 0.0-1.0 range, as required by the filter_detections() function.
        # TODO: pass the min/max_size params to filter_detections() call below.
        # TODO: when opening a new image or folder of images, reset boundary
        # sliders to their default values (min=0%, max=10%).
        # filtered_detections = filter_detections(
        #     detections,
        #     min_size=self.object_size["min_size"],
        #     max_size=self.object_size["max_size"],
        # )
        filtered_detections = detections
        self.detectionCount = filtered_detections.shape[0]
        self.prediction_image = None

        return self._prediction_result(
            filtered_detections,
            original_image=self.original_image,
            inference_image=inference_image,
        )

