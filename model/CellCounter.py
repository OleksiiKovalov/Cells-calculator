"""
In this module the CellCounter class is defined which is used
to calculate cells on a given contrast microimage.
"""

import os
import cv2
import numpy as np
import pandas as pd
from model.BaseModel import BaseModel
from model.utils import draw_bounding_box, filter_detections

CLASSES = ['Cell']
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
        self.model = cv2.dnn.readNetFromONNX(path_to_model)

    def init_x10_model(self, path_to_model):
        self.model_x10 = None

    def count_x10(self, input_image, filename):
        return self.count_x20(input_image, filename)

    def count_x20(self, input_image, filename):
        # # NOTE: this function is deprecated and no longer used, because we have implemented ultralytics-based inference pipeline for simplicity
        """
        Main function to load ONNX model, perform inference, draw bounding boxes,
        and display the output image.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.

        Returns:
            list: List of dictionaries containing detection information such as class_id,
            class_name, confidence, etc.
        """
        # Read the input image
        if self.detections is None:
            original_image: np.ndarray = cv2.imread(input_image)
            self.original_image = original_image.copy()
            [height, width, _] = original_image.shape

            # Prepare a square image for inference
            length = max((height, width))
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = original_image

            # Calculate scale factor
            scale = length / 512

            # Preprocess the image and prepare blob for model
            blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(512, 512), swapRB=True)
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

            # Iterate through output to collect bounding boxes, confidence scores, and class IDs
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
            boxes_to_filter = np.array(boxes)[result_boxes,:]

            detections = []

            # Iterate through NMS results to draw bounding boxes and labels
            for i, _ in enumerate(result_boxes):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    "class_id": class_ids[index],
                    "class_name": CLASSES[class_ids[index]],
                    "confidence": scores[index],
                    "box": np.array(box),
                    "scale": scale,
                }
                detections.append(detection)
            # perform square-based filtering of bboxes
            detections = pd.DataFrame(detections)
            self.detections = detections
            csv_data = self.detections.copy()
            csv_data['width'] = csv_data['box'].apply(lambda b: b[2] / length)
            csv_data['height'] = csv_data['box'].apply(lambda b: b[3] / length)
            csv_data['bbox_area'] = csv_data['width'] * csv_data['height'] / length**2
            csv_data[['confidence', 'width', 'height',
                      'bbox_area']].to_csv(self.out_dir / "cell_data.csv", sep=';', index=False)
            self.scale = scale
            # change object_size for detection
            self.object_size['signal']("set_size", detections['box'].copy())

        detections = self.detections
        self.object_size['signal']("set_size", detections['box'].copy())
        original_image = self.original_image.copy()
        scale = self.scale
        # TODO: in this codeline, calculate max and min squares of obtained bboxes and automatically
        # set them as lower and upper bounds for the filtering sliders if the sliders currently
        # have default values (0 and 10) set up. Otherwise do not re-set up them.
        # TODO: in this codeline, add initialization of min_size and max_size params where their
        # values are read from the boundary sliders. Scale them to be in 0.0-1.0 range, as required
        # by the filter_detections() function.
        # TODO: pass the min/max_size params to filter_detections() call below.
        # TODO: when opening a new image or folder of images, reset boundary sliders to their default values (min=0%, max=10%).
        filtered_detections = filter_detections(detections, min_size = self.object_size['min_size'], max_size= self.object_size['max_size'])
        for i in range(filtered_detections.shape[0]):
            draw_bounding_box(
                original_image,
                filtered_detections.iloc[i,0],
                filtered_detections.iloc[i,2],
                round(filtered_detections.iloc[i,3][0] * scale),
                round(filtered_detections.iloc[i,3][1] * scale),
                round((filtered_detections.iloc[i,-2][0] + filtered_detections.iloc[i,-2][2]) * filtered_detections.iloc[i,-1]),
                round((filtered_detections.iloc[i,-2][1] + filtered_detections.iloc[i,-2][3]) * filtered_detections.iloc[i,-1]),
            )
        try:
            os.remove(filename)
        except:
            pass
        cv2.imwrite(filename, original_image)

        return filtered_detections
        