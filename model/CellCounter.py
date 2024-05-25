"""
In this module the CellCounter class is defined which is used
to calculate cells on a given contrast microimage.
"""

import os
import shutil
import cv2
import numpy as np
from model.utils import draw_bounding_box

CLASSES = ['Cell']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class CellCounter():
    """
    The class for object which performs cell counting.
    This is a part of the general model for obtaining target percentage.
    The objects of this class are not to be used explicitly - they function
    inside of the general Model class defined further.

    Input param is the path to pre-trained object detection model
    which calculates cells by detecting them using 'regression on
    tales' approach.

    Output value is the number of cells detected.
    """
    def __init__(self, path):
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(path)

    def count(self, model, input_image):
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
        original_image: np.ndarray = cv2.imread(input_image)
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 512

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(512, 512), swapRB=True)
        model.setInput(blob)

        # Perform inference
        outputs = model.forward()

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

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )
        cv2.imwrite('.cache/cell_tmp_img_with_detections.png', original_image)

        return detections

    def countCells(self, img_path):
        """
        By calling this method, the CellCounter class instance calculates cells on a given image.
        The counting is done by using 'regression on tales' approach.
        The limit is 1,200 cells per image maximum (4 parts with 300 cells).

        The input param is the path to RGB image of cells.

        The output param is optimized count of cells.
        """
        dst = os.path.join('.cache', 'cell_tmp_img.png')
        shutil.copy2(img_path, dst)
        detections = self.count(self.model, dst)
        return len(detections)
        