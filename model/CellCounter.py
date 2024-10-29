"""
In this module the CellCounter class is defined which is used
to calculate cells on a given contrast microimage.
"""

import os
import shutil
import cv2
import numpy as np
import pandas as pd

from model.utils import draw_bounding_box, filter_detections

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
    def __init__(self, path, object_size):
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(path)
        self.object_size = object_size
        self.detections=None
        self.original_image = None

    def count(self, model, input_image):
        """
        Main function to load ONNX model, perform inference, draw bounding boxes,
        and display the output image.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
            object_size (dict): !!!!!!!!

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
            # boxes_to_filter = np.array(boxes)[result_boxes,:]

            detections = []

            # Iterate through NMS results to draw bounding boxes and labels
            for i in range(len(result_boxes)):
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
            self.scale = scale
            # change object_size for detection
            self.object_size['set_size'](detections['box'].copy())

        detections = self.detections
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
            os.remove('.cache/cell_tmp_img_with_detections.png')
        except:
            pass
        cv2.imwrite('.cache/cell_tmp_img_with_detections.png', original_image)

        return filtered_detections

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
        