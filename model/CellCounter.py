"""
In this module the CellCounter class is defined which is used
to calculate cells on a given contrast microimage.
"""

import os
# import shutil
import cv2
# import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
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
    def __init__(self, path, object_size):
        super().__init__(path, object_size)
        # self.model = YOLO(path, task="detect")
        self.detections=None

    def init_model(self, path_to_model: str):
        self.model = cv2.dnn.readNetFromONNX(path_to_model)

    # def count_cells(self, img_path):
        # """
        # By calling this method, the CellCounter class instance calculates cells on a given image.
        # The counting is done by using 'regression on tales' approach.
        # The limit is 1,200 cells per image maximum (4 parts with 300 cells).

        # The input param is the path to RGB image of cells.

        # The output param is optimized count of cells.
        # """
        # dst = os.path.join('.cache', 'cell_tmp_img.png')
        # shutil.copy2(img_path, dst)
        # detections = self.count(self.model, dst)
        # return len(detections)
    
    # def count(self, model, input_image, **kwargs):
        # # TODO: 1) correct scale variable and plotting util function to plot bboxes properly;
        # # 2) perform 1) by saving model results explicitly into cache;
        # # 3) check out whether the dataset in pandas has correct structure
        # # e.g., we need x-y-w-h format for filtering (or change filtering function);
        # # 4) get to process segmentation results;
        # # 5) build EXE file and see if it is working properly;
        # # 6) change pipelines so that everything is working on the new basis;
        # # 7) create list of TODOs for Alex for him to complete;
        # # 8) start working on the tracker for the spheroids
        # """
        # This function performs inference on a given image using a pre-trained given model.
        # The general pipeline can be described through the following steps:
        # 1. Load model, load image;
        # 2. Perform forward propagation and get results: bboxes, masks, confs;
        # 3. Structure the output;
        # 4. Save output in RAM cache as pandas DataFrame for further possible recalculations;
        # 5. Display obtained results through masks, if available, or simply through bboxes.

        # Args:
        #     - model: loaded ultralytics YOLO model;
        #     - input_image: path to input image;
        #     - **kwargs: additional configurations for model inference: conf, iou etc.

        # Returns:
        #     - list of dictionaries containing detections information.
        # """
        # if self.detections is None:
        #     outputs = self.model(input_image, imgsz=512, conf=0.2, iou=0.6, **kwargs)[0]  # TODO: change the config definition point to a higher level
        #     orig_img = outputs.orig_img
        #     self.original_image = orig_img.copy()
        #     orig_shape = outputs.orig_shape
        #     boxes = outputs.boxes.xyxy.cpu().detach().numpy()  # 0-1 normalized boxes in x1y1x2y2 format
        #     boxes[:,2] = boxes[:,2] - boxes[:,0]
        #     boxes[:,3] = boxes[:,3] - boxes[:,1]  # converted boxes to x1-y1-w-h format, which was used before
        #     boxes = boxes.tolist()
        #     # masks = list(outputs.masks.xy)  # 0-1 normalized masks in x1y1...xnyn YOLO format
        #     scores = list(outputs.boxes.conf)
        #     # boxes = boxes * torch.Tensor([orig_shape[1], orig_shape[0], orig_shape[1], orig_shape[0]])

        #     detections = {
        #             "class_id": list(outputs.names.keys())[0] * len(boxes),
        #             "class_name": list(outputs.names.values())[0] * len(boxes),
        #             "confidence": scores,
        #             "box": boxes,
        #             "scale": 512 / orig_shape[0],
        #     }

        #     # perform square-based filtering of bboxes
        #     detections = pd.DataFrame(detections)
        #     self.detections = detections
        #     self.scale = 512 / orig_shape[0]
        #     # change object_size for detection
        #     self.object_size['set_size'](detections['box'].copy())

        # detections = self.detections
        # original_image = self.original_image.copy()
        # scale = self.scale

        # filtered_detections = filter_detections(detections, min_size = self.object_size['min_size'], max_size= self.object_size['max_size'])
        # for i in range(filtered_detections.shape[0]):
        #     draw_bounding_box(
        #         original_image,
        #         filtered_detections.iloc[i,0],
        #         filtered_detections.iloc[i,2],
        #         round(filtered_detections.iloc[i,3][0] * scale),
        #         round(filtered_detections.iloc[i,3][1] * scale),
        #         round((filtered_detections.iloc[i,-2][0] + filtered_detections.iloc[i,-2][2]) * filtered_detections.iloc[i,-1]),
        #         round((filtered_detections.iloc[i,-2][1] + filtered_detections.iloc[i,-2][3]) * filtered_detections.iloc[i,-1]),
        #     )
        # try:
        #     os.remove('.cache/cell_tmp_img_with_detections.png')
        # except:
        #     pass
        # cv2.imwrite('.cache/cell_tmp_img_with_detections.png', original_image)

        # return filtered_detections

    def count(self, input_image):
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
            self.scale = scale
            # change object_size for detection
            self.object_size['signal']("set_size", detections['box'].copy())

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
        