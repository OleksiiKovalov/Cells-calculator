"""
Here we define the general class for segmentation models used within the application.
These are the models for:
- segmenting L929 cellular monolayer;
- segmenting spherical MSCs;
- segmenting spheroids.
"""
import os
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO

from model.BaseModel import BaseModel
from model.utils import *

class Segmenter(BaseModel):
    def __init__(self, path, object_size):
        super().__init__(path, object_size)
        self.detections=None

    def init_model(self, path_to_model: str):
        self.model = YOLO(path_to_model, task="segment")

    def count(self, input_image, conf=False, labels=False, boxes=False,
              masks=True, probs=False, show=False, save=True, color_mode="instance",
              filename=".cache/cell_tmp_img_with_detections.png", min_score=0.2, **kwargs):
        # TODO: 1) correct scale variable and plotting util function to plot bboxes properly;
        # 2) perform 1) by saving model results explicitly into cache;
        # 3) check out whether the dataset in pandas has correct structure
        # e.g., we need x-y-w-h format for filtering (or change filtering function);
        # 4) get to process segmentation results;
        # 5) build EXE file and see if it is working properly;
        # 6) change pipelines so that everything is working on the new basis;
        # 7) create list of TODOs for Alex for him to complete;
        # 8) start working on the tracker for the spheroids
        """
        This function performs inference on a given image using a pre-trained given model.
        The general pipeline can be described through the following steps:
        1. Load model, load image;
        2. Perform forward propagation and get results: bboxes, masks, confs;
        3. Structure the output;
        4. Save output in RAM cache as pandas DataFrame for further possible recalculations;
        5. Display obtained results through masks, if available, or simply through bboxes.

        Args:
            - model: loaded ultralytics YOLO model;
            - input_image: path to input image;
            - **kwargs: additional configurations for model inference: conf, iou etc.

        Returns:
            - list of dictionaries containing detections information.
        """
        try:
            os.remove(filename)
        except:
            pass
        if self.detections is None:
            outputs = self.model(input_image, conf=0.2, iou=0.6, retina_masks=True, **kwargs)[0]  # TODO: change the config definition point to a higher level
            self.original_image = outputs.orig_img
            outputs.plot(conf=conf, labels=labels, boxes=boxes,
                                             masks=masks, probs=probs, show=show, save=save,
                                             color_mode=color_mode, filename=filename)
            self.detections = results_to_pandas(outputs)
            self.h, self.w = outputs.orig_img.shape[0], outputs.orig_img.shape[1]
            self.detections['box'] = self.detections['box'].apply(lambda b: b * np.array([self.w, self.h, self.w, self.h]))
            # self.object_size['set_size'](self.detections[detections['confidence'] >= min_score]['box'].copy())
            self.object_size['set_size'](self.detections['box'].copy())

        detections = self.detections[self.detections['confidence'] >= min_score]
        # self.object_size['set_size'](detections['box'].copy())
        original_image = self.original_image.copy()

        filtered_detections = filter_detections(detections,
                                                min_size = self.object_size['min_size'],
                                                max_size= self.object_size['max_size'])
        # self.detections['box'] = filtered_detections['box'].apply(lambda b: b / np.array([self.w, self.h, self.w, self.h]))
        current_results = pandas_to_ultralytics(filtered_detections, original_image)
        if current_results is None:
            return None
        current_image = current_results.plot(conf=conf, labels=labels, boxes=boxes,
                                             masks=masks, probs=probs, show=show, save=save,
                                             color_mode=color_mode, filename=filename)
        return filtered_detections
    