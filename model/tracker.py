"""
Here we define tracker for cellular spheroids.
This class differs a lot from an ordinary model, so we write it from scratch.
"""
import os
from pathlib import Path
from model.segmenter import Segmenter
import numpy as np
import pandas as pd
import cv2
from model.utils import *

class Tracker():
    def __init__(self, path_to_model: str, size):
        self.model = Segmenter(path_to_model, size)
        self.output_dir = Path("output")

    def track(self, img_seq_folder: str):
        """
        Tracks spheroid instances through the sequence of frames.

        Input params include:
        - img_seq_folder: str - directory containing sequence of frames.

        Output params:
        - TODO
        """
        # TODO: we need the following algorithm here:
        # 1) we process frame #0 and find all the spheroids there. We assign a unique ID to each spheroid on this frame.
        # 2) we process the following frames one-by-one by doing the following:
        # 3) obtain the mask of each spheroid;
        # 4) calculate IoU values of each spheroid on the current frame with spheroids from the previous frame;
        # 5) if a certain spheroid has max IoU with a spheroid from the previous frame, and this value exceeds some minimal required threshold, we assume this is the same instance
        # 6) if some spheroid is lost somewhere in the middle, we keep waiting for it arriving back - in standard terms, we have infinite memory value here.
        # 7) all the results are stored in one large pandas dataframe which has the following columns (each row describes certain spheroid instance on a certain frame):
        #   - frame_num: number of current frame in the frame sequence;
        #   - id_label: id of a spheroid about which this record in the dataframe is;
        #   - box: array representing the bbox of this spheroid;
        #   - mask: array representing the mask of this spheroid;
        #   - conf: confidence score of this spheroid instance;
        #   - diameter: diameter of this spheroid instance approximated to a circle shape. Is in ratio relative to the square root of the image square;
        #   - area: area of this spheroid. Is in ratio relative to the area of the image;
        #   - volume: volume of the spheroid extrapolated to a spherical shape. Is in ratio relative to image square multiplied by the square root of the image square
        # 8) based on all of that, we build resulting pandas dataframe representing the time series data obtained. It has these columns:
        #   - t: time point in seconds (by default defined as 0-15-30-45-60-...);
        #   - diameter: diameter obtained above;
        #   - area: area obtained above;
        #   - volume: volume obtained above.
        # 9) if any values in the middle are missing, we interpolate them linearly by taking the average value of the 2 values surrounding the missing one.
        # 10) if any first or last values are missing, we do not extrapolate them and instead just mark them as NaN (-100);
        # 11) if a spheroid instance is presented in fewer than minimal threshold value frames, we eliminate it as the one that got lost during the experiments.
        self.results = pd.DataFrame(columns=[
            "frame_num",
            "id_label",
            "box",
            "mask",
            "conf",
            "diameter",
            "area",
            "volume"
        ])
        frame_names = os.listdir(img_seq_folder)
        zero_frame = True
        i = 0
        os.makedirs(self.output_dir, exist_ok=True)
        for frame_name in frame_names:
            path = os.path.join(img_seq_folder, frame_name)
            output = self.model.count(path, calculate_morphology=True, filename=self.output_dir / ("frame_" + str(i).zfill(3) + ".png"))
            if zero_frame:
                zero_frame_results = output.copy()
                c = -1
                for c, _ in enumerate(zero_frame_results.masks.xyn):
                    if len(zero_frame_results.masks.xyn[c]) == 0:
                        pass
                    else:
                        _, morphology = plot_mask(zero_frame_results.masks.xyn[c])
                        record = pd.DataFrame({
                            "frame_num": i,
                            "id_label": c,
                            "box": zero_frame_results['box'].iloc[c],
                            "mask": zero_frame_results['mask'].iloc[c],
                            "conf": zero_frame_results['confidence'].iloc[c],
                            "diameter": morphology['diameter'],
                            "area": morphology['area'],
                            "volume": morphology['volume'],
                        })
                        self.results = pd.concat(self.results, record, ignore_index=True)
                        c += 1
            else:
                iou_matrix, morphology = compute_iou(zero_frame_results.masks.xyn, output.masks.xyn)
                id_indices = np.argmax(iou_matrix, axis=0)                
                i = 0
                for _, record in output.iterrows():
                    record["id_label"] = id_indices[i]
                    i += 1
                    self.results = pd.concat(self.results, record, ignore_index=True)
            zero_frame = False
        unique_spheroids = self.results['id_label'].unique().tolist()
        general_t_series_data = pd.DataFrame(columns=[
                "t",
                "diameter",
                "square",
                "volume",
            ])
        for spheroid in unique_spheroids:
            spheroid_records = self.results[self.results['id_label'] == spheroid]
            t_series_data = pd.DataFrame(columns=[
                "t",
                "diameter",
                "square",
                "volume",
            ])
            i = 0
            for _, record in spheroid_records.iterrows():
                record = record.to_dict()
                record['t'] = i * 15
                i += 1
                t_series_data = pd.concat(t_series_data, pd.DataFrame(record), ignore_index=True)
            t_series_data.to_csv("spheroid_" + str(spheroid).zfill(2) + ".csv")
            general_t_series_data = pd.concat(general_t_series_data, t_series_data, ignore_index=True)
        general_t_series_data.to_csv("general_t_series_data.csv")
