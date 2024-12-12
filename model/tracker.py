"""
Here we define tracker for cellular spheroids.
This class differs a lot from an ordinary model, so we write it from scratch.
"""
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from model.segmenter import Segmenter
from model.utils import *

class Tracker():
    """
    Class for spheroid tracking model.
    The model implements tracking-by-detection approach and is built on top of
    a pre-trained YOLO11x instance segmentation model (defined as Segmenter class instance).
    """
    def __init__(self, path_to_model: str, size):
        self.path = path_to_model
        self.model = Segmenter(path_to_model, size)
        self.output_dir = Path("tracker_output")
        self.img_dir = self.output_dir / "frames"
        self.table_dir = self.output_dir / "tabular data"

    def track(self, img_seq_folder: str, time_period: float = 15):
        """
        Tracks spheroid instances through the sequence of frames.
        Uses Segmenter segmentation model for segmenting given frames.
        Afterwards, it saves the results in the tracker_output folder.
        The obtained saved results include:
        - all processed frames with masks visualized above;
        - general CSV table of all spheroid instances and their size params at all frames;
        - individual CSV tables for each spheroid instance with its size params at each frame.

        Input params include:
        - img_seq_folder: str - directory containing sequence of frames;
        - time_period: time period (presumably, in seconds) between frame shots. Default to 15.

        Output:
        - None
        """

        try:
            shutil.rmtree(self.output_dir)
        except:
            pass

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

        self.results = {
            "frame_num": [],
            "id_label": [],
            "old_label": [],
            "box": [],
            "mask": [],
            "confidence": [],
            "diameter": [],
            "area": [],
            "volume": []
        }

        frame_names = os.listdir(img_seq_folder)
        zero_frame = True
        i = -1
        # we process each frame one-by-one in the loop below
        for frame_name in frame_names:
            i += 1
            path = os.path.join(img_seq_folder, frame_name)
            filename = str(self.img_dir / ("frame_" + str(i).zfill(3) + ".png"))
            output = self.model.count_x20(path, plot=False, filename=filename,
                                          store_bin_mask=True, tracking=True)
            self.model.clear_cached_detections()
            # if it is the first frame in the sequence, we need to process it a bit differently
            if zero_frame:
                current_results = pandas_to_ultralytics(output, cv2.imread(path),
                                                        path=filename, frame_num=i)
                if current_results is None:
                    continue
                current_image = current_results.plot(conf=True, labels=True, boxes=True,
                                                     masks=True, probs=False, show=False,
                                                     save=True, color_mode="class",
                                                     filename=filename)
                zero_frame_results = output.copy()
                for c, _ in enumerate(zero_frame_results['mask'].tolist()):
                    # some masks are of 0 length, so we must filter and skip them
                    if len(zero_frame_results['mask'].iloc[c]) == 0:
                        pass
                    else:
                        # but in general we just mine the needed data and save it
                        _, morphology = plot_mask(zero_frame_results['mask'].iloc[c])

                        self.results["frame_num"].append(i)
                        self.results["id_label"].append(c)
                        self.results["old_label"].append(c)
                        self.results["box"].append(zero_frame_results['box'].iloc[c])
                        self.results["mask"].append(zero_frame_results['mask'].iloc[c])
                        self.results["confidence"].append(zero_frame_results['confidence'].iloc[c])
                        self.results["diameter"].append(morphology['diameter'])
                        self.results["area"].append(morphology['area'])
                        self.results["volume"].append(morphology['volume'])
                
            # below we process the other sequence frames
            else:
                # firstly, converting current results to pandas dataframe
                self.results = pd.DataFrame(self.results)
                # then, we need to correspond the spheroid from current frame with spheroids
                # from the very first frame to assign the same IDs to them
                iou_matrix, morphology = compute_iou(zero_frame_results['mask'].tolist(), output['mask'].tolist())

                # Mask to identify columns with non-zero elements
                non_zero_columns = (iou_matrix != 0).any(axis=0)

                # Create a reduced matrix with only the desired columns
                filtered_matrix = iou_matrix[:, non_zero_columns]

                # Compute argmax for the filtered matrix
                id_indices = np.full(iou_matrix.shape[1], -1)  # Default value for excluded columns (-1 or another value)
                iou_values = np.full(iou_matrix.shape[1], -1).astype(float)  # Default value for excluded columns (-1 or another value)
                id_indices[non_zero_columns] = np.argmax(filtered_matrix, axis=0)
                iou_values[non_zero_columns] = np.max(filtered_matrix, axis=0)

                # now we need to filter out those instances which correspond to 2+
                # original ones at the same time by choosing the "real" one
                # Step 1: Identify unique id_indices (excluding -1 values) and their indices
                unique_indices = np.unique(id_indices[non_zero_columns])

                # Step 2: Iterate over unique indices and handle duplicates
                for index in unique_indices:
                    if np.sum(id_indices == index) > 1:  # If the index appears more than once
                        # Find the position where iou_values are maximized for this index
                        max_iou_index = np.argmax(iou_values[id_indices == index])

                        # Set all occurrences of this index to -1 except the one with the maximum IOU value
                        indices_to_update = np.where(id_indices == index)[0]
                        for itu in indices_to_update:
                            if itu != indices_to_update[max_iou_index]:
                                id_indices[itu] = -1

                for k, row in output.iterrows():
                    if k >= len(id_indices):
                        break
                    row["id_label"] = id_indices[k]
                    record = pd.DataFrame({
                        "frame_num": [i],
                        "id_label": [row['id_label']],
                        "old_label": [k],
                        "box": [row['box']],
                        "mask": [row['mask']],
                        "confidence": [row['confidence']],
                        "diameter": [morphology[k]['diameter']],
                        "area": [morphology[k]['area']],
                        "volume": [morphology[k]['volume']]
                    })
                    # here we assigned all the iIDs, mined all the data and then save it below
                    self.results = pd.concat((self.results, record), ignore_index=True)
                filtered_results = self.results[(self.results['frame_num'] == i)
                                                & (self.results['id_label'] != -1)]
                columns_of_interest = ["frame_num", "old_label", "box", "mask",
                                       "confidence", "diameter", "area", "volume"]
                current_results = pd.merge(filtered_results[columns_of_interest],
                                           output[['id_label', 'bin_mask']],
                                           left_on='old_label', right_on='id_label', how='inner')
                current_results['id_label'] = filtered_results['id_label'].tolist()
                current_results = pandas_to_ultralytics(current_results, cv2.imread(path),
                                                        path=filename, frame_num=i)
                if current_results is None:
                    continue
                current_image = current_results.plot(conf=True, labels=True, boxes=True,
                                                     masks=True, probs=False, show=False,
                                                     save=True, color_mode="class",
                                                     filename=filename)
            zero_frame = False
        # now we processed all frames and start saving the results
        self.results['id_label'].dropna(inplace=True)
        self.results = self.results[self.results['id_label'] != -1]
        unique_spheroids = self.results['id_label'].unique().tolist()
        for spheroid in unique_spheroids:
            spheroid_records = self.results[self.results['id_label'] == spheroid]
            columns_of_interest = ["frame_num", "confidence", "diameter", "area", "volume"]
            filename = str(self.table_dir / ("spheroid_" + str(spheroid).zfill(2) + ".csv"))
            spheroid_records[columns_of_interest].to_csv(filename, sep=';', index=False)
        columns_of_interest = ["frame_num", "id_label",
                               "confidence", "diameter", "area", "volume"]
        general_t_series_data = self.results[columns_of_interest]
        filename = str(self.table_dir / ("general_t_series_data.csv"))
        general_t_series_data.to_csv(filename, sep=';', index=False)
