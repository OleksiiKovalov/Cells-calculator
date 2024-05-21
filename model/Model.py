import os
import numpy as np
import cv2
import tiffile

from model.CellCounter import CellCounter
from model.NucleiCounter import NucleiCounter
from model.utils import read_img, is_image_valid, calculate_lsm, calculate_standard

class Model():
    """
    The class for object of general model.
    To use the class instances, define a new instance and use
    'instance.calculate(img_path)' command, where img_path is
    the path to .lsm image of proper quality.

    Input params are:
    - path: the path param for CellCounter;
    - threshold param for NucleiCounter;
    - eps param for NucleiCounter;
    - min_samples param for NucleiCounter.

    Output values are returned as dictionary and represent:
    - 'Nuclei': the number of nuclei detected;
    - 'Cells': the number of cells detected;
    - '%': the target percentage value obtained.
    """
    def __init__(self, path=os.path.join('model', 'best_m.onnx'),
                 threshold=100, eps=5, min_samples=10):
        self.nuclei_counter = NucleiCounter(threshold=threshold,
                                            eps=eps, min_samples=min_samples)
        self.cell_counter = CellCounter(path=path)

    def calculate(self, img_path, cell_channel=0, nuclei_channel=1):
        """
        Calculates the resulting target values.
        Input params are:
        - img_path: path to lsm/jpg/png/tif/bmp image;
        - cell_channel: channel with cells. Default to 0;
        - nuclei_channel: channel with stained nuclei. Default to 1.

        Returns the result as a dictionary with the following fields:
        - Nuclei: count for stained nuclei detected (given lsm image only);
        - Cells: count for all the cells detected;
        - %: the target percentage for alive cells (given lsm image only).
        """
        if img_path.endswith('lsm'):
            return calculate_lsm(self.cell_counter, self.nuclei_counter,
                  img_path, cell_channel, nuclei_channel)
        elif is_image_valid(img_path):
            return calculate_standard(self.cell_counter, img_path)
