import os
import numpy as np
import cv2
import tiffile

from model.CellCounter import CellCounter
from model.NucleiCounter import NucleiCounter

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
    def __init__(self, path=os.path.join('model', 'best_m.onnx'), threshold=100, eps=5, min_samples=10):
        self.nuclei_counter = NucleiCounter(threshold=threshold, eps=eps, min_samples=min_samples)
        self.cell_counter = CellCounter(path=path)

    def read_img(self, img_path, cell_channel=0, nuclei_channel=1):
        """Reads lsm image and returns as array."""
        with tiffile.TiffFile(img_path) as tif:
            image = tif.pages[0].asarray()
        if np.transpose(image, (1, 2, 0)).shape[-1] == 1:
            return cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_GRAY2RGB)
        elif np.transpose(image, (1, 2, 0)).shape[-1] == 2:
            img = np.transpose(image, (1, 2, 0))
            stacked_array = np.dstack((img, np.zeros((512,512)).astype('uint8')))
            return stacked_array #cv2.cvtColor(stacked_array, cv2.COLOR_RGB2BGR)
        elif np.transpose(image, (1, 2, 0)).shape[-1] == 3:
            return np.transpose(image, (1, 2, 0)) #cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_RGB2BGR)
        else:
            img = np.transpose(image, (1, 2, 0))
            stacked_array = np.dstack((img[cell_channel], img[nuclei_channel], np.zeros((512,512)).astype('uint8')))
            return stacked_array #cv2.cvtColor(stacked_array, cv2.COLOR_RGB2BGR)

    def calculate(self, img_path, cell_channel=0, nuclei_channel=1):
        """
        Calculates the resulting target values.
        Input params are:
        - img_path: path to lsm image;
        - cell_channel: channel with cells. Default to 0;
        - nuclei_channel: channel with stained nuclei. Default to.

        Returns the result as a dictionary with the following fields:
        - Nuclei: count for stained nuclei detected;
        - Cells: count for all the cells detected;
        - %: the target percentage for alive cells.
        """
        img = self.read_img(img_path)

        cell_img = cv2.cvtColor(img[:,:,cell_channel], cv2.COLOR_GRAY2BGR)
        tmp_path = 'cell_tmp_img.png'
        cv2.imwrite(tmp_path, cell_img)
        cell_count = self.cell_counter.countCells(tmp_path)
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

        nuclei_count = self.nuclei_counter.countNuclei(img[:,:,nuclei_channel])

        percentage = (1 - nuclei_count/cell_count) * 100

        return {'Nuclei': nuclei_count, 'Cells': cell_count, '%': round(percentage,1)}
