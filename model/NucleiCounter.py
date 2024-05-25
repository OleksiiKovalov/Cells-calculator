"""
In this module the NucleiCounter class is defined which is used
to calculate stained nuclei presented in LSM images.
"""

import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import DBSCAN

class NucleiCounter():
    """
    The class for object which performs marked nuclei counting.
    This is a part of the general model for obtaining target percentage.
    The objects of this class are not to be used explicitly - they function
    inside of the general Model class defined further.

    Input params are:
    - threshold value for nuclei image binarization;
    - eps param for DBSCAN algorithm;
    - min_samples value for DBSCAN algorithm.

    Output value is the number of marked nuclei detected.
    """
    def __init__(self, threshold=100, eps=2, min_samples=5):
        self.threshold = threshold
        self.eps = eps
        self.min_samples = min_samples

    def preprocess(self, channel, kernel_size=4, threshold=30):
        """Eliminates noise and structures binary component."""
        _, img = cv2.threshold(channel,
                               np.median(channel[channel > threshold]), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened_img = cv2.dilate(img, kernel)
        eroded_img = cv2.erode(opened_img, kernel)
        opened_img = cv2.dilate(eroded_img, kernel)
        eroded_img = cv2.erode(opened_img, kernel)
        return eroded_img

    def channel2points(self, channel):
        """Converts given binary channel to a set of points in 2D space."""
        indices = np.argwhere(channel > self.threshold)
        Y_coordinates = channel.shape[0] - indices[:, 0]
        X_coordinates = indices[:, 1]
        return pd.DataFrame({'x': X_coordinates, 'y': Y_coordinates})

    def groupNuclei(self, points):
        """Groups marked cell nuclei represented as points using DBSCAN clustering algorithm."""
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        if points.shape[0] == 0:
            return 0
        cluster_labels = dbscan.fit_predict(points[['x', 'y']])
        return np.max(cluster_labels)+1

    def countNuclei(self, img_channel):
        """Unites the functions above and calculates marked cell nuclei on a given image channel."""
        return self.groupNuclei(self.channel2points(self.preprocess(img_channel)))
    