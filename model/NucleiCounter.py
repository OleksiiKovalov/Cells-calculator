"""
In this module the NucleiCounter class is defined which is used
to calculate stained nuclei presented in LSM images.
"""

# Third-party imports
import cv2
import numpy as np
import pandas as pd
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
        """
        Initialize nuclei counter with DBSCAN parameters.
        
        Args:
            threshold (int): Binary threshold for nuclei channel binarization. Defaults to 100.
            eps (float): DBSCAN epsilon parameter for clustering radius. Defaults to 2.
            min_samples (int): DBSCAN minimum samples per cluster. Defaults to 5.
        """
        self.threshold = threshold
        self.eps = eps
        self.min_samples = min_samples

    def preprocess(self, channel, kernel_size=4, threshold=30):
        """
        Preprocess nuclei channel to remove noise and enhance structures.
        
        Applies binary thresholding followed by morphological opening and closing
        operations to eliminate noise while preserving nuclei structure.
        
        Args:
            channel (np.ndarray): Input channel image (2D grayscale array)
            kernel_size (int): Size of morphological structuring element. Defaults to 4.
            threshold (int): Threshold value for binarization. Defaults to 30.
        
        Returns:
            np.ndarray: Preprocessed binary image with same shape as input
        """
        bright_pixels = channel[channel > threshold]
        if bright_pixels.size == 0:
            return np.zeros_like(channel, dtype=np.uint8)

        _, img = cv2.threshold(
            channel,
            float(np.median(bright_pixels)),
            255,
            cv2.THRESH_BINARY,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened_img = cv2.dilate(img, kernel)
        eroded_img = cv2.erode(opened_img, kernel)
        opened_img = cv2.dilate(eroded_img, kernel)
        eroded_img = cv2.erode(opened_img, kernel)
        return eroded_img

    def channel2points(self, channel):
        """
        Convert binary channel to point cloud in 2D space.
        
        Extracts coordinates of all foreground pixels (> threshold) and returns
        as a 2D point cloud for clustering. Y-coordinates are inverted to match
        standard image coordinate system.
        
        Args:
            channel (np.ndarray): Binary or grayscale image (2D array)
        
        Returns:
            pd.DataFrame: Point coordinates with columns ['x', 'y']
        """
        indices = np.argwhere(channel > self.threshold)
        y_coordinates = channel.shape[0] - indices[:, 0]
        x_coordinates = indices[:, 1]
        return pd.DataFrame({'x': x_coordinates, 'y': y_coordinates})

    def groupNuclei(self, points):
        """
        Cluster nuclei points using DBSCAN algorithm.
        
        Groups foreground pixels into distinct nuclei clusters using density-based
        spatial clustering. Noise points are ignored.
        
        Args:
            points (pd.DataFrame): 2D points with columns ['x', 'y']
        
        Returns:
            int: Count of distinct nuclei clusters detected (noise not counted)
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        if points.shape[0] == 0:
            return 0
        cluster_labels = dbscan.fit_predict(points[['x', 'y']])
        return np.max(cluster_labels) + 1

    def countNuclei(self, img_channel):
        """
        Count marked cell nuclei in an image channel.
        
        Orchestrates the complete pipeline: preprocess -> point extraction -> clustering.
        
        Args:
            img_channel (np.ndarray): Channel containing stained nuclei (2D grayscale)
        
        Returns:
            int: Total count of distinct nuclei detected
        """
        return self.groupNuclei(self.channel2points(self.preprocess(img_channel)))
    
