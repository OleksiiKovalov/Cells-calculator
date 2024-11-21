"""
Here we define for baseline class of all segmenting or detecting models used in the application.
We define both the structure and the main functionality utils.
"""
import os
import shutil

class BaseModel():
    def __init__(self, path_to_model: str, object_size):
        """Model constructor. Slightly differs for detectors and segmenters."""
        self.init_model(path_to_model)
        self.object_size = object_size
        self.original_image = None
    
    def init_model(self, path_to_model: str):
        raise NotImplementedError
    
    def count_cells(self, img_path):
        """
        By calling this method, the model class instance calculates cells on a given image.
        This method fully relies on the self.count() method.
        The input param is the path to RGB image of cells.
        The output param is optimized count of cells.
        """
        dst = os.path.join('.cache', 'cell_tmp_img.png')
        shutil.copy2(img_path, dst)
        detections = self.count(dst)
        if detections is None:
            return 0
        return len(detections)
    
    def count(self, input_image):
        raise NotImplementedError