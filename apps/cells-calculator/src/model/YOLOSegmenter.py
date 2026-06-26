# Standard library imports

# Third-party imports
from ultralytics import YOLO

# Local application imports
from model.BaseSegmenter import BaseSegmenter
from model.utils import results_to_pandas

class YoloSegmenter(BaseSegmenter):
    """
    Instance segmentation using YOLO11 model.

    Provides interface to YOLOv8/YOLO11 for single-image instance segmentation.

    Attributes:
        model (YOLO): YOLOv8/YOLO11 segmentation model.
    """
    DETECTION_COLUMNS = [
        "id_label",
        "box",
        "mask",
        "confidence",
        "diameter",
        "area",
        "volume",
    ]

    def init_model(self, path_to_model: str):
        """
        Load the YOLO segmentation model for full-image inference.

        Args:
            path_to_model (str): Path to YOLO model weights file.
        """
        self.model = YOLO(path_to_model, task="segment")

    def call_inference(
        self,
        input_image,
        **kwargs
    ):
        """
        Perform inference on a given image using the pre-trained YOLO model.

        Runs forward propagation to obtain bboxes, masks and confidences, then
        structures the output as a pandas DataFrame.

        Args:
            input_image: Input image (path or array) to segment.
            **kwargs: Additional configuration for model inference (conf, iou, etc.).

        Returns:
            pd.DataFrame: Detections with the standard detection columns.
        """
        outputs = self.model(
            input_image,
            device=self.device,
            conf=0.3,
            iou=0.6,
            max_det=2000,
            retina_masks=True,
            **kwargs
        )[0]
        detections = results_to_pandas(outputs, True)
        #self.detections['box'] = self.detections['box'].apply(lambda b: b * np.array([self.w, self.h, self.w, self.h])       )
        return detections

