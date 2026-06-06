"""
Image normalization dialog for previewing and adjusting image preprocessing.

Provides a PyQt5 dialog for loading an image and interactively
adjusting normalization parameters (percentile bounds, scaling).
Displays real-time preview of normalized image output.

Key components:
- ImageNormalizeDialog: Main dialog for image normalization adjustment
"""

# Third-party imports
import numpy as np
from csbdeep.utils import normalize
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QSlider,
    QHBoxLayout, QSizePolicy
)

class ImageNormalizeDialog(QDialog):
    """
    Dialog for viewing and adjusting image normalization parameters.
    """

    def __init__(self, image, parent=None):
        """
        Initialize the dialog.

        Args:
            image: The image to normalize.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Normalized Image Viewer")
        self.image_original = self._validate_image(image).astype(np.float32)
        self._init_ui()
        self.resize(800, 600)

    def _validate_image(self, image):
        if image is None:
            raise ValueError("Image must not be None")
        if image.ndim != 2:
            raise ValueError("Only 2D grayscale images are supported")
        return image

    def _init_ui(self):
        """
        Initialize the user interface.
        """
        layout = QVBoxLayout(self)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        self.label_start = QLabel()
        self.slider_start = self._create_slider(10)
        self.label_stop = QLabel()
        self.slider_stop = self._create_slider(998)

        layout.addLayout(self._create_slider_row(self.label_start, self.slider_start))
        layout.addLayout(self._create_slider_row(self.label_stop, self.slider_stop))

        self.update_image()

    def _create_slider(self, value):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(value)
        slider.valueChanged.connect(self.update_image)
        return slider

    def _create_slider_row(self, label, slider):
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(slider)
        return row

    def update_image(self):
        pmin, pmax = self._get_normalization_bounds()
        self.label_start.setText(f"Start: {pmin:.1f}")
        self.label_stop.setText(f"Stop: {pmax:.1f}")

        norm_img = self._normalized_image(pmin, pmax)
        qimg = self._create_qimage(norm_img)

        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def _get_normalization_bounds(self):
        pmin, pmax = sorted((self.slider_start.value(), self.slider_stop.value()))
        if pmin == pmax:
            pmax = pmin + 1
        return pmin / 10.0, pmax / 10.0

    def _normalized_image(self, pmin, pmax):
        norm_img = normalize(self.image_original, pmin=pmin, pmax=pmax)
        return np.clip(norm_img * 255, 0, 255).astype(np.uint8)

    def _create_qimage(self, gray_image):
        height, width = gray_image.shape
        return QImage(
            gray_image.data,
            width,
            height,
            gray_image.strides[0],
            QImage.Format_Grayscale8,
        )

    def resizeEvent(self, event):
        """
        Handle resize event to update image scaling.
        """
        super().resizeEvent(event)
        self.update_image()