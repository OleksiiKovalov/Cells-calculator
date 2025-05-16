from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QSlider,
    QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from csbdeep.utils import normalize
import numpy as np

class ImageNormalizeDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Normalized Image Viewer")
        self.image_original = image.astype(np.float32)
        self.init_ui()
        self.resize(800, 600)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        # Slider + label for Start
        self.label_start = QLabel("Start: 0.0")
        self.slider_start = QSlider(Qt.Horizontal)
        self.slider_start.setRange(0, 1000)
        self.slider_start.setValue(10)
        self.slider_start.valueChanged.connect(self.update_image)

        slider_layout_start = QHBoxLayout()
        slider_layout_start.addWidget(self.label_start)
        slider_layout_start.addWidget(self.slider_start)

        # Slider + label for Stop
        self.label_stop = QLabel("Stop: 100.0")
        self.slider_stop = QSlider(Qt.Horizontal)
        self.slider_stop.setRange(0, 1000)
        self.slider_stop.setValue(998)
        self.slider_stop.valueChanged.connect(self.update_image)

        slider_layout_stop = QHBoxLayout()
        slider_layout_stop.addWidget(self.label_stop)
        slider_layout_stop.addWidget(self.slider_stop)

        layout.addLayout(slider_layout_start)
        layout.addLayout(slider_layout_stop)

        self.update_image()

    def update_image(self):
        pmin = min(self.slider_start.value(), self.slider_stop.value())
        pmax = max(self.slider_start.value(), self.slider_stop.value())

        if pmax == pmin:
            pmax = pmin + 1

        # Update labels
        self.label_start.setText(f"Start: {pmin / 10:.1f}")
        self.label_stop.setText(f"Stop: {pmax / 10:.1f}")

        # Normalize image
        norm_img = normalize(self.image_original, pmin=pmin / 10.0, pmax=pmax / 10.0)
        norm_img = np.clip(norm_img * 255, 0, 255).astype(np.uint8)

        if norm_img.ndim == 2:
            qimg = QImage(norm_img.data, norm_img.shape[1], norm_img.shape[0],
                          norm_img.strides[0], QImage.Format_Grayscale8)
        else:
            raise ValueError("Only 2D grayscale images are supported")

        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()