# Standard library imports
import sys

# Third-party imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QSlider, QLineEdit
)


class CustomSlider(QSlider):
    """
    Class for filtering sliders in the right layout menu.
    """

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)

    def mousePressEvent(self, event):
        """
        Handle mouse press event to set slider value based on click position.
        """
        if event.button() == Qt.LeftButton:
            # Determine new value based on click position
            new_value = (self.minimum() +
                         (self.maximum() - self.minimum()) *
                         event.pos().x() / self.width())
            self.setValue(int(new_value))  # Set the new value
            event.accept()  # Mark the event as handled
        super().mousePressEvent(event)

class Slider(QWidget):
    """
    Class for filtering sliders in the right layout menu.
    """

    def __init__(self, object_size: dict, default_object_size: dict, key: str):
        """
        Initialize the Slider widget.

        Args:
            object_size (dict): Dictionary containing object size parameters.
            default_object_size (dict): Dictionary containing default object size parameters.
            key (str): Key for the size parameter (e.g., 'min_size', 'max_size').
        """
        self.object_size = object_size
        self.key = key
        self.round_parametr_slider = object_size['round_parametr_slider']
        self.round_parametr_value_input = object_size['round_parametr_value_input']
        self.default_object_size = default_object_size
        super().__init__()
        self.initUI()

    def change_default(self, min_size, max_size):
        """
        Change the default min and max sizes.

        Args:
            min_size: New minimum size value.
            max_size: New maximum size value.
        """
        # TODO: add some validation for min and max
        if min_size is None:
            min_size = 100
        if max_size is None:
            max_size = 0
        self.default_object_size['min_size'] = min_size - min_size / 100
        self.default_object_size['max_size'] = max_size + max_size / 100
        self.value_slider.setMinimum(
            int(self.default_object_size['min_size'] * self.round_parametr_slider))
        self.value_slider.setMaximum(
            int(self.default_object_size['max_size'] * self.round_parametr_slider))
        self.set_default()

    def set_default(self):
        """
        Set the slider and input to the default value for the current key.
        """
        value = self.default_object_size[self.key]
        self.object_size[self.key] = value
        self._update_input_display(value)
        self.value_slider.setValue(int(value * self.round_parametr_slider))

    def initUI(self):
        """
        Initialize the user interface components.
        """
        # Variable to store the value
        # Main vertical layout
        main_layout = QVBoxLayout()

        # First row with 'Value' label
        label_layout = QHBoxLayout()

        value_label = QLabel(self.key.capitalize().replace('_size', ''))
        font = QFont()
        font.setPointSize(16)  # Set font size to 16 points
        value_label.setFont(font)
        label_layout.addWidget(value_label)
        main_layout.addLayout(label_layout)

        # Second row: QLineEdit and QSlider
        slider_layout = QHBoxLayout()

        # Create QLineEdit for displaying the value
        self.value_input = QLineEdit("")
        self.value_input.setFixedWidth(50)
        self.value_input.setAlignment(Qt.AlignCenter)

        # Set validator to allow only digits from 0 to 10
        # self.value_input.setValidator(QDoubleValidator(
        #     self.object_size['min_size'], self.object_size['max_size'], 3))

        slider_layout.addWidget(self.value_input)

        # Create QSlider
        self.value_slider = CustomSlider(Qt.Horizontal)
        self.value_slider.setMinimum(
            int(self.object_size['min_size'] * self.round_parametr_slider))
        self.value_slider.setMaximum(
            int(self.object_size['max_size'] * self.round_parametr_slider))
        self.value_slider.setValue(
            int(self.object_size[self.key] * self.round_parametr_slider))
        slider_layout.addWidget(self.value_slider)

        # Add slider and input field to the main layout
        main_layout.addLayout(slider_layout)

        # Connect signals and slots
        self.value_slider.valueChanged.connect(self.update_value_from_slider)
        self.value_input.returnPressed.connect(self.update_value_from_input)

        # Window settings
        self.setLayout(main_layout)
        self.setWindowTitle('Slider and Input Example')
        self.show()

    def update_value_from_slider(self):
        """
        Update the value from slider changes.
        """
        # Update variable and QLineEdit when slider value changes
        min_size = self.default_object_size['min_size']
        max_size = self.default_object_size['max_size']

        value = self.value_slider.value() / self.round_parametr_slider
        # Check value for correctness, if out of range - set boundaries
        value = self._clamp_value(value, min_size, max_size)
        self.value_slider.setValue(int(value * self.round_parametr_slider))
        self.object_size[self.key] = value
        self._update_input_display(value)

    def update_value_from_input(self):
        """
        Update the value from input field changes.
        """
        # Update variable and slider when QLineEdit value changes
        min_size = self.default_object_size['min_size']
        max_size = self.default_object_size['max_size']
        try:
            value = (float(self.value_input.text()) /
                     self.round_parametr_value_input)
            # Check value for correctness, if out of range - set boundaries
            value = self._clamp_value(value, min_size, max_size)

            self.object_size[self.key] = value
            self.value_slider.setValue(int(value * self.round_parametr_slider))
            self._update_input_display(value)
        except ValueError:
            self._update_input_display(self.object_size[self.key])

    def _clamp_value(self, value, min_size, max_size):
        """
        Clamp value to valid range based on key type and object sizes.

        Args:
            value: The value to clamp.
            min_size: Minimum allowed value.
            max_size: Maximum allowed value.

        Returns:
            The clamped value.
        """
        if self.key == 'min_size' and value > self.object_size['max_size']:
            value = self.object_size['max_size']
        elif self.key == 'max_size' and value < self.object_size['min_size']:
            value = self.object_size['min_size']
        return max(min_size, min(max_size, value))

    def _update_input_display(self, value):
        """
        Update the input field display with the formatted value.

        Args:
            value: The value to display.
        """
        self.value_input.setText(f"{value * self.round_parametr_value_input:.2f}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Slider()
    sys.exit(app.exec_())
