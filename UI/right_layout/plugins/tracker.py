import os
import traceback
from PyQt5.QtWidgets import QPushButton, QGraphicsView, QMessageBox, QGraphicsView, QGraphicsScene,\
     QComboBox, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from model.tracker import Tracker as T

from UI.right_layout.plugins.BasePlagin import BasePlugin
from UI.Slider import Slider

class Tracker(BasePlugin):
    """
    Class for Tracker plugin of the application.
    This plugin is used for tracking cellular spheroids on given sequential image frames.
    """
    def get_name(self):
        return "Tracker"

    def __init__(self, *arg):
        super().__init__(*arg)
        self.plugin_signal.emit("Open_lsm", False)
        self.plugin_signal.emit("Open_folder", True)
        self.plugin_signal.emit("Settings", False)
        self.plugin_signal.emit("Save_as", False)

    def init_value(self, parent, parametrs, object_size, default_object_size, models):
        self.models = models
        default_model = models[list(models.keys())[0]]
        self.model = T(default_model['path'], default_model['size'])
        self.parametrs = parametrs
        self.object_size = object_size
        self.default_object_size = default_object_size
        self.right_layout = parent
        self.lsm_filesList = None

    def handle_action(self, action_name, value):
        if action_name == "reset_detection":
            self.reset_detection()
        # elif action_name == "set_size":
        #     self.set_size(value)

        elif action_name == "open_folder":
            self.reset_detection()
            self.right_scene.clear()
            if value:
                self.lsm_filesList = [os.path.join(value, file) \
        for file in os.listdir(value)\
            if file.lower().endswith(('.png', '.jpg', '.bmp', '.lsm', '.tif'))]
                self.folder_path = value
                # self.max_range_slider.set_default()
                # self.min_range_slider.set_default()
                self.button.setEnabled(True)
            else:
                self.lsm_filesList = None
                self.button.setEnabled(False)

    def init_rightLayout(self):
        plugin_label = QLabel(self.get_name())
        plugin_label.setFont(QFont("Arial", 32))
        # Create a combo box to choose models
        self.combo_box = QComboBox()

        # Create a label to prompt user to choose a model
        label = QLabel("Choose model:")
        label.setFont(QFont("Arial", 24))

        # Create a graphics scene for the right view
        self.right_scene = QGraphicsScene()
        # Create a graphics view for the right scene
        self.right_view = QGraphicsView(self.right_scene)

        # Align the graphics view to the top-left corner
        self.right_view.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Create a button for calculating
        self.button = QPushButton("Calculate")
        self.button.setEnabled(False)

        # Set font for the combo box
        self.combo_box.setFont(QFont("Arial", 24))

        # Add 'All_method' option and method names to the combo box
        #self.combo_box.addItems(['All_models'])
        self.combo_box.addItems([key for key in self.models])

        # Set current index to 1
        self.combo_box.setCurrentIndex(0)

        # Set font for the calculate button
        self.button.setFont(QFont("Arial", 32))

        # Connect button click event to calculate_button function
        self.button.clicked.connect(self.calculate_button)
        ###self.right_layout.addWidget(label)
        self.right_layout.addWidget(plugin_label)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.combo_box)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.right_view)
        self.right_layout.addSpacing(20)

        self.right_layout.addSpacing(20)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.button)
        self.right_layout.addSpacing(20)

    def update_colormap(self, colormap):
        self.object_size["color_map"] = colormap

    def reset_detection(self):
        print("reset_detection")
        return
        for key, model in self.models.items():
            model.cell_counter.detections = None

    def calculate_button(self):
        model = self.combo_box.currentText()
        # Check if a method and file are selected
        if model == "" or self.folder_path is None:
            # If not, show a warning dialog and return
            self.plugin_signal.emit("show_warning", "Warning\n\nChoose model and folder.")
            return 0
        try:
            if self.model.path == self.models[model]['path']:
                self.model.track(img_seq_folder=self.folder_path, time_period = 15)
                text_to_show ="""Success!"""
                self.show_result(text_to_show)
            else:
                self.model = T(path=self.models[model]['path'],
                                     size=self.models[model]['object_size'])
                self.model.track(img_seq_folder=self.folder_path, time_period = 15)
                text_to_show ="""Success!"""
                self.show_result(text_to_show)
        except Exception as e:
            traceback.print_exc()
            # self.plugin_signal.emit("show_warning", traceback.format_exc())
            self.plugin_signal.emit("show_warning", str(e))

    def show_result(self, text):
        msgBox = QMessageBox()

        # Set the icon of the message box to a warning icon
        msgBox.setIcon(QMessageBox.Information)

        # Set the text of the message box
        msgBox.setText(text)

        # Set the title of the message box
        msgBox.setWindowTitle("Success!")

        # Adjust the size of the message box
        msgBox.adjustSize()

        # Execute the message box (display it)
        msgBox.exec_()
