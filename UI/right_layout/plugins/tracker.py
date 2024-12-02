import sys
from PyQt5.QtWidgets import QAbstractItemView, QCheckBox,QGraphicsPixmapItem,\
    QSizePolicy, QGraphicsProxyWidget, QGraphicsRectItem, QHeaderView, \
        QMessageBox, QTableWidget, QTableWidgetItem, QPushButton, QGraphicsView,\
            QApplication, QMainWindow, QAction, QGraphicsView, QGraphicsScene, \
                QVBoxLayout, QWidget, QFileDialog, QGraphicsTextItem, QComboBox, \
                    QLabel, QHBoxLayout, QSlider, QLineEdit
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPen
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot
import numpy as np
import tifffile

from model.tracker import Tracker as T

import os



from UI.right_layout.plugins.BasePlagin import BasePlugin
from UI.Slider import Slider
import traceback


class Tracker(BasePlugin):
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
        elif action_name == "set_size":
            self.set_size(value)

        elif action_name == "open_folder":
            self.reset_detection()
            self.right_scene.clear()
            if value:
               
                self.lsm_filesList = [os.path.join(value, file) \
        for file in os.listdir(value)\
            if file.lower().endswith(('.png', '.jpg', '.bmp', '.lsm', '.tif'))]
                
                
                self.folder_path = value
                self.max_range_slider.set_default()
                self.min_range_slider.set_default()
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

        range_lable = QLabel("Object Size:")
        font = QFont()
        font.setPointSize(16) 
        
        range_lable.setFont(font)
       
        #self.right_layout.addSpacing(1)
        self.min_range_slider = Slider(self.object_size, self.default_object_size, 'min_size')
        self.max_range_slider = Slider(self.object_size, self.default_object_size, 'max_size')

        # LineWidth_label = QLabel("Line Width:")
        # LineWidth_label.setFont(QFont("Arial", 16))

        # self.LineWidth_edit = QLineEdit()
        # size = self.object_size['line_width']
        # self.LineWidth_edit.setText(f"{size:.2f}")
        # self.LineWidth_edit.setFont(QFont("Arial", 12))
        # self.LineWidth_edit.returnPressed.connect(self.update_lineWidth)

        # LineWidth_layout = QHBoxLayout()
        # LineWidth_layout.addWidget(LineWidth_label)
        # LineWidth_layout.addWidget(self.LineWidth_edit)
       
        colormap_label = QLabel("Colormap:")
        colormap_label.setFont(QFont("Arial", 24))

        self.colormap_combo = QComboBox()
        self.colormap_combo.setFont(QFont("Arial", 16))

        self.colormaps =  self.object_size["color_map_list"]
        self.colormap_combo.addItems(self.colormaps)
        self.colormap_combo.setCurrentText(self.object_size['color_map'])  # Установить "Viridis" по умолчанию
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        
        # Add widgets to the right layout with spacing
        ###self.right_layout.addWidget(label)

        self.right_layout.addWidget(plugin_label)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.combo_box)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.right_view)
        self.right_layout.addSpacing(20)

        self.right_layout.addWidget(colormap_label)
        self.right_layout.addWidget(self.colormap_combo)
        self.right_layout.addSpacing(20)
        # self.right_layout.addLayout(LineWidth_layout)
        self.right_layout.addSpacing(20)

        self.right_layout.addWidget(range_lable)
        self.right_layout.addSpacing(20)

        self.right_layout.addWidget(self.min_range_slider)
        self.right_layout.addWidget(self.max_range_slider)

        self.right_layout.addSpacing(20)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.button)
        self.right_layout.addSpacing(20)

    def update_colormap(self, colormap):
        
        self.object_size["color_map"] = colormap
    
    # def update_lineWidth(self):
    #     # Получаем значение из QLineEdit
    #     input_text = self.LineWidth_edit.text()

    #     # Проверяем, является ли введённое значение числом
    #     try:
    #         # Преобразуем в число с плавающей точкой
    #         line_width = float(input_text)

    #         self.object_size["line_width"] = round(line_width, 2)
    #         self.LineWidth_edit.setText(f"{float(input_text):.2f}")

    #     except ValueError:
    #         # Если введено некорректное значение, устанавливаем стандартное значение
    #         size = self.object_size["line_width"]
    #         self.LineWidth_edit.setText(f"{size:.2f}") 

        
    def reset_detection(self):
        print("reset_detection")
        return
        for key, model in self.models.items():
            model.cell_counter.detections = None
    def set_size(self, detection, img_size : tuple = (512,512)):
        min_size, max_size = self.default_object_size["min_size"], self.default_object_size["max_size"]
        model = self.combo_box.currentText()
        if all(len(cell) >= 4 for cell in detection):
            img_sq = img_size[0] * img_size[1]
            # Вычисляем произведения для каждого 
            values = [cell[2] * cell[3] for cell in detection]

            # Находим максимальное и минимальное произведение
            min_size_from_detection = min(values) / img_sq
            max_size_from_detection = max(values) / img_sq
            if self.lsm_filesList or model != "All_models":
                if min_size_from_detection >= min_size:
                    min_size = None
                else:
                    min_size = min_size_from_detection
                if max_size_from_detection <= max_size:
                    max_size = None
                else:
                    max_size = max_size_from_detection
            else:
                min_size = min_size_from_detection
                max_size = max_size_from_detection
                 
            
        
        if min_size is not None:
            self.min_range_slider.change_default(min_size = min_size, max_size = max_size)
        if max_size is not None:
            self.max_range_slider.change_default(min_size = min_size, max_size = max_size)
        
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
