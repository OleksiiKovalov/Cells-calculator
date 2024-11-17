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

import os
import shutil

import pyqtgraph as pg
from UI.Slider import Slider

from model.Model import Model as model1

import traceback


class CellDetector(QObject):
    def get_name(self):
        return "CellDetector"
    plugin_signal = pyqtSignal(str, object)
    def __init__(self, parent, parametrs, object_size, default_object_size, models):
        super().__init__()
        self.show_boundry = 0
        self.draw_bounding = 0
        self.models = models
        self.parametrs = parametrs
        self.object_size = object_size
        self.default_object_size = default_object_size
        self.right_layout = parent
        self.lsm_filesList = None

        try:
            self.init_rightLayout()
        except:
            self.plugin_signal.emit("error",None)
    def handle_action(self, action_name, value):
        if action_name == "reset_detection":
            self.reset_detection()
        elif action_name == "set_size":
            self.set_size(value)
        elif action_name == "open_lsm":
            self.right_scene.clear()
            self.reset_detection()
            self.max_range_slider.set_default()
            self.min_range_slider.set_default()
            self.lsm_filesList = None
            self.folder_path = None
            if value:
                self.lsm_path = value
                self.button.setEnabled(True)
                
            else:
                self.lsm_path = None
                self.button.setEnabled(False)
                

        elif action_name == "open_image":
            self.reset_detection()
            self.right_scene.clear()
            self.max_range_slider.set_default()
            self.min_range_slider.set_default()
            self.lsm_path = value
            self.lsm_filesList = None
            self.folder_path = None
            self.button.setEnabled(True)
         

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
                self.lsm_path = None
                self.draw_bounding = 0
                self.button.setEnabled(True)
            else:
                self.folder_path = None
                self.lsm_filesList = None
                self.button.setEnabled(False)
            

    def init_rightLayout(self):
        """
        Initialize the layout for the right-side panel.

        Notes:
        - Create a combo box to choose models.
        - Create a label to prompt the user to choose a model.
        - Create a graphics scene for the right view.
        - Create a graphics view for the right scene.
        - Align the graphics view to the top-left corner.
        - Create a button for calculating.
        - Set font for the save button and combo box.
        - Add 'All_method' option and method names to the combo box.
        - Set current index to 1.
        - Set font for the label.
        - Create a checkbox for showing detected cells.
        - Connect checkbox state change to the handler function.
        - Set font for the checkbox.
        - Set style sheet for the checkbox to customize its indicator size.
        - Set font for the calculate button.
        - Connect button click event to calculate_button function.
        - Add widgets to the right layout with spacing.
        """
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
        self.combo_box.addItems(['All_models'])
        self.combo_box.addItems([key for key in self.models])
        
        # Set current index to 1
        self.combo_box.setCurrentIndex(1)
        
      
        
        
        # Create a checkbox for showing detected cells
        self.checkbox = QCheckBox("Show Detected Cells")
        
        # Connect checkbox state change to handler function
        self.checkbox.stateChanged.connect(self.on_state_changed)
        
        # Set font for the checkbox
        self.checkbox.setFont(QFont("Arial", 24))
        
        # Set style sheet for the checkbox to customize its indicator size
        self.checkbox.setStyleSheet('''
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
            }
        ''')
        
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

        LineWidth_label = QLabel("Line Width:")
        LineWidth_label.setFont(QFont("Arial", 16))

        self.LineWidth_edit = QLineEdit()
        size = self.object_size['line_width']
        self.LineWidth_edit.setText(f"{size:.2f}")
        self.LineWidth_edit.setFont(QFont("Arial", 12))
        self.LineWidth_edit.returnPressed.connect(self.update_lineWidth)

        LineWidth_layout = QHBoxLayout()
        LineWidth_layout.addWidget(LineWidth_label)
        LineWidth_layout.addWidget(self.LineWidth_edit)
       
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
        self.right_layout.addLayout(LineWidth_layout)
        self.right_layout.addSpacing(20)

        self.right_layout.addWidget(range_lable)
        self.right_layout.addSpacing(20)

        self.right_layout.addWidget(self.min_range_slider)
        self.right_layout.addWidget(self.max_range_slider)

        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.checkbox)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.button)
        self.right_layout.addSpacing(20)


        


       

        # Устанавливаем layout
    def update_colormap(self, colormap):
        
        self.object_size["color_map"] = colormap
    
    def update_lineWidth(self):
        # Получаем значение из QLineEdit
        input_text = self.LineWidth_edit.text()

        # Проверяем, является ли введённое значение числом
        try:
            # Преобразуем в число с плавающей точкой
            line_width = float(input_text)

            self.object_size["line_width"] = round(line_width, 2)
            self.LineWidth_edit.setText(f"{float(input_text):.2f}")

        except ValueError:
            # Если введено некорректное значение, устанавливаем стандартное значение
            size = self.object_size["line_width"]
            self.LineWidth_edit.setText(f"{size:.2f}") 

        
    def reset_detection(self):
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
        """
        Calculate the cells using the selected method and display the results.

        Notes:
        - Get the selected method from the combo box.
        - Check if a method and file are selected.
        - If a specific method is selected:
            - Attempt to calculate the result using the selected method.
            - If an error occurs, try without channel information.
            - If still not successful, show an error dialog.
            - Create QGraphicsTextItems to display the results.
            - Clear the right scene.
            - Add the results to the right scene.
            - Set flag to draw bounding boxes.
            - Draw bounding boxes.
        - If "All_models" is selected:
            - Create a table widget.
            - Configure table properties.
            - Iterate over methods.
            - Attempt to calculate the result using the method.
            - Populate the table with calculated results.
            - Set minimum size and resize rows/columns to fit content.
            - Clear the right scene.
            - Add the table to the right scene.
            - Set flag to draw bounding boxes.
            - Draw bounding boxes.
        """
        # Get the selected method from the combo box

        if self.lsm_filesList:
            self.plugin_signal.emit("create_table", None)
            return
        model = self.combo_box.currentText()
        
        # Check if a method and file are selected
        if model == "" or self.lsm_path is None:
            # If not, show a warning dialog and return
            self.plugin_signal.emit("show_warning", "Warning\n\nChoose model and file.")
            return 0

        # If a specific method is selected
        if model != "All_models":
            try:
                # Attempt to calculate the result using the selected method
                result = self.models[model].calculate(
                    img_path=self.lsm_path, cell_channel=self.parametrs['Cell'],\
                        nuclei_channel=self.parametrs['Nuclei'])
            except:
                traceback.print_exc()
                try:
                    # If an error occurs, try without channel information
                    result = self.models[model].calculate(img_path=self.lsm_path)
                except:
                    traceback.print_exc()
                    # If still not successful, show an error dialog
                    self.plugin_signal.emit("show_warning", "Error during calculation \n\nChoose another model or change channels settings")
                    result = None
                    self.draw_bounding = 0
            
            # If no result, return
            if not result:
                return 0
            
            # Create QGraphicsTextItems to display the results
            label_cells = QGraphicsTextItem(f'Cells: {result["Cells"]}')
            label_cells.setFont(QFont('Arial', 24))
            label_cells.setPos(0, 0)
            
            if result["Nuclei"] == -100:
                label_nuclei = QGraphicsTextItem(f'Nuclei: -')
            else:
                label_nuclei = QGraphicsTextItem(f'Nuclei: {result["Nuclei"]}')
            label_nuclei.setFont(QFont('Arial', 24))
            label_nuclei.setPos(0, 100)
            
            if result["%"] == -100:
                label_alive = QGraphicsTextItem(f'Alive: -')
            else:
                label_alive = QGraphicsTextItem(f'Alive: {result["%"]}%')
            label_alive.setFont(QFont('Arial', 24))
            label_alive.setPos(0, 200)

            # Clear the right scene
            self.right_scene.clear()
            
            # Add the results to the right scene
            self.right_scene.addItem(label_cells)
            self.right_scene.addItem(label_nuclei)
            self.right_scene.addItem(label_alive)
            self.right_view.update()
            
            # Set flag to draw bounding boxes
            self.draw_bounding = 1
            
            # Draw bounding boxes
            self.draw_bounding_box()
        
        else:
            # If "All_models" is selected
            
            # Create a table widget
            table = QTableWidget()
            view_width = self.right_view.viewport().width()
            view_height = self.right_view.viewport().height()
            
            # Configure table properties
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            columns = ['Cells', 'Nuclei', 'Alive']
            table.setRowCount(len(self.models))
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(['Model'] + columns)
            
            row = 0
            # Iterate over methods
            for model_name, model in self.models.items():
                colum_number = 0
                table.setItem(row, colum_number, QTableWidgetItem(model_name))
                colum_number += 1
                
                # If method is "All_models", continue to the next iteration
                if model_name == "All_models":
                    continue
                
                try:
                    # Attempt to calculate the result using the method
                    result = model.calculate(
                        img_path=self.lsm_path, cell_channel=self.parametrs['Cell'],\
                            nuclei_channel=self.parametrs['Nuclei'])
                except:
                    result = None
                    
                if result:
                    row_toAdd = "-"
                    for colum in columns:
                        if colum == "Alive":
                            if result["%"] == -100:
                                row_toAdd = "-"
                            else:
                                row_toAdd = f"{result['%']}%"
                        else:
                            if result[colum] == -100:
                                row_toAdd = "-"
                            else:
                                row_toAdd = f"{result[colum]}"
                        table.setItem(row, colum_number, QTableWidgetItem(row_toAdd))
                        colum_number += 1
                else:
                    row_toAdd = "-"
                    
                row += 1
            
            # Set minimum size and resize rows/columns to fit content
            table.setMinimumSize(view_width, view_height)
            table.resizeRowsToContents()
            table.resizeColumnsToContents()
            
            # Clear the right scene
            self.right_scene.clear()
            
            # Add the table to the right scene
            self.right_scene.addWidget(table)
            self.right_view.update()
            # Set flag to draw bounding boxes
            self.draw_bounding = 1
            
            # Draw bounding boxes
            self.draw_bounding_box()

    def draw_bounding_box(self):
        """
        Draw bounding boxes on the main scene if the checkbox is checked.

        Notes:
        - Check if the draw bounding flag is set.
        - If not set, return without performing any action.
        - Clear the main scene.
        - Check if the show boundary flag is set.
        - If set, add an image with bounding box detections to the scene.
        - If not set, add the original image to the scene.
        - If an error occurs, print the traceback and show a warning dialog.
        """
        # Check if the draw bounding flag is set to 0
        if self.draw_bounding == 0:
            # If not set, return without performing any action
            return
        
        # Clear the main scene
        
        try:
            # Check if the show boundary flag is set
            if self.show_boundry:
                # If set, add an image with bounding box detections to the scene
                self.plugin_signal.emit("add_image", ".cache\cell_tmp_img_with_detections.png" )
            else:
                # If not set, add the original image to the scene
                self.plugin_signal.emit("add_image", ".cache\cell_tmp_img.png")
        except Exception as e:
            # If an error occurs, print the traceback, show a warning dialog
          
            traceback.print_exc()
            self.plugin_signal.emit("show_warning", "Error during opening image.")
    def on_state_changed(self, state):
        """
        Handle the state change of the checkbox for showing bounding boxes.
        
        Args:
        state (int): The new state of the checkbox.
    
        """
        # Update the show_boundry flag with the new state of the checkbox
        self.show_boundry = state
        
        # Redraw bounding boxes
        self.draw_bounding_box()