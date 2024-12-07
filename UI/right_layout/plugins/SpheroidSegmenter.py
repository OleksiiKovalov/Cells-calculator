import os
import traceback
from PyQt5.QtWidgets import QLabel, QPushButton, QGraphicsView, QGraphicsView, QGraphicsScene, \
    QGraphicsTextItem, QComboBox, QPushButton, QGraphicsView, QCheckBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


from UI.Slider import Slider
from UI.right_layout.plugins.BasePlagin import BasePlugin


class SpheroidSegmenter(BasePlugin):
    """
    Plugin for segmenting cellular spheroids on a single given image.
    Currently this plugin is not used in the app and is unavailable.
    Instead, Tracker plugin should be used, where a chosen image folder contains just 1 image.
    """
    def get_name(self):
        return "Spheroid Segmenter"
    def __init__(self, *arg):

        super().__init__(*arg)
        self.plugin_signal.emit("Open_lsm", True)
        self.plugin_signal.emit("Open_folder", False)
        self.plugin_signal.emit("Settings", False)
        self.plugin_signal.emit("Save_as", False)

    def init_value(self, parent, parametrs, object_size, default_object_size, models):
        self.show_boundry = 0
        self.draw_bounding = 0
        self.models = models
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

    # def update_colormap(self, colormap):

    #     self.object_size["color_map"] = colormap
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
        model = self.combo_box.currentText()

        # Check if a method and file are selected
        if model == "" or self.lsm_path is None:
            # If not, show a warning dialog and return
            self.plugin_signal.emit("show_warning", "Warning\n\nChoose model and file.")
            return 0
        # If a specific method is selected
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
        self.right_scene.clear()
        self.print_result(result)

        # Set flag to draw bounding boxes
        self.draw_bounding = 1

        # Draw bounding boxes
        self.draw_bounding_box()

    def print_result(self, result):
        model = self.combo_box.currentText()

        if model == "Baseline Segmenter":
            self.print_result_segmenter(result)

    def print_result_segmenter(self, result):
        spheroid_df = result["Cells"]
        # Вычисление средних значений и количества строк
        try:
            avg_diameter = round(spheroid_df["diameter"].mean(), 2)
            avg_area = round(spheroid_df["area"].mean(), 2)
            avg_volume = round(spheroid_df["volume"].mean(), 2)
        except:
            avg_diameter = "-"
            avg_area = "-"
            avg_volume = "-"
        num_cells = spheroid_df.shape[0]

        # Создание строк для вывода
        results = [
            f"Spheroid detected: {num_cells}",
            f"Mean D: {avg_diameter}",
            f"Mean S: {avg_area}",
            f"Mean V: {avg_volume}",
        ]

        # Настройки для шрифта и отображения
        font = QFont('Arial', 12)
        y_offset = 0
        line_height = 25  # Интервал между строками

        # Добавление строк в сцену
        for text in results:
            label = QGraphicsTextItem(text)
            label.setFont(font)
            label.setPos(0, y_offset)
            self.right_scene.addItem(label)
            y_offset += line_height  # Сдвиг вниз для следующей строки

        # Обновить вид
        self.right_view.update()
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
                self.plugin_signal.emit("add_image", self.lsm_path)
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

        # Create a checkbox for showing detected cells
        self.checkbox = QCheckBox("Show Detected Spheroids")

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

        self.right_layout.addWidget(plugin_label)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.combo_box)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.right_view)
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
