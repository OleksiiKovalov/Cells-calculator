import os
import traceback
from PyQt5.QtWidgets import QCheckBox, QPushButton, QGraphicsView, QGraphicsView, QGraphicsScene,\
    QGraphicsTextItem, QComboBox, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from UI.Slider import Slider
from UI.right_layout.plugins.BasePlagin import BasePlugin
from model.Model import Model


class CellDetector(BasePlugin):
    def get_name(self):
        return "Cell Processor"
    def __init__(self, *arg):

        super().__init__(*arg)
        self.plugin_signal.emit("Open_lsm", True)
        self.plugin_signal.emit("Open_folder", False)
        self.plugin_signal.emit("Settings", False)
        self.plugin_signal.emit("Save_as", False)
        self.model = Model(path=arg[-1][self.combo_box.currentText()]['path'],
                           object_size=arg[-1][self.combo_box.currentText()]['object_size'])

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
            self.reset_sliders()
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
        try:
            self.model.cell_counter.detections = None
        except:
            pass

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
            if self.models[model]['path'] == self.model.path:
                # result = self.models[model].calculate(
                #     img_path=self.lsm_path, cell_channel=self.parametrs['Cell'],\
                #         nuclei_channel=self.parametrs['Nuclei'])
                result = self.model.calculate(
                    img_path=self.lsm_path, cell_channel=self.parametrs['Cell'],\
                        nuclei_channel=self.parametrs['Nuclei'])
            else:
                del self.model
                self.model = Model(path=self.models[model]['path'],
                                   object_size=self.models[model]['object_size'])
                result = self.model.calculate(
                    img_path=self.lsm_path, cell_channel=self.parametrs['Cell'],\
                        nuclei_channel=self.parametrs['Nuclei'])
        except:
            traceback.print_exc()
            try:
                # If an error occurs, try without channel information
                if self.models[model]['path'] == self.model.path:
                    # result = self.models[model].calculate(img_path=self.lsm_path)
                    result = self.model.calculate(img_path=self.lsm_path)
                else:
                    del self.model
                    self.model = Model(path=self.models[model]['path'],
                                       object_size=self.models[model]['object_size'])
                    result = self.model.calculate(img_path=self.lsm_path)
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
        if model == "Detector":
            self.print_result_detector(result)
        else:
            self.print_result_segmenter(result)

    def print_result_detector(self, result):
        results = []
        # Добавить количество клеток
        results.append(f'Cells: {result["Cells"]["box"].shape[0]}')
        # try:
        #     results.append(f'Cells: {result["Cells"]["box"].shape[0]}')
        # except:
        #     results.append(f'Cells: {result["Cells"]}')
        try:
            boxes = result["Cells"]["box"]

            # Извлечение длины и ширины (второй и третий элемент в массивах)
            lengths = boxes.apply(lambda x: x[2])
            widths = boxes.apply(lambda x: x[3])

            img_area = self.model.cell_counter.original_image.shape[0] * self.model.cell_counter.original_image.shape[0]

            # Вычисление диагоналей (диаметры)
            arithmetic_diameters = (lengths + widths) / 2
            geometric_diameters = (lengths * widths)**(1/2)

            # Вычисление площадей
            areas = lengths * widths

            average_arithmetic_diameter = arithmetic_diameters.mean() / img_area * 10000
            average_geometric_diameters = geometric_diameters.mean() / img_area * 10000
            average_area = areas.mean() / img_area * 10000
        except:
            average_arithmetic_diameter = "-"
            average_geometric_diameters = "-"
            average_area = "-"

        results.append(f"Mean S: {average_area}‱")
        results.append(f"Mean Arithmetic D: {round(average_arithmetic_diameter, 2)}‱")
        results.append(f"Mean Geometric D: {round(average_geometric_diameters, 2)}‱")
        results.append("")

        # Добавить количество ядер
        if result["Nuclei"] == -100:
            results.append('Nuclei: -')
        else:
            results.append(f'Nuclei: {result["Nuclei"]}')

        # Добавить процент живых
        if result["%"] == -100:
            results.append('Alive: -')
        else:
            results.append(f'Alive: {result["%"]}%')

        # Шрифт для всех элементов
        font = QFont('Arial', 12)

        # Текущая вертикальная позиция
        y_offset = 0
        line_height = 25  # Высота строки (можно корректировать для нужного интервала)

        # Добавляем текстовые элементы в сцену
        for text in results:
            label = QGraphicsTextItem(text)
            label.setFont(font)
            label.setPos(0, y_offset)
            self.right_scene.addItem(label)
            y_offset += line_height  # Увеличиваем позицию для следующего текста

        # Обновить вид
        self.right_view.update()

    def print_result_segmenter(self, result):
        spheroid_df = result["Cells"]

        # Вычисление средних значений и количества строк
        try:
            avg_diameter = spheroid_df["diameter"].mean()
            avg_area = spheroid_df["area"].mean()
            avg_volume = spheroid_df["volume"].mean()
        except:
            avg_diameter = "-"
            avg_area = "-"
            avg_volume = "-"
        try:
            num_cells = spheroid_df.shape[0]
            # Создание строк для вывода
            results = [
                f"Objects detected: {num_cells}",
                f"Mean D: {round(avg_diameter*10000, 3)}‱",
                f"Mean S: {round(avg_area*10000, 3)}‱",
                f"Mean V: {round(avg_volume*10000, 3)}‱",
            ]
        except AttributeError:
            num_cells = spheroid_df
            # Создание строк для вывода
            results = [
                f"Cells detected: {num_cells}"
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

    def on_state_changed_scale(self, state):
        if state:
            self.object_size["scale"] = 10
        else:
            self.object_size["scale"] = 20
        self.reset_detection()

    def reset_sliders(self):
        self.default_object_size['min_size'] = 100
        self.default_object_size['max_size'] = 0

        self.reset_detection()

    def init_rightLayout(self):
        plugin_label = QLabel(self.get_name())
        plugin_label.setFont(QFont("Arial", 32))
        # Create a combo box to choose models
        self.combo_box = QComboBox()
        self.combo_box.currentIndexChanged.connect(self.reset_sliders)

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

        self.checkbox_scale = QCheckBox("Process at x10 scale")

        # Connect checkbox state change to handler function
        self.checkbox_scale.stateChanged.connect(self.on_state_changed_scale)

        # Set font for the checkbox
        self.checkbox_scale.setFont(QFont("Arial", 24))

        # Set style sheet for the checkbox to customize its indicator size
        self.checkbox_scale.setStyleSheet('''
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

        self.right_layout.addWidget(self.checkbox_scale)
        self.right_layout.addSpacing(20)

        self.right_layout.addWidget(self.checkbox)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.button)
        self.right_layout.addSpacing(20)
