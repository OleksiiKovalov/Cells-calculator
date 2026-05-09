"""
Cell Detector plugin module for segmentation and detection UI.

This module implements the CellDetectorPlugin and related UI helpers used for
selecting models, configuring object filters, and running detection on images.
"""

# Standard library imports
import os
import math
import time
import traceback

# Third-party imports
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox, QPushButton, QTextEdit,
    QComboBox, QLabel, QRadioButton, QButtonGroup, 
    QDoubleSpinBox, QHBoxLayout, QWidget, QVBoxLayout, QFileDialog, QApplication
)

# Local application imports
from UI.WaitWindow import run_with_wait_window
from UI.app_globals import get_global, set_global
from UI.errorhandling import app_logger
from UI.ModelsCheckList import ModelsCheckListDialog
from UI.rangeslider import RangeSlider
from UI.right_layout.plugins.BasePlugin import BasePlugin
from model.Model import Model
from model.utils import create_image_grid, draw_bounding_box, filter_segmentation_detections, plot_predictions, safe_image_write
from UI.app_globals import IMAGE_FILE_NAME_DETECTION, IMAGE_FILE_NAME_GRID, IMAGE_FILE_NAME_INGFERENCE


class RangeSliderWrapper(QWidget):
    """
    Wrapper class that adapts RangeSlider to work with the existing object_size interface.
    """
    # Signal emitted when range values change (min_value, max_value)
    rangeChanged = pyqtSignal(float, float)
    # Signal emitted when apply button is clicked
    applyRequested = pyqtSignal(float, float)
    lockCount = 0
    
    def __init__(self, object_size: dict, default_object_size: dict):
        super().__init__()
        self.object_size = object_size
        self.default_object_size = default_object_size
        self.round_parametr_slider = object_size['round_parametr_slider']
        self.round_parametr_value_input = object_size['round_parametr_value_input']
        self.initUI()

    def _to_slider_floor(self, value):
        return int(math.floor(float(value) * self.round_parametr_slider))

    def _to_slider_ceil(self, value):
        return int(math.ceil(float(value) * self.round_parametr_slider))
            
    def initUI(self):
        layout = QVBoxLayout()
        
        # Create horizontal layout for slider and controls
        slider_controls_layout = QHBoxLayout()
        
        # Create the range slider
        self.range_slider = RangeSlider(
            minimum=self._to_slider_floor(self.object_size['min_size']),
            maximum=self._to_slider_ceil(self.object_size['max_size']),
            low=self._to_slider_floor(self.object_size['min_size']),
            high=self._to_slider_ceil(self.object_size['max_size'])
        )
        
        # Connect signals
        self.range_slider.valueChanged.connect(self.on_range_changed)
        
        # Create Auto Apply controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 0, 0, 0)
        
        self.auto_apply_checkbox = QCheckBox("Auto Apply")
        self.auto_apply_checkbox.setChecked(False)  # Default to auto apply
        self.auto_apply_checkbox.setFont(QFont("Arial", 10))
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.setFont(QFont("Arial", 10))
        self.apply_button.setEnabled(False)  # Disabled when auto apply is on
        self.apply_button.setMaximumWidth(80)
        
        # Connect control signals
        self.auto_apply_checkbox.stateChanged.connect(self.on_auto_apply_changed)
        self.apply_button.clicked.connect(self.on_apply_clicked)
        
        controls_layout.addWidget(self.auto_apply_checkbox)
        controls_layout.addWidget(self.apply_button)
        controls_layout.addStretch()
        controls_widget.setLayout(controls_layout)
        
        # Add slider and controls to horizontal layout
        slider_controls_layout.addWidget(self.range_slider, 1)  # Give slider more space
        slider_controls_layout.addWidget(controls_widget, 0)  # Fixed size for controls
        
        # Create value display labels
        value_layout = QHBoxLayout()
        self.min_label = QLabel(f"Min: {self.object_size['min_size'] * self.round_parametr_value_input:.2f}")
        self.max_label = QLabel(f"Max: {self.object_size['max_size'] * self.round_parametr_value_input:.2f}")
        
        font = QFont("Arial", 12)
        self.min_label.setFont(font)
        self.max_label.setFont(font)
        
        value_layout.addWidget(self.min_label)
        value_layout.addStretch()
        value_layout.addWidget(self.max_label)
        
        layout.addLayout(slider_controls_layout)
        layout.addLayout(value_layout)
        
        self.setLayout(layout)
    
    def on_range_changed(self, low, high):
        """
        Handle range slider value changes.
        """
        if self.lockCount > 0:
            return  
        slider_min = self.range_slider.minimum()
        slider_max = self.range_slider.maximum()
        min_value = (
            float(self.default_object_size['min_size'])
            if low == slider_min
            else low / self.round_parametr_slider
        )
        max_value = (
            float(self.default_object_size['max_size'])
            if high == slider_max
            else high / self.round_parametr_slider
        )
        
        # Update object_size
        self.object_size['min_size'] = min_value
        self.object_size['max_size'] = max_value
        
        # Update display labels
        self.min_label.setText(f"Min: {min_value * self.round_parametr_value_input:.2f}")
        self.max_label.setText(f"Max: {max_value * self.round_parametr_value_input:.2f}")
        
        # Emit signal based on Auto Apply setting
        if self.auto_apply_checkbox.isChecked():
            self.rangeChanged.emit(min_value, max_value)
        else:
            # Enable apply button when not auto-applying
            self.apply_button.setEnabled(True)
    
    def set_default(self):
        """
        Set slider to default values.
        """
        try:
            self.lockCount += 1
            min_val = float(self.default_object_size['min_size'])
            max_val = float(self.default_object_size['max_size'])
        
            self.object_size['min_size'] = min_val
            self.object_size['max_size'] = max_val
        
            self.range_slider.setValues(
                self._to_slider_floor(min_val),
                self._to_slider_ceil(max_val)
            )
            self.min_label.setText(f"Min: {min_val * self.round_parametr_value_input:.2f}")
            self.max_label.setText(f"Max: {max_val * self.round_parametr_value_input:.2f}")
        finally:
            self.lockCount -= 1 
    
    def on_auto_apply_changed(self, state):
        """
        Handle Auto Apply checkbox state change.
        """
        if state == Qt.Checked:
            # Auto apply is enabled
            self.apply_button.setEnabled(False)
            # Immediately apply current values
            min_value = self.object_size['min_size']
            max_value = self.object_size['max_size']
            self.rangeChanged.emit(min_value, max_value)
        else:
            # Auto apply is disabled
            self.apply_button.setEnabled(True)
    
    def on_apply_clicked(self):
        """
        Handle Apply button click.
        """
        min_value = self.object_size['min_size']
        max_value = self.object_size['max_size']
        self.applyRequested.emit(min_value, max_value)
        self.apply_button.setEnabled(False)  # Disable until next change 
    
    def change_default(self, min_size, max_size):
        """
        Change the range and default values.
        """
        try:
            self.lockCount += 1
            if min_size is None:
                min_size = 0.0
            if max_size is None:
                max_size = 1.0

            # Keep normalized metrics within sane bounds
            min_size = max(0.0, min(min_size, 1.0))
            max_size = max(0.0, min(max_size, 1.0))
            
            # Update default values with some buffer
            self.default_object_size['min_size'] = min_size
            self.default_object_size['max_size'] = max_size 
        
            # Update slider range
            self.range_slider.setRange(
                self._to_slider_floor(self.default_object_size['min_size']),
                self._to_slider_ceil(self.default_object_size['max_size'])
            )
        
            # Set to default values
            self.set_default()
        finally:
            self.lockCount -= 1


class CellDetectorPlugin(BasePlugin):
    """
    Plugin for cell detection processing.
    """
    def get_name(self):
        return "Cell Processor"
    
    def __init__(self, *arg):
        """
        Initialize the cell detector plugin.
        """
        super().__init__(*arg)
        self.checked_indices = None
        self.plugin_signal.emit("Open_lsm", True)
        self.plugin_signal.emit("Open_folder", False)
        self.plugin_signal.emit("Settings", False)
        self.plugin_signal.emit("Save_as", False)
        currentModel = self.combo_box.currentText()
        self.model = Model(path=arg[-1][currentModel]['path'],
                           object_size=arg[-1][currentModel]['object_size'],
                           model_type = arg[-1][currentModel]['model_type']
                           )
        self.lsm_path = None

    def init_value(self, parent, parametrs, object_size, default_object_size, models):
        """
        Initialize plugin values.
        """
        self.show_boundry = 0
        self.draw_bounding = 0
        self.models = models
        self.parametrs = parametrs
        self.object_size = object_size
        self.default_object_size = default_object_size
        self.right_layout = parent
        self.lsm_filesList = None

    def handle_action(self, action_name, value):
        """
        Handle an action.
        """
        if action_name == "reset_detection":
            self.reset_detection()
        elif action_name == "set_size":
            self.set_size(value)
        elif action_name == "open_lsm":
            self.results_text.clear()
            self.reset_detection()
            self.range_slider.set_default()
            self.lsm_filesList = None
            self.folder_path = None
            if value:
                self.lsm_path = value
                self.button.setEnabled(True)
            else:
                self.lsm_path = None
                self.button.setEnabled(False)
        elif action_name == "open_image":
            self.currentModelChanged()
            self.reset_detection()

            self.results_text.clear()
            # self.max_range_slider.set_default()
            # self.min_range_slider.set_default()
            # workaround for some weird bug when app crashes after opening new image after prev recognition
            self.range_slider.change_default(0.0, 1.0)

            self.lsm_path = value
            self.lsm_filesList = None
            self.folder_path = None
            self.button.setEnabled(True)
            self.batchProcessButton.setEnabled(True)
        elif action_name == "open_folder":
            self.reset_detection()
            self.results_text.clear()
            if value:
                self.lsm_filesList = [os.path.join(value, file) \
        for file in os.listdir(value)\
            if file.lower().endswith(('.png', '.jpg', '.bmp', '.lsm', '.tif'))]
                self.folder_path = value
                self.range_slider.set_default()
                self.lsm_path = None
                self.draw_bounding = 0
                self.button.setEnabled(True)
                self.batchProcessButton.setEnabled(True)
            else:
                self.folder_path = None
                self.lsm_filesList = None
                self.button.setEnabled(False)

    def update_colormap(self, colormap):
        """
        Update the colormap.
        """
        self.object_size["color_map"] = colormap
        
        # Trigger redraw with new colormap using current range slider values
        current_min = self.object_size['min_size']
        current_max = self.object_size['max_size']
        self.on_range_slider_changed(current_min, current_max)
    
    def update_alpha(self, alpha_text):
        """
        Handle alpha combo box changes.
        """
        # Convert percentage text to float value (e.g., "75%" -> 0.75)
        alpha_value = float(alpha_text.rstrip('%')) / 100.0
        self.object_size["alpha"] = alpha_value
        
        # Trigger redraw with new alpha using current range slider values
        current_min = self.object_size['min_size']
        current_max = self.object_size['max_size']
        self.on_range_slider_changed(current_min, current_max)

    def update_um_per_px(self, value):
        """
        Update micrometers-per-pixel calibration.
        """
        self.object_size["um_per_px"] = float(value)

        if getattr(self, 'result', None) is not None:
            self.print_result(self.result)
    # def update_lineWidth(self):
    #     # Get value from QLineEdit
    #     input_text = self.LineWidth_edit.text()

    #     # Check if the entered value is a number
    #     try:
    #         # Convert to float
    #         line_width = float(input_text)

    #         self.object_size["line_width"] = round(line_width, 2)
    #         self.LineWidth_edit.setText(f"{float(input_text):.2f}")

    #     except ValueError:
    #         # If invalid value, set default
    #         size = self.object_size["line_width"]
    #         self.LineWidth_edit.setText(f"{size:.2f}") 

    def reset_detection(self):
        try:
            self.model.cell_counter.detections = None
        except:
            pass

    def set_size(self, detection, img_size : tuple = (512,512)):
        """
        Set the size based on detection.
        
        Args:
            detection: Detection data.
            img_size: Image size.
        """
        if detection is None or len(detection) == 0:
            self.range_slider.change_default(0.0, 1.0)
            return

        try:
            # Full dataframe from segmentation models: use morphology area directly
            if hasattr(detection, "columns") and "area" in detection.columns:
                values = detection["area"].dropna().tolist()
                if values:
                    self.range_slider.change_default(min(values), max(values))
                else:
                    self.range_slider.change_default(0.0, 1.0)
                return

            # Series/list of boxes
            values = []
            if all(len(cell) >= 4 for cell in detection):
                for cell in detection:
                    area = float(cell[2]) * float(cell[3])

                    # If widths/heights are normalized, area is already normalized
                    if 0.0 <= float(cell[2]) <= 1.0 and 0.0 <= float(cell[3]) <= 1.0:
                        values.append(area)
                    else:
                        img_sq = img_size[0] * img_size[1]
                        values.append(area / img_sq)

            if values:
                self.range_slider.change_default(min(values), max(values))
            else:
                self.range_slider.change_default(0.0, 1.0)

        except Exception as e:
            app_logger().exception(e)
            self.range_slider.change_default(0.0, 1.0)

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
        button_enabled = self.button.isEnabled()
        self.button.setText("Calculating.....")
        self.button.setEnabled(False)
        self.button.repaint()
        
        try:
            # Run inference in a thread using WaitWindow
            # Create a wrapper to handle progress_callback that call_inference doesn't expect
            def inference_wrapper(*args, **kwargs):
                # Remove progress_callback from kwargs if present since call_inference doesn't need it
                kwargs.pop('progress_callback', None)
                return self.call_inference(*args, **kwargs)
            
            wait_window = run_with_wait_window(
                inference_wrapper, 
                model,
                title="Image Processing", 
                info_text="Processing image...", 
                parent=self.parent(),
                threaded=True
            )
            
            # Connect signals to handle completion
            def on_completion(result):
                self.result = result
                
            def on_error(error_msg):
                self.result = None
                self.plugin_signal.emit("show_warning", f"Error during calculation: {error_msg}")

            if hasattr(wait_window, 'process_completed'):
                wait_window.process_completed.connect(on_completion)
            if hasattr(wait_window, 'process_failed'):
                wait_window.process_failed.connect(on_error)

            # Wait for completion (this will block until thread finishes)
            if hasattr(wait_window, 'isVisible'):
                while wait_window.isVisible():
                    QApplication.processEvents()
                    time.sleep(0.01)
                
            result = getattr(self, 'result', None)
        finally:
            self.button.setText("Calculate")
            self.button.setEnabled(button_enabled)
            

        # If no result, return
        if not result:
            return 0

        set_global('detections',self.model.cell_counter.detections if self.model and self.model.cell_counter else None)
        
        # Create QGraphicsTextItems to display the results
        self.results_text.clear()
        self.print_result(result)

        # Set flag to draw bounding boxes
        self.draw_bounding = 1

        # Draw bounding boxes
        self.draw_bounding_box()

    def call_inference(self, model):
        """
        Call inference for the given model.
        
        Args:
            model: Model name.
            
        Returns:
            Result of inference.
        """
        try:
                    # Attempt to calculate the result using the selected method
            if self.model and (self.model == self.model.model_name):
                result = self.model.calculate(
                            img_path=self.lsm_path, cell_channel=self.parametrs['Cell'],\
                                nuclei_channel=self.parametrs['Nuclei'])
            else:
                if self.model:
                    del self.model
                    self.model = None
                self.model = Model(path=self.models[model]['path'],
                                        object_size=self.models[model]['object_size'],
                                        model_type=self.models[model]['model_type'],
                                        model_data=self.models[model],
                                        model_name=model
                                        )
                self.model.cell_counter.original_image_path = self.lsm_path
                result = self.model.calculate(
                            img_path=self.lsm_path, cell_channel=self.parametrs['Cell'],\
                                nuclei_channel=self.parametrs['Nuclei'])
        except  Exception as e:
            traceback.print_exc()
            app_logger().exception(e)
            try:
                        # If an error occurs, try without channel information
                if self.models[model]['path'] == self.model.path:
                            # result = self.models[model].calculate(img_path=self.lsm_path)
                    result = self.model.calculate(img_path=self.lsm_path)
                else:
                    del self.model
                    self.model = None
                    a_path = self.models[model]['path']
                    self.model = Model(path=a_path,
                                            object_size=self.models[model]['object_size'],
                                            model_type=self.models[model]['model_type'],
                                            model_data=self.models[model],
                                            model_name=model)
                    self.model.cell_counter.original_image_path = self.lsm_path
                    result = self.model.calculate(img_path=self.lsm_path)
            except  Exception as e:
                traceback.print_exc()
                app_logger().error(e)
                        # If still not successful, show an error dialog
                self.plugin_signal.emit("show_warning", f"Error during calculation:{e} \n\nChoose another model or change channels settings")
                result = None
                if self.model:
                    del self.model
                    self.model = None
                        # clear the model so it can be restarted 
                self.draw_bounding = 0
        return result

    def calculate_single_model(self, modeltype,modelpath,object_size, image_path,model_data = None, model_name = "<not set>"):
        """
        Calculate for a single model.
        
        Args:
            modeltype: Model type.
            modelpath: Model path.
            object_size: Object size.
            image_path: Image path.
            model_data: Model data.
            model_name: Model name.
            
        Returns:
            Tuple of images and data.
        """
        model = None
        model = Model(path=modelpath,object_size=object_size,model_type=modeltype,model_data=model_data,model_name=model)
        model.cell_counter.original_image_path = self.lsm_path
        try:
            model.calculate(img_path=image_path, cell_channel=self.parametrs['Cell'],nuclei_channel=self.parametrs['Nuclei'])
        except  Exception as e:
            traceback.print_exc()
            app_logger().error(e)
            try:
                model.calculate(img_path=image_path)
            except  Exception as e:
                traceback.print_exc()
                app_logger().error(e)
                # If still not successful, show an error dialog
                self.plugin_signal.emit("show_warning", f"Error during calculation:{e} \n\nChoose another model or change channels settings")
                if model:
                    del model
                    model = None
        if model:
            return model.cell_counter.original_image, model.cell_counter.prediction_image,model.cell_counter.inference_duration,model.cell_counter.detectionCount
        else:
            return None, None, None, None

    def batchProcess_ProcessModelList(self,model_list):
        """
        Process a list of models in batch.
        
        Args:
            model_list: List of model names.
        """
        i = 1
        j = 1
        for model_name in model_list:
            if self.batch_wait_window is not None:
                self.batch_wait_window.set_info_text(f"Processing {model_name} ({j}/{len(model_list)})")
            j = j + 1
            model_data = self.models[model_name]
            modepath = model_data['path']
            model_type = model_data['model_type']
            _, processedImage,duration,counted = self.calculate_single_model(model_type, modepath, self.object_size, self.lsm_path,model_data=model_data, model_name = model_name)
            if counted is not None:
                self.batch_processedImages.append( processedImage)
                self.batch_labels.append(f"{i} {model_name}:{counted} cells in {duration:.2f} seconds")
                i = i + 1
                try:
                    imageGrid = create_image_grid(self.batch_processedImages,self.batch_labels,total_images=len(self.models))
                    safe_image_write(imageGrid, IMAGE_FILE_NAME_GRID)
                    self.plugin_signal.emit("add_image", IMAGE_FILE_NAME_GRID )
                except Exception as e:
                    app_logger().critical("Unhandled exception caught:", e)
        return

    def batchProcessButton_click(self):
        """
        Handle batch process button click.
        """
        self.batch_processedImages = []
        self.batch_labels = []
        try:
            items = list(self.models.keys())
            if self.checked_indices is None:
                self.checked_indices = list(range(0, len(self.models)))  # B and D checked
            dlg = ModelsCheckListDialog(items, self.checked_indices, parent=self.parent())
            #set to resonable height to fit most of models
            dlg.resize(300,383)
            if dlg.Execute():
                checked = dlg.get_checked_items()
            else:
                return
           
            model_list = [s for _, s in checked]
            self.checked_indices = [i for i, _ in checked]

            # Run inference in a thread using WaitWindow
            # Create a wrapper to handle progress_callback that call_inference doesn't expect
            def inference_wrapper(*args, **kwargs):
                # Remove progress_callback from kwargs if present since call_inference doesn't need it
                kwargs.pop('progress_callback', None)
                self.batchProcess_ProcessModelList(model_list)
                return
            
            wait_window = run_with_wait_window(
                inference_wrapper, 
                title="Image Processing", 
                info_text="Processing image...", 
                parent=self.parent(),
                threaded=True
            )
            self.batch_wait_window = wait_window

            # Connect signals to handle completion
            def on_completion(result):
                pass
                
            def on_error(error_msg):
                self.plugin_signal.emit("show_warning", f"Error during calculation: {error_msg}")

            if hasattr(wait_window, 'process_completed'):
                wait_window.process_completed.connect(on_completion)
            if hasattr(wait_window, 'process_failed'):
                wait_window.process_failed.connect(on_error)

            # Wait for completion (this will block until thread finishes)
            if hasattr(wait_window, 'isVisible'):
                while wait_window.isVisible():
                    QApplication.processEvents()
                    time.sleep(0.01)

            imageGrid = create_image_grid(self.batch_processedImages,self.batch_labels)
            self.batch_processedImages, self.batch_labels = None, None

            safe_image_write(imageGrid, IMAGE_FILE_NAME_GRID)
            plt.imshow(imageGrid)
            plt.axis('off')  # Hide axes
            plt.title("Image")
            plt.show()            
        finally:
            pass


    def batchProcess_MultiImage(self, model_type, modepath, object_size, file_path, model_data , model_name, files):
        """
        Process multiple images in batch.
        
        Args:
            model_type: Model type.
            modepath: Model path.
            object_size: Object size.
            file_path: File path.
            model_data: Model data.
            model_name: Model name.
            files: List of files.
        """
        i = 1
        for file_path in files:
            image_name = os.path.basename(file_path)
            self.lsm_path = file_path
            #self.batchProcessButtonMultiImage.setText(f"Processing {image_name} ({i}/{len(files)})")
            #self.batchProcessButtonMultiImage.repaint()
            self.batch_wait_window.set_info_text(f"Processing {image_name} ({i}/{len(files)})")
            _, processedImage,duration,counted = self.calculate_single_model(model_type, modepath, self.object_size, file_path,model_data = model_data , model_name = model_name)
            self.batch_processedImages.append( processedImage)
            self.batch_labels.append(f"{i} {image_name}:{counted} cells in {duration:.2f} seconds")
            i = i + 1
        pass

    def batchProcessMultiImageButton_click(self):
        """
        Handle batch process multi image button click.
        """
        try:
            savedEnabled = self.batchProcessButtonMultiImage.isEnabled()
            saved_lsm_path = self.lsm_path
            self.batchProcessButtonMultiImage.setEnabled(False)
            files, _ = QFileDialog.getOpenFileNames(
                None,
                "Select Images",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
            )            

            self.batch_processedImages = []
            self.batch_labels = []

            model = self.combo_box.currentText()
            modepath = self.models[model]['path']
            model_type = self.models[model]['model_type']
            model_data = self.models[model]

            def inference_wrapper(*args, **kwargs):
                # Remove progress_callback from kwargs if present since call_inference doesn't need it
                kwargs.pop('progress_callback', None)
                self.batchProcess_MultiImage(model_type= model_type,modepath= modepath,object_size= self.object_size,file_path= self.lsm_path,
                                            model_data= model_data,model_name = model,files= files)
                return

            # Connect signals to handle completion
            def on_completion(result):
                pass
            
            def on_error(error_msg):
                self.plugin_signal.emit("show_warning", f"Error during calculation: {error_msg}")

            if files:
                wait_window = run_with_wait_window(
                    inference_wrapper, 
                    title="Image Processing", 
                    info_text="Processing image...", 
                    parent=self.parent(),
                    threaded=True
                )
                self.batch_wait_window = wait_window

                if hasattr(wait_window, 'process_completed'):
                    wait_window.process_completed.connect(on_completion)
                if hasattr(wait_window, 'process_failed'):
                    wait_window.process_failed.connect(on_error)

                # Wait for completion (this will block until thread finishes)
                if hasattr(wait_window, 'isVisible'):
                    while wait_window.isVisible():
                        QApplication.processEvents()
                        time.sleep(0.01)
                       
                imageGrid = create_image_grid(self.batch_processedImages,self.batch_labels)
                safe_image_write(imageGrid, IMAGE_FILE_NAME_GRID)
                self.plugin_signal.emit("add_image", IMAGE_FILE_NAME_GRID )

                plt.imshow(imageGrid)
                plt.axis('off')  # Hide axes
                plt.title("Image")
                plt.show()            
                self.batch_processedImages,self.batch_labels = None,None


        finally:
            self.lsm_path = saved_lsm_path
            self.batchProcessButtonMultiImage.setEnabled(savedEnabled)    
            self.batchProcessButtonMultiImage.setText(f"Current model on multiple images")
        
    def print_result(self, result):
        """
        Print the result based on model type.
        
        Args:
            result: Result data.
        """
        model = self.combo_box.currentText()
        if model == "Detector":
            self.print_result_detector(result)
        else:
            self.print_result_segmenter(result)

    def _format_value(self, label, value_permyriad, value_um=None, unit_um="µm"):
        """
        Format value for display.
        
        Args:
            label: Label for the value.
            value_permyriad: Value in permyriad.
            value_um: Value in micrometers.
            unit_um: Unit for micrometers.
            
        Returns:
            Formatted string.
        """
        if value_permyriad in ("-", None):
            return f"{label}: -"

        value_permyriad = f"{float(value_permyriad):.2f}"

        if value_um is None:
            return f"{label}: {value_permyriad}‱"

        value_um = f"{float(value_um):.2f}"
        return f"{label}: {value_permyriad}‱ ({value_um} {unit_um})"

    def print_result_detector(self, result):
        """
        Print result for detector model.
        
        Args:
            result: Result data.
        """
        results = []
        # Add number of cells
        results.append(f'Cells: {result["Cells"]["box"].shape[0]}')
        # try:
        #     results.append(f'Cells: {result["Cells"]["box"].shape[0]}')
        # except:
        #     results.append(f'Cells: {result["Cells"]}')

        average_arithmetic_diameter_permyriad = "-"
        average_geometric_diameters_permyriad = "-"
        average_area_permyriad = "-"

        average_arithmetic_diameter_um = None
        average_geometrics_diameter_um = None
        average_area_um2 = None

        try:
            boxes = result["Cells"]["box"]

            # Extract length and width in pixels (second and third elements in arrays)
            lengths = boxes.apply(lambda x: x[2])
            widths = boxes.apply(lambda x: x[3])

            image_h = self.model.cell_counter.original_image.shape[0]
            image_w = self.model.cell_counter.original_image.shape[1]
            img_area = image_h * image_w

            # Calculate diagonals (diameters)
            arithmetic_diameters_px = (lengths + widths) / 2
            geometric_diameters_px = (lengths * widths)**(1/2)

            # Calculate areas
            areas_px2 = lengths * widths
            average_area_px2 = areas_px2.mean()

            img_linear = (image_h + image_w) / 2.0
            average_arithmetic_diameter_permyriad = round(arithmetic_diameters_px.mean() / img_linear * 10000, 2)
            average_geometric_diameters_permyriad = round(geometric_diameters_px.mean() / img_linear * 10000, 2)
            average_area_permyriad = round(average_area_px2 / img_area * 10000, 2)

            um_per_px = self.object_size.get("um_per_px")
            if um_per_px is not None:
                average_arithmetic_diameter_um = round(arithmetic_diameters_px.mean() * um_per_px, 2)
                average_geometrics_diameter_um = round(geometric_diameters_px.mean() * um_per_px, 2)
                average_area_um2 = round(average_area_px2 * (um_per_px ** 2), 2)

        except Exception:
            pass

        results.append(
            self._format_value(
                "Mean S",
                average_area_permyriad,
                average_area_um2,
                "µm²"
            )
        )

        results.append(
            self._format_value(
                "Mean Arithmetic D",
                average_arithmetic_diameter_permyriad,
                average_arithmetic_diameter_um,
                "µm"
            )
        )

        results.append(
            self._format_value(
                "Mean Geometric D",
                average_geometric_diameters_permyriad,
                average_geometrics_diameter_um,
                "µm"
            )
        )

        results.append("")

        # Add number of nuclei
        if result["Nuclei"] == -100:
            results.append('Nuclei: -')
        else:
            results.append(f'Nuclei: {result["Nuclei"]}')

        # Add percentage alive
        if result["%"] == -100:
            results.append('Alive: -')
        else:
            results.append(f'Alive: {result["%"]}%')

        # Font for all elements - not needed for text edit
        
        # Add text elements to text edit
        result_text = "\n".join(results)
        self.results_text.setPlainText(result_text)

    def print_result_segmenter(self, result):
        """
        Print result for segmenter model.
        
        Args:
            result: Result data.
        """
        spheroid_df = result["Cells"]

        # Calculate average values and number of rows
        try:
            avg_diameter = spheroid_df["diameter"].mean()
            avg_area = spheroid_df["area"].mean()
            avg_volume = spheroid_df["volume"].mean()
            num_cells = spheroid_df.shape[0]
            inference_duration = -1
            if self.model:
                inference_duration = self.model.inference_duration

            avg_diameter_permyriad = round(avg_diameter * 10000, 3)
            avg_area_permyriad = round(avg_area * 10000, 3)
            avg_volume_permyriad = round(avg_volume * 10000, 3)

            avg_diameter_um = None
            avg_area_um2 = None
            avg_volume_um3 = None

            um_per_px = self.object_size.get("um_per_px")
            if um_per_px is not None:
                image_h = self.model.cell_counter.original_image.shape[0]
                image_w = self.model.cell_counter.original_image.shape[1]

                avg_diameter_px = avg_diameter * image_w
                avg_area_px2 = avg_area * image_h * image_w
                linear_scale_px = (image_h + image_w) / 2
                avg_volume_px3 = avg_volume * (linear_scale_px ** 3)

                avg_diameter_um = round(avg_diameter_px * um_per_px, 2)
                avg_area_um2 = round(avg_area_px2 * (um_per_px ** 2), 2)
                avg_volume_um3 = round(avg_volume_px3 * (um_per_px ** 3), 2)

            # Create strings for output
            results = [
                f"Objects detected: {num_cells}",
                f"Duration        : {inference_duration:.2f} seconds",
                self._format_value("Mean D", avg_diameter_permyriad, avg_diameter_um, "µm"),
                self._format_value("Mean S", avg_area_permyriad, avg_area_um2, "µm²"),
                self._format_value("Mean V", avg_volume_permyriad, avg_volume_um3, "µm³"),
            ]
        except:
            results = [
                f"Objects detected: - ",
                f"Duration        : - ",
                f"Mean D: -",
                f"Mean S: -",
                f"Mean V: -",
            ]
        # try:
        # except AttributeError:
        #     num_cells = spheroid_df
        #     # Create strings for output
        #     results = [
        #         f"Cells detected: {num_cells}"
        #     ]

        # Settings for font and display - not needed for text edit
        
        # Add strings to text edit
        result_text = "\n".join(results)
        self.results_text.setPlainText(result_text)

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
            # Handle different display modes based on show_boundry flag
            if self.show_boundry == 0:  # Original
                # Show the original image
                self.plugin_signal.emit("add_image", self.lsm_path)
            elif self.show_boundry == 2:  # Inference
                # Show the inference image (if available)
                    self.plugin_signal.emit("add_image", IMAGE_FILE_NAME_INGFERENCE)
            elif self.show_boundry == 1:  # Detections
                # Show the image with bounding box detections
                self.plugin_signal.emit("add_image", IMAGE_FILE_NAME_DETECTION)
            else:
                # Default to original image
                self.plugin_signal.emit("add_image", self.lsm_path)
        except Exception as e:
            # If an error occurs, print the traceback, show a warning dialog

            traceback.print_exc()
            self.plugin_signal.emit("show_warning", "Error during opening image.")

    def on_display_mode_changed(self, button):
        """
        Handle the radio button selection change for display mode.

        Args:
        button: The selected radio button.
        """
        # Get the ID of the selected button (0=Original, 1=Detections, 2=Inference)
        selected_id = self.display_group.id(button)
        
        if selected_id == 0:  # Original
            self.show_boundry = 0
        elif selected_id == 1:  # Detections
            self.show_boundry = 1
        elif selected_id == 2:  # Inference
            self.show_boundry = 2
            
        # Redraw with the new display mode
        self.draw_bounding_box()

    def on_range_slider_changed(self, min_value, max_value):
        """
        Handle range slider value changes.
        
        Args:
            min_value (float): New minimum size value
            max_value (float): New maximum size value
        """
        detections = get_global('detections')
        if detections is None:
            return
        
        if hasattr(detections, "columns") and all(c in detections.columns for c in ["area", "mask"]):
            filtered_detections = filter_segmentation_detections(
                detections,
                min_size=min_value,
                max_size=max_value,
                size_metric=self.object_size.get("size_metric", "area")
            )
        else:
            from model.utils import filter_detections
            filtered_detections = filter_detections(
                detections,
                min_size=min_value,
                max_size=max_value
            )

        # Check if filtered_detections has 'mask' key and is valid
        if filtered_detections is not None and 'mask' in filtered_detections and filtered_detections['mask'] is not None:
            base_image = get_global('image_display_base')
            plot_predictions(
                base_image.copy(),
                filtered_detections['mask'].tolist(), 
                filename=IMAGE_FILE_NAME_DETECTION, 
                colormap=self.object_size["color_map"], 
                alpha=self.object_size.get("alpha", 0.75),
                color_ids=filtered_detections['id_label'].tolist() if 'id_label' in filtered_detections else None
            )
        else:
            image = get_global('image_inference').copy()
            for i in range(filtered_detections.shape[0]):
                draw_bounding_box(
                    image,
                    filtered_detections.iloc[i,0],
                    filtered_detections.iloc[i,2],
                    round(filtered_detections.iloc[i,3][0] * filtered_detections.iloc[i,4]),
                    round(filtered_detections.iloc[i,3][1] * filtered_detections.iloc[i,4]),
                    round((filtered_detections.iloc[i,-2][0] + filtered_detections.iloc[i,-2][2]) * filtered_detections.iloc[i,-1]),
                    round((filtered_detections.iloc[i,-2][1] + filtered_detections.iloc[i,-2][3]) * filtered_detections.iloc[i,-1]),
                )
                
            try:
                os.remove(IMAGE_FILE_NAME_DETECTION)
            except:
                pass
            safe_image_write(image, IMAGE_FILE_NAME_DETECTION)

        self.draw_bounding_box()
        return

    def on_state_changed_scale(self, state):
        """
        Handle scale checkbox state change.
        
        Args:
            state: Checkbox state.
        """
        if state and self.x10checkbox.isEnabled():
            self.object_size["scale"] = 10
        else:
            self.object_size["scale"] = 20
        self.reset_detection()

    def currentModelChanged(self):
        """
        Handle model selection change.
        """
        self.default_object_size['min_size'] = 0.0
        self.default_object_size['max_size'] = 1.0
        
        # Reset range slider to default values
        if hasattr(self, 'range_slider'):
            self.range_slider.set_default()

        self.reset_detection()
        if hasattr(self, 'x10checkbox'):
            model_data=self.models[self.combo_box.currentText()]
            self.x10checkbox.setEnabled("x10" in model_data)

    def init_rightLayout(self):
        """
        Initialize the right layout UI components.
        """
        plugin_label = QLabel(self.get_name())
        plugin_label.setFont(QFont("Arial", 32))
        # Create a combo box to choose models
        self.combo_box = QComboBox()
        self.combo_box.currentIndexChanged.connect(self.currentModelChanged)

        # Create a label to prompt user to choose a model
        label = QLabel("Choose model:")
        label.setFont(QFont("Arial", 24))

        # Create a text edit widget for displaying results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Arial", 12))
        self.results_text.setMaximumHeight(200)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)

        # Create a button for calculating
        self.button = QPushButton("Calculate")
        self.button.setEnabled(False)

        # Create a button for calculating
        self.batchProcessButton = QPushButton("All models on current image")
        self.batchProcessButton.setEnabled(False)

        # Create a button for calculating
        self.batchProcessButtonMultiImage = QPushButton("Current model on multiple images")
        self.batchProcessButtonMultiImage.setEnabled(True)

        # Set font for the combo box
        self.combo_box.setFont(QFont("Arial", 24))

        # Add 'All_method' option and method names to the combo box
        #self.combo_box.addItems(['All_models'])
        self.combo_box.addItems([key for key in self.models])

        # Set current index to 1
        self.combo_box.setCurrentIndex(0)

        # Create radio buttons for image display options
        self.radio_original = QRadioButton("Original")
        self.radio_detections = QRadioButton("Detection")
        self.radio_inference = QRadioButton("Inference")
        
        # Create button group to manage radio buttons
        self.display_group = QButtonGroup()
        self.display_group.addButton(self.radio_original, 0)
        self.display_group.addButton(self.radio_detections, 1)
        self.display_group.addButton(self.radio_inference, 2)
        
        # Set default selection to Original
        self.radio_original.setChecked(True)
        
        # Connect radio button changes to handler function
        self.display_group.buttonClicked.connect(self.on_display_mode_changed)
        
        # Set font for radio buttons
        font_radio = QFont("Arial", 12)
        self.radio_original.setFont(font_radio)
        self.radio_inference.setFont(font_radio)
        self.radio_detections.setFont(font_radio)
        
        # Set style sheet for radio buttons to customize their indicator size
        radio_style = '''
            QRadioButton::indicator {
                width: 24px;
                height: 24px;
            }
        '''
        self.radio_original.setStyleSheet(radio_style)
        self.radio_inference.setStyleSheet(radio_style)
        self.radio_detections.setStyleSheet(radio_style)
        
        # Create horizontal layout for radio buttons
        self.radio_layout = QHBoxLayout()
        self.radio_layout.addWidget(self.radio_original)
        self.radio_layout.addWidget(self.radio_inference)
        self.radio_layout.addWidget(self.radio_detections)
        
        # Create widget to contain the horizontal layout
        self.radio_widget = QWidget()
        self.radio_widget.setLayout(self.radio_layout)

        self.x10checkbox = QCheckBox("Process at x10 scale")
        self.x10checkbox.setEnabled(False)  

        # Connect checkbox state change to handler function
        self.x10checkbox.stateChanged.connect(self.on_state_changed_scale)

        # Set font for the checkbox
        self.x10checkbox.setFont(QFont("Arial", 24))

        # Set style sheet for the checkbox to customize its indicator size
        self.x10checkbox.setStyleSheet('''
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
            }
        ''')

        # Set font for the calculate button
        self.button.setFont(QFont("Arial", 32))

        # Connect button click event to calculate_button function
        self.button.clicked.connect(self.calculate_button)
        self.batchProcessButton.clicked.connect(self.batchProcessButton_click)
        self.batchProcessButtonMultiImage.clicked.connect(self.batchProcessMultiImageButton_click)
        
        range_lable = QLabel("Object Size:")
        font = QFont()
        font.setPointSize(16) 

        range_lable.setFont(font)

        # Create range slider wrapper
        self.range_slider = RangeSliderWrapper(self.object_size, self.default_object_size)
        
        # Connect range slider change signals
        self.range_slider.rangeChanged.connect(self.on_range_slider_changed)
        self.range_slider.applyRequested.connect(self.on_range_slider_changed)

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
        colormap_label.setFont(QFont("Arial", 16))

        self.colormap_combo = QComboBox()
        self.colormap_combo.setFont(QFont("Arial", 16))

        self.colormaps =  self.object_size["color_map_list"]
        self.colormap_combo.addItems(self.colormaps)
        self.colormap_combo.setCurrentText(self.object_size['color_map'])  # Set "Viridis" as default
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        
        # Create alpha combo box
        alpha_label = QLabel("Alpha:")
        alpha_label.setFont(QFont("Arial", 24))
        
        self.alpha_combo = QComboBox()
        self.alpha_combo.setFont(QFont("Arial", 16))
        
        # Add alpha values
        alpha_values = ["100%", "75%", "50%", "25%"]
        self.alpha_combo.addItems(alpha_values)
        self.alpha_combo.setCurrentText("75%")  # Default to 75%
        self.alpha_combo.currentTextChanged.connect(self.update_alpha)
        
        # Initialize alpha in object_size
        self.object_size["alpha"] = 0.75  # Default 75%

        um_per_px_label = QLabel("um per px:")
        um_per_px_label.setFont(QFont("Arial", 16))

        self.um_per_px_spin = QDoubleSpinBox()
        self.um_per_px_spin.setFont(QFont("Arial", 16))
        self.um_per_px_spin.setDecimals(6)
        self.um_per_px_spin.setRange(0.000001, 1000.0)
        self.um_per_px_spin.setSingleStep(0.001)
        self.um_per_px_spin.setValue(float(self.object_size.get("um_per_px", 0.325)))
        self.um_per_px_spin.valueChanged.connect(self.update_um_per_px)
        
        # Create horizontal layout for colormap and alpha
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_combo)
        colormap_layout.addSpacing(20)
        #colormap_layout.addWidget(alpha_label)
        colormap_layout.addWidget(self.alpha_combo)
        colormap_layout.addSpacing(20)
        colormap_layout.addWidget(um_per_px_label)
        colormap_layout.addWidget(self.um_per_px_spin)
        
        # Create widget to contain the horizontal layout
        colormap_widget = QWidget()
        colormap_widget.setLayout(colormap_layout)

        # Add widgets to the right layout with spacing
        ###self.right_layout.addWidget(label)
        self.right_layout.addWidget(plugin_label)
        self.right_layout.addSpacing(15)
        self.right_layout.addWidget(self.combo_box)
        self.right_layout.addSpacing(15)
        self.right_layout.addWidget(self.results_text)
        self.right_layout.addSpacing(15)

        self.right_layout.addWidget(colormap_widget)
        self.right_layout.addSpacing(15)
        # self.right_layout.addLayout(LineWidth_layout)
        self.right_layout.addSpacing(15)

        self.right_layout.addWidget(range_lable)
        self.right_layout.addSpacing(15)

        self.right_layout.addWidget(self.range_slider)

        self.right_layout.addSpacing(15)

        self.right_layout.addWidget(self.x10checkbox)
        self.right_layout.addSpacing(15)

        # Add horizontal radio buttons widget for display mode
        self.right_layout.addWidget(self.radio_widget)
        self.right_layout.addSpacing(15)
        self.right_layout.addWidget(self.button)
        self.right_layout.addSpacing(15)
        
        self.right_layout.addWidget(self.batchProcessButton)
        self.right_layout.addSpacing(15)
        self.right_layout.addWidget(self.batchProcessButtonMultiImage)
        self.right_layout.addSpacing(15)
