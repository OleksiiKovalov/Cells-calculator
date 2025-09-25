"""
MainWindow module for Cells Calculator application.

This module defines the MainWindow class that represents the main window of the Cells Calculator application.
The MainWindow class handles the initialization of the UI, loading and processing images, 
and interacting with various models to perform cell calculations.
"""
from  UI.splashscreen import update_splash
from datetime import datetime
import os
import shutil
import traceback
import tifffile
import json
from PyQt5.QtWidgets import (QAbstractItemView, QMessageBox, QTableWidget, QTableWidgetItem,
    QGraphicsView, QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QWidget, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from UI.SettingsWindow import SettingsWindow
from UI.menubar import menubar
from UI.table import calculate_table
from UI.right_layout.right_layout import right_layout
from UI.right_layout.plugins.CellDetector import CellDetector as CellDetector_plugin
from UI.right_layout.plugins.tracker import Tracker as Tracker_plugin
from model.utils import COLOR_NUMBER as color_number, clear_cache

class MainWindow(QMainWindow):
    """
    The MainWindow class represents the main window of the Cells Calculator application.
    This class handles the initialization of the UI, loading and processing images, 
    and interacting with various models to perform cell calculations.
    """
    mainWindow_signal = pyqtSignal(str, object)
    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, progress_callback=None):
        """
        Initialize the MainWindow.

        This method sets up the initial state of the main window.
        Args:
            progress_callback: Optional callback function to report initialization progress
        """
        super().__init__()
        
        self.progress_callback = progress_callback
        self._update_progress(15, "Initializing window...")

        self._update_progress(20, "Setting up cache directory...")
        # Add cache directory creation
        clear_cache()

        self._update_progress(30, "Configuring window layout...")
        # Get desktop information
        desktop = QApplication.desktop()
        screen_geometry = desktop.availableGeometry()

        # Set the fixed size of the main window to match the screen width and height minus the height of the menu bar
        self.setFixedSize(screen_geometry.width(),desktop.availableGeometry().height() - self.menuBar().height())

        # Set the window title
        self.current_plugin_name = "Cell Processor"
        self.setWindowTitle(self.current_plugin_name)

        self._update_progress(40, "Loading configuration and models...")
        self.init_value()
        
        self._update_progress(60, "Initializing menu bar...")
        # Initialize the user interface
        self.menu_bar = menubar(self, list(self.plugin_list.keys()), self.current_plugin_name)
        self.setMenuBar(self.menu_bar)
        
        self._update_progress(75, "Setting up right panel...")
        self.right_layout = right_layout(current_plugin_name=self.current_plugin_name,
                                         plugin_list= self.plugin_list)

        self._update_progress(85, "Connecting signals...")
        self.menu_bar.menubar_signal.connect(self.handle_menubar_action)
        self.menu_bar.menubar_signal.connect(self.right_layout.handle_menubar_action)

        self.right_layout.rightLayout_signal.connect(self.handle_rightLayout_action)
        self.right_layout.rightLayout_signal.connect(self.menu_bar.handle_rightLayout_action)

        self.mainWindow_signal.connect(self.menu_bar.handle_mainWindow_action)
        self.mainWindow_signal.connect(self.right_layout.handle_mainWindow_action)

        self._update_progress(95, "Finalizing layout...")
        self.right_layout.init_rightLayout()
        self.init_mainScene()
        
        self._update_progress(100, "Ready!")
    
    def _update_progress(self, value, message):
        """Update progress during initialization"""
        if self.progress_callback:
            self.progress_callback(value, message)
        QApplication.processEvents()

    @pyqtSlot(str, object)
    def handle_menubar_action(self, action_name, value):
        """Обрабатываем сигнал от menubar"""
        if action_name == "open_file":
            self.open_file(value)
        if action_name == "open_folder":
            self.open_folder(value)
        elif action_name == "open_settings":
            self.open_settings()
        elif action_name == "open_normalize":
            self.open_normalize()

        elif action_name == "show_warning":
            self.show_warning_dialog(value)
        elif action_name == "change_plugin":
            if value in self.plugin_list:
                self.main_scene.clear()
                self.current_plugin_name = value
                self.setWindowTitle(self.current_plugin_name)
                self.setWindowTitle(value)
                self.init_value()
                self.right_layout.set_current_plugin(value, self.plugin_list)

    
    @pyqtSlot(str, object)
    def handle_rightLayout_action(self, action_name, value):
        if action_name == "show_warning":
            self.show_warning_dialog(value)
        elif action_name == "add_image":
            self.add_image(value)
        pass
    
    def open_normalize(self):
        from UI.ImageNormalizeDialog import ImageNormalizeDialog
        from skimage.io import imread
        image = imread(self.lsm_path)
        from model.utils import safergb2gray
        image = safergb2gray(image)
        dlg = ImageNormalizeDialog(image)
        dlg.exec_()        
        pass

    def init_value(self):
        # TODO: сделать так что бы параметры собиралиьс относительно выброного plugin
        # Initialize DataFrame to None
        
        self._update_progress(42, "Setting up object parameters...")
        self.object_size = { 
                'min_size' : 100,
                   'max_size' : 0.000,
                   'signal' : self.mainWindow_signal.emit,
                   'round_parametr_slider' : 10**6,
                   'round_parametr_value_input' : 10**4,
                   'color_map' : "viridis",
                   'color_map_list' : list(color_number.keys()),
                   'line_width' : 100.00,
                   'scale' : 20
        }
        
        self.default_object_size = self.object_size.copy()

        # Default parameters for cell and nuclei channels
        self.parametrs = {'Cell': 0,
                    'Nuclei': 1
        }
        
        self._update_progress(45, "Loading model configuration...")
        self.models_celldetector = {}
        # self.models_celldetector = {
        # 'Detector': {"path": 'model/yolov8m-det.onnx', "object_size": self.object_size, "model_type":"cellcounter"},
        # 'YOLO-512 Segmenter': {"path": 'model/YOLO11x-512-seg.pt', "object_size": self.object_size, "model_type":"segmenter"},
        # 'YOLO-680 Segmenter': {"path": 'model/YOLO11x-680-seg.pt', "object_size": self.object_size, "model_type":"segmenter"},
        # 'Cellpose': {"path": 'cellpose', "object_size": self.object_size, "model_type":"cellpose"},
        # 'InstanSeg Flu_nc': {"path": 'fluorescence_nuclei_and_cells', "object_size": self.object_size, "model_type":"instanseg"},
        # 'InstanSeg bright_nuc': {"path": 'brightfield_nuclei', "object_size": self.object_size, "model_type":"instanseg"},
        # 'InstanSeg trained': {"path": 'model/instanseg_model_weights_best.pth.pt', "object_size": self.object_size, "model_type":"instanseg"},
        # }

        #loading detectors from config file
        from collections import OrderedDict
        
        self._update_progress(48, "Reading model configuration file...")
        with open('modelconfig.json', 'r') as f:
            loaded_models = json.load(f, object_pairs_hook=OrderedDict)

        self._update_progress(52, "Configuring models...")
        for model_name, model_data in loaded_models.items():
            # Set the 'object_size' parameter for each loaded model
            model_data['object_size'] = self.object_size
            self.models_celldetector[model_name] = model_data

        print("Models loaded and updated successfully!")
                    
        self._update_progress(55, "Setting up tracker models...")            
        self.models_tracker = {
            'Baseline Segmenter' : {"path": 'trainedmodels/YOLO11x-sphero-seg.pt', "size": self.object_size}
        }
        
        self._update_progress(58, "Initializing plugins...")
        self.plugin_list = {
            "Cell Processor" : {
                "init" : CellDetector_plugin,
                "arg" : [self.parametrs, self.object_size, self.default_object_size,
                         self.models_celldetector],
                "file_callback" : self.change_image,
                "folder_callback" : self.create_table
            },
            "Tracker" : {
                "init" :  Tracker_plugin,
                "arg" : [self.parametrs, self.object_size, self.default_object_size,
                         self.models_tracker],
                "file_callback" : print,
                "folder_callback" : print
            }
        }

        # Dictionary containing available models and their corresponding methods
        # Initialize variables to None or default values
        self.lsm_path = None
        self.lsm_filesList = None
        self.lsm_folder = None
        self.image_mru  = {}
        self.last_opened_file = ""
        self.currentImageWidth = 0
        self.currentImageHeight = 0

    def init_mainScene(self):
        """
        Initialize the main scene and its layout.

        Notes:
        - Create a central widget.
        - Create a horizontal layout for the main scene and other widgets.
        - Initialize the main scene and its view.
        - Create a vertical layout for the widgets on the right side.
        - Add the main view and right layout to the main layout.
        - Set the main layout for the central widget.
        - Set the scene rectangle to match the size of the main view.
        """
        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a horizontal layout for the main scene and other widgets
        self.main_layout = QHBoxLayout()

        # Initialize the main scene and its view
        self.main_scene = QGraphicsScene()
        self.main_view = QGraphicsView(self.main_scene)
        self.main_view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.main_view.setFixedWidth(int(self.width() * 0.75))

        # Add the main view and right layout to the main layout
        self.main_layout.addWidget(self.main_view)
        self.main_layout.addLayout(self.right_layout)

        # Set the main layout for the central widget
        self.central_widget.setLayout(self.main_layout)

        # Set the scene rectangle to match the size of the main view
        self.main_scene.setSceneRect(
            0, 0, self.main_view.width()-10, self.main_view.height())

    def show_warning_dialog(self, text):
        """
        Display a warning dialog.

        Args:
            text (str): The warning message text.
        """
        # Create a QMessageBox instance
        msgBox = QMessageBox()

        # Set the icon of the message box to a warning icon
        msgBox.setIcon(QMessageBox.Warning)

        # Set the text of the message box
        msgBox.setText(text)

        # Set the title of the message box
        msgBox.setWindowTitle("Warning")

        # Adjust the size of the message box
        msgBox.adjustSize()

        # Execute the message box (display it)
        msgBox.exec_()

    def open_folder(self, folder_path):
        """
        Open a folder and load all image files within it.

        Notes:
        - Open a dialog window to select a folder.
        - If a folder is selected, create a list of image files within it.
        - If image files are found, clear the main scene, reset certain variables, and enable certain actions.
        - Open a dialog window to choose settings.
        - Set the window title to include the selected folder name.
        - If no image files are found, reset variables and show a warning dialog.
        """

        # Create a list of image files within the selected folder
        lsm_folder = folder_path
        self.lsm_filesList = [os.path.join(folder_path, file) \
        for file in os.listdir(folder_path)\
            if file.lower().endswith(('.png', '.jpg', '.bmp', '.lsm', '.tif'))]

        # If image files are found
        if self.lsm_filesList:
            # Clear the main scene
            self.main_scene.clear()
            self.mainWindow_signal.emit("open_folder", folder_path)

            # Open a dialog window
            #self.open_settings()

            # Set the window title to include the selected folder name
            self.setWindowTitle(f"Cells Calculator - {os.path.basename(folder_path)}/")
        else:
            self.mainWindow_signal.emit("open_folder", None)
            self.lsm_path = None
            self.lsm_filesList = None
            self.lsm_folder = None
            # If no image files are found, reset variables and show a warning dialog
            self.show_warning_dialog("No Image files found in the selected folder")

    def create_table(self):
        """
        Create a table with calculated results for multiple files.

        Notes:
        - Checks if there are any files in the list.
        - Clears the main scene.
        - Attempts to create a table widget and calculate table data using given methods, files, and parameters.
        - If an exception occurs during calculation, disables certain actions, resets file list and data frame, and shows a warning dialog.
        - Configures table properties, populates the table with data from the data frame, sets minimum size, and resizes rows and columns to fit content.
        - Adds the table to the main scene.
        """
        print("Expired")
        return
        # Checks if there are any files in the list
        if not self.lsm_filesList:
            return
        
        # Clears the main scene
        self.main_scene.clear()
        
        try:
            # Attempts to create a table widget
            table = QTableWidget()
            
            try:
                # Attempts to calculate table data using given methods, files, and parameters
                df = calculate_table(
                    model_dict=self.models_celldetector, files_name=self.lsm_filesList, parametrs=self.parametrs)
            except Exception as e:
                traceback.print_exc()
                # If an exception occurs during calculation disables certain actions,
                # resets file list and data frame, and shows a warning dialog
                self.mainWindow_signal.emit('open_folder' ,None)
                self.df = None
                self.lsm_filesList = None
                self.show_warning_dialog("Error during calculation.")
                return

            # Stores the calculated data frame
            self.df = df
            
            # Retrieves the dimensions of the view and table cells
            view_width = self.main_view.viewport().width()
            view_height = self.main_view.viewport().height()
            cell_width = table.horizontalHeader().defaultSectionSize()
            cell_height = table.verticalHeader().defaultSectionSize()
            
            # Configures table properties
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setRowCount(df.shape[0])
            table.setColumnCount(df.shape[1])
            table.setHorizontalHeaderLabels(df.columns)
            
            # Populates the table with data from the data frame
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    item = QTableWidgetItem(str(df.iloc[i, j]))
                    table.setItem(i, j, item)
            
            # Sets minimum size and resizes rows and columns to fit content
            table.setMinimumSize(view_width, view_height)
            table.resizeRowsToContents()
            table.resizeColumnsToContents()
            
            # Adds the table to the main scene
            self.main_scene.addWidget(table)
            
        except:
            traceback.print_exc()
            # If an exception occurs during the process, disables certain actions,
            # resets file list and data frame, and shows a warning dialog
            self.mainWindow_signal.emit('open_folder' ,None)
            self.lsm_filesList = None
            self.df = None
            self.show_warning_dialog("Error during creating table")

    def add_image(self, lsm_file:str):
        """
        Open an image file and display it.
        
        Args:
        lsm_file (str or numpy.ndarray): Path to the image file or a numpy array representing an image.
            If it's a string (file path), it opens the image file. If it's a numpy array, it assumes
            it represents an image.

        Notes:
        If lsm_file is a string (file path), it creates a QImage from the file path.
            If lsm_file is a numpy array, it creates a QImage from it with grayscale format.
        """
        self.main_scene.clear()
        if isinstance(lsm_file, str) and not os.path.exists(lsm_file):
            image = self.create_no_image_qimage()
        elif isinstance(lsm_file, str) and not lsm_file.endswith(".lsm"):
            image = QImage(lsm_file)
        elif isinstance(lsm_file, str):
            with tifffile.TiffFile(lsm_file) as tif:
                # Read the first page of the LSM file as an array
                lsm_file = tif.pages[0].asarray()
            image = QImage(lsm_file[self.parametrs['Cell']], lsm_file.shape[1],
                        lsm_file.shape[2], QImage.Format_Grayscale8)
        else:
            # If it's not a string, assume it's a numpy array representing an image
            # and create a QImage from it with grayscale format
            image = QImage(lsm_file[self.parametrs['Cell']], lsm_file.shape[1],
                        lsm_file.shape[2], QImage.Format_Grayscale8)

        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(image)
        self.currentImageWidth = image.width()
        self.currentImageHeight = image.height()

        # Get the dimensions of the main view
        view_width = self.main_view.viewport().width()
        view_height = self.main_view.viewport().height()

        # Calculate the aspect ratio of the image
        if pixmap.height() > 0: 
            pixmap_aspect_ratio = pixmap.width() / pixmap.height()
            # Scale the image to fit within the view while maintaining aspect ratio
            if view_width / view_height > pixmap_aspect_ratio:
                pixmap_width = view_height * pixmap_aspect_ratio
                pixmap_height = view_height
            else:
                pixmap_width = view_width
                pixmap_height = view_width / pixmap_aspect_ratio
        else:
            pixmap_width = 0
            pixmap_height = 0
        pixmap_height = int(pixmap_height)
        pixmap_width = int(pixmap_width)
        # Scale the pixmap
        pixmap = pixmap.scaled(pixmap_width, pixmap_height,
                            aspectRatioMode=Qt.KeepAspectRatio)

        # Add the scaled pixmap to the main scene
        pixmap_item = self.main_scene.addPixmap(pixmap)
        

        # Calculate the position to center the image within the view
        x_pos = (view_width - pixmap.width()) / 2
        y_pos = (view_height - pixmap.height()) / 2

        # Set the position of the pixmap item within the scene
        pixmap_item.setPos(x_pos, y_pos)
        self.main_view.repaint()
        QApplication.processEvents()
        
    def change_image(self):
        """
        Change displayed image. 

        Notes:
            This method is called when there is a need to change the currently displayed image, typically when 
                switching between different LSM files. It attempts to open the specified LSM file, reads the first 
                page of the file as an array, and then adds the new image to the main scene.

        Raises:
            Exception: If an error occurs during the process of opening and displaying the new image, 
                    a warning dialog is shown to notify the user about the error.
        """
        try:
            # Attempt to open the LSM file using tifffile
            with tifffile.TiffFile(self.lsm_path) as tif:
                # Read the first page of the LSM file as an array
                lsm_file = tif.pages[0].asarray()

            # Clear the main scene
            self.main_scene.clear()

            # Add the new image to the scene
            self.add_image(lsm_file)
        except Exception as e:
            traceback.print_exc()
            # If an error occurs, show a warning dialog
            self.show_warning_dialog("Error during opening image.")

    def open_file(self, lsm_path):
        """
        Open an image file (*.png *.jpg *.bmp *.lsm *.TIF) and display it.

        Arguments:
        - self: The instance of the class.
        
        Notes:
        - This method opens a dialog window to select an image file.
        - If no file is selected, it returns.
        - If the selected file is an LSM file, it calls the open_lsm function.
        - If the selected file is not an LSM file, it stores the file path, clears the main scene, and attempts to add the image to the scene.
        - If an error occurs during the process, it shows a warning dialog, resets variables, and clears the main scene.
        """

        clear_cache()

        # If the selected file is an LSM file, call the open_lsm function
        if lsm_path.endswith(".lsm"):
            self.mainWindow_signal.emit("open_lsm", lsm_path)
            self.open_lsm(lsm_path)
        else:
            self.mainWindow_signal.emit("open_image", lsm_path)
            # If the selected file is not an LSM file
            # Store the file path
            self.lsm_path = lsm_path

            # Clear the main scene
            self.main_scene.clear()
            # Disable certain actions
            # Reset file list
            try:
                # Try to add the image to the scene and set the window title
                self.add_image(self.lsm_path)
                self.setWindowTitle(
                    f"Cells Calculator - {os.path.basename(lsm_path)} ({self.currentImageWidth} x {self.currentImageHeight})")
            except Exception as e:
                traceback.print_exc()
                # If an error occurs, show a warning dialog,
                # reset variables, and clear the main scene
                self.show_warning_dialog("Error during opening file.")
                self.setWindowTitle(f"Cells Calculator")
                self.mainWindow_signal.emit("open_lsm", None)
                self.lsm_path = None
                self.lsm_filesList = None
                self.lsm_folder = None
                self.main_scene.clear()

                return 0
        self.image_mru[lsm_path] = datetime.min


    def open_lsm(self, lsm_path):
        """
        Open an image file (*.LSM) and display it.

        Arguments:
        lsm_path (str): The file path of the LSM image to be opened.

        """
        # Store the LSM file path
        self.lsm_path = lsm_path

        # Clear the main scene
        self.main_scene.clear()
        # Reset the file list
        try:
            # Attempt to open the LSM file using tifffile
            with tifffile.TiffFile(self.lsm_path) as tif:
                # Read the first page of the LSM file as an array
                lsm_file = tif.pages[0].asarray()

            # Check if the number of channels in the LSM file is less than the maximum channel index specified in parameters
            if lsm_file.shape[0] < max([value+1 for key, value in self.parametrs.items()]):
                # If so, reset the parameters to default
                self.parametrs['Cell'] = 0
                self.parametrs['Nuclei'] = 1

            # Check if the number of channels in the LSM file is less than or equal to 1
            if lsm_file.shape[0] <= 1:
                # If so, show a warning dialog and return
                self.show_warning_dialog("File is wrong\n\nAmount of Channel less than 2")
                return

            # Add the LSM file image to the scene and set the window title
            self.add_image(lsm_file)
            self.setWindowTitle(
                f"Cells Calculator - {os.path.basename(lsm_path)}")
        except Exception as e:
            traceback.print_exc()
            # If an error occurs, show a warning dialog,
            # reset variables, and clear the main scene
            self.show_warning_dialog("Error during opening file.")
            self.setWindowTitle(f"Cells Calculator")
            self.mainWindow_signal.emit("open_lsm", None)
            self.main_scene.clear()
            self.lsm_path = None
            self.lsm_filesList = None
            self.lsm_folder = None
            return 0

    def open_settings(self):
        """
        Open the Settings window for adjusting channel settings.
        
        Notes:
            This method is responsible for opening the Settings window, allowing the user to adjust 
            channel settings such as choosing different channels for analysis or modifying parameters. 
            It checks if there are LSM files available and sets appropriate callback functions based on 
            whether there are LSM files in the list or not. If a valid LSM path is found, it creates an 
            instance of the SettingsWindow class and displays it modally, blocking other windows until 
            it's closed. If any error occurs during the process, a warning dialog is displayed to 
            notify the user about the error.
        """

        try:
           # Check if there are LSM files in the list
            if self.lsm_filesList:
               # If LSM files are present, set the LSM path and callback function for table creation
               lsm_path = self.lsm_filesList
                #TODO переделать callback так что бы он зависил от плагина
               call_back = self.plugin_list[self.current_plugin_name]["folder_callback"]
            else:
               # If no LSM files in the list, set the callback function for image change
               call_back = self.plugin_list[self.current_plugin_name]["file_callback"]
               lsm_path = self.lsm_path

            # If there's a valid LSM path
            if lsm_path:
                # Create an instance of SettingsWindow
                dialog = SettingsWindow(
                    parent=self, lsm_path=lsm_path, parametrs=self.parametrs, call_back=call_back)

                # Set window modality to block other windows until this one is closed
                dialog.setWindowModality(Qt.ApplicationModal)

                # Show the dialog window
                dialog.show()

                # Center the dialog window on the screen
                dialog.center()
        except:
            traceback.print_exc()
            # If an error occurs, show a warning dialog
            self.show_warning_dialog("Error during opening channels settings")
    
    def create_no_image_qimage(self, width=400, height=300):
        """
        Create a QImage with red bold "NO IMAGE" text.
        
        Args:
            width (int): Width of the image in pixels (default: 400)
            height (int): Height of the image in pixels (default: 300)
            
        Returns:
            QImage: A QImage object with red bold "NO IMAGE" text centered on white background
        """
        # Create a QImage with white background
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(QColor(255, 255, 255))  # White background
        
        # Create a QPainter to draw on the image
        painter = QPainter(image)
        
        try:
            # Set up the font - bold and large
            font = QFont()
            font.setBold(True)
            font.setPointSize(max(16, min(width, height) // 15))  # Scale font size based on image size
            painter.setFont(font)
            
            # Set the text color to red
            painter.setPen(QColor(255, 0, 0))  # Red color
            
            # Draw the text centered in the image
            text = "NO IMAGE"
            text_rect = painter.fontMetrics().boundingRect(text)
            
            # Calculate center position
            x = (width - text_rect.width()) // 2
            y = (height - text_rect.height()) // 2 + text_rect.height()
            
            # Draw the text
            painter.drawText(x, y, text)
            
        finally:
            # Always end the painter
            painter.end()
        
        return image