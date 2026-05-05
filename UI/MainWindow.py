"""
MainWindow module for Cells Calculator application.

This module defines the MainWindow class that represents the main window of the Cells Calculator application.
The MainWindow class handles the initialization of the UI, loading and processing images, 
and interacting with various models to perform cell calculations.
"""

# Standard library imports
import os
import string
import traceback
from datetime import datetime

# Third-party imports
import numpy as np
import tifffile
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor, QWheelEvent
from PyQt5.QtWidgets import (
    QAbstractItemView, QMessageBox, QTableWidget, QTableWidgetItem,
    QGraphicsView, QApplication, QMainWindow, QGraphicsScene, 
    QWidget, QHBoxLayout, QSplitter, QStatusBar, QLabel
)
from skimage.io import imread

# Local application imports
from UI.app_globals import get_global
from UI.errorhandling import connect_to_log_events
from UI.ImageNormalizeDialog import ImageNormalizeDialog
from UI.menubar import menubar
from UI.right_layout.plugins.CellDetectorPlugin import CellDetectorPlugin as CellDetector_plugin
from UI.right_layout.plugins.TrackerPlugin import TrackerPlugin as Tracker_plugin
from UI.right_layout.right_layout import right_layout
from UI.SettingsWindow import SettingsWindow
from UI.table import calculate_table
from model.utils import (
    COLOR_NUMBER as color_number,
    clear_cache,
    lsm_to_channels_last,
    read_lsm_array,
    safergb2gray,
)


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
        #self.setFixedSize(screen_geometry.width(),desktop.availableGeometry().height() - self.menuBar().height())
        self.setMinimumSize(850, 600)

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
        self.init_status_bar()
        connect_to_log_events(self.on_log_line_added)        
        
        self._update_progress(100, "Ready!")
    
    def _update_progress(self, value, message):
        """Update progress during initialization"""
        if self.progress_callback:
            self.progress_callback(value, message)
        QApplication.processEvents()

    @pyqtSlot(str, object)
    def handle_menubar_action(self, action_name, value):
        """
        Handle actions from the menubar.
        """
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
        elif action_name == "filter_and_draw_predictions":
            self.filter_and_draw_predictions(get_global('predictions'), get_global('image_inference'))
        pass
    
    def open_normalize(self):
        image = imread(self.lsm_path)
        image = safergb2gray(image)
        dlg = ImageNormalizeDialog(image)
        dlg.exec_()        
        pass

    def init_value(self):
        # TODO: make parameters collect relative to the selected plugin
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
                   'scale' : 20,
                   'um_per_px' : 0.325
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
        
        self._update_progress(48, "Reading model configuration file...")
        loaded_models = get_global('loaded_models')
        
        self._update_progress(52, "Configuring models...")
        for model_name, model_data in loaded_models.items():
            # Set the 'object_size' parameter for each loaded model
            model_data['object_size'] = self.object_size
            self.models_celldetector[model_name] = model_data
                    
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
        - Create a horizontal splitter for the main scene and right layout widget.
        - Initialize the main scene and its view.
        - Create a widget container for the right layout.
        - Add both widgets to the splitter.
        - Set the splitter as the central widget's layout.
        - Set the scene rectangle to match the size of the main view.
        """
        # Create a central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a horizontal splitter for the main scene and other widgets
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Initialize the main scene and its view
        self.main_scene = QGraphicsScene()
        self.main_view = QGraphicsView(self.main_scene)
        self.main_view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        # Initialize zoom functionality
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 1.15
        
        # Initialize drag functionality for panning
        self.last_pan_point = None
        self.is_panning = False
        
        # Enable zoom with Ctrl+Mouse wheel and pan with left mouse drag
        self.main_view.wheelEvent = self._on_wheel_event
        self.main_view.mousePressEvent = self._on_mouse_press
        self.main_view.mouseMoveEvent = self._on_mouse_move
        self.main_view.mouseReleaseEvent = self._on_mouse_release
        
        # Set drag mode to NoDrag since we're handling dragging manually
        self.main_view.setDragMode(QGraphicsView.NoDrag)

        # Create a widget container for the right layout
        self.right_layout_widget = QWidget()
        self.right_layout_widget.setMaximumWidth(800)
        self.right_layout_widget.setMinimumWidth(300)
        self.right_layout_widget.setLayout(self.right_layout)

        # Add the main view and right layout widget to the splitter
        self.main_splitter.addWidget(self.main_view)
        self.main_splitter.addWidget(self.right_layout_widget)

        # Set stretch factors: main view can expand, right panel stays fixed-ish
        self.main_splitter.setStretchFactor(0, 1)  # main_view can stretch
        self.main_splitter.setStretchFactor(1, 0)  # right_layout_widget minimal stretch
        
        # Set initial splitter proportions (75% for main view, 25% for right panel)
        self.main_splitter.setSizes([750, 400])
        
        # Set collapsible behavior - right panel cannot be collapsed completely
        self.main_splitter.setCollapsible(0, False)   # main_view can be collapsed
        self.main_splitter.setCollapsible(1, False)  # right panel cannot be collapsed
        
        # Create a layout for the central widget and add the splitter
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for full space usage
        self.main_layout.addWidget(self.main_splitter)

        # Set the main layout for the central widget
        self.central_widget.setLayout(self.main_layout)

        # Set the scene rectangle to match the size of the main view
        self.main_scene.setSceneRect(0, 0, self.main_view.width()-10, self.main_view.height())
        
        # Connect resize event for automatic image scaling
        self.main_view.resizeEvent = self._on_main_view_resize

    def init_status_bar(self):
        """
        Initialize and configure the status bar.
        
        Creates a status bar with multiple sections:
        - Main status message
        - Current file information
        - Processing status
        - Application state
        """
        # Create the status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create status bar widgets
        self.status_message = QLabel("Ready")
        self.status_file = QLabel("No file loaded")
        self.status_processing = QLabel("Idle")
        
        # Add main status message (left side)
        self.status_bar.addWidget(self.status_message, 1)  # Stretch factor 1
        
        # Add separator and file info
        self.status_bar.addPermanentWidget(QLabel(" | "))
        self.status_bar.addPermanentWidget(self.status_file)
        
        # Add separator and processing status
        self.status_bar.addPermanentWidget(QLabel(" | "))
        self.status_bar.addPermanentWidget(self.status_processing)
        
        # Set initial status
        self.update_status("Application started successfully")

    def update_status(self, message, file_info=None, processing_status=None):
        """
        Update the status bar with new information.
        
        Args:
            message (str): Main status message
            file_info (str, optional): Current file information
            processing_status (str, optional): Processing status
        """
        if hasattr(self, 'status_message'):
            self.status_message.setText(self.remove_non_printable(message))
        
        if file_info is not None and hasattr(self, 'status_file'):
            self.status_file.setText(file_info)
            
        if processing_status is not None and hasattr(self, 'status_processing'):
            self.status_processing.setText(processing_status)
        QApplication.processEvents()

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

    def add_image(self, lsm_file):
        """
        Create and display an image in the main scene with automatic resizing.
        
        Args:
            lsm_file (str or numpy.ndarray): Path to image file or numpy array representing image.
                - str: File path to image (supports .png, .jpg, .bmp, .lsm, .tif)
                - numpy.ndarray: Image data array (for LSM files or processed images)
        
        Notes:
            - Automatically handles different image formats and sources
            - Scales image to fit main scene while maintaining aspect ratio
            - Centers image in the scene
            - Updates scene rect for proper resizing behavior
        """
        # Clear the current scene
        self.main_scene.clear()
        
        # Step 1: Create QImage from various sources
        image = self._create_qimage_from_source(lsm_file)
        
        # Step 2: Add image to scene with auto-resize functionality
        self._add_image_to_scene(image)
        
        # Update the view
        self.main_view.repaint()
        QApplication.processEvents()
    
    def _create_qimage_from_source(self, lsm_file):
        """
        Create a QImage from various source types.
        
        Args:
            lsm_file (str or numpy.ndarray): Image source
            
        Returns:
            QImage: Created image object
        """
        try:
            if isinstance(lsm_file, str):
                # Handle file path input
                if not os.path.exists(lsm_file):
                    # File doesn't exist - create "NO IMAGE" placeholder
                    return self.create_no_image_qimage()
                
                elif lsm_file.lower().endswith('.lsm'):
                    # Handle LSM files
                    return self._create_lsm_qimage(read_lsm_array(lsm_file))
                
                else:
                    # Handle regular image files (png, jpg, bmp, tif)
                    return QImage(lsm_file)
            
            else:
                # Handle numpy array input
                if isinstance(lsm_file, np.ndarray):
                    if len(lsm_file.shape) >= 3:
                        # Multi-channel array (LSM data)
                        return self._create_lsm_qimage(lsm_file)
                    else:
                        # Single channel 2D array
                        return self._create_grayscale_qimage(lsm_file)
                else:
                    # Unknown type - return placeholder
                    return self.create_no_image_qimage()
                    
        except Exception as e:
            # If any error occurs during image creation, return placeholder
            print(f"Error creating image: {e}")
            return self.create_no_image_qimage()

    def _create_lsm_qimage(self, lsm_array):
        channels_last = lsm_to_channels_last(lsm_array)
        cell_channel = self.parametrs['Cell']
        if channels_last.shape[-1] <= cell_channel:
            cell_channel = 0

        return self._create_grayscale_qimage(channels_last[:, :, cell_channel])

    def _create_grayscale_qimage(self, image_array):
        image_array = np.asarray(image_array)
        if image_array.ndim != 2:
            return self.create_no_image_qimage()

        if image_array.dtype != np.uint8:
            max_value = np.max(image_array) if image_array.size else 0
            if max_value > 0:
                image_array = image_array.astype(np.float32) / max_value * 255
            image_array = image_array.astype(np.uint8)

        image_array = np.ascontiguousarray(image_array)
        return QImage(
            image_array.data,
            image_array.shape[1],
            image_array.shape[0],
            image_array.strides[0],
            QImage.Format_Grayscale8,
        ).copy()
    
    def _add_image_to_scene(self, image):
        """
        Add QImage to scene with automatic scaling and centering.
        
        Args:
            image (QImage): Image to add to the scene
        """
        if image.isNull():
            image = self.create_no_image_qimage()
        
        # Store original image dimensions
        self.currentImageWidth = image.width()
        self.currentImageHeight = image.height()
        
        # Convert to pixmap
        original_pixmap = QPixmap.fromImage(image)
        
        # Get current view dimensions
        view_rect = self.main_view.viewport().rect()
        view_width = view_rect.width()
        view_height = view_rect.height()
        
        # Calculate scaled size while maintaining aspect ratio
        scaled_pixmap = self._scale_pixmap_to_fit(original_pixmap, view_width, view_height)
        
        # Add pixmap to scene
        self.current_pixmap_item = self.main_scene.addPixmap(scaled_pixmap)
        
        # Center the image in the scene
        self._center_image_in_scene(scaled_pixmap, view_width, view_height)
        
        # Update scene rect to match view size for proper resizing
        self.main_scene.setSceneRect(0, 0, view_width, view_height)
        
        # Store original pixmap for resizing
        self.original_image_pixmap = original_pixmap
        
        # Reset zoom when new image is loaded
        self.zoom_factor = 1.0
        self.main_view.resetTransform()
        self._update_zoom_status()  

    def _scale_pixmap_to_fit(self, pixmap, view_width, view_height):
        """
        Scale pixmap to fit within view while maintaining aspect ratio.
        
        Args:
            pixmap (QPixmap): Original pixmap
            view_width (int): Available width
            view_height (int): Available height
            
        Returns:
            QPixmap: Scaled pixmap
        """
        if pixmap.isNull() or view_width <= 0 or view_height <= 0:
            return pixmap
        
        # Calculate aspect ratios
        pixmap_ratio = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 1
        view_ratio = view_width / view_height if view_height > 0 else 1
        
        # Determine scaling dimensions
        if pixmap_ratio > view_ratio:
            # Image is wider - fit to width
            new_width = min(view_width, pixmap.width())
            new_height = int(new_width / pixmap_ratio)
        else:
            # Image is taller - fit to height
            new_height = min(view_height, pixmap.height())
            new_width = int(new_height * pixmap_ratio)
        
        # Scale the pixmap
        return pixmap.scaled(new_width, new_height, 
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    def _center_image_in_scene(self, pixmap, view_width, view_height):
        """
        Center the image pixmap in the scene.
        
        Args:
            pixmap (QPixmap): Pixmap to center
            view_width (int): View width
            view_height (int): View height
        """
        if hasattr(self, 'current_pixmap_item') and self.current_pixmap_item:
            # Calculate center position
            x_pos = max(0, (view_width - pixmap.width()) / 2)
            y_pos = max(0, (view_height - pixmap.height()) / 2)
            
            # Set position
            self.current_pixmap_item.setPos(x_pos, y_pos)
    
    def _on_main_view_resize(self, event):
        """
        Handle main view resize events to automatically rescale the image.
        
        Args:
            event: QResizeEvent
        """
        # Call the original resize event first
        QGraphicsView.resizeEvent(self.main_view, event)
        
        # If we have an image loaded, rescale it
        if hasattr(self, 'original_image_pixmap') and hasattr(self, 'current_pixmap_item'):
            if self.original_image_pixmap and self.current_pixmap_item:
                # Get new view dimensions
                view_rect = self.main_view.viewport().rect()
                view_width = view_rect.width()
                view_height = view_rect.height()
                
                # Rescale the original image to fit new dimensions
                scaled_pixmap = self._scale_pixmap_to_fit(
                    self.original_image_pixmap, view_width, view_height)
                
                # Update the pixmap item
                self.current_pixmap_item.setPixmap(scaled_pixmap)
                
                # Recenter the image
                self._center_image_in_scene(scaled_pixmap, view_width, view_height)
                
                # Update scene rect
                self.main_scene.setSceneRect(0, 0, view_width, view_height)
    
    def _on_wheel_event(self, event):
        """
        Handle mouse wheel events for zooming with Ctrl key.
        
        Args:
            event (QWheelEvent): The wheel event
        """
        # Check if Ctrl key is pressed
        if event.modifiers() & Qt.ControlModifier:
            # Get the wheel delta (positive for zoom in, negative for zoom out)
            delta = event.angleDelta().y()
            
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
                
            # Accept the event to prevent scrolling
            event.accept()
        else:
            # Call the default wheel event handler for normal scrolling
            QGraphicsView.wheelEvent(self.main_view, event)
    
    def zoom_in(self):
        """
        Zoom in the image view.
        """
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor *= self.zoom_step
            self._apply_zoom()
    
    def zoom_out(self):
        """
        Zoom out the image view.
        """
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor /= self.zoom_step
            self._apply_zoom()
    
    def zoom_to_fit(self):
        """
        Reset zoom to fit the image in the view.
        """
        self.zoom_factor = 1.0
        self.main_view.resetTransform()
        self._update_zoom_status()  

        # If we have an image, rescale it to fit
        if hasattr(self, 'original_image_pixmap') and self.original_image_pixmap:
            view_rect = self.main_view.viewport().rect()
            view_width = view_rect.width()
            view_height = view_rect.height()
            
            scaled_pixmap = self._scale_pixmap_to_fit(
                self.original_image_pixmap, view_width, view_height)
            
            if hasattr(self, 'current_pixmap_item') and self.current_pixmap_item:
                self.current_pixmap_item.setPixmap(scaled_pixmap)
                self._center_image_in_scene(scaled_pixmap, view_width, view_height)

    def _update_zoom_status(self):
        # Update cursor based on zoom level - show hand cursor when image is larger than view
        self._update_cursor_for_zoom()
        # Update status bar with zoom level
        zoom_percentage = int(self.zoom_factor * 100)
        if hasattr(self, 'update_status'):
            self.update_status(
                message=f"Zoom: {zoom_percentage}%",
                processing_status=f"Zoom: {zoom_percentage}%"
            )


    def _apply_zoom(self):
        """
        Apply the current zoom factor to the view.
        """
        # Reset transform and apply zoom
        self.main_view.resetTransform()
        self.main_view.scale(self.zoom_factor, self.zoom_factor)
        self._update_zoom_status()

    def _on_mouse_press(self, event):
        """
        Handle mouse press events for panning.
        
        Args:
            event (QMouseEvent): The mouse press event
        """
        if event.button() == Qt.LeftButton:
            # Start panning mode
            self.is_panning = True
            self.last_pan_point = event.pos()
            self.main_view.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            # Call the default mouse press event handler
            QGraphicsView.mousePressEvent(self.main_view, event)
    
    def _on_mouse_move(self, event):
        """
        Handle mouse move events for panning.
        
        Args:
            event (QMouseEvent): The mouse move event
        """
        if self.is_panning and self.last_pan_point is not None:
            # Calculate the delta movement
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            
            # Pan the view by adjusting the scrollbars
            h_scroll = self.main_view.horizontalScrollBar()
            v_scroll = self.main_view.verticalScrollBar()
            
            h_scroll.setValue(h_scroll.value() - delta.x())
            v_scroll.setValue(v_scroll.value() - delta.y())
            
            event.accept()
        else:
            # Call the default mouse move event handler
            QGraphicsView.mouseMoveEvent(self.main_view, event)
    
    def _on_mouse_release(self, event):
        """
        Handle mouse release events for panning.
        
        Args:
            event (QMouseEvent): The mouse release event
        """
        if event.button() == Qt.LeftButton and self.is_panning:
            # End panning mode
            self.is_panning = False
            self.last_pan_point = None
            self.main_view.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            # Call the default mouse release event handler
            QGraphicsView.mouseReleaseEvent(self.main_view, event)
    
    def _update_cursor_for_zoom(self):
        """
        Update the cursor based on current zoom level.
        Shows hand cursor when image is zoomed and can be panned.
        """
        if not self.is_panning:
            if self.zoom_factor > 1.0 and hasattr(self, 'current_pixmap_item'):
                # Image is zoomed in and might need panning - show open hand cursor
                self.main_view.setCursor(Qt.OpenHandCursor)
            else:
                # Normal zoom level - show default cursor
                self.main_view.setCursor(Qt.ArrowCursor)
    
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
        
        # Update status bar
        self.update_status("Opening file...", processing_status="Loading")

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
                
                # Update status bar with success
                self.update_status("File loaded successfully", 
                                 file_info=f"{os.path.basename(lsm_path)} ({self.currentImageWidth}x{self.currentImageHeight})",
                                 processing_status="Ready")
                                 
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
                
                # Update status bar with error
                self.update_status("Error loading file", file_info="No file", processing_status="Error")

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
                #TODO refactor callback to depend on plugin
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
    
    def on_log_line_added(self, log_line):
        """
        Handle log line added event.
        
        Args:
            log_line: The log line to display.
        """
        # Shrink the log line to fit status bar
        shortened_line = self.shrink_text(log_line, max_length=100)
        self.update_status(shortened_line)
    
    def shrink_text(self, text, max_length=100, separator="..."):
        """
        Shrink text to maximum length by replacing middle portion with separator.
        
        Args:
            text (str): Text to shrink
            max_length (int): Maximum allowed length (default: 100)
            separator (str): String to use as separator (default: "...")
            
        Returns:
            str: Shortened text with middle portion replaced by separator
            
        Examples:
            shrink_text("This is a very long text that needs to be shortened", 30)
            # Returns: "This is a very...o be shortened"
            
            shrink_text("Short text", 100)
            # Returns: "Short text" (unchanged if under limit)
        """
        if not isinstance(text, str):
            text = str(text)
            
        # If text is already short enough, return as-is
        if len(text) <= max_length:
            return text
        
        # If max_length is too small to accommodate separator, just truncate
        if max_length <= len(separator):
            return text[:max_length]
        
        # Calculate how much space we have for actual text
        available_space = max_length - len(separator)
        
        # Split available space between start and end
        # Give preference to the start (useful for file paths, log messages, etc.)
        start_length = (available_space + 1) // 2  # Add 1 to give start preference when odd
        end_length = available_space - start_length
        
        # Extract start and end portions
        start_part = text[:start_length]
        end_part = text[-end_length:] if end_length > 0 else ""
        
        # Combine with separator
        return start_part + separator + end_part

    def remove_non_printable(self,text):
        """Remove non-printable characters using string.printable"""
        printable = set(string.printable)
        return ''.join(char for char in text if char in printable)

    def filter_and_draw_predictions(self, image, predictions):
        """
        Filter and draw predictions on the image.
        
        Args:
            image: The image data.
            predictions: The predictions to draw.
        """
        mask_image = None
        self.add_image(mask_image)
        pass
