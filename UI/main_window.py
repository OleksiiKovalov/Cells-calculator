"""
This module defines the MainWindow class for the Cells Calculator application.
The application is designed to open, process, and analyze cell images.
"""
import sys
from PyQt5.QtWidgets import QAbstractItemView, QCheckBox,QGraphicsPixmapItem,\
    QSizePolicy, QGraphicsProxyWidget, QGraphicsRectItem, QHeaderView, \
        QMessageBox, QTableWidget, QTableWidgetItem, QPushButton, QGraphicsView,\
            QApplication, QMainWindow, QAction, QGraphicsView, QGraphicsScene, \
                QVBoxLayout, QWidget, QFileDialog, QGraphicsTextItem, QComboBox, \
                    QLabel, QHBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPen
from PyQt5.QtCore import Qt
import numpy as np
import tifffile
from UI.cahnal_setting import DialogWindow
from model.Model import Model as model1
from UI.table import calculate_table
import os
import shutil

import pyqtgraph as pg
from UI.Slider import Slider

import traceback

class MainWindow(QMainWindow):
    """
    The MainWindow class represents the main window of the Cells Calculator application.
    This class handles the initialization of the UI, loading and processing images, 
    and interacting with various models to perform cell calculations.
    """
    
    object_size = { 'min_size' : 100,
                   'max_size' : 0.000,
                   'set_size' : 'set_size',
                   'round_parametr_slider' : 10**6,
                   'round_parametr_value_input' : 10**4
                   
    }
    default_object_size = object_size.copy()
    # Default parameters for cell and nuclei channels
    parametrs = {'Cell': 0,
                 'Nuclei': 1
    }

    # Dictionary containing available models and their corresponding methods
    models = {
        'Detector': model1(path='model/yolov8m-det.onnx', object_size = object_size),
        'General Segmenter': model1(path='model/yolov8n-seg.pt', object_size = object_size)
    }

    # Initialize variables to None or default values
    lsm_path = None
    lsm_filesList = None
    folder_path = None
    show_boundry = 0
    draw_bounding = 0
    
    def __init__(self):
        """
        Initialize the MainWindow.

        This method sets up the initial state of the main window.
        """
        self.object_size['set_size'] = self.set_size
        self.default_object_size = self.object_size.copy()
        super().__init__()
        # Add cache directory creation
        try:
            # Remove the existing cache directory if it exists
            shutil.rmtree('.cache', ignore_errors=True)  # Ignore errors if directory is empty
        except OSError as e:
            pass
        # Create a new cache directory if it doesn't exist
        os.makedirs('.cache', exist_ok=True)

        # Get desktop information
        desktop = QApplication.desktop()
        screen_geometry = desktop.availableGeometry()

        # Set the fixed size of the main window to match the screen width and height minus the height of the menu bar
        self.setFixedSize(screen_geometry.width(), desktop.availableGeometry().height() - self.menuBar().height())

        # Set the window title
        self.setWindowTitle("Cells Calculator")

        # Initialize the user interface
        self.initUI()


    def initUI(self):
        """
        Initialize the user interface, create menu actions and main scene.

        Notes:
        - Initialize DataFrame to None.
        - Create menu actions for opening image, opening folder, settings, and saving as.
        - Create 'File' and 'Settings' menus.
        - Add actions to the 'File' and 'Settings' menus.
        - Initialize the main scene.
        """
        # Initialize DataFrame to None
        self.df = None
        
        # Create menu actions
        self.open_lsm_action = QAction("Open Image", self)
        self.open_lsm_action.triggered.connect(self.open_file)

        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        self.settings_action = QAction("Settings", self)
        self.settings_action.setEnabled(False)
        self.settings_action.triggered.connect(self.open_dialogWindow)

        self.save_as_action = QAction("Save As", self)
        self.save_as_action.setEnabled(False)
        self.save_as_action.triggered.connect(self.save_as)
        
        # Create a menu bar
        menubar = self.menuBar()

        # Add 'File' and 'Settings' menus to the menu bar
        file_menu = menubar.addMenu("File")
        settings_menu = menubar.addMenu("Settings")

        # Add actions to the 'File' menu
        file_menu.addAction(self.open_lsm_action)
        file_menu.addAction(self.open_folder_action)
        file_menu.addAction(self.save_as_action)

        # Add action to the 'Settings' menu
        settings_menu.addAction(self.settings_action)

        # Initialize the main scene
        self.init_mainScene()


    def init_mainView(self):
        """
        Initialize the main view area.
        """
        self.main_view.setFixedWidth(int(self.width() * 0.75))
        
    def save_as(self):
        """
        Save the current table data to a file.

        Notes:
        - Check if there is data in the DataFrame.
        - If there's no data, show a warning dialog and disable the "Save As" action.
        - Prompt the user to choose the file name and type.
        - Check if the user provided a file name.
        - If the file name ends with '.xlsx', save as Excel file.
        - If the file name ends with '.csv', save as CSV file.
        - If an error occurs during saving, show a warning dialog and disable the "Save As" action.
        """
        # Check if there is data in the DataFrame
        if self.df is None:
            # If there's no data, show a warning dialog
            self.show_warning_dialog("Nothing to Save")
            
            # Disable the "Save As" action
            self.save_as_action.setEnabled(False)
            return
        
        try:
            # Prompt the user to choose the file name and type
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)")
            
            # Check if the user provided a file name
            if file_name:
                # If the file name ends with '.xlsx', save as Excel file
                if file_name.endswith('.xlsx'):
                    self.df.to_excel(file_name, index=False)
                # If the file name ends with '.csv', save as CSV file
                elif file_name.endswith('.csv'):
                    self.df.to_csv(file_name, index=False)
        except Exception as e:
            traceback.print_exc()
            # If an error occurs during saving, show a warning dialog
            self.show_warning_dialog("Error during saving table")
            
            # Disable the "Save As" action
            self.save_as_action.setEnabled(False)
            return


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
        # Create a combo box to choose models
        self.combo_box = QComboBox()
        
        # Create a label to prompt user to choose a model
        label = QLabel("Choose model:")
        
        # Create a graphics scene for the right view
        self.right_scene = QGraphicsScene()

        # Create a graphics view for the right scene
        self.right_view = QGraphicsView(self.right_scene)
        
        # Align the graphics view to the top-left corner
        self.right_view.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Create a button for calculating
        self.right_button = QPushButton("Calculate")
        self.right_button.setEnabled(False)

        # Set font for the combo box
        self.combo_box.setFont(QFont("Arial", 24))
        
        # Add 'All_method' option and method names to the combo box
        self.combo_box.addItems(['All_models'])
        self.combo_box.addItems([key for key in self.models])
        
        # Set current index to 1
        self.combo_box.setCurrentIndex(1)
        
        # Set font for the label
        label.setFont(QFont("Arial", 32))
        
        # Create a checkbox for showing detected cells
        self.checkbox = QCheckBox("Show Detected Cells", self)
        
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
        self.right_button.setFont(QFont("Arial", 32))
        
        # Connect button click event to calculate_button function
        self.right_button.clicked.connect(self.calculate_button)
        
        # Add widgets to the right layout with spacing
        self.right_layout.addWidget(label)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.combo_box)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.right_view)
        self.right_layout.addSpacing(20)
        range_lable = QLabel("Object Size:")
        font = QFont()
        font.setPointSize(16) 
        range_lable.setFont(font)
        self.right_layout.addWidget(range_lable)
        #self.right_layout.addSpacing(1)
        self.min_range_slider = Slider(self.object_size, self.default_object_size, 'min_size')
        self.max_range_slider = Slider(self.object_size, self.default_object_size, 'max_size')
        
        self.right_layout.addWidget(self.min_range_slider)
        self.right_layout.addWidget(self.max_range_slider)


        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.checkbox)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.right_button)
        self.right_layout.addSpacing(20)

     
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
        self.init_mainView()

        # Create a vertical layout for the widgets on the right side
        self.right_layout = QVBoxLayout()
        self.init_rightLayout()

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
        if self.lsm_filesList:
            self.create_table()
            return
        # Get the selected method from the combo box
        model = self.combo_box.currentText()
        
        # Check if a method and file are selected
        if model == "" or self.lsm_path is None:
            # If not, show a warning dialog and return
            self.show_warning_dialog("Warning\n\nChoose model and file.")
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
                    msg = traceback.format_exc()
                    # If still not successful, show an error dialog
                    # self.show_warning_dialog(
                    #     "Error during calculation \n\nChoose another model or change channels settings")
                    self.show_warning_dialog(
                        msg)
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
        self.main_scene.clear()
        
        try:
            # Check if the show boundary flag is set
            if self.show_boundry:
                # If set, add an image with bounding box detections to the scene
                self.add_image(".cache\cell_tmp_img_with_detections.png")
            else:
                # If not set, add the original image to the scene
                self.add_image(".cache\cell_tmp_img.png")
        except Exception as e:
            # If an error occurs, print the traceback, show a warning dialog
          
            traceback.print_exc()
            self.show_warning_dialog("Error during opening image.")


    def open_folder(self):
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
        self.reset_detection()
        # Open a dialog window to select a folder
        self.folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", "")

        # If a folder is selected
        if self.folder_path:
            # Create a list of image files within the selected folder
            self.lsm_filesList = [os.path.join(self.folder_path, file) \
                for file in os.listdir(self.folder_path)\
                    if file.lower().endswith(('.png', '.jpg', '.bmp', '.lsm', '.tif'))]

            # If image files are found
            if self.lsm_filesList:
                # Clear the main scene
                self.main_scene.clear()
                #self.max_range_slider.set_default()
                #self.min_range_slider.set_default()
                # Reset certain variables
                self.lsm_path = None
                self.draw_bounding = 0
                
                # Enable certain actions
                self.settings_action.setEnabled(True)
                self.save_as_action.setEnabled(True)
                self.right_button.setEnabled(False)
                
                # Open a dialog window
                self.open_dialogWindow()

                # Set the window title to include the selected folder name
                self.setWindowTitle(f"Cells Calculator - {os.path.basename(self.folder_path)}/")
            else:
                # If no image files are found, reset variables and show a warning dialog
                self.folder_path = None
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
        # Checks if there are any files in the list
        if not self.lsm_filesList:
            return
        
        # Clears the main scene
        self.main_scene.clear()
        
        try:
            # Attempts to create a table widget
            table = QTableWidget()
            
            try:
                current_model = self.combo_box.currentText()
                if current_model == "All_models":
                    models = self.models
                else:
                    models = {current_model: self.models[current_model]} if current_model in self.models else {}
                # Attempts to calculate table data using given methods, files, and parameters
                df = calculate_table(
                    model_dict=models, files_name=self.lsm_filesList, parametrs=self.parametrs)
            except Exception as e:
                traceback.print_exc()
                # If an exception occurs during calculation disables certain actions,
                # resets file list and data frame, and shows a warning dialog
                self.settings_action.setEnabled(False)
                self.save_as_action.setEnabled(False)
                self.right_button.setEnabled(False)
                self.lsm_filesList = None
                self.show_warning_dialog("Error during calculation. \nTry another Model")
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
            self.right_button.setEnabled(True)
            
        except:
            traceback.print_exc()
            # If an exception occurs during the process, disables certain actions,
            # resets file list and data frame, and shows a warning dialog
            self.settings_action.setEnabled(False)
            self.save_as_action.setEnabled(False)
            self.right_button.setEnabled(False)
            self.lsm_filesList = None
            self.df = None
            self.show_warning_dialog("Error during creating table. \nTry another Model")

    def add_image(self, lsm_file):
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
        # self.models['Model 1'].cell_counter.detections = None
        # Check if the input is a string (file path)
        if isinstance(lsm_file, str):
            # If it's a string, create a QImage from the file path
            image = QImage(lsm_file)
        else:
            # If it's not a string, assume it's a numpy array representing an image
            # and create a QImage from it with grayscale format
            image = QImage(lsm_file[self.parametrs['Cell']], lsm_file.shape[1],
                        lsm_file.shape[2], QImage.Format_Grayscale8)
        
        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(image)

        # Get the dimensions of the main view
        view_width = self.main_view.viewport().width()
        view_height = self.main_view.viewport().height()

        # Calculate the aspect ratio of the image
        pixmap_aspect_ratio = pixmap.width() / pixmap.height()

        # Scale the image to fit within the view while maintaining aspect ratio
        if view_width / view_height > pixmap_aspect_ratio:
            pixmap_width = view_height * pixmap_aspect_ratio
            pixmap_height = view_height
        else:
            pixmap_width = view_width
            pixmap_height = view_width / pixmap_aspect_ratio
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

    def open_file(self):
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
        self.reset_detection()
        # Open a dialog window to select an image file
        lsm_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.lsm *.TIF)")
        
        # If no file is selected, return
        if not lsm_path:
            return 0
        #After opening new image set sliders to default
        #self.max_range_slider.set_default()
        #self.min_range_slider.set_default()
        # If the selected file is an LSM file, call the open_lsm function
        if lsm_path.endswith(".lsm"):
            self.open_lsm(lsm_path)
        else:
            # If the selected file is not an LSM file
            # Store the file path
            self.lsm_path = lsm_path
            
            # Clear the main scene
            self.main_scene.clear()
            
            # Disable certain actions
            self.settings_action.setEnabled(False)
            self.save_as_action.setEnabled(False)
            self.right_button.setEnabled(True)
            
            # Reset file list
            self.lsm_filesList = None
            
            try:
                # Try to add the image to the scene and set the window title
                self.add_image(self.lsm_path)
                self.setWindowTitle(
                    f"Cells Calculator - {os.path.basename(lsm_path)}")
            except Exception as e:
                traceback.print_exc()
                # If an error occurs, show a warning dialog,
                # reset variables, and clear the main scene
                self.show_warning_dialog("Error during opening file.")
                self.setWindowTitle(f"Cells Calculator")
                self.lsm_path = None
                self.main_scene.clear()
                self.settings_action.setEnabled(False)
                self.save_as_action.setEnabled(False)
                self.right_button.setEnabled(False)
                self.lsm_filesList = None
                return 0


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
        
        # Enable certain actions
        self.settings_action.setEnabled(True)
        self.save_as_action.setEnabled(False)
        self.right_button.setEnabled(True)
        
        # Reset the file list
        self.lsm_filesList = None
        
        try:
            # Attempt to open the LSM file using tifffile
            with tifffile.TiffFile(self.lsm_path) as tif:
                # Read the first page of the LSM file as an array
                lsm_file = tif.pages[0].asarray()
            
            # Check if the number of channels in the LSM file is less than the maximum channel index specified in parameters
            if lsm_file.shape[0] < max([value+1 for key, value in self.parametrs.items()]):
                # If so, reset the parameters to default
                self.parametrs = {'Cell': 0,
                                'Nuclei': 1
                                }
            
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
            self.lsm_path = None
            self.main_scene.clear()
            self.settings_action.setEnabled(False)
            self.save_as_action.setEnabled(False)
            self.right_button.setEnabled(False)
            self.lsm_filesList = None
            return 0

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


    def open_dialogWindow(self):
        """
        Open the Settings window for adjusting channel settings.
        
        Notes:
            This method is responsible for opening the Settings window, allowing the user to adjust 
            channel settings such as choosing different channels for analysis or modifying parameters. 
            It checks if there are LSM files available and sets appropriate callback functions based on 
            whether there are LSM files in the list or not. If a valid LSM path is found, it creates an 
            instance of the DialogWindow class and displays it modally, blocking other windows until 
            it's closed. If any error occurs during the process, a warning dialog is displayed to 
            notify the user about the error.
        """
        try:
            # Check if there are LSM files in the list
            if self.lsm_filesList:
                # If LSM files are present, set the LSM path and callback function for table creation
                lsm_path = self.lsm_filesList
                call_back = self.create_table
            else:
                # If no LSM files in the list, set the callback function for image change
                call_back = self.change_image
                lsm_path = self.lsm_path
            
            # If there's a valid LSM path
            if lsm_path:
                # Create an instance of DialogWindow
                dialog = DialogWindow(
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

    def reset_detection(self):
        for key, model in self.models.items():
            model.cell_counter.detections = None
    def set_size(self, detection : list = [], img_size : tuple = (512,512), min_size : float = default_object_size["min_size"], max_size : float = default_object_size["max_size"]):
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
        
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        
        window = MainWindow()
        window.showMaximized()
        sys.exit(app.exec_())
        
    except Exception as e:
        QMessageBox.critical(None, "Critical Error", str(e), QMessageBox.Ok)
        sys.exit(1)
