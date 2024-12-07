"""
This module defines the dialog window for selecting channels in the Cells Calculator application.
"""
import os
import traceback
import tifffile
from PyQt5.QtWidgets import QGraphicsPixmapItem,\
        QMessageBox, QPushButton, QGraphicsView, QMainWindow, QGraphicsView, QGraphicsScene, \
                QVBoxLayout, QWidget, QGraphicsTextItem, QComboBox, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage


def has_duplicates(lst):
    """
    Check if a list contains duplicate elements.

    Args:
    lst (list): The list to be checked.

    Returns:
    bool: True if the list contains duplicates, False otherwise.
    """
    seen = set()
    for item in lst:
        if item in seen:
            return True
        seen.add(item)
    return False
class SettingsWindow(QMainWindow):
    """
    The DialogWindow class represents the dialog window for selecting channels
    in the Cells Calculator application.

    Args:
        parametrs (dict): A dictionary containing parameters.
        lsm_path (str): Path to the LSM image.
        parent: The parent widget.
        call_back: The callback function.

    Attributes:
        warning_count (int): Count of warning messages.
        first_warning (int): Flag indicating the first warning message.
        lsm_list (list): List of LSM images.
        parametrs (dict): Parameters dictionary.
        number (int): Number indicating the current image.
        num_channels (int): Number of image channels.
    """
    def show_warning_dialog(self, text):
        """
        Display a warning dialog.

        Args:
            text (str): The warning message text.
        """
        # Create a QMessageBox for displaying the warning
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(text)

        msgBox.setWindowTitle("Warning")
        msgBox.adjustSize()
        msgBox.exec_()
        # Check if it's the first warning and a callback function is defined
        if self.first_warning and self.call_back:
            self.first_warning = 0
            self.show_warning_dialog("If you have problem to choose Channel press Cancel\
                                     \n\n It will calculate with standart parameters")
    def __init__(self, parametrs: dict, lsm_path: str, parent=None, call_back=None):
        """
        Initialize the DialogWindow.

        Args:
            parametrs (dict): Dictionary containing parameters.
            lsm_path (str): Path to the LSM file or list of paths if multiple files are selected.
            parent: Parent widget.
            call_back: Callback function.
        """
        super().__init__(parent)

        # Initialize instance variables
        self.parent_ = parent
        self.warning_count = 0
        self.first_warning = 1
        self.call_back = call_back
        self.setStyleSheet('''
            QWidget {
                font-size: 32px;
            }
        ''')
        self.lsm_list = None
        self.parametrs = parametrs
        self.number = 0

        # If multiple LSM files are selected, use the first one
        if isinstance(lsm_path, list):
            self.lsm_path = lsm_path[0]
            self.lsm_list = lsm_path
        else:
            self.lsm_path = lsm_path

        self.num_channels = 0
        self.setWindowTitle("Dialog Window")
        self.initUI()

    def initUI(self):
        """
        Initialize the user interface.
        """
        # Calculate dimensions based on parent size
        self.parent_width = self.parent().width()
        self.parent_height = self.parent().height()

        # Set fixed size for the dialog window
        self.setFixedSize(int(self.parent_width * 0.75), int(self.parent_height * 0.75))

        # Set window title with basename of the LSM path
        self.setWindowTitle(f'Settings - {os.path.basename(self.lsm_path)}')

        # Create the scene
        self.makeScene()

    def makeScene(self):
        """
        Create the scene and layout for the dialog window.
        
        Note:
        - This method is responsible for creating the main scene and layout of the dialog window.
        - It sets up the QGraphicsView for displaying images and the layout for selecting channels.
        - Combo boxes are created for each parameter option to choose the corresponding channel.
        - Buttons for canceling, choosing, and navigating between images are also added to the layout.
        """
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout()
        self.scene = QGraphicsScene()

        # Create QGraphicsView for the scene
        self.view = QGraphicsView(self.scene)
        self.view.setFixedWidth(int(self.parent_width * 0.75 * 0.75))
        self.view.setFixedHeight(int(self.parent_height * 0.70))

        # Create layout for right side widgets
        right_layout = QVBoxLayout()
        self.combo_box_dict = None

        try:
            self.add_images()
        except:
            traceback.print_exc()
            
            # Handle error during layout image
            if self.warning_count == 0:
                # self.show_warning_dialog("Error during layout images\n\nUse Next button")
                self.show_warning_dialog(msg)
                self.warning_count = 1
            if not self.call_back and self.warning_count == 0:
                self.show_warning_dialog("Error during layout images")
                self.warning_count = 1
            self.num_channels = 5
            self.scene.clear()

        options = list(self.parametrs.keys())

        # Determine the maximum number of channels from parameters
        for option in options:
            self.num_channels = max(self.num_channels, self.parametrs[option] + 1)

        self.combo_box_dict = self.parametrs.copy()

        # Create combo boxes for channel selection
        for option in options:
            label = QLabel(option)
            combo_box = QComboBox()
            combo_box.addItems([f'Channel {i + 1}' for i in range(self.num_channels)])
            combo_box.setCurrentText(f"Channel {self.parametrs[option] + 1}")
            self.combo_box_dict[option] = combo_box
            label_combo_layout = QHBoxLayout()
            label_combo_layout.addWidget(label)
            label_combo_layout.addWidget(combo_box)
            right_layout.addLayout(label_combo_layout)

        # Create layout for buttons
        buttons_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_action)
        choose_button = QPushButton("Choose")
        choose_button.clicked.connect(self.choose_function)
        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(choose_button)
        right_layout.addLayout(buttons_layout)

        # Create layout for navigation buttons
        buttons_next_layout = QHBoxLayout()
        next_button = QPushButton("Next")
        if not self.lsm_list:
            next_button.setEnabled(False)
        next_button.clicked.connect(self.next_action)
        buttons_next_layout.addWidget(next_button)
        right_layout.addLayout(buttons_next_layout)

        # Add main view and right layout to the main layout
        main_layout.addWidget(self.view)
        main_layout.addLayout(right_layout)
        central_widget.setLayout(main_layout)

    def cancel_action(self):
        """
        Handle the cancel action.

        Note:
        - This method is called when the cancel button is clicked.
        - If a callback function is provided, it is invoked.
        - The dialog window is closed.
        """
        if self.call_back:
            self.call_back()
        self.close()
    def choose_function(self):
        """
        Handle the choose function.

        Note:
        - This method is called when the choose button is clicked.
        - It updates the parameters based on the selected channels.
        - If there are duplicated channels, it shows a warning dialog and returns.
        - If a callback function is provided, it is invoked.
        - The dialog window is closed.
        """
        self.parent_.draw_bounding = 0
        options_list = []
        for option, combo_box in self.combo_box_dict.items():
            options_list.append(int(combo_box.currentText()[len("Channel "):]) - 1)
        if has_duplicates(options_list):
            self.show_warning_dialog("Channels cannot be duplicated")
            return
        for option, combo_box in self.combo_box_dict.items():
            self.parametrs[option] = int(combo_box.currentText()[len("Channel "):]) - 1
        #For recalculation
        self.parent_.mainWindow_signal.emit("reset_detection", None)
        if self.call_back:
            self.call_back()
        self.close()

    def next_action(self):
        """
        Handle the next action.

        Note:
        - This method is called when the next button is clicked.
        - It increments the image number and updates the image path accordingly.
        - If the end of the image list is reached, it goes back to the first image.
        - It attempts to add the images to the scene.
        - If there's an error during layout, it shows a warning dialog.
        - The window title is updated with the new image path.
        """
        self.number += 1
        if self.number == len(self.lsm_list):
            self.number = 0
        self.lsm_path = self.lsm_list[self.number]
        try:
            self.add_images()
        except:
            traceback.print_exc()
            if self.warning_count == 0:
                self.warning_count = 1
                self.show_warning_dialog("Error during layout images\n\nUse Next button")
            self.scene.clear()
        self.setWindowTitle(f'Settings - {os.path.basename(self.lsm_path)}')

    def add_images(self):
        """
        Add images to the scene.

        Note:
        - This method clears the scene.
        - It reads the LSM file using tifffile.
        - It determines the number of channels in the LSM file.
        - If combo_box_dict is not None, it adjusts the combo box items based on the number of channels.
        - It calculates image dimensions for display on the screen.
        - It iterates over each channel, creates a QPixmap, and adds it to the scene.
        - It adds a text item "Channel X" below each image.
        """
        self.scene.clear()
        with tifffile.TiffFile(self.lsm_path) as tif:
            lsm = tif.pages[0].asarray()
        self.num_channels = lsm.shape[0]
        if self.combo_box_dict:
            options = list(self.parametrs.keys())
            for option in options:
                current_amount = self.combo_box_dict[option].count()
                if len(self.combo_box_dict[option]) < self.num_channels:
                    self.combo_box_dict[option].addItems([f'Channel {i}' \
                        for i in range(current_amount + 1, self.num_channels + 1)])

        # Image dimensions for display
        image_width = int(self.parent_width * 0.75 * 0.75 / 2)
        image_height = int(self.parent_height * 0.75 / 2)

        for i in range(self.num_channels):
            pixmap = QPixmap.fromImage(
                QImage(lsm[i], lsm.shape[1], lsm.shape[2], QImage.Format_Grayscale8))
            pixmap_item = QGraphicsPixmapItem(pixmap)
            current_image_height = (i // 2) * (image_height + 40)
            pixmap_item.setPos((i % 2) * image_width, current_image_height)
            pixmap_item.setScale(min(image_width / pixmap.width(), image_height / pixmap.height()))
            self.scene.addItem(pixmap_item)

            # Add text "Channel X" below each image
            text_item = QGraphicsTextItem(f"Channel {i + 1}")
            text_item.setPos((i % 2) * (image_width + 20) + image_width / 4,\
                current_image_height + image_height)
            self.scene.addItem(text_item)

    def center(self):
        """
        Center the dialog window relative to its parent widget.
        """
        parent_rect = self.parent().frameGeometry()
        dialog_rect = self.frameGeometry()
        dialog_rect.moveCenter(parent_rect.center())
        self.move(dialog_rect.topLeft())
