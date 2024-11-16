from PyQt5.QtWidgets import QAction, QFileDialog, QMenuBar
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPen
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot

class menubar(QMenuBar):
    menubar_signal = pyqtSignal(str, object)

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_menubar()

    def init_menubar(self):
        file_menu = self.addMenu("File")
        settings_menu = self.addMenu("Settings")

        # Add action to the 'Settings' menu
        
        self.open_lsm_action = QAction("Open Image", self)
        self.open_lsm_action.triggered.connect(self.open_file)

        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        self.settings_action = QAction("Settings", self)
        self.settings_action.setEnabled(False)
        self.settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(self.settings_action)

        self.save_as_action = QAction("Save As", self)
        self.save_as_action.setEnabled(False)
        self.save_as_action.triggered.connect(self.save_as)


        # Add actions to the 'File' menu
        file_menu.addAction(self.open_lsm_action)
        file_menu.addAction(self.open_folder_action)
        file_menu.addAction(self.save_as_action)

      

    @pyqtSlot(str, object)
    def handle_mainWindow_action(self, action_name, value):
        if action_name == "open_lsm":
            if value:
                self.settings_action.setEnabled(True)
                self.save_as_action.setEnabled(False)
            else:
                self.settings_action.setEnabled(False)
                self.save_as_action.setEnabled(False)

        elif action_name == "open_image":
            self.settings_action.setEnabled(False)
            self.save_as_action.setEnabled(False)
            

        elif action_name == "open_folder":
            if value:
                self.settings_action.setEnabled(True)
                self.save_as_action.setEnabled(True)
            else:
                self.settings_action.setEnabled(False)
                self.save_as_action.setEnabled(False)
        elif action_name == "error_save_as":
            self.save_as_action.setEnabled(False)

    @pyqtSlot(str, object)
    def handle_rightLayout_action(self, action_name, value):
        pass
        

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            caption="Open Image File",
            directory="",
            filter="Image Files (*.png *.jpg *.bmp *.lsm *.TIF)",
        )
        if file_path:
            self.menubar_signal.emit("open_file", file_path)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            caption="Open Folder", directory=""
        )
        if folder_path:
            self.menubar_signal.emit("open_folder", folder_path)

    def open_settings(self):
        self.menubar_signal.emit("open_settings", None)

    def save_as(self):
        self.menubar_signal.emit("save_as", None)
