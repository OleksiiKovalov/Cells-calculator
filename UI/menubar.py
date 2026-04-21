# Standard library imports
from pathlib import Path
from typing import Optional

# Third-party imports
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QFileInfo
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QAction, QFileDialog, QMenuBar, QDialog

# Local application imports
from UI.CustomFileDialog import CustomFileDialog
from UI.settings_manager import get_setting, set_setting

class menubar(QMenuBar):
    menubar_signal = pyqtSignal(str, object)

    def __init__(self, parent, plugin_list, current_plugin_name):
        super().__init__()
        self.parent = parent
        self.plugin_list = plugin_list
        self.current_plugin_name = current_plugin_name
        self.init_menubar()

    def init_menubar(self):
        file_menu = self.addMenu("File")
        settings_menu = self.addMenu("Settings")
        plugin_menu = self.addMenu("plugin")
        
        self.open_lsm_action = QAction("Open Image", self)
        self.open_lsm_action.triggered.connect(self.open_file)
        self.open_lsm_action.setShortcut("Ctrl+O") 
        self.open_lsm_action.setEnabled(False)

        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)
        self.open_folder_action.setEnabled(False)

        self.save_as_action = QAction("Save As", self)
        self.save_as_action.setEnabled(False)
        self.save_as_action.triggered.connect(self.save_as)

        file_menu.addAction(self.open_lsm_action)
        file_menu.addAction(self.open_folder_action)
        file_menu.addAction(self.save_as_action)

        self.settings_action = QAction("Settings", self)
        self.settings_action.setEnabled(False)
        self.settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(self.settings_action)

        self.normalize_action = QAction("normalize", self)
        self.normalize_action.setEnabled(False)
        self.normalize_action.triggered.connect(self.open_normalize)
        settings_menu.addAction(self.normalize_action)

        self.plugin_actions = {}

        # Добавляем плагины в меню
        for plugin in self.plugin_list:
            action = QAction(plugin, self, checkable=True)
            action.triggered.connect(self.select_plugin)
            plugin_menu.addAction(action)
            self.plugin_actions[plugin] = action
            if plugin == self.current_plugin_name:
                action.setChecked(True)

    def select_plugin(self):
        # Получаем действие, вызвавшее сигнал
        action = self.sender()
        if action and action.isCheckable():
            # Сбрасываем состояние всех действий
            for act in self.plugin_actions.values():
                act.setChecked(False)
            # Устанавливаем состояние выбранного действия
            action.setChecked(True)
            self.current_plugin_name = action.text()
            self.menubar_signal.emit("change_plugin", self.current_plugin_name)

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
                if self.current_plugin_name != "Tracker":
                    self.settings_action.setEnabled(True)
                #self.save_as_action.setEnabled(True)
            else:
                self.settings_action.setEnabled(False)
                self.save_as_action.setEnabled(False)
        elif action_name == "error_save_as":
            self.save_as_action.setEnabled(False)

    @pyqtSlot(str, object)
    def handle_rightLayout_action(self, action_name, value):
        if action_name == "Open_lsm":
            self.open_lsm_action.setEnabled(value)
                 
        if action_name == "Open_folder":
            self.open_folder_action.setEnabled(value)
          
        if action_name == "Settings":
           self.settings_action.setEnabled(value)

        if action_name == "Save_as":
            self.save_as_action.setEnabled(value)

    def get_process_time(self, file_info: QFileInfo) -> str:
        """Example function for process time column"""
        if file_info.isDir():
            return ""
        
        size = file_info.size()
        if size < 1024 * 1024:  # < 1MB
            return "~1s"
        elif size < 10 * 1024 * 1024:  # < 10MB
            return "~5s"
        elif size < 100 * 1024 * 1024:  # < 100MB
            return "~30s"
        else:
            return "~2m"
        
    def example_text_color_rule(self, file_info: QFileInfo) -> Optional[QColor]:
        """Example function to set text color based on file type"""
#        if file_info.isDir():
#            return QColor(100, 100, 100)  # Gray for directories

        if file_info.isFile():
            filename = str(Path(file_info.absoluteFilePath()))
            val = self.parent.image_mru.get(filename)
            if val is not None:
                if val.year == 1:
                    return QColor(0, 150, 0)  # Green for recently opened files
                return QColor(0, 0, 150)  # Blue for processed files
        return None  # Use default color

    def open_file(self):
        dialog = CustomFileDialog(
            caption="Select Image File",
            directory=Path.home(),
            parent=self.parent
        )
        dialog.add_custom_column("Process Time", 100, self.get_process_time)
        dialog.set_color_rule(self.example_text_color_rule)
        
        # Set file filters
        dialog.set_file_filters([
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
            "Documents (*.txt *.doc *.pdf)",
            "All files (*.*)"
        ])
        
        last_opened_file = get_setting("paths.last_opened_file","")
        dialog.openAt(get_setting("paths.last_opened_file",""),True)
        
        if dialog.exec_() == QDialog.Accepted:
            set_setting("paths.last_opened_file", str(dialog.get_selected_file()))
            selected_file =  str(Path(dialog.get_selected_file()))
            self.menubar_signal.emit("open_file", str(selected_file))
        dialog = None

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

    def open_normalize(self):
        self.menubar_signal.emit("open_normalize", None)
