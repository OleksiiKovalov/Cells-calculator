"""
Application menu bar with file, settings, and plugin menus.

Defines the menubar class which provides the main menu bar for the
application with standard menus (File, Settings) and dynamic plugin
selection. Emits signals for menu actions to communicate with the
main application window.

Key components:
- menubar: QMenuBar subclass with customized menus and actions
"""

# Standard library imports
from pathlib import Path
from typing import Optional, List, Dict

# Third-party imports
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QFileInfo
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QAction, QFileDialog, QMenuBar, QDialog

# Local application imports
from UI.CustomFileDialog import CustomFileDialog
from UI.settings_manager import get_setting, set_setting

class MenuBar(QMenuBar):
    """
    Menu bar for the application with file, settings, and plugin menus.
    """

    # Menu and action labels
    LABEL_FILE = "File"
    LABEL_SETTINGS = "Settings"
    LABEL_PLUGIN = "plugin"
    LABEL_OPEN_IMAGE = "Open Image"
    LABEL_OPEN_FOLDER = "Open Folder"
    LABEL_SAVE_AS = "Save As"
    LABEL_SETTINGS_ACTION = "Settings"
    LABEL_NORMALIZE = "normalize"

    # Keyboard shortcuts
    SHORTCUT_OPEN_FILE = "Ctrl+O"

    menubar_signal = pyqtSignal(str, object)

    def __init__(self, parent, plugin_list: List[str], current_plugin_name: str) -> None:
        """
        Initialize the menu bar.

        Args:
            parent: Parent widget.
            plugin_list: List of available plugins.
            current_plugin_name: Name of the current plugin.
        """
        super().__init__()
        self.parent = parent
        self.plugin_list = plugin_list
        self.current_plugin_name = current_plugin_name
        self.init_menubar()

    def init_menubar(self) -> None:
        """
        Initialize the menu bar with menus and actions.
        """
        file_menu = self.addMenu(self.LABEL_FILE)
        settings_menu = self.addMenu(self.LABEL_SETTINGS)
        plugin_menu = self.addMenu(self.LABEL_PLUGIN)
        
        self.open_lsm_action = QAction(self.LABEL_OPEN_IMAGE, self)
        self.open_lsm_action.triggered.connect(self.open_file)
        self.open_lsm_action.setShortcut(self.SHORTCUT_OPEN_FILE)
        self.open_lsm_action.setEnabled(False)

        self.open_folder_action = QAction(self.LABEL_OPEN_FOLDER, self)
        self.open_folder_action.triggered.connect(self.open_folder)
        self.open_folder_action.setEnabled(False)

        self.save_as_action = QAction(self.LABEL_SAVE_AS, self)
        self.save_as_action.setEnabled(False)
        self.save_as_action.triggered.connect(self.save_as)

        file_menu.addAction(self.open_lsm_action)
        file_menu.addAction(self.open_folder_action)
        file_menu.addAction(self.save_as_action)

        self.settings_action = QAction(self.LABEL_SETTINGS_ACTION, self)
        self.settings_action.setEnabled(False)
        self.settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(self.settings_action)

        self.normalize_action = QAction(self.LABEL_NORMALIZE, self)
        self.normalize_action.setEnabled(False)
        self.normalize_action.triggered.connect(self.open_normalize)
        settings_menu.addAction(self.normalize_action)

        self.plugin_actions = {}

        # Add plugins to menu
        for plugin in self.plugin_list:
            action = QAction(plugin, self, checkable=True)
            action.triggered.connect(self.select_plugin)
            plugin_menu.addAction(action)
            self.plugin_actions[plugin] = action
            if plugin == self.current_plugin_name:
                action.setChecked(True)

    def select_plugin(self) -> None:
        """
        Handle plugin selection from the menu.
        """
        # Get the action that triggered the signal
        action = self.sender()
        if action and action.isCheckable():
            # Reset state of all actions
            for act in self.plugin_actions.values():
                act.setChecked(False)
            # Set state of selected action
            action.setChecked(True)
            self.current_plugin_name = action.text()
            self.menubar_signal.emit("change_plugin", self.current_plugin_name)

    @pyqtSlot(str, object)
    def handle_mainWindow_action(self, action_name: str, value: object) -> None:
        """
        Handle actions from the main window.

        Args:
            action_name: Name of the action.
            value: Value associated with the action.
        """
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
    def handle_rightLayout_action(self, action_name: str, value: object) -> None:
        """
        Handle actions from the right layout.

        Args:
            action_name: Name of the action.
            value: Value associated with the action.
        """
        if action_name == "Open_lsm":
            self.open_lsm_action.setEnabled(value)
        elif action_name == "Open_folder":
            self.open_folder_action.setEnabled(value)
        elif action_name == "Settings":
            self.settings_action.setEnabled(value)
        elif action_name == "Save_as":
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
            parent_object = self.parent()
            if parent_object is not None and hasattr(parent_object, 'image_mru'):
                image_mru = getattr(parent_object, 'image_mru')
                val = image_mru.get(filename)
                if val is not None:
                    if val.year == 1:
                        return QColor(0, 150, 0)  # Green for recently opened files
                    return QColor(0, 0, 150)  # Blue for processed files
        return None  # Use default color

    def open_file(self) -> None:
        """
        Open file dialog for selecting an image file.
        """
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
        
        dialog.openAt(get_setting("paths.last_opened_file", ""), True)
        
        if dialog.exec_() == QDialog.Accepted:
            set_setting("paths.last_opened_file", str(dialog.get_selected_file()))
            selected_file =  str(Path(dialog.get_selected_file()))
            self.menubar_signal.emit("open_file", str(selected_file))
        dialog = None

    def open_folder(self) -> None:
        """
        Open folder dialog for selecting a directory.
        """
        folder_path = QFileDialog.getExistingDirectory(
            caption="Open Folder", directory=""
        )
        if folder_path:
            self.menubar_signal.emit("open_folder", folder_path)

    def open_settings(self) -> None:
        """
        Emit signal to open settings.
        """
        self.menubar_signal.emit("open_settings", None)

    def save_as(self) -> None:
        """
        Emit signal to save as.
        """
        self.menubar_signal.emit("save_as", None)

    def open_normalize(self) -> None:
        """
        Emit signal to open normalize dialog.
        """
        self.menubar_signal.emit("open_normalize", None)
