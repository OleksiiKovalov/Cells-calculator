from PyQt5.QtWidgets import QAction, QFileDialog, QMenuBar
from PyQt5.QtCore import pyqtSignal, pyqtSlot

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
