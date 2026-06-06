"""
Right layout module for managing plugin display and lifecycle.

This module provides the right-side layout container used in the application UI.
It handles plugin switching, layout clearing, and recursive cleanup of widgets
and nested layouts.
"""

# Third-party imports
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QVBoxLayout


class right_layout(QVBoxLayout):
    """
    Right layout for the application, managing plugins.
    """
    rightLayout_signal = pyqtSignal(str, object)
    current_plugin_name = None

    ACTION_SHOW_WARNING = "show_warning"
    ACTION_CHANGE_PLUGIN = "change_plugin"
    ACTION_OPEN_LSM = "open_lsm"
    ACTION_OPEN_FOLDER = "open_folder"

    def __init__(self, current_plugin_name, plugin_list):
        """
        Initialize the right layout.
        
        Args:
            current_plugin_name: Name of the current plugin.
            plugin_list: List of available plugins.
        """
        super().__init__()
        self.current_plugin = None
        self.current_plugin_name = current_plugin_name
        self.plugin_list = plugin_list

    def clear(self):
        """
        Clear the layout by removing all widgets and sub-layouts.
        """
        while self.count():
            item = self.takeAt(0)  # Extract item from layout
            widget = item.widget()  # Check if there is an associated widget
            if widget is not None:
                widget.setParent(None)  # Remove from parent layout
                widget.deleteLater()  # Mark for deletion
            else:
                # If it's another layout, clear recursively
                layout = item.layout()
                if layout is not None:
                    self._clear_layout(layout)

    def _clear_layout(self, layout):
        """
        Recursively clear a layout.
        
        Args:
            layout: The layout to clear.
        """
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            else:
                sub_layout = item.layout()
                if sub_layout is not None:
                    self._clear_layout(sub_layout)

    def set_current_plugin(self, plugin_name, plugin_list):
        """
        Set the current plugin.
        
        Args:
            plugin_name: Name of the plugin to set.
            plugin_list: List of plugins.
        """
        self.current_plugin_name = plugin_name
        self.plugin_list = plugin_list
        self.clear()
        self.current_plugin = None
        self.init_rightLayout()

    def init_rightLayout(self):
        """
        Initialize the right layout with the current plugin.
        """
        plugin_config = self.plugin_list.get(self.current_plugin_name)
        if plugin_config is None:
            raise ValueError(f"Unknown plugin: {self.current_plugin_name!r}")

        plugin = plugin_config["init"]
        arg = plugin_config["arg"]
        self.current_plugin = plugin(self.handel_plugin_signal, self, *arg)

    @pyqtSlot(str, object)
    def handle_mainWindow_action(self, action_name, value):
        """
        Handle actions from the main window.
        
        Args:
            action_name: Name of the action.
            value: Value associated with the action.
        """
        if self.current_plugin is None:
            return

        self.current_plugin.handle_action(action_name, value)
    # TODO: signal with current_plugin
    @pyqtSlot(str, object)
    def handel_plugin_signal(self, action_name, value):
        """
        Handle signals from the plugin.
        
        Args:
            action_name: Name of the action.
            value: Value associated with the action.
        """
        if action_name == self.ACTION_SHOW_WARNING:
            self.rightLayout_signal.emit(self.ACTION_SHOW_WARNING, value)
        elif action_name:
            self.rightLayout_signal.emit(action_name, value)

    @pyqtSlot(str, object)
    def handle_menubar_action(self, action_name, value):
        """
        Handle actions from the menubar.
        
        Args:
            action_name: Name of the action.
            value: Value associated with the action.
        """
        if action_name == self.ACTION_CHANGE_PLUGIN and value in self.plugin_list:
            self.set_current_plugin(plugin_name=value, plugin_list=self.plugin_list)
            return

        if self.current_plugin is None:
            return

        self.current_plugin.handle_action(action_name, value)
