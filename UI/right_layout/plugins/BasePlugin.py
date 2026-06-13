"""
Base plugin module providing the shared plugin interface.

Defines BasePlugin, the parent class for right-layout plugins. It standardizes
signal handling, initialization behavior, and required plugin methods.
"""

# Standard library imports
from typing import Protocol, Any

# Third-party imports
from PyQt5.QtCore import QObject, pyqtSignal


class SignalLike(Protocol):
    """Protocol for signal-like objects used in testing and production."""
    
    def emit(self, *args: Any) -> None:
        """Emit a signal with given arguments."""
        ...
    
    def connect(self, handler: Any) -> None:
        """Connect a handler to the signal."""
        ...


class BasePlugin(QObject):
    """
    Base class for application plugins (CellProcessor and Tracker).
    Implements general functionality and architectural decisions.
    """
    def get_name(self):
        """
        Get the name of the plugin.
        
        Returns:
            str: Plugin name.
        """
        raise NotImplementedError
    
    plugin_signal: Any = pyqtSignal(str, object)

    def __init__(self, handel_plugin_signal, *arg):
        """
        Initialize the base plugin.
        
        Args:
            handel_plugin_signal: Signal handler function.
            *arg: Additional arguments.
        """
        super().__init__()
        self.plugin_signal.connect(handel_plugin_signal)
        self.init_value(*arg)
        try:
            self.init_rightLayout()
        except Exception:
            self.plugin_signal.emit("error",None)

    def handle_action(self, action_name, value):
        """
        Handle an action.
        
        Args:
            action_name: Name of the action.
            value: Value associated with the action.
        """
        raise NotImplementedError

    def init_value(self):
        """
        Initialize plugin values.
        """
        raise NotImplementedError

    def init_rightLayout(self):
        """
        Initialize the right layout for the plugin.
        """
        raise NotImplementedError
