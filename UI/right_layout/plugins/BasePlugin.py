# Third-party imports
from PyQt5.QtCore import QObject, pyqtSignal


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
    
    plugin_signal = pyqtSignal(str, object)

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
        except:
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
