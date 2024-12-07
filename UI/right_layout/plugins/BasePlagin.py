from PyQt5.QtCore import QObject, pyqtSignal


class BasePlugin(QObject):
    """
    Base class for application plugins (CellProcessor and Tracker).
    Implements general functionality and architectural decisions.
    """
    def get_name(self):
        raise NotImplementedError
    plugin_signal = pyqtSignal(str, object)

    def __init__(self, handel_plugin_signal, *arg):
        super().__init__()
        self.plugin_signal.connect(handel_plugin_signal)
        self.init_value(*arg)
        try:
            self.init_rightLayout()
        except:
            self.plugin_signal.emit("error",None)

    def handle_action(self, action_name, value):
        raise NotImplementedError

    def init_value(self):
        raise NotImplementedError

    def init_rightLayout(self):
        raise NotImplementedError
