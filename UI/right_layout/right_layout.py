from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtCore import pyqtSignal, pyqtSlot


class right_layout(QVBoxLayout):
    rightLayout_signal = pyqtSignal(str, object)
    current_plugin_name = None
    def __init__(self, current_plugin_name, plugin_list):
        super().__init__()
        self.current_plugin = None
        self.current_plugin_name = current_plugin_name
        self.plugin_list = plugin_list

    def clear(self):
        while self.count():
            item = self.takeAt(0)  # Извлекаем элемент из макета
            widget = item.widget()  # Проверяем, есть ли связанный виджет
            if widget is not None:
                widget.setParent(None)  # Убираем из родительского макета
                widget.deleteLater()  # Помечаем на удаление
            else:
                # Если это другой макет, очищаем рекурсивно
                layout = item.layout()
                if layout is not None:
                    self._clear_layout(layout)

    def _clear_layout(self, layout):
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
        self.current_plugin_name = plugin_name
        self.plugin_list = plugin_list
        self.clear()
        del self.current_plugin
        self.init_rightLayout()

    def init_rightLayout(self):
        plugin =  self.plugin_list[self.current_plugin_name]['init']
        arg = self.plugin_list[self.current_plugin_name]['arg']
        self.current_plugin = plugin(self.handel_plugin_signal, self, *arg)

    @pyqtSlot(str, object)
    def handle_mainWindow_action(self, action_name, value):
        self.current_plugin.handle_action(action_name, value)
        if action_name == "open_lsm":
            pass 
    #TODO: сигнал с current_plugin
    @pyqtSlot(str, object)
    def handel_plugin_signal(self, action_name, value):
        if action_name == "show_warning":
            self.rightLayout_signal.emit("show_warning", value)
        elif action_name == "":
            pass
        else:
            self.rightLayout_signal.emit(action_name, value)
        pass

    @pyqtSlot(str, object)
    def handle_menubar_action(self, action_name, value):
        if action_name == "change_plugin":
            #self.set_current_plugin(plugin_name = value)
            pass
        else:
            self.current_plugin.handle_action(action_name, value)
