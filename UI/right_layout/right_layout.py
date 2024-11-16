import sys
from PyQt5.QtWidgets import QAbstractItemView, QCheckBox,QGraphicsPixmapItem,\
    QSizePolicy, QGraphicsProxyWidget, QGraphicsRectItem, QHeaderView, \
        QMessageBox, QTableWidget, QTableWidgetItem, QPushButton, QGraphicsView,\
            QApplication, QMainWindow, QAction, QGraphicsView, QGraphicsScene, \
                QVBoxLayout, QWidget, QFileDialog, QGraphicsTextItem, QComboBox, \
                    QLabel, QHBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPen
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot
import numpy as np
import tifffile


import os
import shutil

import pyqtgraph as pg
from UI.Slider import Slider
from UI.SettingsWindow import SettingsWindow
from model.Model import Model as model1
from UI.table import calculate_table


import traceback

class right_layout(QVBoxLayout):
    rightLayout_signal = pyqtSignal(str, object)
    current_plugin_name = None
    def __init__(self, current_plugin_name, plugin_list):
        super().__init__()
        self.current_plugin = None
        self.current_plugin_name = current_plugin_name
        self.plugin_list = plugin_list
        self.init_rightLayout()
        

    def clear(self):
        while self.count():
            item = self.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                
    def set_current_plugin(self, plugin_name, plugin_list):
        self.current_plugin_name = plugin_name
        self.plugin_list = plugin_list
        self.clear()
        del self.current_plugin
        self.init_rightLayout()

    def init_rightLayout(self):
        plugin =  self.plugin_list[self.current_plugin_name]['init']
        arg = self.plugin_list[self.current_plugin_name]['arg']
        self.current_plugin = plugin(self, *arg)
        self.current_plugin.plugin_signal.connect(self.handel_plugin_signal)

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
        if action_name == "plugin":
            self.set_current_plugin(plugin_name = value)


