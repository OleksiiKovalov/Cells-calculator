import sys
from PyQt5.QtWidgets import QAbstractItemView, QCheckBox,QGraphicsPixmapItem,\
    QSizePolicy, QGraphicsProxyWidget, QGraphicsRectItem, QHeaderView, \
        QMessageBox, QTableWidget, QTableWidgetItem, QPushButton, QGraphicsView,\
            QApplication, QMainWindow, QAction, QGraphicsView, QGraphicsScene, \
                QVBoxLayout, QWidget, QFileDialog, QGraphicsTextItem, QComboBox, \
                    QLabel, QHBoxLayout, QSlider, QLineEdit
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPen
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot
import numpy as np
import tifffile

import os
import shutil

import pyqtgraph as pg
from UI.Slider import Slider


import traceback

class BasePlugin(QObject):
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