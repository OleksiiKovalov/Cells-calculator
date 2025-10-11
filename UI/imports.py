from  UI.splashscreen import update_splash
curpc = 25

update_splash(curpc, "Loading system modules...")
curpc += 1
from datetime import datetime
import os
import shutil
import traceback
import tifffile
import json
import cv2

update_splash(curpc, "Loading system modules...")
curpc += 1
from pathlib import Path,PureWindowsPath, PurePosixPath
from pyparsing import Optional
from typing import Optional
import torch

update_splash(curpc, "Loading system modules...")
curpc += 1
import os
import numpy as np
from model.BaseModel import BaseModel

update_splash(curpc, "Loading system modules...")
curpc += 1
from model.utils import *
from skimage.color import rgb2gray

update_splash(curpc, "Loading system modules...")
curpc += 1
import pandas as pd
import cv2  # OpenCV for findContours

update_splash(curpc, "Loading system modules...")
curpc += 1
from scipy.ndimage import find_objects  # For efficient bounding box calculation
from typing import Optional, List, Tuple, Dict, Any # For type hinting

update_splash(curpc, "Loading system modules...")
curpc += 1
import numpy as np
import pandas as pd

update_splash(curpc, "Loading system modules...")
curpc += 1
from PyQt5.QtWidgets import (QAbstractItemView, QMessageBox, QTableWidget, QTableWidgetItem,
    QGraphicsView, QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QWidget, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QCheckBox, QPushButton, QGraphicsView, QGraphicsView, QGraphicsScene,\
    QGraphicsTextItem, QComboBox, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction, QFileDialog, QMenuBar
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtCore import QFileInfo, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QAction

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.SettingsWindow import SettingsWindow

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.menubar import menubar

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.table import calculate_table

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.right_layout.right_layout import right_layout

update_splash(curpc, "Loading CellCounter modules..")
curpc += 1
from model.CellCounter import CellCounter

update_splash(curpc, "Loading NucleiCounter modules..")
curpc += 1
from model.NucleiCounter import NucleiCounter

update_splash(curpc, "Loading Segmenter modules..")
curpc += 1
from model.segmenter import Segmenter

update_splash(curpc, "Loading system modules..")
curpc += 1
from model.utils import is_image_valid, calculate_lsm

update_splash(curpc, "Loading CellposeSegmenter modules..")
curpc += 1
from model.CellposeSegmenter import CellposeSegmenter

update_splash(curpc, "Loading InstansegSegmenter modules..")
curpc += 1
from model.InstanSegSegmenter import InstansegSegmenter

update_splash(curpc, "Loading StardistSegmenter modules..")
curpc += 1
from model.StardistSegmenter import StardistSegmenter

update_splash(curpc, "Loading CellDetector modules..")
curpc += 1
from UI.right_layout.plugins.CellDetector import CellDetector as CellDetector_plugin

update_splash(curpc, "Loading Model modules..")
curpc += 1
from model.Model import Model
from UI.errorhandling import app_logger
from model.utils import create_image_grid

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.right_layout.plugins.tracker import Tracker as Tracker_plugin

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.Slider import Slider

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.right_layout.plugins.BasePlagin import BasePlugin

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.errorhandling import app_logger

update_splash(curpc, "Loading system modules...")
curpc += 1
from UI.CustomFileDialog import CustomFileDialog

update_splash(curpc, "Loading system modules...")
curpc += 1
from model.utils import COLOR_NUMBER as color_number

update_splash(curpc, "Loading system modules...")
curpc += 1
from model.utils import create_image_grid

update_splash(curpc, "Loading system modules...")
curpc += 1
