"""
Centralized module imports with progress feedback.

Handles sequential loading of all application dependencies with
progress bar updates through the splash screen. Organizes imports
by category (system, path handling, data structures, GUI, etc.) to
provide clear visibility into initialization state.

Usage:
    Import this module early in application startup to ensure all
    dependencies are loaded with user feedback.
"""

from UI.app_globals import get_registered_model
from UI.splashscreen import splash_manager
curpc = 25

splash_manager.update_splash(curpc, "Loading System and OS...")
curpc += 1
import os
import sys
import shutil
import tempfile
import threading
import traceback
import io
import glob
import logging
import math
import string
import time
import re
from datetime import datetime

splash_manager.update_splash(curpc, "Loading Path and File Handling...")
curpc += 1
from pathlib import Path, PureWindowsPath, PurePosixPath
from contextlib import redirect_stdout, redirect_stderr

splash_manager.update_splash(curpc, "Loading Collections and Data Structures...")
curpc += 1
from collections import OrderedDict
import json
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from pyparsing import Optional

# Windows-specific imports (conditional for cross-platform compatibility)
try:
    import win32api
except ImportError:
    win32api = None
    
try:
    import winsound
except ImportError:
    winsound = None

## Third-Party Scientific Libraries

splash_manager.update_splash(curpc, "Loading NumPy and Data Processing...")
curpc += 1
import numpy as np
import pandas as pd
import cv2  # OpenCV for image processing

splash_manager.update_splash(curpc, "Loading Image Processing...")
curpc += 1

import tifffile
import tiffile  # Alternative tiff library used in some modules
from skimage.io import imread, imsave
from skimage.color import rgb2gray, gray2rgb
from skimage.measure import regionprops
from skimage.transform import resize

splash_manager.update_splash(curpc, "Loading Scientific Computing...")
curpc += 1

from scipy.ndimage import find_objects
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

splash_manager.update_splash(curpc, "Loading Machine Learning and Deep Learning...")
curpc += 1

splash_manager.update_splash(curpc, "Loading torch...")
curpc += 1
import torch


# Machine Learning Libraries
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

splash_manager.update_splash(curpc, "Loading YOLO Models...")
curpc += 1

r = get_registered_model('yolo')
if r is not None and r['preload'] is True:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results, Masks
    splash_manager.update_splash(curpc, "Loading SAHI Models...")
    curpc += 1
    from sahi.utils.cv import read_image
    from sahi.predict import get_sliced_prediction
    from sahi.auto_model import AutoDetectionModel

# Additional deep learning frameworks (optional)
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

# Optional evaluation tools
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None
    COCOeval = None

# Optional advanced packages
try:
    import importlib_metadata
except ImportError:
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        importlib_metadata = None

try:
    import IPython
except ImportError:
    IPython = None

try:
    import fiftyone as fo
except ImportError:
    fo = None

try:
    import imantics
except ImportError:
    imantics = None

try:
    import skimage.io
except ImportError:
    pass  # Already imported individual modules above

### Specialized AI Libraries
splash_manager.update_splash(curpc, "Loading Cellpose Models...")
curpc += 1
r = get_registered_model('cellpose')
if r is not None and r['preload'] is True:
    from cellpose import models as cp_models

splash_manager.update_splash(curpc, "Loading InstanSeg Models...")
curpc += 1
r = get_registered_model('instanseg')
if r is not None and r['preload'] is True:
    from instanseg import InstanSeg
    from instanseg.utils.utils import labels_to_features

r = get_registered_model('stardist')
if r is not None and r['preload'] is True:
    splash_manager.update_splash(curpc, "Loading tensorflow...")
    curpc += 1
    import tensorflow as tf
    splash_manager.update_splash(curpc, "Loading Stardist Models...")
    curpc += 1
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize


splash_manager.update_splash(curpc, "Loading Geometry Processing...")
curpc += 1

from shapely.geometry import shape

## PyQt5 GUI Framework

splash_manager.update_splash(curpc, "Loading GUI Framework...")
curpc += 1

### Core PyQt5 Modules
from PyQt5.QtCore import (
    Qt, QObject, QTimer, QDir, QFileInfo, 
    QAbstractTableModel, QModelIndex, QEvent, QDateTime,
    pyqtSignal, pyqtSlot
)

### PyQt5 GUI Components
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QFont, QColor, QPen, 
    QLinearGradient, QBrush, QIcon, QKeyEvent
)


splash_manager.update_splash(curpc, "Loading PyQt5 Widgets...")
curpc += 1

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QCheckBox, QRadioButton, QButtonGroup,
    QComboBox, QLineEdit, QSlider, QProgressBar,
    QTableWidget, QTableWidgetItem, QAbstractItemView,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem,
    QMenuBar, QAction, QFileDialog,
    QListWidget, QListWidgetItem, QDialogButtonBox,
    QSplashScreen, QTabWidget, QGroupBox, QScrollArea,
    QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit,
    QSplitter
)

## Project-Specific Local Imports

splash_manager.update_splash(curpc, "Loading UI Components...")
curpc += 1

from UI.splashscreen import *
from UI.errorhandling import *
from UI.app_globals import *
from UI.settings_manager import *
from UI.SettingsWindow import *
from UI.menubar import *
from UI.table import *
from UI.Slider import *
from UI.rangeslider import *
from UI.CustomFileDialog import *
from UI.ModelsCheckList import *
from UI.ImageNormalizeDialog import *
from UI.WaitWindow import *

splash_manager.update_splash(curpc, "Loading UI Layout Components...")
curpc += 1

from UI.right_layout.right_layout import *
from UI.right_layout.plugins.BasePlugin import *
from UI.right_layout.plugins.CellDetectorPlugin import CellDetectorPlugin as CellDetector_plugin
from UI.right_layout.plugins.TrackerPlugin import TrackerPlugin as Tracker_plugin
from UI.right_layout.plugins.SpheroidSegmenterPlugin import *

splash_manager.update_splash(curpc, "Loading Model Components...")
curpc += 1
from model.BaseModel import BaseModel
from model.Model import Model

splash_manager.update_splash(curpc, "Loading Model Utilities and SAHI...")
curpc += 1

from model.utils import *

# Additional imports from InstanSeg utilities
from instanseg.utils.utils import export_to_torchscript

splash_manager.update_splash(curpc, "Import loading complete!")
curpc += 1

# Note: This file consolidates all imports used across the Cells-Calculator project
# Some imports are wrapped in try/except blocks for optional dependencies
# that may not be available in all environments.
# Generated on: 2025-10-06
# Total modules covered: All core application files (excluding SAHI library internals)
