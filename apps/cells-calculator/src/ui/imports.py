"""Pre-load the heavy third-party libraries at startup.

main() shows the splash screen and then imports this module, so the user sees a
progress bar advance while the large scientific / deep-learning dependencies
warm up.  Nothing imports names *from* this module — its only job is to pull the
slow packages into ``sys.modules`` (and update the splash) before the first
inference runs.  Model backends are only preloaded when enabled and flagged
``preload`` in modelconfig.json.
"""

from ui.app_globals import get_registered_model
from ui.splashscreen import update_splash

update_splash(0, "Loading NumPy and data processing...")
import numpy as np
import pandas as pd
import cv2

update_splash(0, "Loading image processing...")
import tifffile
from skimage import io, color, measure, transform

update_splash(0, "Loading scientific computing...")
import scipy.ndimage
import matplotlib.pyplot as plt

update_splash(0, "Loading PyTorch...")
import torch

r = get_registered_model('yolo')
if r is not None and r['preload'] == "true":
    update_splash(0, "Loading YOLO models...")
    from ultralytics import YOLO

r = get_registered_model('cellpose')
if r is not None and r['preload'] == "true":
    update_splash(0, "Loading Cellpose models...")
    from cellpose import models as cp_models

r = get_registered_model('instanseg')
if r is not None and r['preload'] == "true":
    update_splash(0, "Loading InstanSeg models...")
    from instanseg import InstanSeg

r = get_registered_model('stardist')
if r is not None and r['preload'] == "true":
    update_splash(0, "Loading TensorFlow...")
    import tensorflow as tf
    update_splash(0, "Loading StarDist models...")
    from stardist.models import StarDist2D

update_splash(0, "Loading geometry processing...")
import shapely.geometry

update_splash(0, "Loading model components...")
from model.BaseSegmenter import BaseSegmenter
from model.Model import Model

update_splash(0, "Startup complete!")
