"""Some useful functions used by the model or its submodels."""

# Standard library imports
import math
import os
import shutil
from collections import OrderedDict
from typing import Any

# Third-party imports
import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import tiffile
import torch
from csbdeep.utils import normalize
from PyQt5.QtWidgets import QMessageBox
from skimage.color import gray2rgb, rgb2gray
from skimage.io import imsave
from skimage.transform import resize
from ultralytics.engine.results import Results

# Local application imports
from UI.app_globals import set_global
from UI.app_globals import (
    IMAGE_FILE_NAME_DETECTION,
    IMAGE_FILE_NAME_GRID,
    IMAGE_FILE_NAME_INGFERENCE,
    IMAGE_FILE_NAME_TMP,
    CASH_DIRECTORY
)



VALID_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'lsm']
CLASSES = ['Cell']
COLORS = [(3, 177, 252)]
COLOR_NUMBER = {
    "gist_rainbow": 20,
    "tab20": 20,
    "tab20b": 20,
    "tab20c": 20,
    "tab10": 10,
    "Set1": 9,
    "Set2": 8,
    "Set3": 12,
    "Paired": 12,
    "viridis": 10,
    "plasma": 10
}


def read_lsm_array(img_path):
    """Reads the primary LSM/TIFF series as an array."""
    with tiffile.TiffFile(img_path) as tif:
        try:
            return tif.series[0].asarray()
        except (IndexError, ValueError):
            return tif.pages[0].asarray()


def lsm_to_channels_last(image):
    """Normalizes LSM arrays to (height, width, channels)."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image[:, :, np.newaxis]

    image = np.squeeze(image)
    if image.ndim == 2:
        return image[:, :, np.newaxis]

    if image.ndim > 3:
        image = image.reshape((-1, image.shape[-2], image.shape[-1]))

    if image.ndim != 3:
        raise ValueError(f"Unsupported LSM image shape: {image.shape}")

    if image.shape[-1] <= 8 and image.shape[0] > 8 and image.shape[1] > 8:
        return image

    return np.transpose(image, (1, 2, 0))


def safe_image_read(img_path, color_mode='color', channel=None):
    """
    Standardized image reading function that handles various formats and edge cases.
    
    Args:
        img_path (str): Path to the image file
        color_mode (str): 'color', 'grayscale', 'unchanged' - how to read the image
        channel (int): Specific channel to extract (for multi-channel images like LSM)
    
    Returns:
        np.ndarray: Image array or None if reading failed
    """
    try:
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return None
            
        extension = img_path.split('.')[-1].lower()
        
        if extension == 'lsm':
            # Handle LSM files with tiffile
            image = read_lsm_array(img_path)
            if channel is not None:
                image = lsm_to_channels_last(image)
                if 0 <= channel < image.shape[-1]:
                    return image[:, :, channel]
                print(f"Channel {channel} not available in LSM file")
                return None
            return image
        else:
            # Handle standard image formats
            if color_mode == 'unchanged':
                cv_flag = cv2.IMREAD_UNCHANGED
            elif color_mode == 'grayscale':
                cv_flag = cv2.IMREAD_GRAYSCALE
            else:  # color
                cv_flag = cv2.IMREAD_COLOR
            
            # Use cv2.imdecode for Unicode path support
            img_array = np.fromfile(img_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv_flag)
            
            if image is None:
                print(f"Failed to read image: {img_path}")
                return None
                
            return image
            
    except Exception as e:
        print(f"Error reading image {img_path}: {str(e)}")
        return None


def safe_image_write(image, filename, quality=95, preserve_dtype=True):
    """
    Standardized image writing function that handles various formats and data types.
    
    Args:
        image (np.ndarray): Image array to save
        filename (str): Output file path
        quality (int): JPEG quality (for JPEG files)
        preserve_dtype (bool): Whether to preserve original data type or convert to uint8
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if image is None:
            print("Cannot save None image")
            return False

        filename = os.fspath(filename)
        image = np.asarray(image)
            
        # Ensure output directory exists
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        extension = filename.split('.')[-1].lower()
        
        # Handle data type conversion
        if not preserve_dtype or image.dtype != np.uint8:
            if image.dtype in [np.float32, np.float64]:
                # Normalize float images to 0-255 range
                if image.max() <= 1.0:
                    image_to_save = (image * 255).astype(np.uint8)
                else:
                    image_to_save = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image_to_save = image.astype(np.uint8)
        else:
            image_to_save = image
        
        # Save based on file extension
        write_success = False
        if extension in ['jpg', 'jpeg']:
            # JPEG with quality control
            write_success = cv2.imwrite(
                filename,
                image_to_save,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
        elif extension == 'png':
            # PNG with compression
            write_success = cv2.imwrite(
                filename,
                image_to_save,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
        elif extension == 'bmp':
            write_success = cv2.imwrite(filename, image_to_save)
        elif extension in ['tif', 'tiff']:
            # Use skimage for better TIFF support
            if len(image_to_save.shape) == 3 and image_to_save.shape[2] == 3:
                # Convert BGR to RGB for skimage
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_BGR2RGB)
            imsave(filename, image_to_save)
            write_success = os.path.exists(filename)
        else:
            print(f"Unsupported image extension: {extension}")
            return False
        
        return bool(write_success and os.path.exists(filename))
        
    except Exception as e:
        print(f"Error saving image {filename}: {str(e)}")
        return False

def read_lsm_img(img_path, cell_channel=0, nuclei_channel=1):
    """Reads lsm image and returns as array."""
    img = lsm_to_channels_last(read_lsm_array(img_path))
    if img.shape[-1] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    if img.shape[-1] == 2:
        stacked_array = np.dstack((img, np.zeros(img.shape[:2], dtype=img.dtype)))
        return stacked_array
    if img.shape[-1] == 3:
        return img

    empty_channel = np.zeros(img.shape[:2], dtype=img.dtype)
    cell = img[:, :, cell_channel] if 0 <= cell_channel < img.shape[-1] else img[:, :, 0]
    nuclei = img[:, :, nuclei_channel] if 0 <= nuclei_channel < img.shape[-1] else empty_channel
    stacked_array = np.dstack((cell, nuclei, empty_channel))
    return stacked_array

def read_standard_img(img_path):
    """Reads image in grayscale jpg/png/tif/bmp which contains cells only."""
    img = safe_image_read(img_path, color_mode='unchanged')
    if img is None:
        return None
    
    # Convert to grayscale and then to RGB for consistency
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def is_image_valid(img_path: str):
    """Checks if provided image is in correct format."""
    return img_path.split('.')[-1].lower() in VALID_IMAGE_EXTENSIONS

def read_img(img_path, cell_channel=0, nuclei_channel=1):
    """Reads any possible image of cells (and/or nuclei)."""
    if img_path.endswith('lsm'):
        return read_lsm_img(img_path, cell_channel, nuclei_channel)
    elif is_image_valid(img_path):
        return read_standard_img(img_path)


def extract_nuclei_channel(img_path, nuclei_channel=1):
    """Extracts the channel used for dead-cell counting from any supported image."""
    if img_path.endswith('lsm'):
        img = lsm_to_channels_last(read_lsm_array(img_path))
        if 0 <= nuclei_channel < img.shape[-1]:
            return img[:, :, nuclei_channel]
        return None

    img = safe_image_read(img_path, color_mode='unchanged')
    if img is None:
        return None

    if img.ndim == 2:
        return img

    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0]
        if 0 <= nuclei_channel < img.shape[2]:
            return img[:, :, nuclei_channel]
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return None

def count_detected_objects(detections) -> int:
    """Returns the number of detected objects for detector- and segmenter-style outputs."""
    if detections is None:
        return 0
    if hasattr(detections, "shape"):
        return int(detections.shape[0])
    return int(detections)


def calculate_alive_percentage(cell_count: int, nuclei_count: int):
    """Calculates alive percentage, returning -100 when it cannot be computed."""
    if cell_count <= 0 or nuclei_count == -100:
        return -100
    return round((1 - nuclei_count / cell_count) * 100, 3)

def _select_channel(image, channel, *, empty_on_missing=False):
    """Return a valid channel, falling back to channel 0 or an empty image."""
    try:
        channel = int(channel)
    except (TypeError, ValueError):
        channel = -1

    if image.ndim == 2:
        if empty_on_missing and channel not in (0, None):
            return np.zeros(image.shape, dtype=image.dtype)
        return image

    if 0 <= channel < image.shape[-1]:
        return image[:, :, channel]
    if empty_on_missing:
        return np.zeros(image.shape[:2], dtype=image.dtype)
    return image[:, :, 0]


def calculate_lsm(cell_counter, nuclei_counter,
                  img_path, cell_channel=0, nuclei_channel=1, nuclei_count=None):
    """
    Calculates the resulting target values.
    Input params are:
    - img_path: path to lsm/jpg/png/tif/bmp image;
    - cell_channel: channel with cells. Default to 0;
    - nuclei_channel: channel with stained nuclei. Default to 1.

    Returns the result as a dictionary with the following fields:
    - Nuclei: count for stained nuclei detected;
    - Cells: count for all the cells detected;
    - %: the target percentage for alive cells.
    """
    img = lsm_to_channels_last(read_lsm_array(img_path))

    cell_img = cv2.cvtColor(
        _select_channel(img, cell_channel),
        cv2.COLOR_GRAY2BGR
    )
    tmp_path = IMAGE_FILE_NAME_TMP
    safe_image_write(cell_img, tmp_path)
    cell_count = cell_counter.count_cells(tmp_path)
    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        pass
    if nuclei_count is None:
        nuclei_img = _select_channel(img, nuclei_channel, empty_on_missing=True)
        nuclei_count = nuclei_counter.countNuclei(nuclei_img)
    percentage = calculate_alive_percentage(
        count_detected_objects(cell_count),
        nuclei_count
    )
    return {'Nuclei': nuclei_count, 'Cells': cell_count, '%': percentage}



def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, draw_mode=0):
    """
    Draws bounding boxes on the input image.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        draw_mode (int): 0 for rectangle, other for circle.
    """
    color = COLORS[0]
    thickness = 1 if img.shape[0] < 800 else 2
    if draw_mode == 0:
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, thickness)
    else:
        cv2.circle(img, (x, y), 2, color, -1)

def filter_detections(
    detections: pd.DataFrame, 
    min_size: float = 0.0, 
    max_size: float = 1.0, 
    img_size: tuple = (512, 512)
) -> pd.DataFrame:
    """
    [DEPRECATED] Filters bounding boxes based on their area.
    No longer used - new inference pipeline implemented.
    
    Bboxes of size < min_size or > max_size are removed.
    Area is measured in % of image size (between 0.0 and 1.0).

    Args:
    - detections: pd.DataFrame of detections with bboxes in [x1, y1, w, h] format
    - min_size: minimal possible size of bbox
    - max_size: maximal possible size of bbox
    - img_size: tuple (width, height) of image

    Returns pd.DataFrame of filtered detections.
    """
    if detections.empty:
        return detections
    img_sq = img_size[0] * img_size[1]
    filtered_detections = detections[detections['box'].apply(lambda b: min_size <= b[2] * b[3] / img_sq <= max_size)]
    return filtered_detections

def filter_segmentation_detections(
    detections: pd.DataFrame,
    min_size: float = 0.0,
    max_size: float = 1.0,
    size_metric: str = "area"
) -> pd.DataFrame:
    """
    Filters segmentation detections using morphology-based metrics.
    
    Supported metrics:
    - area: relative object area / image area
    - diameter: relative object diameter / sqrt(image area)
    - volume: relative object volume / image volume surrogate
    """
    if detections is None or detections.empty:
        return detections

    metric = size_metric if size_metric in detections.columns else "area"

    min_size = 0.0 if min_size is None else float(min_size)
    max_size = 1.0 if max_size is None else float(max_size)

    if min_size > max_size:
        min_size, max_size = max_size, min_size

    values = pd.to_numeric(detections[metric], errors="coerce")
    keep = values.notna() & (values >= min_size) & (values <= max_size)
    return detections.loc[keep].copy()

def results_to_pandas(outputs: Results, store_bin_mask:bool = False) -> pd.DataFrame:
    """Converts ultralytics Results instance to pandas DataFrame for easy filtering."""
    if not store_bin_mask:
        data: dict[str, list[Any]] = {
            "id_label": [],
            "box": [],
            "mask": [],
            "confidence": [],
            "diameter": [],
            "area": [],
            "volume": []
        }
    else:
        data = {
            "id_label": [],
            "box": [],
            "mask": [],
            "confidence": [],
            "diameter": [],
            "area": [],
            "volume": [],
            "bin_mask": []
        }
    for i, _ in enumerate(outputs.masks.xyn):
        if len(outputs.masks.xyn[i]) == 0:
            pass
        else:
            data['id_label'].append(i)
            box = outputs.boxes.xyxyn[i].cpu().detach().numpy()
            box[2:] -= box[:2]
            data['box'].append(box)
            data['mask'].append(outputs.masks.xyn[i])
            data['confidence'].append(outputs.boxes.conf[i].cpu().detach().numpy())
            bin_mask, morphology = plot_mask(outputs.masks.xyn[i], image_size=outputs.orig_shape)
            data['diameter'].append(morphology['diameter'])
            data['area'].append(morphology['area'])
            data['volume'].append(morphology['volume'])
            if store_bin_mask is True:
                data['bin_mask'].append(bin_mask)
    return pd.DataFrame(data)

def sahi_to_pandas(outputs: list, h: int, w: int) -> pd.DataFrame:
    """
    Converts predictions from SAHI model to pandas dataframe for further processing.

    Input args:
    - outputs: list - model predictions in COCO_predictions format (list of dictionaries);
    - h: image height (for normalizing masks);
    - w: image width (for normalizing masks).

    Returns:
    - pd.DataFrame of the standard form with the predictions in it.
    """
    data: dict[str, list[Any]] = {
        "id_label": [],
        "box": [],
        "mask": [],
        "confidence": [],
        "diameter": [],
        "area": [],
        "volume": []
    }
    try:
        for i, obj in enumerate(outputs):
            if len(obj['bbox']) == 4 and len(obj['segmentation']) == 1 and len(obj['segmentation'][0]) >= 8:
                data['id_label'].append(i)
                data['box'].append(np.array(obj['bbox']))
                xs, ys = np.array(obj['segmentation'][0][::2]) / w, np.array(obj['segmentation'][0][1::2]) / h
                mask_array = np.vstack((xs, ys)).T
                data['mask'].append(mask_array)
                data['confidence'].append(obj['score'])
                _, morphology = plot_mask(mask_array)
                data['diameter'].append(morphology['diameter'])
                data['area'].append(morphology['area'])
                data['volume'].append(morphology['volume'])
    except Exception as e:
        print(f"SAHI conversion failed: {e}")
    return pd.DataFrame(data)

def pandas_to_ultralytics(df, original_image, path, frame_num: int = 0):
    """
    Convert detection DataFrame to ultralytics Results object for visualization.
    
    Transforms pandas DataFrame with detection columns into ultralytics Results
    format for easy plotting and further processing.
    
    Args:
        df (pd.DataFrame): Detection dataframe with columns:
            ['confidence', 'id_label', 'box', 'bin_mask']
        original_image (np.ndarray): Original RGB/BGR image array
        path (str): File path for the results object
        frame_num (int): Frame number for tracking context. Defaults to 0.
    
    Returns:
        Results | None: ultralytics Results object or None if empty
    """
    names = {}
    for n in range(100):
        names[n] = str(n)
    conf_array = np.array(df['confidence'].tolist())
    if len(conf_array) == 0:
        return None
    class_array = np.array(df['id_label'].tolist())
    df['box'] = df['box'].apply(lambda b: [b[0], b[1], b[2] + b[0], b[3] + b[1]])
    box_array = np.array(df['box'].tolist())
    box_array = np.hstack((box_array, np.expand_dims(conf_array, axis=1),
                           np.expand_dims(class_array, axis=1)))
    mask_array = np.stack(df['bin_mask'].tolist(), axis=0)
    probs = torch.Tensor(conf_array)
    boxes = torch.Tensor(box_array)
    try:
        masks = torch.Tensor(mask_array)
    except:
        masks = torch.Tensor(mask_array.astype(np.uint8))
    results = Results(orig_img=original_image, path=path, names=names, boxes=boxes,
                      masks=masks, probs=probs, keypoints=None, obb=None, speed=None)
    return results

def compute_iou(masks_1: list, masks_2: list) -> tuple[NDArray, list]:
    """
    Computes IoU matrix for 2 given sets of polygon masks.
    The function is used for spheroid tracjing.

    Input params:
    - masks_1: list - first set of polygon masks defined as ultralytics.engine.Results.Masks.xyn numpy array;
    - masks_2: list - second set of polygon masks defined as ultralytics.engine.Results.Masks.xyn numpy array.

    Returns:
    - iou_matrix: numpy array - matrix of IoU values for corresponding i-th mask from the first set and j-th mask from the second set.
    """
    iou_matrix = np.zeros((len(masks_1), len(masks_2)))
    mask_2_morphologies = []
    for i, _ in enumerate(masks_1):
        for j, _ in enumerate(masks_2):
            mask1, _ = plot_mask(masks_1[i])
            mask2, morphology = plot_mask(masks_2[j])
            mask_2_morphologies.append(morphology)
            intersection = np.sum(mask1 * mask2)
            union = np.sum(np.clip(mask1 + mask2, 0, 1))
            iou_matrix[i, j] = intersection / union
    return iou_matrix, mask_2_morphologies

def plot_mask(in_mask: NDArray, image_size=(1000, 1000)) -> tuple[NDArray, dict]:
    """
    Rasterizes a polygon mask safely and calculates morphology.
    Handles normalized and denormalized coordinates.
    Guards against degenerate contours and out-of-bounds points.

    Input params:
    - in_mask: np.array - np.array of contour points in ultralytics.engine.Results.Masks.xyn format;
    - image_size = 1000 - size of the canvas for drawing. Larger size leads to slightly better
    calculation precision, but it slows the processing significantly, and may be irrelevant in cases
    where the size of input image is rather small.

    Returns:
    - bin_mask: np.array - binary array where 0-values represent background and 1-values represent
    the foreground (the polygon for the given mask).
    """
    bin_mask: NDArray[np.uint8] = np.zeros(image_size, dtype=np.uint8)
    if in_mask is None:
        return bin_mask.astype(bool), calculate_morphology(bin_mask)

    try:
        coords = np.asarray(in_mask, dtype=np.float32)
    except (TypeError, ValueError):
        return bin_mask.astype(bool), calculate_morphology(bin_mask)

    if coords.size < 6:
        return bin_mask.astype(bool), calculate_morphology(bin_mask)

    if coords.ndim == 1 and coords.size % 2 == 1:
        coords = coords[:-1]

    try:
        coords = coords.reshape(-1, 2)
    except ValueError:
        return bin_mask.astype(bool), calculate_morphology(bin_mask)
    coords = coords[np.isfinite(coords).all(axis=1)]

    if coords.shape[0] < 3:
        return bin_mask.astype(bool), calculate_morphology(bin_mask)
    
    if coords.max() <= 1.0 and coords.min() >= 0.0:
        coords = coords * np.array([image_size[1], image_size[0]], dtype=np.float32)

    coords[:, 0] = np.clip(coords[:, 0], 0, image_size[1] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image_size[0] - 1)

    coords = np.round(coords).astype(np.int32)

    if np.unique(coords, axis=0).shape[0] < 3:
        return bin_mask.astype(bool), calculate_morphology(bin_mask)
    
    cv2.fillPoly(bin_mask, [coords], (1,))
    morphology = calculate_morphology(bin_mask)
    return bin_mask.astype(bool), morphology

def colormap_to_hex(cmap_name):
    """
    Convert a matplotlib colormap into a list of discrete HEX colors.
    
    Parameters:
        cmap_name (str): Name of the colormap (e.g., 'viridis', 'plasma', etc.).        
    Returns:
        List[str]: List of HEX color strings.
    """
    color_number = COLOR_NUMBER
    assert cmap_name in color_number, f"incorrect colormap specified: {cmap_name}"
    num_colors = color_number[cmap_name]
    # Get the colormap object
    cmap = plt.get_cmap(cmap_name)
    color_values = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    hex_colors = [mcolors.to_hex(c) for c in color_values]
    return hex_colors

def hex_to_bgr(hex_colors):
    """
    Convert a HEX color string to a BGR tuple for OpenCV.

    Parameters:
        hex_color (str): HEX color string (e.g., '#FF5733').
    Returns:
        Tuple[int, int, int]: BGR color tuple.
    """
    # Convert HEX to RGB
    if isinstance(hex_colors, str):  # Single color
        hex_colors = [hex_colors]
    bgr_colors = []
    for hex_color in hex_colors:
        rgb = [int(c * 255) for c in mcolors.hex2color(hex_color)]
        bgr_colors.append(tuple(reversed(rgb)))
    return bgr_colors

def denormalize_coordinates(coords, image_shape):
    """Converts normalized coords to given image coordinates."""
    return coords * np.array([image_shape[1], image_shape[0]])

def plot_predictions(image, pred_masks, filename: str = IMAGE_FILE_NAME_DETECTION,
                     alpha=0.75, colormap="tab20", color_ids=None):
    """Draws predicted masks on the image."""
    hex_colors = hex_to_bgr(colormap_to_hex(colormap))
    if not pred_masks:
        print("No masks found.")
        safe_image_write(image, filename)
        return image
    overlay = image.copy()
    for i, mask in enumerate(pred_masks):
        coords = np.asarray(mask, dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] < 3:
            continue
        color_index = i if color_ids is None else int(color_ids[i])
        color = hex_colors[color_index % len(hex_colors)]
        if coords.max() <= 1.0:  # Проверка, денормализованы ли координаты (xIn или xIm)
            coords = denormalize_coordinates(coords, image.shape)
        coords[:, 0] = np.clip(coords[:, 0], 0, image.shape[1] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, image.shape[0] - 1)
        coords = np.round(coords).astype(np.int32)
        if len(np.unique(coords, axis=0)) < 3:
            continue
        cv2.fillPoly(overlay, [coords], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    safe_image_write(image, filename)
    return image


def plot_predictions_with_alignment(
    original_image,
    img_inference,
    pred_masks,
    filename: str = IMAGE_FILE_NAME_DETECTION,
    colormap="tab20",
    alpha=0.75,
    color_ids=None
):
    """
    Plot predictions with automatic dimension alignment.
    
    Resizes original image to match inference dimensions if needed, then
    overlays predicted masks. Useful when inference uses different image
    size than original.
    
    Args:
        original_image (np.ndarray): Original input image
        img_inference (np.ndarray): Image dimensions used for inference
        pred_masks (list): List of mask contours (normalized coordinates)
        filename (str): Output image path. Defaults to IMAGE_FILE_NAME_DETECTION.
        colormap (str): Matplotlib colormap. Defaults to 'tab20'.
        alpha (float): Mask transparency (0-1). Defaults to 0.75.
    
    Returns:
        np.ndarray: Image with overlaid masks
    """
    h, w = img_inference.shape[:2]
    o_h, o_w = original_image.shape[:2]
    if h != o_h or w != o_w:
        original_image = resize_and_pad_cv(original_image, w, h)
    set_global('image_display_base', original_image.copy())
    return plot_predictions(
        original_image,
        pred_masks,
        filename=filename,
        colormap=colormap,
        alpha=alpha,
        color_ids=color_ids
    )


def calculate_morphology(bin_mask: NDArray[np.uint8]) -> dict:
    """
    Calculates the morphology for the given segmented object.
    The morphology includes: diameter, area, volume.
    All the values (except area) are calculated under the assumption that the given object is spherical.
    We measure these values in relative ratios as follows:
    - diameter - relative to the square root of image area;
    - area - relative to the image area;
    - volume - relative to the image volume (image area multiplied by square root of image area).
    """
    img_area = bin_mask.shape[0] * bin_mask.shape[1]
    if img_area == 0:
        return {'diameter': 0.0, 'area': 0.0, 'volume': 0.0}

    area = float(np.sum(bin_mask))
    diameter = 2 * np.sqrt(area / np.pi)
    radius = diameter / 2
    volume = (4/3) * np.pi * radius**3
    return {
        'diameter': float(diameter / np.sqrt(img_area)),
        'area': float(area / img_area),
        'volume': float(volume / (img_area * np.sqrt(img_area))),
    }

def create_image_grid(
    images, 
    labels, 
    label_font_scale=0.5, 
    label_thickness=1, 
    font=cv2.FONT_HERSHEY_SIMPLEX, 
    total_images=None
) -> np.ndarray:
    """Create a grid of images with labels."""
    # Resize all images to the same dimensions
    height, width = next(img for img in images if img is not None).shape[:2]
    images = [cv2.resize(img, (width, height)) for img in images]
    
    # Annotate each image with filename
    labeled_images = []
    for i, labeltext in enumerate(labels):
        img_copy = images[i].copy()
        if img_copy.ndim == 2:
            img_copy = gray2rgb(img_copy)
        
        cv2.putText(img_copy, labeltext, (5, height - 10), font, label_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
        cv2.rectangle(img_copy, (0, 0), (width, height), (255, 255, 255), 1)
        labeled_images.append(img_copy)

    # Determine grid size (nearly square)
    n = len(labeled_images) if total_images is None else total_images
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Pad with black images if needed
    black = np.zeros_like(labeled_images[0])
    while len(labeled_images) < rows * cols:
        labeled_images.append(black)

    # Stack images into rows
    grid_rows = []
    for i in range(rows):
        row_imgs = labeled_images[i*cols:(i+1)*cols]
        row = np.hstack(row_imgs)
        grid_rows.append(row)

    # Stack all rows into final image
    grid_image = np.vstack(grid_rows)
    return grid_image            

def resize_and_pad_cv(image, target_width, target_height, anti_aliasing=True):
    """
    Resize image maintaining aspect ratio and pad to target dimensions.
    
    Scales image to fit within target dimensions while preserving aspect ratio,
    then zero-pads to exactly match target size (centered).
    
    Args:
        image (np.ndarray): Input image (2D or 3D array)
        target_width (int): Target image width in pixels
        target_height (int): Target image height in pixels
        anti_aliasing (bool): Apply Gaussian filter before downsampling. Defaults to True.
    
    Returns:
        np.ndarray: Resized and padded image with shape (target_height, target_width, ...)
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = resize(image, (new_h, new_w), anti_aliasing=anti_aliasing, preserve_range=True)
    resized = resized.astype(image.dtype)
    
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    # Calculate padding for numpy
    pad_width = (
        (top, bottom),     # pad rows (height)
        (left, right),     # pad columns (width)
    )
    # For grayscale
    if resized.ndim == 2:
        padded = np.pad(resized, pad_width, mode='constant', constant_values=0)
    # For color or multi-channel
    elif resized.ndim == 3:
        pad_width = pad_width + ((0, 0),)  # no padding on channels
        padded = np.pad(resized, pad_width, mode='constant', constant_values=0)
        
    return padded


def process_loaded_image(image, settings: OrderedDict):
    """
    Apply sequence of preprocessing operations to image.
    
    Executes ordered preprocessing steps specified in settings OrderedDict.
    Supports operations: resize, resizeandpad, gray2rgb, rgb2gray, normalize, clip.
    
    Args:
        image (np.ndarray): Input image array
        settings (OrderedDict): Ordered dict of preprocessing steps.
            Each step is {'operation': 'parameters'}. Supported:
            - {'resize': 'WIDTH:HEIGHT'}
            - {'resizeandpad': 'WIDTH:HEIGHT'}
            - {'gray2rgb': ''}
            - {'rgb2gray': ''}
            - {'normalize': 'PMIN,PMAX'} (percentiles)
            - {'clip': 'MIN,MAX'} (value ranges)
    
    Returns:
        np.ndarray: Processed image
        
    Raises:
        RuntimeError: If unknown operation specified
    """
    for step in settings:
        key, value = next(iter(step.items()))    
        match key:
            case "resize":
                target_width, target_height = map(int, value.strip().split(":"))
                orig_height, orig_width = image.shape[:2]
                scale = min(target_width / orig_width, target_height / orig_height)
                resized_width = int(orig_width * scale)
                resized_height = int(orig_height * scale)
                image = resize(
                    image, 
                    output_shape=(resized_height,resized_width), 
                    order=0, 
                    preserve_range=True, 
                    anti_aliasing=False
                    ).astype(image.dtype)

            case "resizeandpad":
                target_width, target_height = map(int, value.strip().split(":"))
                image = resize_and_pad_cv(image,target_width, target_height )
            case "gray2rgb":
                image = safegray2rgb(image)
            case "rgb2gray":
                image = safergb2gray(image)
            case "normalize":
                p_min, p_max = map(float, value.strip().split(","))
                image = normalize(image, pmin = p_min, pmax=p_max)
            case "clip":
                a_min, a_max = map(int, value.strip().split(","))
                image = np.clip(image, a_min=a_min, a_max=a_max)
            case _:
                raise RuntimeError(f"Unknow process_loaded_image instruction:{key}")
    return image

def safegray2rgb(image):
    """
    Convert grayscale to RGB if needed, otherwise return unchanged.
    
    Args:
        image (np.ndarray): Input image array
        
    Returns:
        np.ndarray: RGB image if input was grayscale, otherwise unchanged
    """
    if image.ndim == 2:
        return gray2rgb(image)
    return image


def safergb2gray(image):
    """
    Convert RGB to grayscale if needed, otherwise return unchanged.
    
    Args:
        image (np.ndarray): Input image array
        
    Returns:
        np.ndarray: Grayscale image if input was RGB, otherwise unchanged
    """
    if image.ndim == 3:
        image = rgb2gray(image)
        return (image * 255).astype("uint8")
    return image


def compute_f1_from_matches(matches, num_ground, num_candidate, iou_threshold=0.5):
    """
    Calculate F1, precision, recall, TP, FP, FN metrics from matches.
    
    Args:
        matches (list): List of (ground_idx, candidate_idx, iou) tuples
        num_ground (int): Total number of ground truth objects
        num_candidate (int): Total number of candidate detections
        iou_threshold (float): IoU threshold for considering match valid. Defaults to 0.5.
    
    Returns:
        dict: Metrics dictionary with keys:
            - 'TP': True positives
            - 'FP': False positives  
            - 'FN': False negatives
            - 'Precision': Precision score
            - 'Recall': Recall score
            - 'F1': F1 score
    """
    tp = sum(1 for (_, _, iou) in matches if iou >= iou_threshold)
    fp = num_candidate - tp
    fn = num_ground - tp

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
      
def clear_cache():
    """
    Remove and recreate application cache directory.
    """
    cache_dir = CASH_DIRECTORY
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir,ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

def show_error_message(title, message):
    """
    Display error dialog to user with title and message.
    
    Args:
        title (str): Dialog window title
        message (str): Error message text
        
    Returns:
        None
    """
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.exec_()
