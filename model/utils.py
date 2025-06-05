"""Some useful functions used by the model or its submodels."""

import os
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from ultralytics.engine.results import Results
import tiffile
from collections import OrderedDict

VALID_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'tif', 'bmp']
CLASSES = ['Cell']
COLORS = [(3,177,252)]

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

def read_lsm_img(img_path, cell_channel=0, nuclei_channel=1):
    """Reads lsm image and returns as array."""
    with tiffile.TiffFile(img_path) as tif:
        image = tif.pages[0].asarray()
    if np.transpose(image, (1, 2, 0)).shape[-1] == 1:
        return cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_GRAY2RGB)
    elif np.transpose(image, (1, 2, 0)).shape[-1] == 2:
        img = np.transpose(image, (1, 2, 0))
        stacked_array = np.dstack((img, np.zeros((512,512)).astype('uint8')))
        return stacked_array
    elif np.transpose(image, (1, 2, 0)).shape[-1] == 3:
        return np.transpose(image, (1, 2, 0))
    else:
        img = np.transpose(image, (1, 2, 0))
        stacked_array = np.dstack((img[cell_channel], img[nuclei_channel],
                                   np.zeros((512,512)).astype('uint8')))
        return stacked_array

def read_standard_img(img_path):
    """Reads image in grayscale jpg/png/tif/bmp which contains cells only."""
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,
                                     cv2.COLOR_GRAY2RGB))

def is_image_valid(img_path: str):
    """Checks if provided image is in correct format."""
    extension = img_path.split('.')[-1]
    if extension.lower() in VALID_IMAGE_EXTENSIONS:
        return True
    return False

def read_img(img_path, cell_channel=0, nuclei_channel=1):
    """Reads any possible image of cells (and/or nuclei)."""
    if img_path.endswith('lsm'):
        return read_lsm_img(img_path, cell_channel, nuclei_channel)
    elif is_image_valid(img_path):
        return read_standard_img(img_path)

def calculate_lsm(cell_counter, nuclei_counter,
                  img_path, cell_channel=0, nuclei_channel=1):
    """
    Calculates the resulting target values.
    Input params are:
    - img_path: path to lsm/jpg/png/tif/bmp image;
    - cell_channel: channel with cells. Default to 0;
    - nuclei_channel: channel with stained nuclei. Default to 1.

    Returns the result as a dictionary with the following fields:
    - Nuclei: count for stained nuclei detected (given lsm image only);
    - Cells: count for all the cells detected;
    - %: the target percentage for alive cells (given lsm image only).
    """
    img = read_lsm_img(img_path)

    cell_img = cv2.cvtColor(img[:,:,cell_channel], cv2.COLOR_GRAY2BGR)
    tmp_path = 'cell_tmp_img.png'
    cv2.imwrite(tmp_path, cell_img)
    cell_count = cell_counter.count_cells(tmp_path)
    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        pass
    nuclei_count = nuclei_counter.countNuclei(img[:,:,nuclei_channel])
    percentage = (1 - nuclei_count/cell_count.shape[0]) * 100
    return {'Nuclei': nuclei_count, 'Cells': cell_count, '%': round(percentage,3)}



def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, draw_mode=0):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        # img_path (str): The path to input image to draw the bounding box on.
        # class_id (int): Class ID of the detected object.
        # confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    color = COLORS[0]
    if img.shape[0] < 800:
        thickness = 1
    else:
        thickness = 2
    if draw_mode == 0:
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, thickness)
    else:
        img = cv2.circle(img, (x, y), 2, color, -1)

def filter_detections(detections: pd.DataFrame, min_size: float = 0.0, max_size: float = 1.0, img_size: tuple = (512,512)) -> pd.DataFrame:
    # NOTE: this function is deprecatedand no longer used, since we have implemented new inference pipeline
    """
    Filters bounding boxes based on their area.
    Bboxes of size < min_size or > max_size are removed.
    Area is measured in % of image size (between 0.0 and 1.0).
    This filtering function is implemented in October, 2024, to reduce the amount of garbage detected as cells.

    Input args:
    - detections: pd.DataFrame of detections, including bboxes in [x1, y1, w, h] format, where x1, y1 - lower-left corner coordinates;
    - min_size: float representing the minimal possible size of bbox;
    - max_size: float representing the maximal possible size of bbox;
    - img_size: tuple of size 2 representing width and height of image.

    Returns np.array of filtered bboxes.
    """
    # assert(min_size <= max_size)
    if detections.empty:
        return detections
    img_sq = img_size[0] * img_size[1]
    # print(type(detections['box']))
    # print(detections.head())
    # print(detections['box'].head())
    filtered_detections = detections[detections['box'].apply(lambda b: min_size <= b[2] * b[3] / img_sq <= max_size)]
    # filtered_detections = filtered_detections[filtered_detections['box'].apply(lambda b: (min_size <= b[2] * b[3] / img_sq <= max_size).item())]
    return filtered_detections

def results_to_pandas(outputs: Results, store_bin_mask:bool = False) -> pd.DataFrame:
    """Converts ultralytics Results instance to pandas DataFrame for easy filtering."""
    if store_bin_mask is False:
        data = {
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
    data = {
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
    except:
        print("Something wrong happenned...")
    return pd.DataFrame(data)

def pandas_to_ultralytics(df, original_image, path, frame_num: int = 0):
    """Converts pandas DataFrame instance to ultralytics Results for easier plotting."""
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

def compute_iou(masks_1: list, masks_2: list) -> np.array:
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
            iou_matrix[i,j] = intersection / union
    return iou_matrix, mask_2_morphologies

def plot_mask(in_mask: np.array, image_size=(1000,1000)) -> np.array:
    """
    Plots given mask on a 1000x1000 canvas for its further processing.
    This util is used as a helper for compute_iou() function above.

    Input params:
    - in_mask: np.array - np.array of contour points in ultralytics.engine.Results.Masks.xyn format;
    - image_size = 1000 - size of the canvas for drawing. Larger size leads to slightly better
    calculation precision, but it slows the processing significantly, and may be irrelevant in cases
    where the size of input image is rather small.

    Returns:
    - bin_mask: np.array - binary array where 0-values represent background and 1-values represent
    the foreground (the polygon for the given mask).
    """
    if in_mask.max() > 1.0:
        in_mask = in_mask /  np.array([image_size[1], image_size[0]])
    coords = in_mask.reshape(-1, 2) * np.array([image_size[1], image_size[0]])
    coords = coords.astype(np.int32)
    bin_mask = np.zeros(image_size, dtype=np.uint8)
    cv2.fillPoly(bin_mask, [coords], 1)
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
    color_number = {
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

def plot_predictions(image, pred_masks, filename: str = ".cache/cell_tmp_img_with_detections.png",
                     alpha=.75, colormap="tab20"):
    """Draws predicted masks on the image."""
    hex_colors = hex_to_bgr(colormap_to_hex(colormap))
    if not pred_masks:
        print("No masks found.")
        return
    overlay = image.copy()
    for i, mask in enumerate(pred_masks):
        coords = np.array(mask)
        color = hex_colors[i % len(hex_colors)]
        if coords.max() <= 1.0:  # Проверка, денормализованы ли координаты (xIn или xIm)
            coords = denormalize_coordinates(coords, image.shape)
        coords = coords.astype(int)
        cv2.fillPoly(overlay, [coords], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.imwrite(filename, image)
    return image

def calculate_morphology(bin_mask: np.array) -> dict:
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
    area = np.sum(bin_mask)
    diameter = 2 * np.sqrt(area / np.pi)
    radius = diameter / 2
    volume = (4/3) * np.pi * radius**3
    return {'diameter': diameter / np.sqrt(img_area), 'area': area / img_area, 'volume': volume / (img_area * np.sqrt(img_area))}

def create_image_grid(images, labels, label_font_scale=0.5, label_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, total_images = None) -> np.ndarray:
    # if any(img is None for img in images):
    #     raise ValueError("One or more images could not be loaded.")
    
    # Resize all images to the same dimensions
    #height, width = [img is not None for img in images][0].shape[:2]
    height, width =  (next((img for img in images if img is not None), None)).shape[:2]
    images = [cv2.resize(img, (width, height)) for img in images]
    
    # Annotate each image with filename
    labeled_images = []
    for i, labeltext in enumerate(labels):
        img_copy = images[i].copy()
        if img_copy.ndim == 2:
            from skimage.color import gray2rgb
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

def resize_and_pad_cv(image, target_width, target_height):
    """Resize image keeping aspect ratio and pad to target size."""
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    from skimage.transform import resize
    resized = resize(image, (new_h, new_w), anti_aliasing=True, preserve_range=True)
    #resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = resized.astype(image.dtype)
    
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

#    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Assuming image shape: (height, width, channels) or (height, width)
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


def process_loaded_image(image, settings:OrderedDict):
    for step in settings:
        key, value = next(iter(step.items()))
        match key:
            case "resize":
                target_width, target_height = map(int, value.strip().split(":"))
                orig_height, orig_width = image.shape[:2]
                scale = min(target_width / orig_width, target_height / orig_height)
                resized_width = int(orig_width * scale)
                resized_height = int(orig_height * scale)
                from skimage.transform import resize
                image = resize(image, output_shape=(resized_height,resized_width), order=0, preserve_range=True, anti_aliasing=False).astype(image.dtype)
            case "gray2rgb":
                image = safegray2rgb(image)
            case "rgb2gray":
                image = safergb2gray(image)
            case "normalize":
                from csbdeep.utils import normalize            
                p_min, p_max = map(float, value.strip().split(","))
                image = normalize(image, pmin = p_min, pmax=p_max)
            case "clip":
                import numpy as np
                a_min, a_max = map(int, value.strip().split(","))
                image = np.clip(image, a_min=a_min, a_max=a_max)
            case _:
                raise RuntimeError(f"Unknow process_loaded_image instruction:{key}")
    return image

def safegray2rgb(image):
    if image.ndim == 2:
        from skimage.color import gray2rgb
        return gray2rgb(image)
    else:
        return image

def safergb2gray(image):
    if image.ndim == 3:
        from skimage.color import rgb2gray
        image = rgb2gray(image)
        return (image * 255).astype("uint8")
    else:
        return image

def compute_f1_from_matches(matches, num_ground, num_candidate, iou_threshold=0.5):
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
    
def compute_iou(mask1, mask2):
    """Compute Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def match_masks(masks_a, masks_b, iou_threshold=0.5):
    """Match masks from List A to List B using IoU and Hungarian algorithm."""
    n, m = len(masks_a), len(masks_b)
    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(masks_a[i], masks_b[j])

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches = [(i, j, iou_matrix[i, j]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= iou_threshold]
    return matches, iou_matrix    
    
def safeimagesave(image,filename):
    from skimage.io import imsave
    if image.dtype == 'uint8':
        #no denormalization needed
        imsave(filename, image.astype('uint8'))
    else:
        #denormalization needed
        imsave(filename, (image * 255).astype('uint8'))
      
