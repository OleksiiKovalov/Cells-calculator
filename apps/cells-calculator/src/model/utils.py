"""Some useful functions used by the model or its submodels."""

# Standard library imports
import logging
import os
from typing import Any

# Third-party imports
import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import tifffile
from csbdeep.utils import normalize
from PySide6.QtWidgets import QMessageBox
from skimage.color import gray2rgb, rgb2gray
from skimage.io import imsave
from skimage.transform import resize
from ultralytics.engine.results import Results

# Local application imports
from ui.app_globals import IMAGE_FILE_NAME_DETECTION

logger = logging.getLogger(__name__)


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
    with tifffile.TiffFile(img_path) as tif:
        try:
            return tif.series[0].asarray()
        except (IndexError, ValueError):
            return tif.pages[0].asarray()


def lsm_to_channels_last(image):
    """Normalizes LSM arrays to (height, width, channels)."""
    image = np.asarray(image)
    if image.ndim == 0:
        return image.reshape(1, 1, 1)
    if image.ndim == 1:
        return image.reshape(1, 1, image.shape[0])
    if image.ndim == 2:
        return image[:, :, np.newaxis]

    if image.ndim > 3:
        image = np.squeeze(image)
        if image.ndim == 0:
            return image.reshape(1, 1, 1)
        if image.ndim == 1:
            return image.reshape(1, 1, image.shape[0])
        if image.ndim == 2:
            return image[:, :, np.newaxis]
        image = image.reshape((-1, image.shape[-2], image.shape[-1]))

    if image.ndim != 3:
        raise ValueError(f"Unsupported LSM image shape: {image.shape}")

    if (
        image.shape[0] <= 8
        and (
            image.shape[1] > 8
            or image.shape[2] > 8
            or (image.shape[0] > 1 and image.shape[2] <= 1)
        )
    ):
        return np.transpose(image, (1, 2, 0))

    if image.shape[-1] <= 8 and image.shape[0] > 8 and image.shape[1] > 8:
        return image

    if image.shape[-1] in (3, 4):
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
            logger.warning("Image file not found: %s", img_path)
            return None
            
        extension = img_path.split('.')[-1].lower()
        
        if extension == 'lsm':
            # Handle LSM files with tifffile
            image = read_lsm_array(img_path)
            if channel is not None:
                image = lsm_to_channels_last(image)
                if 0 <= channel < image.shape[-1]:
                    return image[:, :, channel]
                logger.warning("Channel %s not available in LSM file", channel)
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
                logger.warning("Failed to read image: %s", img_path)
                return None
                
            return image
            
    except Exception:
        logger.exception("Error reading image %s", img_path)
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
            logger.warning("Cannot save None image")
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

        write_success = False

        # Save based on file extension
        if extension in ['jpg', 'jpeg']:
            # JPEG with quality control
            write_success = cv2.imwrite(filename, image_to_save, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif extension == 'png':
            # PNG with compression
            write_success = cv2.imwrite(filename, image_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 1])
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
            logger.warning("Unsupported image extension: %s", extension)
            return False

        return bool(write_success and os.path.exists(filename))

    except Exception:
        logger.exception("Error saving image %s", filename)
        return False

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

def read_lsm_img(img_path, cell_channel=0, nuclei_channel=1):
    """Reads an LSM image and returns it as an (H, W, 3) array."""
    img = lsm_to_channels_last(read_lsm_array(img_path))
    if img.shape[-1] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    if img.shape[-1] == 2:
        return np.dstack((img, np.zeros(img.shape[:2], dtype=img.dtype)))
    if img.shape[-1] == 3:
        return img
    # >3 channels: compose a pseudo-RGB from the cell + nuclei channels.
    empty_channel = np.zeros(img.shape[:2], dtype=img.dtype)
    cell = img[:, :, cell_channel] if 0 <= cell_channel < img.shape[-1] else img[:, :, 0]
    nuclei = img[:, :, nuclei_channel] if 0 <= nuclei_channel < img.shape[-1] else empty_channel
    return np.dstack((cell, nuclei, empty_channel))

def read_img(img_path, cell_channel=0, nuclei_channel=1):
    """Reads any supported image; routes .lsm through the LSM reader."""
    if img_path.lower().endswith('lsm'):
        return read_lsm_img(img_path, cell_channel, nuclei_channel)
    return read_standard_img(img_path)

def filter_detections(
    detections: pd.DataFrame, 
    min_size: float = 0.0, 
    max_size: float = 1.0, 
    img_size: tuple = (512, 512)
) -> pd.DataFrame:
    """
    Filters bounding boxes based on their area.

    Legacy box-only fallback used by the UI when detections carry bounding
    boxes but no segmentation masks (see MainWindow._get_filtered_detections);
    the mask-based pipeline uses filter_segmentation_detections instead.

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
    filtered_detections: pd.DataFrame = detections[detections['box'].apply(lambda b: min_size <= b[2] * b[3] / img_sq <= max_size)]
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


def get_segmentation_detections_range(
    detections: pd.DataFrame,
    size_metric: str = "area"
) -> tuple[float, float]:
    """
    Returns the (min, max) values of the given size metric across all detections.

    Args:
        detections: detections DataFrame as returned by the segmenters.
        size_metric: column to inspect — "area", "diameter", or "volume".

    Returns:
        (min_value, max_value) as floats, or (0.0, 1.0) when the data is empty/invalid.
    """
    if detections is None or detections.empty:
        return 0.0, 1.0

    metric = size_metric if size_metric in detections.columns else "area"
    values = pd.to_numeric(detections[metric], errors="coerce").dropna()

    if values.empty:
        return 0.0, 1.0

    return float(values.min()), float(values.max())

def results_to_pandas(outputs: Results, store_bin_mask:bool = False) -> pd.DataFrame:
    """Converts ultralytics Results instance to pandas DataFrame for easy filtering."""
    data: dict[str, list[Any]] = {
        "id_label": [],
        "box": [],
        "mask": [],
        "confidence": [],
        "diameter": [],
        "area": [],
        "volume": [],
    }
    if store_bin_mask:
        data["bin_mask"] = []
    # No masks at all (zero detections) -> return an empty, correctly-typed frame
    # instead of crashing on outputs.masks.xyn (outputs.masks is None).
    if outputs.masks is None:
        return pd.DataFrame(data)
    for i, _ in enumerate(outputs.masks.xyn):
        if len(outputs.masks.xyn[i]) == 0:
            continue
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
        if store_bin_mask:
            data['bin_mask'].append(bin_mask)
    return pd.DataFrame(data)

def plot_mask(in_mask: NDArray, image_size=(1000, 1000)) -> tuple[NDArray, dict]:
    """
    Rasterizes a polygon mask safely and calculates morphology.
    Guards against degenerate contours and out-of-bounds points.

    Contract: in_mask must contain normalized [0, 1] coordinates in (x, y) order.

    Input params:
    - in_mask: np.array - contour points in normalized [0, 1] (x, y) format;
    - image_size - canvas size for rasterization. Larger = slightly more precise morphology,
      but slower; typically pass the original image shape.

    Returns:
    - bin_mask: np.array - binary array where 0 = background, 1 = foreground polygon.
    """
    bin_mask: NDArray[np.uint8] = np.zeros(image_size, dtype=np.uint8)
    if in_mask is None:
        return bin_mask.astype(bool), calculate_morphology(bin_mask)

    # Contract: coordinates are normalized [0, 1]; multiply by image dimensions to get pixel space.
    arr: np.ndarray = np.asarray(in_mask, dtype=np.float32).ravel()
    if arr.size % 2:  # drop a dangling, unpaired coordinate
        arr = arr[:-1]
    coords: np.ndarray = arr.reshape(-1, 2)
    # Drop non-finite points (NaN / inf) before rasterizing.
    coords = coords[np.isfinite(coords).all(axis=1)]

    if coords.shape[0] < 3:
        return bin_mask.astype(bool), calculate_morphology(bin_mask)

    coords = coords * np.array([image_size[1], image_size[0]], dtype=np.float32)

    coords[:, 0] = np.clip(coords[:, 0], 0, image_size[1] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image_size[0] - 1)

    coords = np.round(coords).astype(np.int32)

    coords = np.unique(coords, axis=0)
    if coords.shape[0] < 3:
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
                     alpha=0.75, colormap="tab20", color_ids=None,
                     filled: bool = True, outline_thickness: int = 1,
                     draw_labels: bool = False):
    """Draws predicted masks on the image.

    Args:
        filled: if True (default) fills the polygon with a semi-transparent color.
                if False draws only the polygon outline.
        outline_thickness: line thickness used when filled=False.
        draw_labels: if True, draws the color_index value at the centroid of each mask.
    """
    hex_colors = hex_to_bgr(colormap_to_hex(colormap))
    if not pred_masks:
        print("No masks found.")
        safe_image_write(image, filename)
        return image
    # Contract: mask coordinates are normalized [0, 1]; denormalize to pixel space for rendering.
    overlay = image.copy()
    label_draws = []  # (centroid_x, centroid_y, label_text) — deferred until after blending
    for i, mask in enumerate(pred_masks):
        coords = np.asarray(mask, dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] < 3:
            continue
        color_index = i if color_ids is None else int(color_ids[i])
        color = hex_colors[color_index % len(hex_colors)]
        coords = denormalize_coordinates(coords, image.shape)
        coords[:, 0] = np.clip(coords[:, 0], 0, image.shape[1] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, image.shape[0] - 1)
        coords = np.round(coords).astype(np.int32)
        if filled:
            cv2.fillPoly(overlay, [coords], color)
        else:
            cv2.polylines(overlay, [coords], isClosed=True, color=color,
                          thickness=outline_thickness)
        if draw_labels:
            cx = int(coords[:, 0].mean())
            cy = int(coords[:, 1].mean())
            label_draws.append((cx, cy, str(color_index)))
    if filled:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    else:
        image = overlay
    if draw_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.3, min(image.shape[0], image.shape[1]) / 1500.0)
        thickness = max(1, int(font_scale * 2))
        for cx, cy, text in label_draws:
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            ox = cx - tw // 2
            oy = cy + th // 2
            # Dark outline for readability on any background
            cv2.putText(image, text, (ox, oy), font, font_scale,
                        (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(image, text, (ox, oy), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
    safe_image_write(image, filename)
    return image


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

def morphology_to_micrometers(diameter_norm, area_norm, volume_norm,
                              img_w, img_h, um_per_px):
    """Convert normalized morphology ratios to absolute micrometers.

    Inverts ``calculate_morphology``'s normalization — with image area
    ``A = img_w * img_h`` it stores diameter/√A, area/A and volume/A^1.5 — to
    recover pixel-space values, then applies the µm/px calibration ``k``:
    a linear measure scales by k, an area by k², a volume by k³.

    Returns (diameter_um, area_um2, volume_um3); zeros if inputs are degenerate.
    """
    img_area = float(img_w) * float(img_h)
    if img_area <= 0 or um_per_px <= 0:
        return 0.0, 0.0, 0.0
    diameter_px = diameter_norm * (img_area ** 0.5)
    area_px2 = area_norm * img_area
    volume_px3 = volume_norm * (img_area ** 1.5)
    return (
        diameter_px * um_per_px,
        area_px2 * (um_per_px ** 2),
        volume_px3 * (um_per_px ** 3),
    )

# A geometric transform recording what a resize/pad step actually did, as
# ``dst_px = src_px * scale + pad`` (uniform scale, centered padding). The
# resize/pad code reports this so detections found on the preprocessed image can
# be mapped back exactly — no guessing the geometry from image shapes.
IDENTITY_TRANSFORM = {"scale": 1.0, "pad_x": 0.0, "pad_y": 0.0}


def compose_transforms(first: dict, second: dict) -> dict:
    """Compose two scale-then-centered-pad transforms.

    ``first`` is applied to the original image, ``second`` to the already
    transformed image. Returns the net transform mapping original-image pixels
    to the final image's pixels: applying ``second`` to ``p * s1 + pad1`` gives
    ``p * s1*s2 + (pad1*s2 + pad2)``.
    """
    return {
        "scale": first["scale"] * second["scale"],
        "pad_x": first["pad_x"] * second["scale"] + second["pad_x"],
        "pad_y": first["pad_y"] * second["scale"] + second["pad_y"],
    }


def invert_transform_points(pts_xy: NDArray, transform: dict) -> NDArray:
    """Map (N, 2) x/y points from preprocessed space back to original pixels.

    Inverts ``p * scale + pad`` → ``(p - pad) / scale``.
    """
    scale = transform["scale"] if transform["scale"] else 1.0
    out = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2).copy()
    out[:, 0] = (out[:, 0] - transform["pad_x"]) / scale
    out[:, 1] = (out[:, 1] - transform["pad_y"]) / scale
    return out


def resize_and_pad_cv(image, target_width, target_height, anti_aliasing=True,
                      return_transform=False):
    """Resize image keeping aspect ratio and pad to target size.

    When ``return_transform`` is True, also returns the transform this call
    applied (uniform scale + centered pad) so detections on the output can be
    mapped back to the input.
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
        pad_width_3d = pad_width + ((0, 0),)  # no padding on channels
        padded = np.pad(resized, pad_width_3d, mode='constant', constant_values=0)
    else:
        raise ValueError(f"resize_and_pad_cv expects a 2D or 3D image, got ndim={resized.ndim}")

    if return_transform:
        return padded, {"scale": scale, "pad_x": float(left), "pad_y": float(top)}
    return padded


def process_loaded_image(image, settings, return_transform=False):
    """Apply a sequence of image processing operations based on settings.

    When ``return_transform`` is True, also returns the net transform the
    geometric steps (resize/resizeandpad) applied; non-geometric steps
    (gray2rgb, normalize, …) leave it unchanged. Callers invert it to map
    detections back onto the original image.
    """
    transform = dict(IDENTITY_TRANSFORM)
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
                # Aspect-preserving resize, no padding: pure uniform scale.
                transform = compose_transforms(
                    transform, {"scale": scale, "pad_x": 0.0, "pad_y": 0.0}
                )

            case "resizeandpad":
                target_width, target_height = map(int, value.strip().split(":"))
                image, step_transform = resize_and_pad_cv(
                    image, target_width, target_height, return_transform=True
                )
                transform = compose_transforms(transform, step_transform)
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
    if return_transform:
        return image, transform
    return image

def safegray2rgb(image):
    """Convert grayscale to RGB if needed."""
    if image.ndim == 2:
        return gray2rgb(image)
    return image


def safergb2gray(image):
    """Convert RGB to grayscale if needed."""
    if image.ndim == 3:
        image = rgb2gray(image)
        return (image * 255).astype("uint8")
    return image

def show_error_message(title, message):
    """Show error message box to user"""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.exec()
