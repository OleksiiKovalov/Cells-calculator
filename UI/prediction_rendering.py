"""Rendering helpers for model predictions shown by the UI."""

import os
from typing import Optional

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from UI.app_globals import (
    IMAGE_FILE_NAME_DETECTION,
    IMAGE_FILE_NAME_INGFERENCE,
    set_global,
)
from model.PredictionResult import get_prediction_images


CLASSES = ["Cell"]
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
    "plasma": 10,
}


def safe_image_write(image, filename, quality=95, preserve_dtype=True):
    """Save UI-rendered image output without importing the model utility module."""
    if image is None:
        print("Cannot save None image")
        return False

    filename = os.fspath(filename)
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image = np.asarray(image)
    if not preserve_dtype or image.dtype != np.uint8:
        if image.dtype in [np.float32, np.float64]:
            if image.size and image.max() <= 1.0:
                image_to_save = (image * 255).astype(np.uint8)
            else:
                image_to_save = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image_to_save = image.astype(np.uint8)
    else:
        image_to_save = image

    extension = filename.split(".")[-1].lower()
    params = []
    if extension in ["jpg", "jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif extension == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    elif extension not in ["bmp", "tif", "tiff"]:
        print(f"Unsupported image extension: {extension}")
        return False

    return bool(cv2.imwrite(filename, image_to_save, params) and os.path.exists(filename))


def _copy_image(image):
    return image.copy() if hasattr(image, "copy") else image


def publish_inference_image(
    model,
    result,
    filename=IMAGE_FILE_NAME_INGFERENCE,
    preserve_dtype=False,
):
    """Publish a model-produced inference image for UI display/cache."""
    _, inference_image = get_prediction_images(result)
    if inference_image is None:
        return None

    image_for_state = _copy_image(inference_image)
    if model is not None and getattr(model, "cell_counter", None):
        model.cell_counter.inference_image = _copy_image(inference_image)

    set_global("image_inference", image_for_state)
    safe_image_write(
        inference_image,
        filename,
        preserve_dtype=preserve_dtype,
    )
    return inference_image


def resize_and_pad_cv(image, target_width, target_height):
    """Resize image to fit target size and pad the remaining area."""
    metadata = get_resize_and_pad_metadata(image.shape, target_width, target_height)
    resized = cv2.resize(
        image,
        (metadata["new_w"], metadata["new_h"]),
        interpolation=metadata["interpolation"],
    )

    if resized.ndim == 2:
        pad_width = (
            (metadata["top"], metadata["bottom"]),
            (metadata["left"], metadata["right"]),
        )
    else:
        pad_width = (
            (metadata["top"], metadata["bottom"]),
            (metadata["left"], metadata["right"]),
            (0, 0),
        )
    return np.pad(resized, pad_width, mode="constant", constant_values=0)


def get_resize_and_pad_metadata(image_shape, target_width, target_height):
    """Return resize/pad values used by ``resize_and_pad_cv``."""
    h, w = image_shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    return {
        "scale": scale,
        "new_w": new_w,
        "new_h": new_h,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "interpolation": interpolation,
        "original_w": w,
        "original_h": h,
        "target_w": target_width,
        "target_h": target_height,
    }


def restore_from_resize_and_pad(image, metadata):
    """Crop padding and resize an aligned image back to its original dimensions."""
    top = metadata["top"]
    left = metadata["left"]
    new_h = metadata["new_h"]
    new_w = metadata["new_w"]
    cropped = image[top:top + new_h, left:left + new_w]
    return cv2.resize(
        cropped,
        (metadata["original_w"], metadata["original_h"]),
        interpolation=cv2.INTER_LINEAR,
    )


def draw_bounding_box(
    img,
    class_id,
    confidence,
    x,
    y,
    x_plus_w,
    y_plus_h,
    draw_mode=0,
):
    """Draw one detector bounding box on ``img`` in place."""
    color = COLORS[0]
    thickness = 1 if img.shape[0] < 800 else 2
    if draw_mode == 0:
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, thickness)
    else:
        cv2.circle(img, (x, y), 2, color, -1)


def colormap_to_hex(cmap_name):
    """Convert a matplotlib colormap into a discrete list of HEX colors."""
    assert cmap_name in COLOR_NUMBER, f"incorrect colormap specified: {cmap_name}"
    num_colors = COLOR_NUMBER[cmap_name]
    cmap = plt.get_cmap(cmap_name)
    color_values = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    return [mcolors.to_hex(c) for c in color_values]


def hex_to_bgr(hex_colors):
    """Convert HEX color strings to OpenCV BGR tuples."""
    if isinstance(hex_colors, str):
        hex_colors = [hex_colors]
    bgr_colors = []
    for hex_color in hex_colors:
        rgb = [int(c * 255) for c in mcolors.hex2color(hex_color)]
        bgr_colors.append(tuple(reversed(rgb)))
    return bgr_colors


def denormalize_coordinates(coords, image_shape):
    """Convert normalized x/y coordinates to pixel coordinates."""
    return coords * np.array([image_shape[1], image_shape[0]])


def plot_predictions(
    image,
    pred_masks,
    filename: Optional[str] = IMAGE_FILE_NAME_DETECTION,
    alpha=0.75,
    colormap="tab20",
    color_ids=None,
):
    """Draw segmentation masks on an image and save the rendered result."""
    hex_colors = hex_to_bgr(colormap_to_hex(colormap))
    if pred_masks is None or len(pred_masks) == 0:
        print("No masks found.")
        if filename is not None:
            safe_image_write(image, filename)
        return image

    overlay = image.copy()
    for i, mask in enumerate(pred_masks):
        coords = np.asarray(mask, dtype=np.float32).reshape(-1, 2)
        if coords.shape[0] < 3:
            continue
        color_index = i if color_ids is None else int(color_ids[i])
        color = hex_colors[color_index % len(hex_colors)]
        if coords.max() <= 1.0:
            coords = denormalize_coordinates(coords, image.shape)
        coords[:, 0] = np.clip(coords[:, 0], 0, image.shape[1] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, image.shape[0] - 1)
        coords = np.round(coords).astype(np.int32)
        if len(np.unique(coords, axis=0)) < 3:
            continue
        cv2.fillPoly(overlay, [coords], color)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    if filename is not None:
        safe_image_write(image, filename)
    return image


def plot_predictions_with_alignment(
    original_image,
    img_inference,
    pred_masks,
    filename: str = IMAGE_FILE_NAME_DETECTION,
    colormap="tab20",
    alpha=0.75,
    color_ids=None,
    mask_coordinate_space="auto",
):
    """Draw masks that may be aligned to a preprocessed inference image."""
    h, w = img_inference.shape[:2]
    o_h, o_w = original_image.shape[:2]

    if mask_coordinate_space == "inference":
        inference_base = img_inference.copy() if hasattr(img_inference, "copy") else img_inference
        set_global(
            "image_display_base",
            inference_base.copy() if hasattr(inference_base, "copy") else inference_base,
        )
        return plot_predictions(
            inference_base,
            pred_masks,
            filename=filename,
            colormap=colormap,
            alpha=alpha,
            color_ids=color_ids,
        )

    if h == o_h and w == o_w:
        set_global("image_display_base", original_image.copy())
        return plot_predictions(
            original_image,
            pred_masks,
            filename=filename,
            colormap=colormap,
            alpha=alpha,
            color_ids=color_ids,
        )

    if mask_coordinate_space == "original":
        set_global("image_display_base", original_image.copy())
        return plot_predictions(
            original_image,
            pred_masks,
            filename=filename,
            colormap=colormap,
            alpha=alpha,
            color_ids=color_ids,
        )

    metadata = get_resize_and_pad_metadata(original_image.shape, w, h)
    aligned_image = resize_and_pad_cv(original_image, w, h)
    rendered_aligned = plot_predictions(
        aligned_image,
        pred_masks,
        filename=filename if mask_coordinate_space != "inference" else None,
        colormap=colormap,
        alpha=alpha,
        color_ids=color_ids,
    )
    rendered_original = restore_from_resize_and_pad(rendered_aligned, metadata)
    set_global("image_display_base", original_image.copy())
    safe_image_write(rendered_original, filename)
    return rendered_original


def render_detector_predictions(
    image,
    detections,
    filename: str = IMAGE_FILE_NAME_DETECTION,
    draw_mode=0,
):
    """Render detector-style bounding boxes from a detection DataFrame."""
    rendered = image.copy()
    if detections is None:
        safe_image_write(rendered, filename)
        return rendered

    for _, row in detections.iterrows():
        box = row["box"]
        scale = row["scale"] if "scale" in row else 1
        x = round(box[0] * scale)
        y = round(box[1] * scale)
        x_plus_w = round((box[0] + box[2]) * scale)
        y_plus_h = round((box[1] + box[3]) * scale)
        draw_bounding_box(
            rendered,
            row["class_id"] if "class_id" in row else 0,
            row["confidence"] if "confidence" in row else 1.0,
            x,
            y,
            x_plus_w,
            y_plus_h,
            draw_mode=draw_mode,
        )

    safe_image_write(rendered, filename)
    return rendered


def render_segmentation_predictions(
    image,
    detections,
    filename: str = IMAGE_FILE_NAME_DETECTION,
    colormap="tab20",
    alpha=0.75,
):
    """Render segmenter-style mask detections from a detection DataFrame."""
    if detections is None or "mask" not in detections:
        return plot_predictions(
            image,
            [],
            filename=filename,
            colormap=colormap,
            alpha=alpha,
        )

    color_ids = detections["id_label"].tolist() if "id_label" in detections else None
    return plot_predictions(
        image,
        detections["mask"].tolist(),
        filename=filename,
        colormap=colormap,
        alpha=alpha,
        color_ids=color_ids,
    )


def render_predictions(
    image,
    detections,
    filename: str = IMAGE_FILE_NAME_DETECTION,
    colormap="tab20",
    alpha=0.75,
):
    """Render either segmenter masks or detector boxes based on DataFrame columns."""
    if detections is not None and "mask" in detections:
        return render_segmentation_predictions(
            image,
            detections,
            filename=filename,
            colormap=colormap,
            alpha=alpha,
        )
    return render_detector_predictions(image, detections, filename=filename)
