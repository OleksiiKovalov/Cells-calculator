"""Window-less image-preview helpers for Training Studio's GUI.

Backs the GUI's *Open Image* / *Open Mask* preview: every function here
*returns* arrays and NEVER opens a window. Imports are limited to the lightweight
stack (numpy, cv2, PIL, skimage); nothing here imports torch or instanseg, so the
previews work in a bare environment.
"""

from pathlib import Path

import numpy as np

# Extensions the app treats as images throughout (readers/scanners/viewer).
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
MASK_SUFFIX = "_masks"


# ===========================================================================
# I/O helpers
# ===========================================================================
def load_image_rgb(path):
    """Load any supported image from disk as an RGB uint8 array.

    Uses skimage.io so multi-page/odd-bit TIFFs load the same way InstanSeg
    reads them; grayscale is promoted to 3 channels, RGBA is truncated to RGB.
    Raises FileNotFoundError if the file cannot be read.
    """
    from skimage import io

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    arr = io.imread(str(path))
    arr = np.asarray(arr)

    if arr.ndim == 2:                       # grayscale -> RGB
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]                  # drop alpha
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        # channel-first (C,H,W) -> (H,W,C)
        arr = np.moveaxis(arr, 0, -1)[..., :3]

    return _to_uint8(arr)


def load_mask_preview(path):
    """Load an instance mask (any format) as a colourised RGB uint8 preview.

    Each distinct positive label gets a distinct colour so the viewer shows the
    instances; background (0) stays black.
    """
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        from skimage import io
        mask = np.asarray(io.imread(str(path)))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return colorize_labels(mask)


def _to_uint8(arr):
    """Best-effort normalisation of any numeric image to uint8 (cell 9 idea)."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= 255 and lo >= 0:
        return arr.astype(np.uint8)
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def colorize_labels(mask):
    """Map an integer label image to a deterministic RGB uint8 image."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        mask = mask.reshape(mask.shape[:2])
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    labels = np.unique(mask)
    labels = labels[labels != 0]
    rng = np.random.default_rng(12345)
    for lab in labels:
        color = rng.integers(60, 256, size=3, dtype=np.int64)
        out[mask == lab] = color.astype(np.uint8)
    return out


# ===========================================================================
# Directory scanning (image <-> mask pairing helpers)
# ===========================================================================
def is_mask_file(name):
    """True if a filename is a mask companion (``*_masks.<ext>``)."""
    stem = Path(name).stem
    return stem.endswith(MASK_SUFFIX)


def find_image_files(folder):
    """Return sorted image files in ``folder`` that are NOT mask companions."""
    folder = Path(folder)
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_mask_file(p.name):
            files.append(p)
    return files


def mask_path_for(image_path, image_format=None):
    """Companion mask path for an image: ``<stem>_masks<ext>``.

    ``image_format`` overrides the extension (e.g. force ``.tif`` masks); when
    omitted the image's own extension is reused.
    """
    image_path = Path(image_path)
    ext = image_format or image_path.suffix
    return image_path.with_name(f"{image_path.stem}{MASK_SUFFIX}{ext}")


def summarize_folder(folder):
    """Return a dict describing image/mask counts in a folder (for the GUI)."""
    folder = Path(folder)
    if not folder.is_dir():
        return {"exists": False, "n_images": 0, "n_masks": 0}
    images = find_image_files(folder)
    masks = [p for p in folder.iterdir()
             if p.is_file() and is_mask_file(p.name) and p.suffix.lower() in IMAGE_EXTS]
    paired = sum(1 for img in images if mask_path_for(img).is_file()
                 or mask_path_for(img, ".tif").is_file())
    return {
        "exists": True,
        "n_images": len(images),
        "n_masks": len(masks),
        "n_paired": paired,
    }
