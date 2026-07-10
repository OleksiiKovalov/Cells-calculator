import os
import atexit
import shutil
import tempfile
import numpy as np
from .base_loader import BaseDatasetLoader
from .rle import encode_mask


def _save_image(array: np.ndarray, path: str) -> None:
    from PIL import Image
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 1:
        img = Image.fromarray(arr[:, :, 0], mode='L')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr, mode='RGB')
    elif arr.ndim == 3 and arr.shape[2] == 4:
        img = Image.fromarray(arr, mode='RGBA')
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    img.save(path)


def _mask_to_annotations(item: dict) -> list[dict]:
    mask = item.get('cell_masks') if 'cell_masks' in item else item.get('nucleus_masks')
    if mask is None:
        return []
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    mask = np.asarray(mask)
    label = 'cell' if 'cell_masks' in item else 'nucleus'

    result = []
    for iid in np.unique(mask):
        if iid == 0:
            continue
        instance = mask == iid
        rows, cols = np.where(instance)
        if not len(rows):
            continue
        x, y = int(cols.min()), int(rows.min())
        w = int(cols.max()) - x + 1
        h = int(rows.max()) - y + 1
        # Real mask shape as uncompressed RLE (the viewer renders 'mask'
        # directly); bbox fields stay for exporters without mask support.
        result.append({
            'class_id': 0,
            'label': label,
            'type': 'mask',
            'rle_counts': encode_mask(instance),
            'rle_size': [int(instance.shape[0]), int(instance.shape[1])],
            'x': float(x), 'y': float(y),
            'w': float(w), 'h': float(h),
        })
    return result


class PTHLoader(BaseDatasetLoader):
    def __init__(self, pth_path: str):
        super().__init__(os.path.dirname(os.path.abspath(pth_path)))
        self._pth_path = pth_path
        self._splits: dict[str, list[dict]] = {}   # split_key -> [image_info]
        self._path_to_item: dict[str, dict] = {}   # abs image path -> pth item
        self._temp_dir = tempfile.mkdtemp(prefix='dv_pth_')
        atexit.register(self._cleanup)
        self._load()

    def _load(self) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to load .pth files: pip install torch")

        raw = torch.load(self._pth_path, map_location='cpu', weights_only=False)
        counter = 0
        for split, items in raw.items():
            if not items:
                continue
            split_key = split.lower()
            self._splits[split_key] = []
            for item in items:
                path = self._resolve_path(item, counter)
                counter += 1
                self._path_to_item[path] = item
                self._splits[split_key].append({'path': path, 'name': os.path.basename(path)})

        # Class names from mask type in the first item
        for split_items in raw.values():
            if split_items:
                first = split_items[0]
                if 'cell_masks' in first:
                    self.class_names = ['cell']
                elif 'nucleus_masks' in first:
                    self.class_names = ['nucleus']
                break
        if not self.class_names:
            self.class_names = ['object']

    def _resolve_path(self, item: dict, counter: int) -> str:
        file_name = item.get('file_name', '')
        if file_name and os.path.isfile(file_name):
            return str(os.path.normpath(file_name))
        stem = os.path.splitext(os.path.basename(file_name or f'image_{counter}'))[0]
        out_path = os.path.join(self._temp_dir, f'{counter:04d}_{stem}.png')
        _save_image(np.asarray(item['image']), out_path)
        return out_path

    def get_splits(self) -> list[str]:
        return list(self._splits.keys()) if len(self._splits) > 1 else []

    def get_images(self, split: str | None = None) -> list[dict]:
        if split:
            return list(self._splits.get(split, []))
        return [img for imgs in self._splits.values() for img in imgs]

    def get_annotations(self, image_path: str) -> list[dict]:
        item = self._path_to_item.get(os.path.normpath(image_path))
        return _mask_to_annotations(item) if item else []

    def _cleanup(self) -> None:
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __del__(self) -> None:
        self._cleanup()
