import os
import numpy as np
from .base_loader import BaseDatasetLoader
from .rle import decode_mask


def _load_image_as_array(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'))


def _fill_polygon(mask: np.ndarray, points: list, value: int) -> None:
    from PIL import Image, ImageDraw
    h, w = mask.shape
    tmp = Image.new('L', (w, h), 0)
    flat = [coord for pt in points for coord in (float(pt[0]), float(pt[1]))]
    ImageDraw.Draw(tmp).polygon(flat, fill=1)
    mask[np.asarray(tmp) > 0] = value


def _anns_to_instance_mask(annotations: list[dict], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.int32)
    for iid, ann in enumerate(annotations, start=1):
        x = int(ann.get('x', 0))
        y = int(ann.get('y', 0))
        bw = int(ann.get('w', 0))
        bh = int(ann.get('h', 0))

        if ann.get('type') == 'polygon':
            points = ann.get('points') or (ann.get('polygons') or [[]])[0]
            if points:
                _fill_polygon(mask, points, iid)
                continue

        if ann.get('type') == 'mask' and ann.get('rle_counts') and ann.get('rle_size'):
            sub = decode_mask(ann['rle_counts'], ann['rle_size'])
            hh, ww = min(h, sub.shape[0]), min(w, sub.shape[1])
            region = sub[:hh, :ww] > 0
            if region.any():
                mask[:hh, :ww][region] = iid
                continue

        x2 = min(x + bw, w)
        y2 = min(y + bh, h)
        if x2 > x and y2 > y:
            mask[y:y2, x:x2] = iid

    return mask


def _split_label(split: str) -> str:
    """Canonical InstanSeg split key for a source split name.

    InstanSeg's training/test loaders key on 'Train' / 'Validation' / 'Test', so
    a YOLO/COCO 'val' (or 'valid') split must map to 'Validation' — not 'Val',
    which the loader would silently ignore.
    """
    low = split.lower()
    if 'train' in low:
        return 'Train'
    if 'val' in low:
        return 'Validation'
    if 'test' in low:
        return 'Test'
    return split.title()


class PTHExporter:
    """Export a loaded dataset to an InstanSeg ``.pth`` (a ``torch.save`` dict).

    Preprocessing and train/val/test splitting are NOT done here — wrap the
    source loader in :class:`~datasets.prepared_loader.PreparedLoader` for that.
    This exporter only turns whatever the loader presents into the InstanSeg
    schema. Options:

    * ``filename`` — output file name (default ``'dataset.pth'``).
    * ``modality`` — ``image_modality`` tag on every item (default ``'Brightfield'``).
    * ``mask_key`` — ``'cell_masks'`` / ``'nucleus_masks'``; ``None`` auto-picks
      from the class names (``'cell'`` → cell, else nucleus).
    """

    def __init__(self, *, filename: str = 'dataset.pth', modality: str = 'Brightfield',
                 mask_key: str | None = None):
        self.filename = filename if filename.endswith('.pth') else f'{filename}.pth'
        self.modality = modality
        self.mask_key = mask_key

    def export(self, loader: BaseDatasetLoader, output_folder: str, progress_cb=None) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to save .pth files: pip install torch")

        splits = loader.get_splits()
        image_sets = [(s, loader.get_images(s)) for s in splits] if splits else [('default', loader.get_images())]
        total = sum(len(imgs) for _, imgs in image_sets)
        done = 0

        mask_key = self.mask_key or (
            'cell_masks' if (loader.class_names and loader.class_names[0] == 'cell') else 'nucleus_masks')
        parent_dataset = os.path.basename(os.path.normpath(loader.folder))

        dataset: dict[str, list] = {}
        for split, images in image_sets:
            items = []
            for img_info in images:
                src = img_info['path']
                image = _load_image_as_array(src)
                h, w = image.shape[:2]
                mask = _anns_to_instance_mask(loader.get_annotations(src), h, w)
                items.append({
                    'image': image,
                    'file_name': src,
                    'parent_dataset': parent_dataset,
                    'image_modality': self.modality,
                    mask_key: mask,
                })
                done += 1
                if progress_cb and not progress_cb(done, total):
                    return
            dataset.setdefault(_split_label(split), []).extend(items)

        os.makedirs(output_folder, exist_ok=True)
        torch.save(dataset, os.path.join(output_folder, self.filename))
