import os
import json
import glob
from .base_loader import BaseDatasetLoader


def _decode_compressed_rle(counts_str: str, size: list) -> list | None:
    """Decode COCO LEB128-compressed RLE string to a list of run lengths."""
    # Algorithm from the COCO API (maskApi.c :: rleFrString)
    try:
        cnts: list[int] = []
        m, p = 0, 0
        s = counts_str
        while p < len(s):
            x, k, more = 0, 0, 1
            while more:
                c = ord(s[p]) - 48
                p += 1
                more = c & 32
                x |= (c & 31) << (5 * k)
                k += 1
                if not more and (c & 16):
                    x |= -1 << (5 * k)   # sign-extend negative deltas
            if m > 2:
                x += cnts[m - 2]         # counts are deltas vs. two back
            cnts.append(x)
            m += 1
        return cnts
    except Exception:
        return None


class COCOLoader(BaseDatasetLoader):
    def __init__(self, folder: str):
        super().__init__(folder)
        self._images: dict[int, dict] = {}       # image_id -> info
        self._annotations: dict[int, list] = {}  # image_id -> [ann]
        self._categories: dict[int, str] = {}    # cat_id -> name
        self._path_to_id: dict[str, int] = {}    # resolved abs path -> image_id
        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_all(self):
        for jf in self._find_json_files():
            self._load_json(jf)
        if self._categories and not self.class_names:
            self.class_names = [
                self._categories[k] for k in sorted(self._categories)
            ]
        # Assign class ids only once every file is loaded, so indices are
        # stable even when a later JSON introduces new categories.
        cat_ids = sorted(self._categories)
        for anns in self._annotations.values():
            for entry in anns:
                cat_id = entry.pop('cat_id')
                entry['class_id'] = cat_ids.index(cat_id) if cat_id in self._categories else 0
                entry['label'] = self._categories.get(cat_id, str(cat_id))

    def _find_json_files(self) -> list[str]:
        files = []
        ann_dir = os.path.join(self.folder, 'annotations')
        if os.path.isdir(ann_dir):
            files += glob.glob(os.path.join(ann_dir, '*.json'))
        files += glob.glob(os.path.join(self.folder, '*.json'))
        return files

    def _load_json(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return

        if not all(k in data for k in ('images', 'annotations', 'categories')):
            return

        split = self._split_from_filename(path)

        for cat in data['categories']:
            self._categories[cat['id']] = cat['name']

        for img in data['images']:
            self._images[img['id']] = {
                'id': img['id'],
                'file_name': img['file_name'],
                'width': img.get('width', 0),
                'height': img.get('height', 0),
                'split': split,
            }

        for ann in data['annotations']:
            img_id = ann['image_id']
            bbox = ann.get('bbox', [0, 0, 0, 0])  # [x, y, w, h] absolute pixels
            entry = {
                'cat_id': ann['category_id'],  # class_id/label filled in _load_all
                'x': bbox[0], 'y': bbox[1],
                'w': bbox[2], 'h': bbox[3],
            }
            entry.update(self._parse_segmentation(ann.get('segmentation')))
            self._annotations.setdefault(img_id, []).append(entry)

    @staticmethod
    def _parse_segmentation(seg) -> dict:
        """Return type-specific fields to merge into an annotation entry."""
        if isinstance(seg, list) and seg:
            # Polygon format: [[x1,y1,x2,y2,...], ...]  (pixels, absolute)
            polygons = []
            for part in seg:
                if isinstance(part, list) and len(part) >= 6:
                    pts = [(part[i], part[i + 1]) for i in range(0, len(part) - 1, 2)]
                    polygons.append(pts)
            if polygons:
                return {'type': 'polygon', 'polygons': polygons}

        if isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
            counts = seg['counts']
            size = seg['size']   # [height, width]
            if isinstance(counts, list):
                # Uncompressed RLE
                return {'type': 'mask', 'rle_counts': counts, 'rle_size': size}
            if isinstance(counts, str):
                # LEB128-compressed RLE — decode inline (no pycocotools needed)
                decoded = _decode_compressed_rle(counts, size)
                if decoded is not None:
                    return {'type': 'mask', 'rle_counts': decoded, 'rle_size': size}

        return {'type': 'bbox'}

    @staticmethod
    def _split_from_filename(path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        for s in ('train', 'val', 'test'):
            if s in stem:
                return s
        return 'all'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_splits(self) -> list[str]:
        splits = sorted({info['split'] for info in self._images.values()})
        return splits if len(splits) > 1 else []

    def get_images(self, split: str | None = None) -> list[dict]:
        result = []
        for img_id, info in self._images.items():
            if split and info['split'] != split:
                continue
            path = self._resolve_path(info)
            if path:
                self._path_to_id[path] = img_id
                result.append({'path': path, 'name': info['file_name'], 'id': img_id})
        return sorted(result, key=lambda x: x['name'])

    def get_annotations(self, image_path: str) -> list[dict]:
        norm = os.path.normpath(image_path)
        img_id = self._path_to_id.get(norm)
        if img_id is None:
            # Fallback for paths not seen via get_images: match by basename.
            base = os.path.basename(image_path)
            for iid, info in self._images.items():
                if os.path.basename(info['file_name']) == base:
                    img_id = iid
                    break
        if img_id is None:
            return []
        return list(self._annotations.get(img_id, []))

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    def _resolve_path(self, info: dict) -> str | None:
        file_name = info['file_name']
        split = info['split']
        base = os.path.basename(file_name)
        candidates = [
            os.path.join(self.folder, file_name),
            os.path.join(self.folder, 'images', base),
            os.path.join(self.folder, split, base),
            os.path.join(self.folder, 'images', split, base),
            os.path.join(self.folder, f'{split}2017', base),
            os.path.join(self.folder, f'{split}2014', base),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return os.path.normpath(p)
        return None
