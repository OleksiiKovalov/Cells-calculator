import os
import glob
from .base_loader import BaseDatasetLoader

_IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


class YOLOLoader(BaseDatasetLoader):
    def __init__(self, folder: str):
        super().__init__(folder)
        self._load_class_names()

    # ------------------------------------------------------------------
    # Class names
    # ------------------------------------------------------------------
    def _load_class_names(self):
        for pattern in ('*.yaml', '*.yml'):
            for yf in glob.glob(os.path.join(self.folder, pattern)):
                names = self._names_from_yaml(yf)
                if names:
                    self.class_names = names
                    return

        for nf in glob.glob(os.path.join(self.folder, '*.names')):
            try:
                with open(nf, encoding='utf-8') as f:
                    names = [ln.strip() for ln in f if ln.strip()]
                if names:
                    self.class_names = names
                    return
            except OSError:
                continue

    @staticmethod
    def _names_from_yaml(path: str) -> list:
        try:
            import yaml
            with open(path, encoding='utf-8') as f:
                data = yaml.safe_load(f)
            names = data.get('names')
            if isinstance(names, list):
                return names
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Splits / images
    # ------------------------------------------------------------------
    def get_splits(self) -> list[str]:
        images_root = os.path.join(self.folder, 'images')
        if not os.path.isdir(images_root):
            return []
        return sorted(
            d for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        )

    def get_images(self, split: str | None = None) -> list[dict]:
        candidates = self._image_dirs(split)
        seen: set[str] = set()
        images = []
        for d in candidates:
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                if os.path.splitext(fname)[1].lower() in _IMG_EXT:
                    full = os.path.normpath(os.path.join(d, fname))
                    if full not in seen:
                        seen.add(full)
                        images.append({'path': full, 'name': fname})
        return images

    def _image_dirs(self, split: str | None) -> list[str]:
        if split:
            return [
                os.path.join(self.folder, 'images', split),
                os.path.join(self.folder, split, 'images'),
                os.path.join(self.folder, split),
            ]
        return [
            os.path.join(self.folder, 'images'),
            self.folder,
        ]

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    def get_annotations(self, image_path: str) -> list[dict]:
        label_path = self._label_path(image_path)
        if not label_path or not os.path.isfile(label_path):
            return []

        img_w, img_h = self._image_size(image_path)
        if img_w == 0 or img_h == 0:
            return []

        result = []
        try:
            with open(label_path, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(parts[0])
                        coords = list(map(float, parts[1:]))
                    except ValueError:
                        continue

                    label = (self.class_names[cls]
                             if cls < len(self.class_names) else str(cls))

                    # Segmentation polygon: class_id x1 y1 x2 y2 ... (>4 coords, even count)
                    if len(coords) > 4 and len(coords) % 2 == 0:
                        points = [
                            (coords[i] * img_w, coords[i + 1] * img_h)
                            for i in range(0, len(coords), 2)
                        ]
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x, y = min(xs), min(ys)
                        w, h = max(xs) - x, max(ys) - y
                        result.append({
                            'class_id': cls,
                            'label': label,
                            'type': 'polygon',
                            'points': points,
                            'x': x, 'y': y, 'w': w, 'h': h,
                        })
                    else:
                        # Bounding box: class_id cx cy w h (normalized)
                        cx, cy, nw, nh = coords[:4]
                        x = (cx - nw / 2) * img_w
                        y = (cy - nh / 2) * img_h
                        result.append({
                            'class_id': cls,
                            'label': label,
                            'type': 'bbox',
                            'x': x, 'y': y, 'w': nw * img_w, 'h': nh * img_h,
                        })
        except OSError:
            pass
        return result

    @staticmethod
    def _label_path(image_path: str) -> str:
        """Swap the 'images' component for 'labels' and change extension to .txt."""
        norm = image_path.replace('\\', '/')
        parts = norm.split('/')
        for i, part in enumerate(parts):
            if part == 'images':
                parts[i] = 'labels'
                candidate = os.path.normpath('/'.join(parts))
                return os.path.splitext(candidate)[0] + '.txt'
        # Flat layout: same directory
        return os.path.splitext(image_path)[0] + '.txt'

    @staticmethod
    def _image_size(path: str) -> tuple[int, int]:
        from PySide6.QtGui import QImageReader
        reader = QImageReader(path)
        size = reader.size()
        return (size.width(), size.height()) if size.isValid() else (0, 0)
