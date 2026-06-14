import os
import glob
from xml.etree import ElementTree as ET
from .base_loader import BaseDatasetLoader

_IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
# Candidate image directory names in preference order
_IMG_DIR_NAMES = ('JPEGImages', 'images', 'Images', 'imgs', 'img')
# Only top-level split files (skip class-specific ones like dog_train.txt)
_KNOWN_SPLITS = {'train', 'val', 'test', 'trainval'}


class VOCLoader(BaseDatasetLoader):
    def __init__(self, folder: str):
        super().__init__(folder)
        self._ann_dir = os.path.join(folder, 'Annotations')
        self._img_dirs = self._find_image_dirs()
        self._splits: dict[str, list[str]] = self._load_splits()
        self._collect_classes()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _find_image_dirs(self) -> list[str]:
        for name in _IMG_DIR_NAMES:
            d = os.path.join(self.folder, name)
            if os.path.isdir(d):
                return [d]
        return [self.folder]

    def _load_splits(self) -> dict[str, list[str]]:
        """Parse ImageSets/Main/*.txt → {split_name: [image_stem, ...]}."""
        splits: dict[str, list[str]] = {}
        for candidate in (
            os.path.join(self.folder, 'ImageSets', 'Main'),
            os.path.join(self.folder, 'ImageSets'),
        ):
            if not os.path.isdir(candidate):
                continue
            for txt in glob.glob(os.path.join(candidate, '*.txt')):
                name = os.path.splitext(os.path.basename(txt))[0].lower()
                if name not in _KNOWN_SPLITS:
                    continue
                try:
                    with open(txt, encoding='utf-8') as f:
                        # VOC lines can be "stem" or "stem  1/-1" (difficult flag)
                        stems = [ln.strip().split()[0] for ln in f if ln.strip()]
                    if stems:
                        splits[name] = stems
                except OSError:
                    continue
            break
        return splits

    def _collect_classes(self):
        """Sample up to 500 XMLs to build the class-name list."""
        xml_files = glob.glob(os.path.join(self._ann_dir, '*.xml'))
        names: set[str] = set()
        for xml_file in xml_files[:500]:
            try:
                tree = ET.parse(xml_file)
                for obj in tree.findall('object'):
                    el = obj.find('name')
                    if el is not None and el.text:
                        names.add(el.text.strip())
            except Exception:
                continue
        self.class_names = sorted(names)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_splits(self) -> list[str]:
        return sorted(self._splits.keys())

    def get_images(self, split: str | None = None) -> list[dict]:
        if split and split in self._splits:
            result = []
            for stem in self._splits[split]:
                path = self._find_image(stem)
                if path:
                    result.append({'path': path, 'name': os.path.basename(path)})
            return result

        # No split: enumerate all XMLs that have a matching image
        result = []
        for xml_file in sorted(glob.glob(os.path.join(self._ann_dir, '*.xml'))):
            stem = os.path.splitext(os.path.basename(xml_file))[0]
            path = self._find_image(stem)
            if path:
                result.append({'path': path, 'name': os.path.basename(path)})
        return result

    def get_annotations(self, image_path: str) -> list[dict]:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        xml_path = os.path.join(self._ann_dir, stem + '.xml')
        if not os.path.isfile(xml_path):
            return []
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return []

        result = []
        for obj in root.findall('object'):
            name_el = obj.find('name')
            if name_el is None or not name_el.text:
                continue
            name = name_el.text.strip()
            cls = (self.class_names.index(name)
                   if name in self.class_names else 0)

            entry = self._parse_object(obj, name, cls)
            if entry:
                result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_object(self, obj, name: str, cls: int) -> dict | None:
        # Bounding box (always present in standard VOC)
        bndbox = obj.find('bndbox')
        if bndbox is None:
            return None
        try:
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
        except (AttributeError, ValueError, TypeError):
            return None

        entry: dict = {
            'class_id': cls,
            'label': name,
            'type': 'bbox',
            'x': xmin, 'y': ymin,
            'w': xmax - xmin, 'h': ymax - ymin,
        }

        # Optional polygon (some extended VOC datasets include <polygon>)
        polygon_el = obj.find('polygon')
        if polygon_el is not None:
            pts = self._parse_polygon(polygon_el)
            if pts:
                entry['type'] = 'polygon'
                entry['points'] = pts

        return entry

    @staticmethod
    def _parse_polygon(polygon_el) -> list[tuple]:
        pts = []
        i = 1
        while True:
            x_el = polygon_el.find(f'x{i}')
            y_el = polygon_el.find(f'y{i}')
            if x_el is None or y_el is None:
                break
            try:
                pts.append((float(x_el.text), float(y_el.text)))
            except (ValueError, TypeError):
                break
            i += 1
        return pts if len(pts) >= 3 else []

    def _find_image(self, stem: str) -> str | None:
        for img_dir in self._img_dirs:
            for ext in _IMG_EXT:
                for variant in (ext, ext.upper(), ext.capitalize()):
                    p = os.path.normpath(os.path.join(img_dir, stem + variant))
                    if os.path.isfile(p):
                        return p
        return None
