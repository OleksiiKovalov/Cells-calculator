import os
import shutil
from xml.etree import ElementTree as ET
from .base_loader import BaseDatasetLoader


def _image_size(path: str) -> tuple[int, int]:
    from PySide6.QtGui import QImageReader
    r = QImageReader(path)
    s = r.size()
    return (s.width(), s.height()) if s.isValid() else (0, 0)


class VOCExporter:
    def export(self, loader: BaseDatasetLoader, output_folder: str, progress_cb=None) -> None:
        splits = loader.get_splits()
        image_sets = [(s, loader.get_images(s)) for s in splits] if splits else [(None, loader.get_images())]
        total = sum(len(imgs) for _, imgs in image_sets)
        done = 0

        img_dir = os.path.join(output_folder, 'JPEGImages')
        ann_dir = os.path.join(output_folder, 'Annotations')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        split_stems: dict[str, list[str]] = {}

        for split, images in image_sets:
            stems: list[str] = []
            for img_info in images:
                src = img_info['path']
                base = os.path.basename(src)
                dst = os.path.join(img_dir, base)
                if os.path.normpath(src) != os.path.normpath(dst):
                    shutil.copy2(src, dst)

                stem = os.path.splitext(base)[0]
                stems.append(stem)

                img_w, img_h = _image_size(src)
                xml_str = self._build_xml(base, img_w, img_h, loader.get_annotations(src))
                with open(os.path.join(ann_dir, stem + '.xml'), 'w', encoding='utf-8') as f:
                    f.write(xml_str)

                done += 1
                if progress_cb and not progress_cb(done, total):
                    return

            if split:
                split_stems[split] = stems

        if split_stems:
            sets_dir = os.path.join(output_folder, 'ImageSets', 'Main')
            os.makedirs(sets_dir, exist_ok=True)
            for split, stems in split_stems.items():
                with open(os.path.join(sets_dir, split + '.txt'), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(stems) + '\n')

    @staticmethod
    def _build_xml(filename: str, img_w: int, img_h: int, annotations: list[dict]) -> str:
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = filename

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(img_w)
        ET.SubElement(size, 'height').text = str(img_h)
        ET.SubElement(size, 'depth').text = '3'

        for ann in annotations:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = ann.get('label', str(ann.get('class_id', 0)))
            ET.SubElement(obj, 'difficult').text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            x, y, w, h = ann.get('x', 0), ann.get('y', 0), ann.get('w', 0), ann.get('h', 0)
            ET.SubElement(bndbox, 'xmin').text = str(int(round(x)))
            ET.SubElement(bndbox, 'ymin').text = str(int(round(y)))
            ET.SubElement(bndbox, 'xmax').text = str(int(round(x + w)))
            ET.SubElement(bndbox, 'ymax').text = str(int(round(y + h)))

            # Extended-VOC <polygon> — the same form VOCLoader reads back.
            if ann.get('type') == 'polygon':
                points = ann.get('points') or (ann.get('polygons') or [[]])[0]
                if len(points) >= 3:
                    poly = ET.SubElement(obj, 'polygon')
                    for i, (px, py) in enumerate(points, start=1):
                        ET.SubElement(poly, f'x{i}').text = f'{px:.2f}'
                        ET.SubElement(poly, f'y{i}').text = f'{py:.2f}'

        ET.indent(root, space='  ')
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding='unicode') + '\n'
