import os
import json
import shutil
from .base_loader import BaseDatasetLoader


def _image_size(path: str) -> tuple[int, int]:
    from PyQt5.QtGui import QImageReader
    r = QImageReader(path)
    s = r.size()
    return (s.width(), s.height()) if s.isValid() else (0, 0)


def _segmentation(ann: dict) -> list:
    if ann.get('type') == 'polygon':
        points = ann.get('points')
        if points:
            return [[coord for pt in points for coord in (round(pt[0], 2), round(pt[1], 2))]]
        polygons = ann.get('polygons')
        if polygons:
            return [[coord for pt in poly for coord in (round(pt[0], 2), round(pt[1], 2))]
                    for poly in polygons]
    return []


class COCOExporter:
    def export(self, loader: BaseDatasetLoader, output_folder: str, progress_cb=None) -> None:
        splits = loader.get_splits()
        image_sets = [(s, loader.get_images(s)) for s in splits] if splits else [('default', loader.get_images())]
        total = sum(len(imgs) for _, imgs in image_sets)
        done = 0

        img_dir = os.path.join(output_folder, 'images')
        ann_dir = os.path.join(output_folder, 'annotations')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        categories = [{'id': i + 1, 'name': n, 'supercategory': 'none'}
                      for i, n in enumerate(loader.class_names)]
        cat_id_map = {i: i + 1 for i in range(len(loader.class_names))}

        img_id = 0
        ann_id = 0

        for split, images in image_sets:
            coco_images: list[dict] = []
            coco_anns: list[dict] = []

            for img_info in images:
                src = img_info['path']
                base = os.path.basename(src)
                dst = os.path.join(img_dir, base)
                if os.path.normpath(src) != os.path.normpath(dst):
                    shutil.copy2(src, dst)

                img_w, img_h = _image_size(src)
                img_id += 1
                coco_images.append({'id': img_id, 'file_name': base,
                                    'width': img_w, 'height': img_h})

                for ann in loader.get_annotations(src):
                    cls = ann['class_id']
                    x, y, w, h = ann.get('x', 0), ann.get('y', 0), ann.get('w', 0), ann.get('h', 0)
                    ann_id += 1
                    coco_anns.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': cat_id_map.get(cls, cls + 1),
                        'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                        'area': round(w * h, 2),
                        'segmentation': _segmentation(ann),
                        'iscrowd': 0,
                    })

                done += 1
                if progress_cb and not progress_cb(done, total):
                    return

            json_name = f'instances_{split}.json'
            with open(os.path.join(ann_dir, json_name), 'w', encoding='utf-8') as f:
                json.dump({
                    'info': {'description': 'Exported by Dataset Viewer'},
                    'categories': categories,
                    'images': coco_images,
                    'annotations': coco_anns,
                }, f, indent=2)
