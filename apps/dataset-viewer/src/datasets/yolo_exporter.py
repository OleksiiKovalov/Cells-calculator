import os
import shutil
from .base_loader import BaseDatasetLoader


def _image_size(path: str) -> tuple[int, int]:
    from PySide6.QtGui import QImageReader
    r = QImageReader(path)
    s = r.size()
    return (s.width(), s.height()) if s.isValid() else (0, 0)


def _ann_to_yolo_line(ann: dict, img_w: int, img_h: int) -> str | None:
    cls = ann['class_id']
    if ann.get('type') == 'polygon':
        points = ann.get('points') or (ann.get('polygons') or [[]])[0]
        if points and img_w and img_h:
            coords = ' '.join(f'{px / img_w:.6f} {py / img_h:.6f}' for px, py in points)
            return f'{cls} {coords}'
    # bbox (default)
    x, y, w, h = ann.get('x', 0), ann.get('y', 0), ann.get('w', 0), ann.get('h', 0)
    if img_w and img_h and w and h:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        return f'{cls} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}'
    return None


class YOLOExporter:
    def export(self, loader: BaseDatasetLoader, output_folder: str, progress_cb=None) -> None:
        splits = loader.get_splits()
        image_sets = [(s, loader.get_images(s)) for s in splits] if splits else [(None, loader.get_images())]
        total = sum(len(imgs) for _, imgs in image_sets)
        done = 0

        for split, images in image_sets:
            img_dir = os.path.join(output_folder, 'images', split) if split else os.path.join(output_folder, 'images')
            lbl_dir = os.path.join(output_folder, 'labels', split) if split else os.path.join(output_folder, 'labels')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            for img_info in images:
                src = img_info['path']
                dst = os.path.join(img_dir, os.path.basename(src))
                if os.path.normpath(src) != os.path.normpath(dst):
                    shutil.copy2(src, dst)

                img_w, img_h = _image_size(src)
                stem = os.path.splitext(os.path.basename(src))[0]
                with open(os.path.join(lbl_dir, stem + '.txt'), 'w', encoding='utf-8') as f:
                    for ann in loader.get_annotations(src):
                        line = _ann_to_yolo_line(ann, img_w, img_h)
                        if line:
                            f.write(line + '\n')

                done += 1
                if progress_cb and not progress_cb(done, total):
                    return

        self._write_yaml(loader, output_folder, splits)

    @staticmethod
    def _write_yaml(loader: BaseDatasetLoader, output_folder: str, splits: list[str]) -> None:
        names_str = ', '.join(repr(n) for n in loader.class_names)
        lines = [f'nc: {len(loader.class_names)}', f'names: [{names_str}]']
        if splits:
            lines.append('')
            for s in splits:
                lines.append(f'{s}: images/{s}')
        with open(os.path.join(output_folder, 'data.yaml'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
