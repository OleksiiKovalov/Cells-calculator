import os
import json
import glob


def detect_format(folder: str):
    """Returns (loader_instance, format_name) or (None, None)."""
    # Direct .pth file path
    if folder.lower().endswith('.pth') and os.path.isfile(folder):
        from .pth_loader import PTHLoader
        return PTHLoader(folder), "InstanSeg PTH"

    if not os.path.isdir(folder):
        return None, None

    # --- COCO: annotations/*.json or root *.json with images+annotations+categories ---
    ann_dir = os.path.join(folder, 'annotations')
    if os.path.isdir(ann_dir):
        for jf in glob.glob(os.path.join(ann_dir, '*.json')):
            if _is_coco_json(jf):
                from .coco_loader import COCOLoader
                return COCOLoader(folder), "COCO"

    for jf in glob.glob(os.path.join(folder, '*.json')):
        if _is_coco_json(jf):
            from .coco_loader import COCOLoader
            return COCOLoader(folder), "COCO"

    # --- Pascal VOC: Annotations/*.xml directory ---
    voc_ann_dir = os.path.join(folder, 'Annotations')
    if os.path.isdir(voc_ann_dir) and glob.glob(os.path.join(voc_ann_dir, '*.xml')):
        from .voc_loader import VOCLoader
        return VOCLoader(folder), "Pascal VOC"

    # --- YOLO: images/ + labels/ dirs, or .yaml/.names config file ---
    has_images = os.path.isdir(os.path.join(folder, 'images'))
    has_labels = os.path.isdir(os.path.join(folder, 'labels'))
    has_yaml = bool(
        glob.glob(os.path.join(folder, '*.yaml')) +
        glob.glob(os.path.join(folder, '*.yml'))
    )
    has_names = bool(glob.glob(os.path.join(folder, '*.names')))

    if has_images and has_labels:
        from .yolo_loader import YOLOLoader
        return YOLOLoader(folder), "YOLO"

    if has_yaml or has_names:
        from .yolo_loader import YOLOLoader
        return YOLOLoader(folder), "YOLO"

    try:
        entries = os.listdir(folder)
    except OSError:
        return None, None

    # --- YOLO split layout: <split>/images + <split>/labels subfolders ---
    if any(
        os.path.isdir(os.path.join(folder, d, 'images')) and
        os.path.isdir(os.path.join(folder, d, 'labels'))
        for d in entries
    ):
        from .yolo_loader import YOLOLoader
        return YOLOLoader(folder), "YOLO"

    # --- YOLO flat: image files paired with same-stem .txt files ---

    _IMG = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    txt_stems = {os.path.splitext(f)[0] for f in entries if f.lower().endswith('.txt')}
    img_files = [f for f in entries if os.path.splitext(f)[1].lower() in _IMG]

    if img_files and any(os.path.splitext(f)[0] in txt_stems for f in img_files):
        from .yolo_loader import YOLOLoader
        return YOLOLoader(folder), "YOLO"

    # Folder containing a .pth file
    pth_files = glob.glob(os.path.join(folder, '*.pth'))
    if pth_files:
        from .pth_loader import PTHLoader
        return PTHLoader(sorted(pth_files)[0]), "InstanSeg PTH"

    return None, None


def _is_coco_json(path: str) -> bool:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Cheap partial read — only parse first 8 KB to check keys
            chunk = f.read(8192)
        return ('"images"' in chunk and
                '"annotations"' in chunk and
                '"categories"' in chunk)
    except Exception:
        return False
