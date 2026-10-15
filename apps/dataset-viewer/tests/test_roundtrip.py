"""Exporting a dataset and re-loading it preserves image and annotation counts."""
import pytest

from datasets.format_detector import detect_format
from datasets.yolo_exporter import YOLOExporter
from datasets.coco_exporter import COCOExporter
from datasets.voc_exporter import VOCExporter
from datasets.pth_exporter import PTHExporter
from tests.conftest import sample


def _counts(loader):
    splits = loader.get_splits()
    imgs = ([im for s in splits for im in loader.get_images(s)]
            if splits else loader.get_images())
    anns = sum(len(loader.get_annotations(im["path"])) for im in imgs)
    return len(imgs), anns


@pytest.mark.parametrize("name,exporter_cls", [
    ("YOLO", YOLOExporter),
    ("COCO", COCOExporter),
    ("Pascal VOC", VOCExporter),
    ("InstanSeg PTH", PTHExporter),
])
def test_roundtrip_preserves_counts(name, exporter_cls, tmp_path):
    src, _ = detect_format(sample("yolo_seg"))
    src_images, src_anns = _counts(src)

    out = str(tmp_path / name.replace(" ", "_"))
    exporter_cls().export(src, out, lambda done, total: True)

    reloaded, fmt = detect_format(out)
    assert reloaded is not None, f"{name}: exported dataset not re-detected"
    img_count, ann_count = _counts(reloaded)
    assert img_count == src_images, f"{name}: image count drifted"
    assert ann_count == src_anns, f"{name}: annotation count drifted"


def _write_png(path, w=10, h=10):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(Qt.white)
    assert img.save(str(path))


def test_coco_export_keeps_same_named_images_per_split(tmp_path):
    """Same file name in two splits must not overwrite on COCO export, and
    each split's copy must keep its own annotations on reload."""
    src_root = tmp_path / "src"
    labels = {"train": "0 0.5 0.5 0.2 0.2\n",
              "val": "0 0.3 0.3 0.1 0.1\n0 0.7 0.7 0.1 0.1\n"}
    for split, lines in labels.items():
        (src_root / "images" / split).mkdir(parents=True)
        (src_root / "labels" / split).mkdir(parents=True)
        _write_png(src_root / "images" / split / "a.png")
        (src_root / "labels" / split / "a.txt").write_text(lines)

    src, _ = detect_format(str(src_root))
    out = tmp_path / "coco"
    COCOExporter().export(src, str(out), lambda d, t: True)

    reloaded, fmt = detect_format(str(out))
    assert fmt == "COCO"
    per_split = {
        s: [len(reloaded.get_annotations(im["path"])) for im in reloaded.get_images(s)]
        for s in reloaded.get_splits()
    }
    assert per_split == {"train": [1], "val": [2]}


def test_voc_roundtrip_preserves_polygons(tmp_path):
    """The VOC exporter writes <polygon> elements the VOC loader reads back."""
    src, _ = detect_format(sample("yolo_seg"))
    out = str(tmp_path / "voc")
    VOCExporter().export(src, out, lambda d, t: True)

    reloaded, fmt = detect_format(out)
    assert fmt == "Pascal VOC"
    imgs = [im for s in reloaded.get_splits() for im in reloaded.get_images(s)]
    anns = [a for im in imgs for a in reloaded.get_annotations(im["path"])]
    assert anns and all(a["type"] == "polygon" for a in anns)
    assert all(len(a["points"]) >= 3 for a in anns)


def test_coco_export_uses_global_image_ids(tmp_path):
    """Regression: per-split image-id reset would collide when COCOLoader merges
    the JSON files, dropping the first image of later splits."""
    src, _ = detect_format(sample("yolo_seg"))
    out = str(tmp_path / "coco")
    COCOExporter().export(src, out, lambda d, t: True)

    reloaded, _ = detect_format(out)
    images = [im for s in reloaded.get_splits() for im in reloaded.get_images(s)]
    assert len(images) == 3  # none lost to id collisions
