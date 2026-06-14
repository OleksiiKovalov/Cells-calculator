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


def test_coco_export_uses_global_image_ids(tmp_path):
    """Regression: per-split image-id reset would collide when COCOLoader merges
    the JSON files, dropping the first image of later splits."""
    src, _ = detect_format(sample("yolo_seg"))
    out = str(tmp_path / "coco")
    COCOExporter().export(src, out, lambda d, t: True)

    reloaded, _ = detect_format(out)
    images = [im for s in reloaded.get_splits() for im in reloaded.get_images(s)]
    assert len(images) == 3  # none lost to id collisions
