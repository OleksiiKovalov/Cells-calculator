"""Loaders produce the annotation contract from the bundled sample datasets."""
import pytest

from datasets.format_detector import detect_format
from tests.conftest import sample

REQUIRED_KEYS = {"class_id", "label", "type", "x", "y", "w", "h"}
VALID_TYPES = {"bbox", "polygon", "mask"}


def _all_images(loader):
    splits = loader.get_splits()
    if splits:
        return [im for s in splits for im in loader.get_images(s)]
    return loader.get_images()


@pytest.mark.parametrize("folder", ["yolo_seg", "coco"])
def test_sample_dataset_counts_and_contract(folder):
    loader, _ = detect_format(sample(folder))
    assert loader.class_names == ["cell"]
    assert sorted(loader.get_splits()) == ["train", "val"]

    images = _all_images(loader)
    assert len(images) == 3

    total = 0
    for img in images:
        for ann in loader.get_annotations(img["path"]):
            total += 1
            assert REQUIRED_KEYS <= set(ann), f"missing keys in {ann}"
            assert ann["type"] in VALID_TYPES
            assert ann["w"] >= 0 and ann["h"] >= 0
            if ann["type"] == "polygon":
                assert ann.get("points") or ann.get("polygons")
    assert total == 15


def test_yolo_segmentation_parsed_as_polygons():
    loader, _ = detect_format(sample("yolo_seg"))
    img = loader.get_images("train")[0]
    anns = loader.get_annotations(img["path"])
    assert anns and all(a["type"] == "polygon" for a in anns)
    pts = anns[0]["points"]
    assert len(pts) >= 3 and all(len(p) == 2 for p in pts)


def test_images_resolve_to_existing_files():
    import os
    loader, _ = detect_format(sample("coco"))
    for img in _all_images(loader):
        assert os.path.isfile(img["path"])


def test_yolo_split_images_layout(tmp_path):
    """Datasets laid out as <split>/images + <split>/labels are detected."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    root = tmp_path / "ds"
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        img = QImage(10, 10, QImage.Format_RGB32)
        img.fill(Qt.white)
        assert img.save(str(root / split / "images" / f"{split}_a.png"))
        (root / split / "labels" / f"{split}_a.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    loader, fmt = detect_format(str(root))
    assert fmt == "YOLO"
    assert sorted(loader.get_splits()) == ["train", "val"]
    imgs = _all_images(loader)
    assert len(imgs) == 2
    anns = loader.get_annotations(imgs[0]["path"])
    assert len(anns) == 1 and anns[0]["type"] == "bbox"


def test_pth_instance_masks_become_rle_annotations():
    """PTH-style instance masks surface as 'mask' annotations whose RLE
    decodes back to the exact instance shape (bbox fields kept alongside)."""
    import numpy as np
    from datasets.pth_loader import _mask_to_annotations
    from datasets.rle import decode_mask

    m = np.zeros((8, 8), dtype=np.int32)
    m[1:4, 1:4] = 1
    m[5:8, 2:6] = 2

    anns = _mask_to_annotations({"cell_masks": m})
    assert len(anns) == 2
    for iid, ann in enumerate(anns, start=1):
        assert ann["type"] == "mask"
        assert ann["rle_size"] == [8, 8]
        assert (decode_mask(ann["rle_counts"], ann["rle_size"]) == (m == iid)).all()
        assert ann["w"] > 0 and ann["h"] > 0
