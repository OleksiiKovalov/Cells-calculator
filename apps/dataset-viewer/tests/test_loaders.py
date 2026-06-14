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
