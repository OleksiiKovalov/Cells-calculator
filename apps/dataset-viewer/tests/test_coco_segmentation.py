"""Unit tests for COCO segmentation parsing (polygon / RLE / bbox)."""
from datasets.coco_loader import COCOLoader, _decode_compressed_rle


def test_parse_polygon():
    out = COCOLoader._parse_segmentation([[0, 0, 10, 0, 10, 10, 0, 10]])
    assert out["type"] == "polygon"
    assert out["polygons"] == [[(0, 0), (10, 0), (10, 10), (0, 10)]]


def test_parse_uncompressed_rle():
    out = COCOLoader._parse_segmentation({"counts": [4, 2, 4], "size": [2, 5]})
    assert out["type"] == "mask"
    assert out["rle_counts"] == [4, 2, 4]
    assert out["rle_size"] == [2, 5]


def test_parse_bbox_fallback():
    assert COCOLoader._parse_segmentation(None)["type"] == "bbox"
    assert COCOLoader._parse_segmentation([])["type"] == "bbox"


def test_compressed_rle_decodes_to_runs():
    # A compressed counts string decodes to a non-negative run-length list.
    runs = _decode_compressed_rle("PPNo0", [10, 10])
    assert runs is not None
    assert all(isinstance(r, int) and r >= 0 for r in runs)
