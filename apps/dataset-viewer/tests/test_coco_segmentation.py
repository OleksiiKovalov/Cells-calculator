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


def _encode_rle_string(counts: list) -> str:
    """Reference encoder (pycocotools maskApi.c :: rleToString)."""
    chars = []
    for i, c in enumerate(counts):
        x = int(c) - (int(counts[i - 2]) if i > 2 else 0)
        more = True
        while more:
            ch = x & 0x1F
            x >>= 5
            more = (x != -1) if (ch & 0x10) else (x != 0)
            if more:
                ch |= 0x20
            chars.append(chr(ch + 48))
    return ''.join(chars)


def test_compressed_rle_roundtrip_exact():
    runs = [6, 1, 40, 4, 5, 4, 5, 4, 21]
    assert _decode_compressed_rle(_encode_rle_string(runs), [10, 10]) == runs


def test_compressed_rle_negative_delta_sign_extension():
    # counts[3]-counts[1] and counts[4]-counts[2] are negative deltas, which
    # exercise the decoder's sign-extension branch.
    runs = [0, 50, 120, 3, 2, 8]
    assert _decode_compressed_rle(_encode_rle_string(runs), [16, 16]) == runs


def test_rle_mask_string_roundtrip():
    import numpy as np
    from datasets.rle import encode_mask, decode_mask

    mask = np.zeros((12, 9), dtype=bool)
    mask[2:7, 3:8] = True
    mask[9:11, 0:2] = True

    counts = encode_mask(mask)
    decoded = _decode_compressed_rle(_encode_rle_string(counts), [12, 9])
    assert decoded == counts
    assert (decode_mask(decoded, [12, 9]) == mask).all()
