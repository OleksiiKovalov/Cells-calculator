"""PreparedLoader: split (keep/ratio), image materialization + annotation
scaling (bbox/polygon/RLE), and prepared export to every format."""
import numpy as np
import pytest

from datasets.format_detector import detect_format
from datasets.prepared_loader import PreparedLoader, _scale_annotation
from datasets.rle import decode_mask, encode_mask
from datasets.yolo_exporter import YOLOExporter
from datasets.coco_exporter import COCOExporter
from datasets.voc_exporter import VOCExporter
from datasets.pth_exporter import PTHExporter
from tests.conftest import sample


def _all_images(loader):
    sp = loader.get_splits()
    return [im for s in sp for im in loader.get_images(s)] if sp else loader.get_images()


def test_keep_splits_preserve_counts():
    base, _ = detect_format(sample("yolo_seg"))
    p = PreparedLoader(base, {"split_mode": "keep"})
    assert p.get_splits() == base.get_splits()
    assert len(_all_images(p)) == len(_all_images(base))


def test_ratio_split_partitions_all():
    base, _ = detect_format(sample("yolo_seg"))
    p = PreparedLoader(base, {"split_mode": "ratio", "ratios": (0.6, 0.2, 0.2), "seed": 1})
    assert set(p.get_splits()) <= {"train", "val", "test"}
    assert len(_all_images(p)) == len(_all_images(base))


def test_ratio_split_deterministic():
    base, _ = detect_format(sample("yolo_seg"))

    def counts(seed):
        p = PreparedLoader(base, {"split_mode": "ratio", "seed": seed})
        return {s: len(p.get_images(s)) for s in p.get_splits()}

    assert counts(5) == counts(5)


def test_resize_materializes_and_scales_annotations():
    from PIL import Image

    base, _ = detect_format(sample("yolo_seg"))
    base_imgs = _all_images(base)
    src0 = base_imgs[0]["path"]
    w0, h0 = Image.open(src0).size

    p = PreparedLoader(base, {"resize": True, "resize_target": max(8, min(w0, h0) // 4)})
    pimgs = _all_images(p)

    # Image is materialized to a new (smaller) file, not the original.
    assert pimgs[0]["path"] != src0
    pw, ph = Image.open(pimgs[0]["path"]).size
    assert pw <= w0 and ph <= h0 and (pw < w0 or ph < h0)

    # Annotations are scaled down to match.
    base_ann = base.get_annotations(src0)
    prep_ann = p.get_annotations(pimgs[0]["path"])
    assert len(prep_ann) == len(base_ann)
    if base_ann and "w" in base_ann[0]:
        assert prep_ann[0]["w"] <= base_ann[0]["w"]


def test_contrast_only_keeps_coordinates():
    base, _ = detect_format(sample("yolo_seg"))
    p = PreparedLoader(base, {"contrast": True, "contrast_factor": 1.5})   # no resize
    base_ann = base.get_annotations(_all_images(base)[0]["path"])
    prep_ann = p.get_annotations(_all_images(p)[0]["path"])
    if base_ann and "w" in base_ann[0]:
        assert prep_ann[0]["w"] == base_ann[0]["w"]                        # sx=sy=1


def test_scale_annotation_rle_roundtrip():
    m = np.zeros((20, 20), dtype=np.uint8)
    m[4:12, 6:14] = 1
    ann = {"type": "mask", "class_id": 0, "label": "c", "x": 6, "y": 4, "w": 8, "h": 8,
           "rle_counts": encode_mask(m), "rle_size": [20, 20]}
    out = _scale_annotation(ann, 0.5, 0.5)
    assert out["rle_size"] == [10, 10]
    assert out["w"] == 4.0 and out["h"] == 4.0
    dec = decode_mask(out["rle_counts"], out["rle_size"])
    assert dec.shape == (10, 10) and dec.sum() > 0


@pytest.mark.parametrize("name,exporter_cls", [
    ("YOLO", YOLOExporter), ("COCO", COCOExporter),
    ("Pascal VOC", VOCExporter), ("InstanSeg PTH", PTHExporter),
])
def test_prepared_export_roundtrips_all_formats(tmp_path, name, exporter_cls):
    """A prepared (resized + re-split) dataset exports to every format and
    re-detects with the same image count."""
    base, _ = detect_format(sample("yolo_seg"))
    p = PreparedLoader(base, {"resize": True, "resize_target": 32,
                              "split_mode": "ratio", "seed": 1})
    out = str(tmp_path / name.replace(" ", "_"))
    exporter_cls().export(p, out, lambda d, t: True)

    reloaded, fmt = detect_format(out)
    assert reloaded is not None, f"{name}: not re-detected"
    assert len(_all_images(reloaded)) == len(_all_images(p))
