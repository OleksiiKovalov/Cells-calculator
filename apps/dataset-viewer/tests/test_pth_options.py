"""PTH export options (filename / modality / split-label) and the headless
convert() bridge. Split + transforms are exercised in test_prepared_loader.py;
here we test what PTHExporter itself owns, plus a prepared→PTH integration."""
import pytest

from datasets.format_detector import detect_format
from datasets.pth_exporter import PTHExporter
from datasets.prepared_loader import PreparedLoader
from tests.conftest import sample


def _load_pth(path):
    import torch
    return torch.load(str(path), weights_only=False)


def test_val_split_maps_to_validation(tmp_path):
    """A YOLO 'val' split must become the 'Validation' key InstanSeg reads,
    not 'Val' (which the training/test loaders silently ignore)."""
    src, _ = detect_format(sample("yolo_seg"))
    out = tmp_path / "keep"
    PTHExporter(filename="d.pth").export(src, str(out), lambda d, t: True)

    data = _load_pth(out / "d.pth")
    assert "Validation" in data
    assert "Val" not in data
    assert sum(len(v) for v in data.values()) == 3


def test_custom_filename_and_modality(tmp_path):
    src, _ = detect_format(sample("yolo_seg"))
    out = tmp_path / "named"
    PTHExporter(filename="my_dataset", modality="Fluorescence").export(
        src, str(out), lambda d, t: True)

    assert (out / "my_dataset.pth").is_file()            # extension auto-appended
    item = next(it for v in _load_pth(out / "my_dataset.pth").values() for it in v)
    assert item["image_modality"] == "Fluorescence"


def test_default_export_unchanged(tmp_path):
    """Defaults reproduce the original behaviour: dataset.pth, Brightfield."""
    src, _ = detect_format(sample("yolo_seg"))
    out = tmp_path / "def"
    PTHExporter().export(src, str(out), lambda d, t: True)
    assert (out / "dataset.pth").is_file()
    item = next(it for v in _load_pth(out / "dataset.pth").values() for it in v)
    assert item["image_modality"] == "Brightfield"


def test_prepared_ratio_and_resize_to_pth(tmp_path):
    """Prepared (ratio re-split + resize) → PTH: splits are the InstanSeg keys
    and the mask matches the resized image."""
    src, _ = detect_format(sample("yolo_seg"))
    prepared = PreparedLoader(src, {"resize": True, "resize_target": 32,
                                    "split_mode": "ratio", "seed": 1})
    out = tmp_path / "prep"
    PTHExporter(filename="d.pth").export(prepared, str(out), lambda d, t: True)

    data = _load_pth(out / "d.pth")
    assert set(data.keys()) <= {"Train", "Validation", "Test"}
    assert sum(len(v) for v in data.values()) == 3
    item = next(it for v in data.values() for it in v)
    mask_key = "cell_masks" if "cell_masks" in item else "nucleus_masks"
    assert item["image"].shape[:2] == item[mask_key].shape          # aligned
    assert max(item["image"].shape[:2]) <= 32


def test_convert_prepare_and_target(tmp_path):
    """The headless convert() helper prepares + converts via the same engine."""
    import convert
    out = tmp_path / "conv"
    result = convert.convert_dataset(
        sample("yolo_seg"), str(out), "pth",
        prepare_spec={"split_mode": "ratio", "resize": True, "resize_target": 32},
        pth_options={"filename": "c.pth"})
    assert "InstanSeg PTH" in result
    data = _load_pth(out / "c.pth")
    assert sum(len(v) for v in data.values()) == 3


def test_convert_resolve_format_aliases():
    import convert
    assert convert.resolve_format("pth") == "InstanSeg PTH"
    assert convert.resolve_format("YOLO") == "YOLO"
    assert convert.resolve_format("voc") == "Pascal VOC"
    with pytest.raises(ValueError):
        convert.resolve_format("nonsense")
