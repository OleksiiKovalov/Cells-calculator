"""Format auto-detection against the bundled sample datasets."""
import pytest

from datasets.format_detector import detect_format
from tests.conftest import sample


@pytest.mark.parametrize("folder,expected_fmt", [
    ("yolo_seg", "YOLO"),
    ("coco", "COCO"),
])
def test_detect_known_formats(folder, expected_fmt):
    loader, fmt = detect_format(sample(folder))
    assert fmt == expected_fmt
    assert loader is not None


def test_detect_unknown_returns_none(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing dataset-like here")
    loader, fmt = detect_format(str(tmp_path))
    assert loader is None and fmt is None


def test_detect_missing_path_returns_none():
    loader, fmt = detect_format("Z:/definitely/not/here")
    assert loader is None and fmt is None
