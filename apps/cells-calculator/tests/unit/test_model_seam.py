"""Tests for the model layer seam: Model dispatch + BaseSegmenter.inference."""
import numpy as np
import pandas as pd
import pytest

import model.Model as MM
from model.BaseSegmenter import BaseSegmenter


def test_model_unknown_type_raises_value_error(monkeypatch):
    # Avoid the blocking QMessageBox shown for unknown model types.
    monkeypatch.setattr(MM, "show_error_message", lambda *a, **k: None)
    with pytest.raises(ValueError):
        MM.Model(path="x", model_type="definitely_not_a_model", model_data={})


class _Dummy(BaseSegmenter):
    def init_model(self, path_to_model):
        self.loaded = path_to_model

    def call_inference(self, input_image, **kwargs):
        return pd.DataFrame({"area": [0.1, 0.2]})


def test_basesegmenter_inference_returns_dataframe_and_times():
    seg = _Dummy("nopath")
    assert seg.loaded == "nopath"
    out = seg.inference(np.zeros((4, 4, 3), dtype=np.uint8))
    assert isinstance(out, pd.DataFrame) and len(out) == 2
    assert seg.inference_duration >= 0.0


class _Bare(BaseSegmenter):
    def init_model(self, path_to_model):
        pass


def test_basesegmenter_call_inference_not_implemented_by_default():
    with pytest.raises(NotImplementedError):
        _Bare("x").inference(np.zeros((2, 2), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Shared preprocessing geometry — the standardized mask-mapping seam that every
# segmenter reuses instead of reimplementing (preprocess/add_geometry_step/
# to_original_norm).
# --------------------------------------------------------------------------- #
from collections import OrderedDict

from model.utils import resize_and_pad_cv


def test_preprocess_records_identity_when_no_geometry():
    seg = _Bare("x")
    img = np.zeros((40, 90, 3), dtype=np.uint8)
    out = seg.preprocess(img, [OrderedDict([("gray2rgb", "")])])
    assert out.shape[:2] == (40, 90)
    assert seg._original_shape == (40, 90)
    # No resize/pad -> a point maps straight through, only normalized.
    got = seg.to_original_norm(np.array([[45.0, 20.0]]), src_shape=(40, 90))
    assert np.allclose(got, [[0.5, 0.5]])


def test_preprocess_then_pad_maps_detection_back_to_original():
    """The exact InstanSeg scenario, driven only through base-class helpers.

    A 1000x500 image is resized (aspect-preserved) to 512x256 then padded to a
    512x512 square. A point at the top edge of the content must map back to the
    top of the original (y≈0), not the vertical centre.
    """
    seg = _Bare("x")
    img = np.zeros((500, 1000, 3), dtype=np.uint8)  # H=500, W=1000
    fitted = seg.preprocess(img, [OrderedDict([("resize", "512:512")])])
    assert fitted.shape[:2] == (256, 512)

    padded, tf = resize_and_pad_cv(fitted, 512, 512, return_transform=True)
    seg.add_geometry_step(padded, tf)
    assert seg._inference_shape == (512, 512)

    # Top-centre and bottom-centre of the content band in the 512x512 output.
    pts = np.array([[256.0, 128.0], [256.0, 384.0]])
    norm = seg.to_original_norm(pts, src_shape=(512, 512))
    back = norm * np.array([1000, 500])
    assert np.allclose(back, [[500, 0], [500, 500]], atol=1.0)
