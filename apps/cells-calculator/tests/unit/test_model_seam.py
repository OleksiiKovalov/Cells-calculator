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
