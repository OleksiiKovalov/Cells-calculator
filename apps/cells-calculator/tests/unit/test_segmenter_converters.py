"""Converter tests for the re-added segmenters (Cellpose, StarDist).

These exercise the label-map -> DataFrame conversion directly on synthetic
masks, so they need neither model weights nor a forward pass. StarDist's module
imports TensorFlow at import time; importorskip handles environments without it
(including a torch/PySide6-before-TF DLL load failure, which raises ImportError).
"""
import numpy as np
import pytest

CANON_COLS = ["id_label", "box", "mask", "confidence", "diameter", "area", "volume"]


def _two_blob_labels(h=64, w=64):
    m = np.zeros((h, w), dtype=np.int32)
    m[5:15, 5:20] = 1
    m[30:50, 30:55] = 2
    return m


def test_cellpose_converter_schema_and_normalization():
    pytest.importorskip("cellpose")
    from model.CellposeSegmenter import CellposeSegmenter
    seg = object.__new__(CellposeSegmenter)  # bypass __init__/model load
    masks = _two_blob_labels()
    cellprob = np.ones(masks.shape, dtype=np.float32)
    # No preprocess() called -> identity geometry: masks normalize by their own dims.
    df = seg.cellpose_results_to_pandas(masks, cellprob_map=cellprob)
    assert list(df.columns) == CANON_COLS
    assert df["id_label"].tolist() == [1, 2]
    assert all(0.0 <= v <= 1.0 for box in df["box"] for v in box)
    assert df["confidence"].notna().all()
    assert (df["area"] > 0).all()


def test_cellpose_converter_empty_label_map():
    pytest.importorskip("cellpose")
    from model.CellposeSegmenter import CellposeSegmenter
    seg = object.__new__(CellposeSegmenter)
    df = seg.cellpose_results_to_pandas(np.zeros((32, 32), dtype=np.int32))
    assert list(df.columns) == CANON_COLS and len(df) == 0


def test_cellpose_converter_undoes_pad_and_scale():
    """Cellpose masks map back through recorded resize/pad geometry (base helper)."""
    pytest.importorskip("cellpose")
    from model.CellposeSegmenter import CellposeSegmenter
    seg = object.__new__(CellposeSegmenter)
    masks = np.zeros((512, 512), dtype=np.int32)
    masks[128:148, 240:280] = 1  # top of the content band in a padded 512x512
    seg._original_shape = (500, 1000)
    seg._inference_shape = (512, 512)
    seg._inference_transform = {"scale": 0.512, "pad_x": 0.0, "pad_y": 128.0}
    df = seg.cellpose_results_to_pandas(masks)
    assert len(df) == 1
    mask = np.asarray(df["mask"].iloc[0])
    assert float(mask[:, 1].min()) * 500 < 5      # maps to the top, not the centre
    assert 0.45 < float(mask[:, 0].min()) < 0.49  # horizontally centered


def test_instanseg_converter_undoes_pad_and_scale():
    """A mask in the padded inference space maps back onto the original image.

    Reproduces the misplacement bug: a 1000x500 original under a 512 config is
    resized to 512x256 then padded to 512x512 (128px top/bottom). An object at
    the top of the real content must map to the top of the original (y≈0), not
    to y≈125 as the old (pad-unaware) normalization produced.
    """
    pytest.importorskip("instanseg")
    from model.InstanSegSegmenter import InstansegSegmenter

    seg = object.__new__(InstansegSegmenter)  # bypass __init__/model load
    out = np.zeros((512, 512), dtype=np.int32)
    out[128:148, 240:280] = 1  # top of the content band, horizontally centered
    labeled = out[None, None, :, :]

    # Emulate the geometry recorded by preprocess()/add_geometry_step() for a
    # 1000x500 original fitted+padded into a 512x512 inference image.
    seg._original_shape = (500, 1000)
    seg._inference_shape = (512, 512)
    seg._inference_transform = {"scale": 0.512, "pad_x": 0.0, "pad_y": 128.0}
    df = seg.instanseg_results_to_pandas(labeled)
    assert list(df.columns) == CANON_COLS
    assert len(df) == 1
    mask = np.asarray(df["mask"].iloc[0])
    # y maps back to the very top of the original image, not the vertical centre.
    assert float(mask[:, 1].min()) * 500 < 5
    assert float(mask[:, 1].max()) * 500 < 45
    # x is horizontally centered (~469..547 px of 1000).
    assert 0.45 < float(mask[:, 0].min()) < 0.49
    assert 0.53 < float(mask[:, 0].max()) < 0.57


def test_instanseg_converter_identity_transform_defaults():
    """With no recorded geometry (output space == original), normalize by output dims."""
    pytest.importorskip("instanseg")
    from model.InstanSegSegmenter import InstansegSegmenter

    seg = object.__new__(InstansegSegmenter)  # no preprocess() -> identity geometry
    out = _two_blob_labels()
    df = seg.instanseg_results_to_pandas(out[None, None, :, :])
    assert list(df.columns) == CANON_COLS
    assert len(df) == 2
    assert all(0.0 <= v <= 1.0 for m in df["mask"] for v in np.asarray(m).ravel())


def test_stardist_converter_schema_and_normalization():
    pytest.importorskip("tensorflow")
    StardistMod = pytest.importorskip("model.StardistSegmenter")
    seg = object.__new__(StardistMod.StardistSegmenter)
    instances = _two_blob_labels()
    # No preprocess() called -> identity geometry: contours normalize by label-map dims.
    df = seg.stardist_results_to_pandas(instances, scores=[0.9, 0.7])
    assert list(df.columns) == CANON_COLS
    assert len(df) == 2
    assert df["confidence"].tolist() == [0.9, 0.7]
    # mask contours normalized to [0, 1]
    assert all(float(np.max(m)) <= 1.0 for m in df["mask"])


def test_stardist_converter_undoes_pad_and_scale():
    """StarDist masks map back through recorded resize/pad geometry (base helper)."""
    pytest.importorskip("tensorflow")
    StardistMod = pytest.importorskip("model.StardistSegmenter")
    seg = object.__new__(StardistMod.StardistSegmenter)
    instances = np.zeros((512, 512), dtype=np.int32)
    instances[128:148, 240:280] = 1  # top of the content band in a padded 512x512
    seg._original_shape = (500, 1000)
    seg._inference_shape = (512, 512)
    seg._inference_transform = {"scale": 0.512, "pad_x": 0.0, "pad_y": 128.0}
    df = seg.stardist_results_to_pandas(instances, scores=[0.9])
    assert len(df) == 1
    mask = np.asarray(df["mask"].iloc[0])
    assert float(mask[:, 1].min()) * 500 < 5      # maps to the top, not the centre
    assert 0.45 < float(mask[:, 0].min()) < 0.49  # horizontally centered
