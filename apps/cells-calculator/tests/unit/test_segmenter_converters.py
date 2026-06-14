"""Converter tests for the re-added segmenters (Cellpose, StarDist).

These exercise the label-map -> DataFrame conversion directly on synthetic
masks, so they need neither model weights nor a forward pass. StarDist's module
imports TensorFlow at import time; importorskip handles environments without it
(including a torch/PyQt-before-TF DLL load failure, which raises ImportError).
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
    df = seg.cellpose_results_to_pandas(
        masks, cellprob_map=cellprob, image_shape_for_norm=masks.shape
    )
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


def test_stardist_converter_schema_and_normalization():
    pytest.importorskip("tensorflow")
    StardistMod = pytest.importorskip("model.StardistSegmenter")
    seg = object.__new__(StardistMod.StardistSegmenter)
    instances = _two_blob_labels()
    df = seg.stardist_results_to_pandas(
        instances,
        scores=[0.9, 0.7],
        original_shape=instances.shape,
        inference_shape=instances.shape,
    )
    assert list(df.columns) == CANON_COLS
    assert len(df) == 2
    assert df["confidence"].tolist() == [0.9, 0.7]
    # mask contours normalized to [0, 1]
    assert all(float(np.max(m)) <= 1.0 for m in df["mask"])
