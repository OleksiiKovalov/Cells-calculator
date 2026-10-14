"""Unit tests for model.utils — morphology, micrometer calibration, LSM reading,
mask rasterization, filtering, range, image I/O and preprocessing.

Covers the fixes ported from OLD@fixed (LSM 512 / series handling, zero-area
morphology guard, honest safe_image_write, results_to_pandas empty guard) plus
the new um calibration helper.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import tifffile

import model.utils as U


# --------------------------------------------------------------------------- #
# calculate_morphology
# --------------------------------------------------------------------------- #
def test_calculate_morphology_zero_area_guard():
    out = U.calculate_morphology(np.zeros((0, 0), dtype=np.uint8))
    assert out == {"diameter": 0.0, "area": 0.0, "volume": 0.0}


def test_calculate_morphology_returns_floats_and_relative_area():
    bm = np.zeros((100, 100), dtype=np.uint8)
    bm[40:60, 40:60] = 1  # 400 px of 10000
    out = U.calculate_morphology(bm)
    assert all(isinstance(v, float) for v in out.values())
    assert out["area"] == pytest.approx(400 / 10000)


# --------------------------------------------------------------------------- #
# morphology_to_micrometers
# --------------------------------------------------------------------------- #
def test_morphology_to_micrometers_exact_pixel_roundtrip_square():
    W = H = 512
    k = 0.325
    bm = np.zeros((H, W), dtype=np.uint8)
    bm[100:140, 100:160] = 1  # 40 x 60 = 2400 px
    m = U.calculate_morphology(bm)
    d_um, a_um2, v_um3 = U.morphology_to_micrometers(
        m["diameter"], m["area"], m["volume"], W, H, k
    )
    assert a_um2 / (k ** 2) == pytest.approx(2400, rel=1e-6)
    assert d_um > 0 and v_um3 > 0


def test_morphology_to_micrometers_scaling_powers():
    m = (0.03, 0.001, 1e-5)
    d1, a1, v1 = U.morphology_to_micrometers(*m, 512, 512, 0.3)
    d2, a2, v2 = U.morphology_to_micrometers(*m, 512, 512, 0.6)
    assert d2 / d1 == pytest.approx(2)
    assert a2 / a1 == pytest.approx(4)
    assert v2 / v1 == pytest.approx(8)


@pytest.mark.parametrize("w,h,k", [(0, 512, 0.3), (512, 512, 0.0), (512, 0, -1)])
def test_morphology_to_micrometers_degenerate_returns_zeros(w, h, k):
    assert U.morphology_to_micrometers(0.1, 0.1, 0.1, w, h, k) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# plot_mask
# --------------------------------------------------------------------------- #
def test_plot_mask_degenerate_contour_returns_zero_morphology():
    bm, morph = U.plot_mask(np.array([[0.1, 0.1], [0.2, 0.2]]), image_size=(100, 100))
    assert bm.dtype == bool
    assert morph["area"] == 0.0


def test_plot_mask_valid_polygon_has_area():
    tri = np.array([[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]], dtype=np.float32)
    bm, morph = U.plot_mask(tri, image_size=(100, 100))
    assert bm.sum() > 0
    assert morph["area"] > 0


def test_plot_mask_none_is_safe():
    bm, morph = U.plot_mask(None, image_size=(50, 50))
    assert bm.shape == (50, 50)
    assert morph["area"] == 0.0


# --------------------------------------------------------------------------- #
# filter_segmentation_detections / get_segmentation_detections_range
# --------------------------------------------------------------------------- #
def _det(areas):
    return pd.DataFrame({"area": areas, "mask": [[] for _ in areas]})


def test_filter_segmentation_detections_by_area():
    df = _det([0.1, 0.5, 0.9])
    out = U.filter_segmentation_detections(df, min_size=0.2, max_size=0.8)
    assert out["area"].tolist() == [0.5]


def test_filter_segmentation_detections_swaps_inverted_bounds():
    df = _det([0.1, 0.5, 0.9])
    out = U.filter_segmentation_detections(df, min_size=0.8, max_size=0.2)
    assert out["area"].tolist() == [0.5]


def test_filter_segmentation_detections_empty():
    out = U.filter_segmentation_detections(pd.DataFrame({"area": []}))
    assert out.empty


def test_get_segmentation_detections_range_empty_is_unit():
    assert U.get_segmentation_detections_range(pd.DataFrame({"area": []})) == (0.0, 1.0)


def test_get_segmentation_detections_range_values():
    assert U.get_segmentation_detections_range(_det([0.2, 0.7, 0.4])) == (0.2, 0.7)


# --------------------------------------------------------------------------- #
# results_to_pandas — empty/None masks guard (the YOLO zero-detection fix)
# --------------------------------------------------------------------------- #
class _NoMasks:
    masks = None


@pytest.mark.parametrize("store_bin_mask,ncols", [(False, 7), (True, 8)])
def test_results_to_pandas_handles_none_masks(store_bin_mask, ncols):
    df = U.results_to_pandas(_NoMasks(), store_bin_mask=store_bin_mask)
    assert isinstance(df, pd.DataFrame) and len(df) == 0
    assert len(df.columns) == ncols
    assert df.columns[0] == "id_label"


# --------------------------------------------------------------------------- #
# safe_image_write / safe_image_read / read_standard_img
# --------------------------------------------------------------------------- #
def test_safe_image_write_honest_return(tmp_path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert U.safe_image_write(img, str(tmp_path / "a.png")) is True
    assert (tmp_path / "a.png").exists()
    assert U.safe_image_write(img, str(tmp_path / "a.xyz")) is False
    assert U.safe_image_write(None, str(tmp_path / "b.png")) is False


def test_read_standard_img_roundtrip_and_missing(tmp_path):
    img = np.full((10, 12, 3), 128, dtype=np.uint8)
    U.safe_image_write(img, str(tmp_path / "c.png"))
    out = U.read_standard_img(str(tmp_path / "c.png"))
    assert out.shape == (10, 12, 3)
    assert U.read_standard_img(str(tmp_path / "missing.png")) is None


# --------------------------------------------------------------------------- #
# LSM reading (512 hardcode fix + series/channels-last handling)
# --------------------------------------------------------------------------- #
def test_read_lsm_img_single_channel_2d_series(tmp_path):
    p = tmp_path / "single.lsm"
    tifffile.imwrite(str(p), np.zeros((16, 24), dtype=np.uint8))
    out = U.read_lsm_img(str(p))
    assert out.shape == (16, 24, 3)


def test_read_lsm_img_multichannel_uses_series_shape(tmp_path):
    p = tmp_path / "two.lsm"
    img = np.zeros((2, 20, 30), dtype=np.uint8)
    img[1] = 255
    tifffile.imwrite(str(p), img, metadata={"axes": "CYX"})
    out = U.read_lsm_img(str(p), nuclei_channel=1)
    assert out.shape == (20, 30, 3)  # NOT hardcoded 512x512


def test_read_lsm_img_non_512_does_not_crash(tmp_path):
    # 2-channel 1024x1024 — the exact case the old 512 hardcode corrupted.
    p = tmp_path / "big.lsm"
    img = np.zeros((2, 1024, 1024), dtype=np.uint8)
    tifffile.imwrite(str(p), img, metadata={"axes": "CYX"})
    out = U.read_lsm_img(str(p))
    assert out.shape == (1024, 1024, 3)


def test_lsm_to_channels_last_variants():
    assert U.lsm_to_channels_last(np.zeros((16, 24))).shape == (16, 24, 1)
    assert U.lsm_to_channels_last(np.zeros((2, 16, 24))).shape == (16, 24, 2)


def test_read_img_dispatches_lsm_vs_standard(tmp_path):
    std = tmp_path / "x.png"
    U.safe_image_write(np.zeros((8, 8, 3), np.uint8), str(std))
    assert U.read_img(str(std)).shape == (8, 8, 3)
    lsm = tmp_path / "x.lsm"
    tifffile.imwrite(str(lsm), np.zeros((8, 8), np.uint8))
    assert U.read_img(str(lsm)).shape == (8, 8, 3)


# --------------------------------------------------------------------------- #
# process_loaded_image / resize_and_pad_cv
# --------------------------------------------------------------------------- #
def test_process_loaded_image_gray2rgb_and_resize():
    img = np.zeros((40, 60), dtype=np.uint8)
    settings = [OrderedDict([("gray2rgb", "")]), OrderedDict([("resize", "20:20")])]
    out = U.process_loaded_image(img, settings)
    assert out.ndim == 3 and out.shape[2] == 3
    assert max(out.shape[:2]) == 20


def test_process_loaded_image_unknown_instruction_raises():
    with pytest.raises(RuntimeError):
        U.process_loaded_image(np.zeros((4, 4)), [OrderedDict([("bogus", "")])])


def test_resize_and_pad_cv_target_shape():
    out = U.resize_and_pad_cv(np.zeros((50, 100, 3), dtype=np.uint8), 64, 64)
    assert out.shape[:2] == (64, 64)


# --------------------------------------------------------------------------- #
# geometric transform tracking (mask-mapping correctness for padded inference)
# --------------------------------------------------------------------------- #
def test_resize_and_pad_cv_reports_transform():
    # 100x50 (WxH) -> 64x64: uniform scale 0.64, content 64x32 padded with 16 top/bottom.
    _, tf = U.resize_and_pad_cv(
        np.zeros((50, 100, 3), dtype=np.uint8), 64, 64, return_transform=True
    )
    assert tf["scale"] == pytest.approx(0.64)
    assert tf["pad_x"] == pytest.approx(0.0)
    assert tf["pad_y"] == pytest.approx(16.0)


def test_process_loaded_image_reports_resize_transform():
    img = np.zeros((500, 1000), dtype=np.uint8)  # H=500, W=1000
    settings = [OrderedDict([("gray2rgb", "")]), OrderedDict([("resize", "512:512")])]
    out, tf = U.process_loaded_image(img, settings, return_transform=True)
    assert out.shape[:2] == (256, 512)          # aspect preserved, no padding
    assert tf["scale"] == pytest.approx(0.512)
    assert tf["pad_x"] == 0.0 and tf["pad_y"] == 0.0


def test_compose_and_invert_round_trip():
    # resize (scale 0.512, no pad) then centered-pad by 128 in y — the exact
    # chain InstanSeg applies to a non-square image under a 512 config.
    tf = U.compose_transforms(
        {"scale": 0.512, "pad_x": 0.0, "pad_y": 0.0},
        {"scale": 1.0, "pad_x": 0.0, "pad_y": 128.0},
    )
    assert tf["scale"] == pytest.approx(0.512)
    assert tf["pad_y"] == pytest.approx(128.0)
    # A point at the top edge of the original (y=0) forward-maps to y=128 in the
    # padded image; inverting must return it to y=0 (the buggy path left it at 128).
    orig = np.array([[500.0, 0.0], [1000.0, 500.0]], dtype=np.float32)
    fwd = orig * tf["scale"] + np.array([tf["pad_x"], tf["pad_y"]], dtype=np.float32)
    back = U.invert_transform_points(fwd, tf)
    assert np.allclose(back, orig, atol=1e-3)


# --------------------------------------------------------------------------- #
# colormap helpers
# --------------------------------------------------------------------------- #
def test_colormap_to_hex_and_bgr():
    hexes = U.colormap_to_hex("tab20")
    assert len(hexes) == 20
    bgr = U.hex_to_bgr(hexes)
    assert len(bgr) == 20 and all(len(c) == 3 for c in bgr)


def test_denormalize_coordinates():
    coords = np.array([[0.5, 0.5]])
    out = U.denormalize_coordinates(coords, (100, 200))  # (H, W)
    assert out.tolist() == [[100.0, 50.0]]
