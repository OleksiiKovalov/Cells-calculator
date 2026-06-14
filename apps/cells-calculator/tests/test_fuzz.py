"""Property-based fuzzing of the pure (model-free) pipeline with Hypothesis.

These probe the morphology / mask-rasterization / filtering / image-I/O helpers
with adversarial inputs (empty arrays, NaN/inf coords, odd lengths, weird image
shapes/dtypes) and assert they never crash and return sane values.
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from hypothesis.extra.numpy import arrays, array_shapes

import model.utils as U

_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

masks = arrays(
    np.uint8,
    array_shapes(min_dims=2, max_dims=2, min_side=0, max_side=48),
    elements=st.integers(0, 1),
)
# Coordinate blobs include NaN/inf and odd lengths on purpose.
coord_blobs = arrays(
    np.float32,
    array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=30),
    elements=st.floats(allow_nan=True, allow_infinity=True, width=32),
)
gray_or_rgb = st.one_of(
    arrays(np.uint8, array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=40),
           elements=st.integers(0, 255)),
    arrays(np.uint8, st.tuples(st.integers(1, 40), st.integers(1, 40), st.just(3)),
           elements=st.integers(0, 255)),
)


@_SETTINGS
@given(bm=masks)
def test_fuzz_calculate_morphology(bm):
    out = U.calculate_morphology(bm)
    for v in out.values():
        assert np.isfinite(v) and v >= 0.0
    assert out["area"] <= 1.0 + 1e-9


@_SETTINGS
@given(
    d=st.floats(0, 1, allow_nan=False),
    a=st.floats(0, 1, allow_nan=False),
    v=st.floats(0, 1, allow_nan=False),
    w=st.integers(1, 4000),
    h=st.integers(1, 4000),
    k=st.floats(1e-6, 1000, allow_nan=False),
)
def test_fuzz_morphology_to_micrometers(d, a, v, w, h, k):
    d_um, a_um, v_um = U.morphology_to_micrometers(d, a, v, w, h, k)
    assert all(np.isfinite(x) and x >= 0 for x in (d_um, a_um, v_um))


@_SETTINGS
@given(blob=coord_blobs, side=st.integers(1, 64))
def test_fuzz_plot_mask_never_crashes(blob, side):
    bm, morph = U.plot_mask(blob, image_size=(side, side))
    assert bm.shape == (side, side)
    assert bm.dtype == bool
    for val in morph.values():
        assert np.isfinite(val) and val >= 0.0


@_SETTINGS
@given(
    areas=arrays(np.float64, array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=30),
                 elements=st.floats(0, 1, allow_nan=False)),
    lo=st.floats(0, 1), hi=st.floats(0, 1),
)
def test_fuzz_filter_and_range(areas, lo, hi):
    df = pd.DataFrame({"area": areas, "mask": [[] for _ in areas]})
    out = U.filter_segmentation_detections(df, min_size=lo, max_size=hi)
    assert len(out) <= len(df)
    lo2, hi2 = min(lo, hi), max(lo, hi)
    for a in out["area"]:
        assert lo2 - 1e-9 <= a <= hi2 + 1e-9
    rmin, rmax = U.get_segmentation_detections_range(df)
    assert rmin <= rmax


@_SETTINGS
@given(img=gray_or_rgb, ext=st.sampled_from(["png", "jpg", "tif", "bmp"]))
def test_fuzz_safe_image_write_read_roundtrip(img, ext, tmp_path_factory):
    d = tmp_path_factory.mktemp("fuzz_img")
    path = str(d / f"img.{ext}")
    ok = U.safe_image_write(img, path)
    assert isinstance(ok, bool)
    if ok:
        back = U.read_standard_img(path)
        assert back is None or back.ndim == 3


@_SETTINGS
@given(img=gray_or_rgb)
def test_fuzz_process_loaded_image(img):
    from collections import OrderedDict
    settings_list = [OrderedDict([("gray2rgb", "")]), OrderedDict([("clip", "0,255")])]
    out = U.process_loaded_image(img, settings_list)
    assert out is not None and out.ndim == 3
