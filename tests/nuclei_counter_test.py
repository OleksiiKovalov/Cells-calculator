"""Tests for model.NucleiCounter.NucleiCounter."""

import cv2
import numpy as np
import pandas as pd

from model.NucleiCounter import NucleiCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense_cluster(cx: int, cy: int) -> tuple[list[int], list[int]]:
    """Return a 3×3 grid of 9 points centred on (cx, cy).

    Every point is within distance sqrt(2) ≈ 1.41 of the centre,
    so all 9 are mutually reachable under the default eps=2.
    """
    xs, ys = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            xs.append(cx + dx)
            ys.append(cy + dy)
    return xs, ys


# ---------------------------------------------------------------------------
# channel2points
# ---------------------------------------------------------------------------

def test_channel2points_black_image_gives_empty_dataframe():
    counter = NucleiCounter(threshold=100)
    channel = np.zeros((50, 50), dtype=np.uint8)
    result = counter.channel2points(channel)
    assert result.empty


def test_channel2points_single_bright_pixel_gives_one_row():
    counter = NucleiCounter(threshold=100)
    channel = np.zeros((100, 100), dtype=np.uint8)
    channel[30, 40] = 255
    result = counter.channel2points(channel)
    assert len(result) == 1


def test_channel2points_y_is_row_inverted():
    """y = image_height - row maps array (row, col) → Cartesian (x, y)."""
    height = 100
    counter = NucleiCounter(threshold=100)
    channel = np.zeros((height, 100), dtype=np.uint8)
    channel[30, 40] = 255  # row=30, col=40
    result = counter.channel2points(channel)
    assert result.iloc[0]["x"] == 40
    assert result.iloc[0]["y"] == height - 30


def test_channel2points_pixel_at_threshold_is_excluded():
    # channel2points filters with >, so a pixel exactly at self.threshold is excluded.
    counter = NucleiCounter(threshold=100)
    channel = np.zeros((50, 50), dtype=np.uint8)
    channel[10, 10] = 100  # equal to threshold, not above it
    result = counter.channel2points(channel)
    assert result.empty


# ---------------------------------------------------------------------------
# groupNuclei
# ---------------------------------------------------------------------------

def test_groupnuclei_empty_dataframe_returns_zero():
    counter = NucleiCounter()
    result = counter.groupNuclei(pd.DataFrame({"x": [], "y": []}))
    assert result == 0


def test_groupnuclei_one_dense_cluster_returns_one():
    xs, ys = _dense_cluster(0, 0)
    points = pd.DataFrame({"x": xs, "y": ys})
    counter = NucleiCounter(eps=2, min_samples=5)
    assert counter.groupNuclei(points) == 1


def test_groupnuclei_two_separated_clusters_returns_two():
    xs1, ys1 = _dense_cluster(0, 0)
    xs2, ys2 = _dense_cluster(100, 100)
    points = pd.DataFrame({"x": xs1 + xs2, "y": ys1 + ys2})
    counter = NucleiCounter(eps=2, min_samples=5)
    assert counter.groupNuclei(points) == 2


def test_groupnuclei_sparse_isolated_points_are_noise():
    # Each point is far from every other, so no cluster meets min_samples=5.
    xs = [0, 50, 100, 150]
    ys = [0, 50, 100, 150]
    points = pd.DataFrame({"x": xs, "y": ys})
    counter = NucleiCounter(eps=2, min_samples=5)
    assert counter.groupNuclei(points) == 0


# ---------------------------------------------------------------------------
# countNuclei  (integration)
# ---------------------------------------------------------------------------

def test_countnuclei_detects_two_blobs():
    """End-to-end: two filled circles on a uniform background → count == 2.

    Background is set to 100 so that the median of all bright pixels is 100,
    and the circles (value 200) survive the binary threshold in preprocess().
    The circles are far enough apart that two DBSCAN clusters emerge.
    """
    channel = np.full((200, 200), 100, dtype=np.uint8)
    cv2.circle(channel, (50, 50), 15, 200, -1)
    cv2.circle(channel, (150, 150), 15, 200, -1)
    counter = NucleiCounter()
    assert counter.countNuclei(channel) == 2


def test_countnuclei_blank_image_returns_zero():
    counter = NucleiCounter()
    channel = np.zeros((50, 50), dtype=np.uint8)
    assert counter.countNuclei(channel) == 0
