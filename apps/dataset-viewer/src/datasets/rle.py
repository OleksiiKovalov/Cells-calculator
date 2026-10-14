"""Uncompressed COCO RLE helpers.

COCO run-length encoding is column-major (Fortran order): the mask is
flattened column by column and ``counts`` alternates runs of 0s and 1s,
always starting with a run of zeros (possibly of length 0).
"""
import numpy as np


def encode_mask(binary: np.ndarray) -> list[int]:
    """Encode a 2-D binary mask as uncompressed COCO RLE counts."""
    flat = np.asarray(binary, dtype=bool).flatten(order='F')
    if flat.size == 0:
        return []
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    counts = [int(c) for c in np.diff(boundaries)]
    if flat[0]:
        counts.insert(0, 0)
    return counts


def decode_mask(counts: list, size: list) -> np.ndarray:
    """Decode uncompressed COCO RLE counts to a (h, w) uint8 mask."""
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=np.uint8)
    idx, val = 0, 0
    for cnt in counts:
        end = min(idx + int(cnt), h * w)
        if val:
            flat[idx:end] = 1
        idx = end
        val ^= 1
    return flat.reshape((h, w), order='F')
