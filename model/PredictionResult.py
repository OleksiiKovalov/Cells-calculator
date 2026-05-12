"""Explicit result wrapper for model prediction output."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    """Raw model output plus the images used to produce/display it."""

    cells: Any
    original_image: Any = None
    inference_image: Any = None

    @property
    def detections(self):
        """Alias for callers that use detection-oriented wording."""
        return self.cells

    def __len__(self):
        return len(self.cells) if self.cells is not None else 0

    def __bool__(self):
        return self.cells is not None

    def __getattr__(self, name):
        return getattr(self.cells, name)

    def __getitem__(self, key):
        return self.cells[key]

    def __setitem__(self, key, value):
        self.cells[key] = value

    def __contains__(self, key):
        return self.cells is not None and key in self.cells

    def __iter__(self):
        return iter(self.cells) if self.cells is not None else iter(())

    def copy(self, *args, **kwargs):
        return self.cells.copy(*args, **kwargs)


def unwrap_prediction_cells(result):
    """Return the raw cells/detections payload from supported result shapes."""
    if isinstance(result, PredictionResult):
        return result.cells
    return result


def get_prediction_images(result):
    """Return ``(original_image, inference_image)`` from a result wrapper/dict."""
    if isinstance(result, PredictionResult):
        return result.original_image, result.inference_image

    if isinstance(result, dict):
        cells = result.get("Cells")
        if isinstance(cells, PredictionResult):
            original = result.get("original_image")
            inference = result.get("inference_image")
            if original is None:
                original = cells.original_image
            if inference is None:
                inference = cells.inference_image
            return original, inference
        return result.get("original_image"), result.get("inference_image")

    return None, None


def count_prediction_cells(result) -> int:
    """Count rows/items in a prediction result or raw detections payload."""
    cells = unwrap_prediction_cells(result)
    if cells is None:
        return 0
    if hasattr(cells, "shape"):
        return int(cells.shape[0])
    return int(cells)
