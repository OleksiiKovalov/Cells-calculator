"""Output invariants checked for every fuzz case."""
import numpy as np
import pandas as pd

CANON_COLS = ["id_label", "box", "mask", "confidence", "diameter", "area", "volume"]


def check_detections(det):
    """Return a list of invariant violations (empty == OK)."""
    violations = []
    if not isinstance(det, pd.DataFrame):
        return [f"result is not a DataFrame: {type(det).__name__}"]
    if len(det) == 0:
        return violations

    for col in ("id_label", "mask", "area", "diameter", "volume", "confidence"):
        if col not in det.columns:
            violations.append(f"missing column '{col}'")

    for col in ("area", "diameter", "volume"):
        if col in det:
            vals = pd.to_numeric(det[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                violations.append(f"{col} contains non-finite values")
            if (vals < 0).any():
                violations.append(f"{col} contains negative values")

    if "area" in det:
        areas = pd.to_numeric(det["area"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(areas).any() and np.nanmax(areas) > 1.5:
            violations.append("area > 1.5 (expected normalized <= 1)")

    return violations
