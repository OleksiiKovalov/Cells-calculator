"""Golden-image regression tests.

Each enabled model with local weights is run on a committed sample image; a
small numeric summary of its detections (count + area/diameter/confidence
aggregates) is compared against a committed baseline within tolerances, so an
accidental change to the inference/morphology pipeline is caught.

Regenerate the baseline after an intentional change:

    python tests/test_image_golden_regressions.py
"""
import json
from collections import OrderedDict
from pathlib import Path

import pytest

from tests._models import BACKEND_DEP, KNOWN_MODELS, ROOT

SAMPLE_IMAGE = Path(__file__).parent / "data" / "TYPE_13_10.jpg"
BASELINE_PATH = Path(__file__).parent / "golden" / "baseline.json"

COUNT_ABS, COUNT_REL = 2, 0.05
FLOAT_TOL = {  # metric -> (abs, rel)
    "area_sum": (0.02, 0.15),
    "area_mean": (0.0005, 0.20),
    "diameter_mean": (0.005, 0.15),
    "confidence_sum": (5.0, 0.15),
    "confidence_mean": (0.05, 0.15),
}


def _enabled_local_models():
    """Enabled models backed by a local trainedmodels/ weight file (deterministic,
    offline)."""
    cfg = json.loads((ROOT / "modelconfig.json").read_text(encoding="utf-8"),
                     object_pairs_hook=OrderedDict)
    out = OrderedDict()
    for name, data in cfg.items():
        if str(data.get("enabled", "true")).lower() != "true":
            continue
        path = str(data.get("path", ""))
        if path.startswith("trainedmodels") and (ROOT / path).exists():
            out[name] = data
    return out


def _summarize(det) -> dict:
    summary = {"count": int(det.shape[0])}
    if "area" in det:
        a = det["area"].astype(float)
        summary["area_sum"] = round(float(a.sum()), 6)
        summary["area_mean"] = round(float(a.mean()), 6) if len(a) else 0.0
    if "diameter" in det:
        d = det["diameter"].astype(float)
        summary["diameter_mean"] = round(float(d.mean()), 6) if len(d) else 0.0
    if "confidence" in det:
        c = det["confidence"].astype(float)
        summary["confidence_sum"] = round(float(c.sum()), 6)
        summary["confidence_mean"] = round(float(c.mean()), 6) if len(c) else 0.0
    return summary


def _run_summary(name, data):
    from ui.app_globals import register_model
    from model.Model import Model
    from model.utils import read_img
    register_model(data["model_type"], KNOWN_MODELS[data["model_type"]], False)
    img = read_img(str(SAMPLE_IMAGE))
    model = Model(path=data["path"], model_type=data["model_type"],
                  model_data=data, model_name=name)
    return _summarize(model.inference(img))


def _close(actual, expected, abs_tol, rel_tol=0.0):
    return abs(float(actual) - float(expected)) <= max(abs_tol, abs(float(expected)) * rel_tol)


def _load_baseline():
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


_BASELINE = _load_baseline()
_CASES = list(_BASELINE.keys())


@pytest.mark.skipif(not _CASES, reason="no golden baseline committed")
@pytest.mark.parametrize("name", _CASES)
def test_model_matches_golden_baseline(name):
    expected = _BASELINE[name]
    enabled = _enabled_local_models()
    if name not in enabled:
        pytest.skip(f"{name} not enabled / weights missing")
    if (dep := BACKEND_DEP.get(enabled[name]["model_type"])):
        pytest.importorskip(dep)

    actual = _run_summary(name, enabled[name])

    assert _close(actual["count"], expected["count"], COUNT_ABS, COUNT_REL), \
        f"{name}.count: expected {expected['count']}, got {actual['count']}"
    for metric, (abs_tol, rel_tol) in FLOAT_TOL.items():
        if metric in expected:
            assert metric in actual, f"{name}.{metric} missing in output"
            assert _close(actual[metric], expected[metric], abs_tol, rel_tol), \
                f"{name}.{metric}: expected {expected[metric]} +/-, got {actual[metric]}"


def _generate_baseline():
    """Run every enabled local model and (re)write the baseline JSON."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = {}
    for name, data in _enabled_local_models().items():
        try:
            baseline[name] = _run_summary(name, data)
            print(f"  {name}: {baseline[name]}")
        except Exception as exc:
            print(f"  {name}: SKIPPED ({exc})")
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"wrote {BASELINE_PATH} with {len(baseline)} models")


if __name__ == "__main__":
    _generate_baseline()
