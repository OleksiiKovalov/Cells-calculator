"""Runtime fuzz of the model pipeline — one parametrized case per enabled model.

Part of the normal ``python -m pytest`` run: every enabled model is fuzzed with
FUZZ_MAX_CASES (default 20) generated/edge/corpus images and must finish with no
crash or oracle violation. Models with missing weights / backends / downloadable
weights that can't be fetched are skipped.

Tune locally:  FUZZ_MAX_CASES=100 python -m pytest tests/test_fuzz_runtime.py
Deeper ad-hoc: python -m tests.fuzzing --model "<name>" --max-cases 500
"""
import os

import pytest

from tests._models import BACKEND_DEP, DOWNLOAD_HINTS, ROOT, enabled_models
from tests.fuzzing.runner import run

MAX_CASES = int(os.environ.get("FUZZ_MAX_CASES", "20"))
SEED = int(os.environ.get("FUZZ_SEED", "20240614"))

_MODELS = list(enabled_models().items())


@pytest.mark.parametrize("name,data", _MODELS, ids=[m[0] for m in _MODELS])
def test_runtime_fuzz_each_model(name, data):
    dep = BACKEND_DEP.get(data["model_type"])
    if dep:
        pytest.importorskip(dep)
    path = data["path"]
    if str(path).startswith("trainedmodels") and not (ROOT / path).exists():
        pytest.skip(f"weights missing: {path}")

    try:
        rc = run(name, max_cases=MAX_CASES, seed=SEED, profile="mixed", corpus_dir=None)
    except Exception as exc:  # model weights download / offline backend -> skip
        if any(h in str(exc).lower() for h in DOWNLOAD_HINTS):
            pytest.skip(f"{name}: backend/weights unavailable ({exc})")
        raise

    assert rc == 0, f"{name}: runtime fuzzer found failures (see .cache/fuzz_failures/)"
