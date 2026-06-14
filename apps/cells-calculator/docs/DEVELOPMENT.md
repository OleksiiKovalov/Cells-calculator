# Cells Calculator (V6) — Developer Manual

## Setup
```bash
pip install -r requirements-dev.txt
```
This installs `requirements.txt` plus the test tooling (`pytest`, `hypothesis`,
`pillow`).

Environment notes:
- **Python 3.13** is the reference interpreter.
- **numpy** must stay in `>=2.1,<2.3` (pinned in `requirements.txt`):
  TensorFlow/ml_dtypes need ≥2.1 on py3.13, numba/StarDist need <2.3.
- **TensorFlow** is only required for StarDist. On Windows it must initialise
  before torch/PyQt5 (a native DLL load‑order constraint); `main.py` and the test
  `conftest.py` both import it first when needed.

## Running the app
```bash
python main.py
```

## Type checking
The project is type-checked with mypy (config in `pyproject.toml`): strict-ish on
`main.py` + `model/`, lenient on `ui.*` (Qt is dynamic), with third-party libs
that lack usable stubs set to `ignore_missing_imports`.
```bash
python -m mypy src
```
This runs in CI and must stay clean. No stub packages are needed (pandas etc.
are treated as untyped to avoid fighting the `numpy>=2.1,<2.3` pin).

## Tests
The whole suite runs with one command:
```bash
python -m pytest
```
It is configured by `pyproject.toml` (`testpaths = tests`, `pythonpath = src`) and
`conftest.py` (off‑screen Qt, TensorFlow preloaded first, a `qapp` fixture).

### Layout
- **`tests/unit/`** — fast, model‑free unit tests for `model/utils`
  (morphology, µm, LSM, mask rasterization, filtering, image I/O), the
  segmenter converters, the `Model`/`BaseSegmenter` seam, the cp1252‑safe
  logger, and the GUI helpers (off‑screen).
- **`tests/`** — heavier tests that load real models:
  - `test_smoke.py` — every enabled model runs through `Model.inference` without
    crashing.
  - `test_image_golden_regressions.py` — numeric regression vs.
    `tests/golden/baseline.json` for the local‑weight models.
  - `test_fuzz.py` — Hypothesis property‑fuzzing of the pure pipeline.
  - `test_fuzz_runtime.py` — bounded runtime fuzz of **each** enabled model.
  - `fuzzing/` — the standalone runtime fuzzer (also usable from the CLI).

Models with missing weights, missing backends, or undownloadable built‑in
weights are **skipped**, not failed.

### Focused runs
```bash
python -m pytest tests/unit                       # just the fast unit tests
python -m pytest tests/test_smoke.py
python -m pytest tests/test_image_golden_regressions.py
python -m pytest tests/test_smoke.py -k "Stardist"   # one model by keyword
```

### Golden‑image regressions
Each enabled model with **local** weights is run on a committed sample image; a
small numeric summary (count, area/diameter/confidence aggregates) is compared
against `tests/golden/baseline.json` within tolerances, so an accidental
change to the inference/morphology pipeline is caught with a readable diff.

Regenerate the baseline after an **intentional** change (TensorFlow is imported
first so StarDist loads):
```bash
python -c "import tensorflow, runpy; runpy.run_path('tests/test_image_golden_regressions.py', run_name='__main__')"
```

### Runtime fuzzing
`test_fuzz_runtime.py` fuzzes every enabled model as part of `python -m pytest`
(default 20 cases each). Tune without editing code:
```bash
FUZZ_MAX_CASES=100 python -m pytest tests/test_fuzz_runtime.py   # deeper
FUZZ_MAX_CASES=5   python -m pytest                                    # quick full run
```

For deeper ad‑hoc runs use the CLI harness directly:
```bash
python -m tests.fuzzing --list-models
python -m tests.fuzzing --model "YOLO-512 Segmenter" --max-cases 500
python -m tests.fuzzing --model "Cellpose cyto3" --max-cases 200 \
    --profile mixed --seed 0 \
    --corpus "path/to/seed/images" --corpus-probability 0.7 -v
```
Flags: `--model`, `--max-cases`, `--seed`, `--profile {random,edge,corpus,mixed}`,
`--corpus DIR`, `--corpus-probability`, `--out-dir`, `-v`. The fuzzer generates
3‑channel uint8 images (the contract `read_img` guarantees the model), checks
output invariants, and writes failing cases to `.cache/fuzz_failures/` for
replay. Exit code is non‑zero if any case crashes or violates an oracle.

### Running tests in VS Code
Select **pytest** as the framework; `.vscode/settings.json`:
```json
{
    "python.testing.pytestArgs": ["tests"],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```
