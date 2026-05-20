# Cells Calculator Developer Manual

## Command Reference

#### Setup
Install dev dependencies.
```commandline
pip install -r requirements-dev.txt
```
This installs `mypy` and `pytest` libraries and binaries,
and other dependencies required by them.
This will also automatically install everything from `requirements.txt`.

#### Running
```commandline
python main.py
```

#### Type Checking

```commandline
mypy .
```
To check only specific folder, run
```commandline
mypy model/
```

#### Running tests
Tests inside `tests` directory are run on CI for every PR.
You can run them locally with this command:
```commandline
pytest -v tests
```

Tests inside `tests_local` require models (which are not part of the repo)
and can take a long time,
thus they aren't running on PRs.
Run them locally with this command:
```commandline
pytest -v tests_local
```

Image golden regressions run as part of the same `tests_local` suite. They process
selected committed images through nuclei counting and every enabled model from
`modelconfig.json`, then compare readable metrics with
`tests_local/golden/image_regression_baseline.json`.
Numeric comparisons use small tolerances so failures report the changed image
and field instead of a raw image hash mismatch.
For a focused run, use:
```commandline
python -m pytest -q tests_local/test_image_golden_regressions.py
```

This focused command is also suitable for bisecting a batch of patches:
```commandline
git bisect run python -m pytest -q tests_local/test_image_golden_regressions.py
```

Smoke tests run as part of the same `tests_local` suite. They load every model
listed in `tests_local/test_smoke.py` and run each one against the committed
sample images from that file. These tests only check that model loading and
inference complete without exceptions; they do not validate prediction counts,
metrics, masks, or image output correctness.
For a focused run, use:
```commandline
python -m pytest -q tests_local/test_smoke.py
```

To run one model or image while debugging, combine the focused command with a
pytest keyword expression:
```commandline
python -m pytest -q tests_local/test_smoke.py -k "StarDist and SK-N-DZ"
```

## Visual Studio Code

#### Running tests

You can run tests from the "Testing" panel in VS Code.
Make sure to set `pytest` as the testing framework for the project.
To run both `tests` and `local_tests`,
make sure both directories are present in `.vscode/settings.json`.
It should look something like this:
```json
{
    "python.testing.pytestArgs": [
        "tests",
        "tests_local"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```
