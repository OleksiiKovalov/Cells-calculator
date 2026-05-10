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

Fuzz/property tests are marked with `fuzz` and require the dev dependencies.
Run them with:
```commandline
pytest -m fuzz
```

To run the deterministic tests without fuzzing:
```commandline
pytest -m "not fuzz"
```

For runtime fuzzing against real models, use the subprocess harness.
It generates images plus UI-like parameters, runs selected models with a timeout,
and keeps failing cases under `.cache/runtime_fuzz/failures`.
```commandline
python scripts/fuzz_runtime.py --max-cases 25 --timeout 180
```

Run forever until Ctrl+C:
```commandline
python scripts/fuzz_runtime.py --max-cases 0 --timeout 180
```

Limit by time or model name:
```commandline
python scripts/fuzz_runtime.py --seconds 3600 --models "Detector,YOLO-512 Segmenter"
```

YOLO-specific fuzzing can use the model type directly. The `auto` image profile
already switches YOLO models to a YOLO-focused generator, but it can also be
requested explicitly:
```commandline
python scripts/fuzz_runtime.py --max-cases 20 --models yolo --timeout 600 --image-profile yolo
```

Keep YOLO on regular full-image inference only:
```commandline
python scripts/fuzz_runtime.py --models yolo --scales 20 --max-cases 20
```

Replay a failure:
```commandline
python scripts/fuzz_runtime.py --replay .cache/runtime_fuzz/failures/<case>/case.json
```

Tests inside `tests_local` require models (which are not part of the repo)
and can take a long time,
thus they aren't running on PRs.
Run them locally with this command:
```commandline
pytest -v tests_local
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
