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

Runtime fuzzing uses subprocess isolation so native crashes, hangs, and sanitizer
failures are kept as replayable cases:
```commandline
py -3.11 -m tests.fuzzing --max-cases 100 --workflow stateful --sanitizers all --oracle-level strict --timeout 600
```

The fuzzer also writes `coverage_summary.json` with input buckets such as
profile, extension, size, aspect ratio, workflow operation, and detected cell
count, plus behavioral buckets such as failure kind, exception type, oracle
failure class, and result shape. Use it to spot generator saturation. A
seed-stable case-spec generator is available without replacing the subprocess
harness:
```commandline
py -3.11 -m tests.fuzzing --max-cases 100 --generation-engine strategy --workflow stateful --sanitizers all --oracle-level strict --timeout 600
```

For more biologically useful corpus fuzzing, use cell-preserving mutations.
These crop around foreground blobs and apply mild microscopy-like augmentation
instead of turning most seeds into pure stress/noise cases. Optional Python
line coverage traces repo `model/` and `UI/` code in child processes and
aggregates the executed lines into `coverage_summary.json`:
```commandline
py -3.11 -m tests.fuzzing --max-cases 100 --generation-engine strategy --corpus-mutation-mode cell-preserving --python-coverage model --workflow stateful --sanitizers all --oracle-level strict --timeout 600
```

Coverage-guided fuzzing turns that line coverage into feedback. Cases that
execute new `model/` or `UI/` lines are saved under `interesting/`; future cases
prefer mutating those saved inputs, similar in spirit to a small Python-level
kcov loop:
```commandline
py -3.11 -m tests.fuzzing --max-cases 100 --generation-engine strategy --coverage-guided --corpus-mutation-mode cell-preserving --workflow stateful --sanitizers all --oracle-level strict --timeout 600
```

Leak-oriented sanitizers can check CPython allocated block growth and process
handle/file-descriptor growth. Thresholds are opt-in because ML libraries may
intentionally keep model caches alive:
```commandline
py -3.11 -m tests.fuzzing --max-cases 100 --sanitizers leaks --max-handle-growth 64 --max-allocated-block-growth 250000 --timeout 600
```

Python tracemalloc is also available, but it is expensive around model loading,
so enable it only for targeted investigations:
```commandline
py -3.11 -m tests.fuzzing --max-cases 20 --sanitizers leaks --max-tracemalloc-mb 2048 --timeout 900
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
