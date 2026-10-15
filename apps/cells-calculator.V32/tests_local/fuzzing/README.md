# Runtime Fuzzing Harness

The runtime fuzzer lives here so it can be shipped together with its unit tests
and kept separate from production application modules.

Entry points:

```powershell
py -3.11 -m tests.fuzzing --list-models
py -3.11 -m tests.fuzzing --max-cases 500 --generation-engine grammar-mutational --workflow stateful --max-workflow-steps 8 --model-strategy random --seed-corpus testimages --image-profile mixed --corpus-probability 0.7 --coverage-guided --sanitizers all --oracle-level paranoid --determinism-check counts --timeout 600
```

Module layout:

- `config.py`: CLI parsing, settings, constants, model discovery.
- `generation.py`: generated images, semantic grammar recipes, corpus mutation.
- `subprocess_runner.py`: child-process execution and environment isolation.
- `instrumentation.py`: sanitizers, memory probes, Python line/MC/DC coverage.
- `oracles.py`: result validation, mock models, UI workflow exercising.
- `child_runner.py`: single-case child process execution.
- `signatures.py`: failure signatures, coverage summaries, saved cases.
- `minimizer.py`: failure minimization and replay preservation checks.
- `runner.py`: parent fuzzing loop.
