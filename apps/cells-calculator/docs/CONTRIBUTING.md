# Contributing

Thanks for contributing to Cells Calculator. This document describes what is
expected of a change. See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and the
exact commands referenced below.

## One atomic contribution per change
Keep each change one clear, logically self‑contained thing. If you find yourself
bundling unrelated work (e.g. a bug fix plus an unrelated refactor), split it.
Smaller, focused changes are easier to review, revert and bisect. If you have a
good reason to combine changes, say so in the description.

## New functionality is covered by tests
New behavior should come with tests. For most code that means a unit test under
`tests/unittests/`. For things that are hard to unit‑test in isolation — model
integration, image‑pipeline output, GUI workflows — coverage can take the form
of a smoke test, an image golden‑regression entry, or a fuzz case under
`tests_local/`. See [DEVELOPMENT.md](DEVELOPMENT.md) for both suites.

A pure refactor that existing tests already cover, or a docs‑only change, needs
no new test — just say so.

## The suite passes
CI (GitHub Actions, `.github/workflows/ci.yml`) runs on every push and pull
request: it checks the environment (numpy pin + core imports + a headless Qt
smoke), type-checks with **mypy** (`python -m mypy src`), and runs
the weight-free unit tests (`python -m pytest tests/unit`) on Windows / Python 3.13.
Keep mypy clean — see [DEVELOPMENT.md](DEVELOPMENT.md).

CI does **not** run `tests_local/` (smoke, golden regressions, runtime fuzzing)
because those need model weights that aren't in the repo — run them yourself
before sharing a change:
```bash
python -m pytest          # everything, incl. tests_local
```
`tests_local/` loads real models and can take a few minutes; missing
weights/backends are skipped automatically. Use the focused commands in
[DEVELOPMENT.md](DEVELOPMENT.md) while iterating.

If you changed inference or morphology on purpose, **regenerate the golden
baseline** (see DEVELOPMENT.md) and review the diff before committing it.

## No runtime artifacts committed
The app and tests write files that should **not** be committed:
- `logs/`
- `.cache/` (cached renders, `fuzz_failures/`, ultralytics config)
- `ui_settings.json`
- `app_screenshot.png` and any other generated images

Check your working tree before committing.

## Dependencies
If a change adds or removes a Python dependency, update `requirements.txt`
(runtime) and/or `requirements-dev.txt` (tooling). Remember the numpy pin
(`>=2.1,<2.3`) and that TensorFlow is only for StarDist.

## Code style
Follow [PEP 8](https://peps.python.org/pep-0008/) and match the surrounding
style. Keep the project lean: no dead code, no unused imports. New segmenters
must follow the `BaseSegmenter` seam (see
[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) §7).
