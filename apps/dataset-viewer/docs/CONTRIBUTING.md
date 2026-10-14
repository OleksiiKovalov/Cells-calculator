# Contributing

Thanks for contributing to Dataset Viewer. See [DEVELOPMENT.md](DEVELOPMENT.md)
for setup and the exact commands.

## One atomic change at a time
Keep each change one clear, self-contained thing; split unrelated work into
separate changes (easier to review, revert and bisect).

## New functionality is covered by tests
New behaviour should come with a test under `tests/`. A new or changed loader /
exporter should add a round-trip or contract assertion against a sample dataset;
UI behaviour can be covered by the off-screen GUI smoke test. A pure refactor
already covered by existing tests, or a docs-only change, needs no new test.

## The checks pass
Run both before sharing a change:
```bash
python -m pytest
python -m mypy src
```
mypy must stay clean. If you add a format, follow the existing module shape
(`{fmt}_loader.py` / `{fmt}_exporter.py`) and the annotation contract in
[SPEC.md](SPEC.md) — the UI is format-agnostic and needs no changes.

## Dependencies
Update `requirements.txt` (runtime) or `requirements-dev.txt` (tooling/soft
deps) if you add an import. Keep `torch` / `Pillow` imported lazily inside the
PTH modules so the app runs without them.

## Code style
Follow [PEP 8](https://peps.python.org/pep-0008/) and match the surrounding
style. Keep coordinates pixel-absolute inside the app; only loaders/exporters
normalize.

## No runtime artifacts committed
Don't commit `__pycache__/`, caches, or exported datasets. Check `git status`
before committing (see `.gitignore`).
