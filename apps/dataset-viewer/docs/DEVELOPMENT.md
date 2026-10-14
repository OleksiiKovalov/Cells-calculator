# Dataset Viewer — Developer Manual

## Setup
```bash
pip install -r requirements-dev.txt
```
Installs the runtime deps plus the test/type tooling (`pytest`, `mypy`) and the
soft deps needed to exercise the InstanSeg `.pth` format (`torch`, `Pillow`).

`src/` is the import root (top-level modules `app`, `main_window` and the
`widgets` / `datasets` / `dialogs` packages). `python main.py` adds `src/` to the
path and launches the app.

## Running the app
```bash
python main.py
```

## Tests
```bash
python -m pytest
```
Configured by `pyproject.toml` (`pythonpath = src`, `testpaths = tests`). The
suite (`tests/`) is fully self-contained — it exercises every loader and exporter
against the bundled demo datasets under `datasets/`, so no external assets are
needed. Coverage:
- `test_format_detector.py` — auto-detection of YOLO / COCO / VOC / unknown.
- `test_loaders.py` — each loader yields the annotation contract from the samples.
- `test_roundtrip.py` — export → re-detect → counts preserved (incl. the COCO
  global-image-id regression).
- `test_coco_segmentation.py` — polygon / RLE / bbox parsing units.
- `test_gui_smoke.py` — off-screen window open → select → render → overlay toggle.

A `QApplication` is created once per session (off-screen) by `tests/conftest.py`
because the loaders use `QImageReader`/`QPixmap`.

## Type checking
```bash
python -m mypy src
```
Config in `pyproject.toml`: strict on the `datasets` loaders/exporters, lenient
on the Qt/UI modules (`widgets`, `dialogs`, `main_window`, `app`), with PySide6 and
the untyped third-party libs (`matplotlib`, `yaml`, `PIL`, `torch`) treated as
`ignore_missing_imports`. Keep it clean.

## Architecture
See [SPEC.md](SPEC.md) for the full annotation contract, per-format behaviour and
the key design invariants (pixel-absolute coordinates, cosmetic pens,
`ItemIgnoresTransformations` labels, the COCO global-id counter, the PTH
`'key' in item` mask check).
