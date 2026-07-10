# Dataset Viewer

A PySide6 desktop app for **browsing and converting annotated image datasets**.
Open a dataset folder — the format is auto-detected — view every image with its
bounding boxes, polygons and masks overlaid, and export the whole thing to a
different format.

## Features
- **Auto-detects** the dataset format on open (no manual picking).
- **Visual overlays**: bounding boxes, polygons (semi-transparent fill) and COCO
  RLE masks, color-cycled per annotation with always-readable labels.
- **Image viewer**: smooth zoom (2 %–3200 %), pan, fit-to-window, 1:1.
- **Split-aware browser**: train / val / test grouped in a dockable tree;
  arrow keys walk through images.
- **Convert** between any supported formats via *Save As…* with a progress bar.

## Supported formats
| Format | Load | Export | Notes |
|--------|:----:|:------:|-------|
| **YOLO** (v5/v8) | ✓ | ✓ | split or flat layout; bboxes + segmentation polygons; `data.yaml`/`.names` |
| **COCO** (instances JSON) | ✓ | ✓ | polygons, uncompressed **and** LEB128-compressed RLE (no pycocotools) |
| **Pascal VOC** | ✓ | ✓ | `Annotations/*.xml`, `ImageSets/Main`, optional `<polygon>` |
| **InstanSeg PTH** | ✓ | ✓ | `torch.save` dict of instance-mask items (needs `torch` + `Pillow`) |

Internally all annotations are **pixel-absolute** dicts (`bbox` / `polygon` /
`mask`); only loaders and exporters do normalization. See [SPEC.md](docs/SPEC.md) for
the full contract and design invariants.

## Install & run

> **Runs in the flagship app's environment.** Within this monorepo, Dataset
> Viewer is meant to run inside the **Cells Calculator** conda environment rather
> than having its own — its dependencies (PySide6, numpy, matplotlib; PyYAML for
> YOLO `data.yaml`; torch + Pillow for `.pth`) are already provided there. After
> setting up the flagship env (`apps/cells-calculator/install.bat`), launch it
> with **`run.bat`** (or `conda run -n cells-calculator python main.py`). No
> separate install step is needed.

To run it **standalone** instead (outside the monorepo), Python 3.10+:
```bash
pip install -r requirements.txt
python main.py
```
`torch` and `Pillow` are only needed for the InstanSeg `.pth` format (imported
lazily, with a helpful message if missing).

## Using it
- **File → Open Folder…** (Ctrl+O) — auto-detect a YOLO / COCO / VOC dataset.
- **File → Open File…** (Ctrl+Shift+O) — open an InstanSeg `.pth`.
- Click an image (or use ← / →) to view it; toggle overlays with **A**.
- **File → Save As…** (Ctrl+Shift+S) — convert to another format.

Three tiny synthetic datasets ship under `datasets/` (`yolo_seg`, `coco`) for trying it out and for the test suite.

## Development
```bash
pip install -r requirements-dev.txt
python -m pytest          # loaders, exporters, round-trips, format detection, GUI smoke
python -m mypy src main.py
```

## Project layout
```
main.py                 # launcher — run `python main.py` from this folder
pyproject.toml          # pytest + mypy config
src/                    # import root
  app.py                #   entry point (Fusion style, High-DPI)
  main_window.py        #   window, menus, toolbar, actions
  widgets/              #   image_viewer (QGraphicsView), file_browser (dock tree)
  datasets/             #   base_loader, format_detector, {yolo,coco,voc,pth}_{loader,exporter}
  dialogs/              #   save_as_dialog
datasets/               # bundled demo datasets
docs/                   # SPEC.md + developer docs
tests/                  # pytest suite
```
