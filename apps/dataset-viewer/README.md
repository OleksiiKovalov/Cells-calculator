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
- **Convert** between any supported formats via *Prepare & Export…* with a
  progress bar, or headlessly via `src/convert.py` (same engine, no GUI).
- **Prepare** while exporting — to **any** format: preprocess (standardize to
  RGB/uint8, resize toward N px, contrast) and (re)split train/val/test (keep the
  source's or re-split by ratio+seed). Resize rescales bbox/polygon/RLE
  coordinates, so a resized YOLO/COCO/VOC export stays correct — not only `.pth`.
  See [Prepare & Export](#prepare--export).

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
- **File → Prepare & Export…** (Ctrl+Shift+S) — preprocess, split and export.

### Prepare & Export
The dialog turns a raw annotation set into a ready dataset in one step, for **any**
target format:
- **Preprocess** — *standardize* images to RGB/uint8, *resize* toward N px, and/or
  *enhance contrast*. Resize rescales every annotation (bbox, polygon, and COCO
  RLE mask), so a resized YOLO/COCO/VOC export stays geometrically correct.
- **Split** — *keep source splits*, or *re-split by ratio* (train/val/test + seed)
  to (re)partition an unsplit or differently-split dataset.
- **InstanSeg PTH** target adds *file name* and *modality*; `val` splits are
  written as **`Validation`** (the key InstanSeg's trainer reads).

### Headless (scripting / batch)
Same engine, no window:
```bash
python src/convert.py --src path/to/coco --out out --to yolo \
    --resize --resize-target 512 --split ratio --train 0.7 --val 0.1 --test 0.2
```
Importable: `convert.convert_dataset(src, out, "pth", prepare_spec={...}, pth_options={...})`.

Two tiny synthetic datasets ship under `datasets/` (`yolo_seg`, `coco`) for trying it out and for the test suite.

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
  convert.py            #   headless prepare & convert (CLI + convert_dataset())
  datasets/             #   base_loader, format_detector, rle, prepared_loader,
                        #   {yolo,coco,voc,pth}_{loader,exporter}
  dialogs/              #   save_as_dialog (Prepare & Export)
datasets/               # bundled demo datasets
docs/                   # SPEC.md + developer docs
tests/                  # pytest suite
```
