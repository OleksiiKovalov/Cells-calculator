# Applications Monorepo

This repository hosts multiple desktop/scientific applications, each
**self-contained** under `apps/`. An app ships as its own folder — code, tests,
docs, packaging, and its runtime data (e.g. model weights) all live together, so
it can be zipped and run as-is.

The flagship application is **Cells Calculator**; the repo also ships
**Dataset Viewer**.

---

## 🔬 Cells Calculator — flagship app

**[apps/cells-calculator/](apps/cells-calculator/README.md)** — a PyQt5 desktop
application for **automated cell & spheroid analysis in microscopy images**.
Biologists and lab researchers open a micrograph, pick a segmentation model, and
get every cell outlined with quantitative morphology — in seconds, with no
manual annotation.

![Cells Calculator](apps/cells-calculator/docs/images/app_screenshot.png)

### What it does
Given a microscopy image (brightfield, fluorescence, phase-contrast, or Zeiss
**LSM**), it runs **instance segmentation** to find each individual cell/spheroid,
draws the masks over the image, and reports per-object and average **morphology**:
- **area**, **equivalent diameter**, and a sphere-extrapolated **volume**;
- reported both relative to the image and, with a **µm/px calibration**, in real
  physical units (**µm / µm² / µm³**).

It's built for real lab workflows: load a folder of images, filter detections by
size, measure distances, inspect per-cell numbers on hover, and switch between
the original and the segmented overlay — all in a responsive UI that keeps
working while a model runs on a background thread.

### Segmentation backends
Four state-of-the-art models are interchangeable behind one uniform interface;
choose per image from the toolbar (configured in `modelconfig.json`):

| Backend | Best for |
|--------|----------|
| **YOLO11** (Ultralytics) | fast, robust general cell segmentation |
| **InstanSeg** | nuclei & cells, fluorescence; tiled large images |
| **Cellpose** | generalist cellular segmentation (cyto/nuclei models) |
| **StarDist** | star-convex nuclei/cell instances (TensorFlow backend) |

Adding another backend is a single class implementing
`call_inference(image) -> DataFrame` — the entire UI is model-agnostic.

### Highlights
- **µm calibration** — morphology in physical units from a pixel-size setting.
- **Broad image support** incl. multi-channel **LSM** of any size.
- **Interactive viewer** — zoom/pan/fit, overlay toggle, hover tooltips,
  distance measurement and region selection.
- **On-the-fly size filtering**, threaded & cancellable inference, live console.
- **Engineered to last** — typed (`mypy`-clean), unit + smoke + golden-image
  regression + runtime fuzzing test suites, and CI.

### Run it
```bash
cd apps/cells-calculator
pip install -r requirements.txt
python main.py
```
Full usage, model configuration, architecture and development docs:
**[apps/cells-calculator/README.md](apps/cells-calculator/README.md)** and its
[docs/](apps/cells-calculator/docs/).

> Cells Calculator is a **redesign and restyling of the well-known previous
> version** of *CellsCalculator* (V3.2), carrying forward its proven
> segmentation core in a cleaner architecture and UI. That original version is
> preserved in this repository at
> **[apps/cells-calculator.V32/](apps/cells-calculator.V32/)**. See also the
> [contributors](apps/cells-calculator/docs/CONTRIBUTORS.md).

---

## 🗂 Dataset Viewer

**[apps/dataset-viewer/](apps/dataset-viewer/README.md)** — a PyQt5 tool for
**browsing and converting annotated image datasets**. Open a dataset folder (the
format is auto-detected), view every image with its bounding boxes, polygons and
masks overlaid, and export the whole dataset to another format.

- **Formats** (load + export): YOLO (v5/v8), COCO JSON, Pascal VOC, InstanSeg
  PTH — including COCO RLE masks and YOLO segmentation polygons.
- **Viewer**: zoom / pan / fit, a split-aware browser, color-cycled overlays.
- Shares the flagship **Cells Calculator** conda environment — no separate
  install; launch with `apps/dataset-viewer/run.bat`.

---

## Repository layout
```
apps/
└─ <app-name>/
   ├─ main.py                 # launcher (run from the app folder)
   ├─ pyproject.toml          # pytest + mypy config
   ├─ requirements.txt  requirements-dev.txt
   ├─ src/                    # import root (top-level modules/packages)
   ├─ tests/                  # per-app tests
   ├─ docs/                   # per-app documentation
   └─ …                       # per-app runtime data (e.g. model weights)
```
New apps drop in as `apps/<name>/` with the same internal shape.

## Continuous integration
`.github/workflows/ci.yml` (GitHub requires workflows at the repo root) runs each
app's checks — type-checking (`mypy`) and the weight-free unit tests — on
Windows via a **matrix** over the apps (`cells-calculator` and `dataset-viewer`
on Python 3.13, plus the legacy V3.2 app on 3.11, non-blocking). New apps are
added as one more matrix entry.

## Contributing
See each app's `docs/CONTRIBUTING.md` (e.g.
[Cells Calculator](apps/cells-calculator/docs/CONTRIBUTING.md)).

## 👥 Contributors
Cells Calculator was created by students of **NTU "KhPI"** across successive
releases (V2 → V4.0). The current v4.0 redesign builds directly on their work —
full credit to everyone who contributed (see also [CONTRIBUTORS.md](CONTRIBUTORS.md)):

**CellsCalculator V2.0**
- **Ponomarov Y.** — team-lead, ML engineer
- **Kuznesova I.** — ML engineer, data labelling
- **Batiuchenko O.** — software developer
- **Noskova K.** — ML engineer
- **Glushchenko D.** — lead of documentation editing assistance, lead of data labelling
- **Baluka A.** — documentation editing assistance, data labelling
- **Ipatko K.** — documentation editing assistance, data labelling

**CellsCalculator V3.0**
- **Cherkashyna I.** — team-lead, lead of documentary editing
- **Fesenko M.** — ML engineer
- **Lytvynenko S.** — ML engineer
- **Olijnyk V.** — ML engineer
- **Kovalov O.** — tech-lead

**CellsCalculator V3.1**
- **Cherkashyna I.** — team-lead, lead of documentary editing
- **Noskova K.** — ML engineer
- **Lytvynenko S.** — ML engineer
- **Kharkivskyi I.** — ML engineer
- **Zharskyi N.** — software engineer
- **Besedina Y.** — documentation editing assistance, data labelling
- **Borysenko M.** — documentation editing assistance, data labelling
- **Tkachenko V.** — documentation editing assistance, data labelling
- **Kovalov O.** — tech-lead

**CellsCalculator V3.2**
- **Kharkivskyi I.** — Software Development team lead, functionality fix
- **Koziuk D.** — software engineer, functionality fix
- **Kremliov R.** — software engineer, functionality fix
- **Savchenko V.** — software engineer, refactoring
- **Rudenko H.** — software engineer, refactoring
- **Ohanjanyan A.** — software engineer, automated testing
- **Hazin H.** — software engineer, automated testing
- **Smirnov S.** — QA, manual testing
- **Lytvynenko S.** — Data Science and Documentation team lead
- **Malakhov R.** — ML engineer
- **Lyndin Y.** — ML engineer
- **Pohasii M.** — ML engineer
- **Moskalenko O.** — ML engineer
- **Li P.** — ML engineer
- **Lysachenko S.** — ML engineer
- **Borysenko M.** — documentation editing assistance
- **Tyshchenko K.** — documentation editing assistance
- **Boiko K.** — documentation editing assistance
- **Fesenko M.** — Data Labeling team lead
- **Dolhodush A.** — Data Labeling
- **Husachenko M.** — Data Labeling
- **Vuziian Y.** — Data Labeling
- **Borovko L.** — Data Labeling
- **Kovalov O.** — tech-lead

**CellsCalculator V4.0** (redesign)
- **Kovalov O.** — tech-lead
