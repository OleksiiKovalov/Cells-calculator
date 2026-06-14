# Applications Monorepo

This repository hosts multiple desktop/scientific applications, each
**self-contained** under `apps/`. An app ships as its own folder — code, tests,
docs, packaging, and its runtime data (e.g. model weights) all live together, so
it can be zipped and run as-is.

The flagship application is **Cells Calculator**.

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

> Cells Calculator is a restyled, restructured rebuild of the original
> *CellsCalculator* project, carrying forward its proven segmentation core. See
> its [contributors](apps/cells-calculator/docs/CONTRIBUTORS.md).

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
Windows / Python 3.13. It's structured so more apps are added as extra jobs / a
matrix.

## Contributing
See each app's `docs/CONTRIBUTING.md` (e.g.
[Cells Calculator](apps/cells-calculator/docs/CONTRIBUTING.md)).
