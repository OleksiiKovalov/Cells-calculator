# Cells Calculator (v4.0) — Architecture Documentation

## 1. Overview

**Cells Calculator** is a PyQt5 desktop application for cell/spheroid instance
segmentation and morphology analysis in microscopy images. v4.0 is a restructured
rebuild of the original project: a single image viewer with floating panels on
top, and a thin, uniform model layer underneath.

Design split (the guiding principle of the rebuild):
- **`src/model/`** is the core — it turns an image into detections. It is
  UI‑agnostic.
- **`src/ui/`** is "a fancy way to use the core and show results" — it owns image
  display, threading, filtering, rendering and stats.

The contract between them is one method:

```
Model(path, model_type, model_data, model_name).inference(image: np.ndarray)
    -> pandas.DataFrame[id_label, box, mask, confidence, diameter, area, volume]
```

`box` is normalized `[x, y, w, h]`; `mask` is an `(N, 2)` normalized contour;
`area`/`diameter`/`volume` are normalized morphology (see §6).

---

## 2. Entry point — `main.py` (launcher) + `src/app.py`

Startup sequence:

```
app.main()  (started by the thin root main.py launcher, which puts src/ on the path)
  └─ _preload_tensorflow_if_needed()   # import TF first iff a StarDist model is enabled (Windows DLL order)
  └─ QApplication + window icon
  └─ init_splash()                     # show splash
  └─ load_models_on_startup()          # read modelconfig.json -> register_model(...)
  └─ import ui.imports                 # staged heavy-lib preload, advancing the splash
  └─ MainWindow() + _register_segmenters() + showMaximized()
  └─ close_splash() + app.exec_()
```

`load_models_on_startup()` reads `modelconfig.json`, keeps the entries whose
`enabled` is not `"false"`, and registers each `model_type` → dotted class path
in the global registry (`src/ui/app_globals.py`).

---

## 3. UI layer (`src/ui/`)

### 3.1 `MainWindow.py`
The `QMainWindow`: a central `ImageViewer`, a top toolbar (Open, Files, zoom /
fit / reset, **model dropdown**, **Calculate**, **Show original**, **size range
slider**, **Filter**, and buttons that toggle the Info / Console / Options /
Tools / File‑Browser panels), a menu and a status bar (image info + zoom).

Responsibilities that used to live in the model layer now live here:
- `load_image()` → `model.utils.read_img()` → `ImageViewer.set_image()`.
- `callInference()` builds a `Model` for the selected entry and runs it on an
  `InferenceWorker` (QThread); `_on_inference_finished()` stores the detections,
  builds the tooltip grid, seeds the size‑filter range, renders the overlay and
  prints stats.
- `refreshPredictionImage()` / `getFilteredDetections()` filter by area and
  re‑render via `model.utils.plot_predictions` over a copy of the original image.
- `show_detection_stats()` prints the morphology summary, including µm values
  when a µm/px scale is set (`morphology_to_micrometers`).
- `_build_detection_cache()` / `_on_mouse_image_pos()` implement an O(1)
  spatial‑grid hover‑tooltip lookup over detections.
- `_save_settings()` / `_load_settings()` persist Options to `ui_settings.json`.

### 3.2 `ImageViewer.py`
A `QGraphicsView` showing the image with: zoom (Ctrl + wheel), pan (drag),
fit‑to‑window / 1:1, distance measurement (Ctrl + click‑click), rubber‑band
region selection (Shift), and a `mouse_image_pos` signal for tooltips. Emits
`zoom_changed`, `measure_distance`, `region_selected`, `mouse_image_pos`.

### 3.3 Floating panels
All float over the viewer, draggable by their title bar and edge‑resizable:
- **`InfoPanel.py`** — a text panel; used twice, as the **Info** (stats) panel
  and the **Console** (live log) panel.
- **`OptionsPanel.py`** — checkboxes (clear ruler/region, draw labels, wrap
  Info/Console, detection tooltip, fill polygons) and the **µm/px** spin box.
- **`ProgressPanel.py`** — elapsed/expected time and a **Cancel** button during
  inference.
- **`FileBrowserPanel.py`** — lists images in a folder with a thumbnail preview;
  double‑click loads.
- **`ToolbarDropDown.py`** — the popup for the **Tools** button.

### 3.4 Support modules
- **`splashscreen.py`** — dark splash with a progress bar driven by `ui.imports`.
- **`imports.py`** — *intentional*: pre‑loads the heavy scientific/ML libraries
  in stages, advancing the splash so the user sees progress while big
  dependencies warm up. Nothing imports names from it.
- **`errorhandling.py`** — root logging to `logs/<date>.log`, a global
  excepthook (logs + shows a dialog), and an **encoding‑safe** stdout/stderr
  redirector (won't crash on non‑cp1252 characters or a missing console).
- **`app_globals.py`** — the model registry (`register_model` /
  `get_registered_model`) and a couple of path constants.
- **`InferenceWorker.py`** — a `QThread` that runs `Model.inference` off the UI
  thread and emits `finished` / `error` / `cancelled`.

---

## 4. Model layer (`src/model/`)

### 4.1 `BaseSegmenter.py`
Abstract base for every segmenter:

```python
class BaseSegmenter:
    def __init__(self, path_to_model, model_data=None): ...   # sets device, calls init_model
    def init_model(self, path_to_model): ...                  # subclass: load the model
    def inference(self, image) -> DataFrame | None:           # times + delegates to call_inference
    def call_inference(self, image) -> DataFrame:             # subclass: the actual forward pass
```

### 4.2 `Model.py` — factory / facade
Resolves a `model_type` to its class via the registry, dynamically imports it
(`importlib`), instantiates it with `(path, model_data)`, and exposes
`inference(image)`. Unknown types raise a `ValueError` (after a user dialog).

### 4.3 Segmenters
All subclass `BaseSegmenter` and return the canonical DataFrame:

```
BaseSegmenter
├── YoloSegmenter        (Ultralytics YOLO11 instance segmentation)
├── InstansegSegmenter   (InstanSeg; x10/x20 config; small-image padding guard)
├── CellposeSegmenter    (Cellpose label map -> contours/morphology)
└── StardistSegmenter    (StarDist2D; NMS distance-clip crash guard)
```

`src/app.py` also maps a `cellcounter` type, but the box‑detector and the LSM
nuclei/%‑alive pipeline from the original project are **not** part of v4.0.

### 4.4 `utils.py`
The shared toolbox: image I/O (`read_img`, `read_standard_img`,
`safe_image_read/write`, LSM via `read_lsm_array` / `lsm_to_channels_last` /
`read_lsm_img`), preprocessing (`process_loaded_image`, `resize_and_pad_cv`),
mask rasterization (`plot_mask`), morphology (`calculate_morphology`,
`morphology_to_micrometers`), result conversion (`results_to_pandas`),
filtering/range helpers, and overlay rendering (`plot_predictions`).

---

## 5. Data flow

```
Open image ── read_img() ──> RGB uint8 ndarray ──> ImageViewer
                                   │
                              Calculate
                                   ▼
                 InferenceWorker (QThread): Model.inference(image)
                                   ▼
                 DataFrame[id_label, box, mask, confidence, diameter, area, volume]
                                   ▼
   _on_inference_finished: build tooltip grid · seed size range ·
                           show_detection_stats() · refreshPredictionImage()
                                   ▼
            plot_predictions(original.copy(), filtered masks) ──> overlay shown
```

Filtering (size slider) and the µm/px scale re‑drive the last two steps without
re‑running the model — the pristine detections DataFrame is never mutated.

---

## 6. Morphology & calibration

`calculate_morphology(bin_mask)` returns values **normalized** to the image:
`area = px_area / (W·H)`, `diameter = px_diameter / √(W·H)`,
`volume = px_volume / (W·H)^1.5` (volume assumes a sphere of the same area).

`morphology_to_micrometers(diameter, area, volume, W, H, µm_per_px)` inverts that
normalization to pixels and applies the calibration `k = µm/px`:
linear `× k`, area `× k²`, volume `× k³`.

---

## 7. Adding a new model

1. Create `src/model/<Name>Segmenter.py` subclassing `BaseSegmenter`; implement
   `init_model(path)` and `call_inference(image) -> DataFrame` (canonical
   columns; coordinates normalized to `[0, 1]`).
2. Add the `model_type` → dotted class path mapping in `src/app.py`'s
   `known_models`.
3. Add an entry to `modelconfig.json` and (if it adds a dependency) to
   `requirements.txt`.

No UI changes are needed — the toolbar, filtering, rendering and stats are all
backend‑agnostic.

---

## 8. Key design choices
- **Thin core, fat UI** — the model layer only produces detections; everything
  else is presentation.
- **One uniform seam** — image in, DataFrame out; no per‑backend special‑casing
  in the UI.
- **Lazy, staged loading** — segmenter modules import their heavy backends only
  when used; `ui.imports` pre‑warms them behind the splash.
- **Off‑main‑thread inference** — `InferenceWorker` keeps the UI responsive.
