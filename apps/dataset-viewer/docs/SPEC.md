# Dataset Viewer — Full Specification

Build a **PyQt5 desktop application** called "Dataset Viewer" for browsing and converting annotated image datasets. Use the Fusion style. Entry point is `main.py`; all app code lives under `app/`.

---

## Dependencies (`requirements.txt`)
```
PyQt5>=5.15.0
PyYAML>=6.0
numpy>=1.21.0
matplotlib>=3.5.0
```
Soft dependencies (import inside methods, show helpful error if missing): `torch` (for `.pth` files), `Pillow` (for PTH image I/O — available transitively via matplotlib).

---

## File structure
```
main.py
app/
  __init__.py
  main_window.py
  widgets/
    __init__.py
    image_viewer.py
    file_browser.py
  datasets/
    __init__.py
    base_loader.py
    format_detector.py
    yolo_loader.py
    coco_loader.py
    voc_loader.py
    pth_loader.py
    yolo_exporter.py
    coco_exporter.py
    voc_exporter.py
    pth_exporter.py
  dialogs/
    __init__.py
    save_as_dialog.py
```

---

## `main.py`
Enable `AA_EnableHighDpiScaling` and `AA_UseHighDpiPixmaps`. Set app name to `"Dataset Viewer"`, style to `"Fusion"`. Show `MainWindow`, call `sys.exit(app.exec_())`.

---

## Annotation contract
All loaders produce pixel-absolute annotations as dicts. Three types:

**Bounding box:**
```python
{'class_id': int, 'label': str, 'type': 'bbox', 'x': float, 'y': float, 'w': float, 'h': float}
```

**Polygon:**
```python
{'class_id': int, 'label': str, 'type': 'polygon',
 'x': float, 'y': float, 'w': float, 'h': float,  # bounding box of the polygon
 'points': [(x,y), ...],      # single polygon (YOLO / VOC)
 'polygons': [[(x,y),...], ...]}  # multi-part (COCO)
```
Either `points` or `polygons` is present (not both).

**Mask (COCO RLE only):**
```python
{'class_id': int, 'label': str, 'type': 'mask',
 'x': float, 'y': float, 'w': float, 'h': float,
 'rle_counts': [int, ...], 'rle_size': [height, width]}
```

---

## `base_loader.py`
Abstract base class `BaseDatasetLoader(ABC)`:
- `__init__(self, folder: str)` — stores `self.folder`, `self.class_names: list[str] = []`
- `get_splits() -> list[str]` — return split names (`['train','val','test']`) or `[]` if unsplit
- `get_images(split: str | None = None) -> list[dict]` — return `[{'path': str, 'name': str}]`
- `get_annotations(image_path: str) -> list[dict]` — return annotation dicts per the contract above

---

## `format_detector.py`
`detect_format(path: str) -> (loader | None, format_name | None)`

Detection order:
1. **Direct `.pth` file**: if `path.lower().endswith('.pth')` and `os.path.isfile(path)` → `PTHLoader(path), "InstanSeg PTH"`
2. Bail if not a directory.
3. **COCO**: look for `annotations/*.json` or root `*.json` where the first 8 KB contains `"images"`, `"annotations"`, `"categories"`.
4. **Pascal VOC**: `Annotations/` directory containing `.xml` files.
5. **YOLO (split layout)**: `images/` + `labels/` subdirs, or a `.yaml`/`.yml`/`.names` file in root.
6. **YOLO (flat layout)**: image files paired with same-stem `.txt` files in root.
7. **PTH in folder**: any `*.pth` file in root → `PTHLoader(sorted_first_match)`.

---

## `yolo_loader.py`
Supports YOLOv5/v8 split layout (`images/{split}/`, `labels/{split}/`) and flat layout.

- Load class names from `*.yaml` (via PyYAML, key `names` as list or dict), `*.yml`, or `*.names`.
- `get_splits()`: list subdirectories of `images/`.
- `get_images(split)`: enumerate image files (`.jpg .jpeg .png .bmp .tif .tiff .webp`) from `images/{split}/` or `images/` or root.
- `get_annotations(image_path)`: find the matching `.txt` label by swapping the `images` path component for `labels`. Parse each line:
  - **Bbox**: `class_id cx cy w h` (normalized) → convert to pixel absolute top-left + size using `QImageReader`.
  - **Polygon**: `class_id x1 y1 x2 y2 ...` (>4 coords, even count, normalized) → denormalize, store as `points`, compute bbox.
  - Use `QImageReader` to get image dimensions.

---

## `coco_loader.py`
- Find all JSON files in `annotations/` and root.
- Detect split from filename stem: if it contains `'train'`, `'val'`, or `'test'`, use that; else `'all'`.
- Parse `images`, `annotations`, `categories` from each JSON.
- Map `category_id` → 0-based `class_id` using sorted category IDs.
- Annotation segmentation parsing:
  - List of lists → `type: 'polygon'`, `polygons: [[(x,y),...]]`
  - Dict with `counts` (list) → uncompressed RLE, `type: 'mask'`
  - Dict with `counts` (str) → LEB128-compressed RLE, decode inline (no pycocotools), `type: 'mask'`
  - Otherwise → `type: 'bbox'`
- `get_splits()`: return sorted unique splits only if more than one exists.
- `_resolve_path`: try multiple candidate locations including `images/`, `{split}/`, `{split}2017/`, `{split}2014/`.

---

## `voc_loader.py`
- `Annotations/` for `.xml` files, candidate image dirs: `JPEGImages/`, `images/`, `Images/`, `imgs/`, `img/`, root.
- Splits from `ImageSets/Main/*.txt` — only files named `train`, `val`, `test`, `trainval`. Lines may be `"stem"` or `"stem  1"` (VOC difficult flag).
- Collect class names by sampling up to 500 XMLs.
- Each XML `<object>`: read `<name>`, `<bndbox>` (xmin/ymin/xmax/ymax → x/y/w/h), optional `<polygon>` with `<x1><y1><x2><y2>...` tags.

---

## `pth_loader.py`
InstanSeg `.pth` format: `torch.save()`-ed dict `{'Train': [item, ...], 'Validation': [...], 'Test': [...]}`.

Each item dict:
```python
{'image': np.ndarray,           # HxW, HxWx1, HxWx3, HxWx4
 'file_name': str,              # original path (may not exist)
 'cell_masks': np.ndarray,      # int32 instance mask (0=bg, 1..N=instances)
 # OR
 'nucleus_masks': np.ndarray,
 'parent_dataset': str, 'image_modality': str, 'pixel_size': float (optional)}
```

Loader behaviour:
- Extract images to `tempfile.mkdtemp(prefix='dv_pth_')` using PIL, normalized to uint8. Filename: `{counter:04d}_{original_stem}.png`. If `file_name` exists on disk, use it directly instead.
- Register cleanup with `atexit.register` and `__del__`.
- Lowercase split keys: `'Train'→'train'`, `'Validation'→'validation'`, etc.
- `get_splits()`: return keys only if more than one non-empty split.
- `get_annotations`: convert instance mask to bboxes — for each unique non-zero value in mask, `np.where(mask==iid)` → `cols.min/max, rows.min/max`. Label is `'cell'` or `'nucleus'` based on which mask key is present.
- Critical: use `'cell_masks' in item` (not `or`) to pick the mask — numpy arrays are not truthy.

---

## `yolo_exporter.py`
Output layout: `images/{split}/`, `labels/{split}/` (or flat `images/`, `labels/` if unsplit). Plus `data.yaml`.

- Copy images with `shutil.copy2` (skip if src == dst).
- Get image dimensions via `QImageReader`.
- Bbox → `class_id cx cy w h` (normalized).
- Polygon → `class_id x1/w y1/h x2/w y2/h ...`; if `polygons` key (COCO multi-part), take `polygons[0]`. Fall back to bbox if no points.
- `data.yaml`: write manually (no PyYAML dependency for writing):
  ```yaml
  nc: N
  names: ['class1', 'class2']

  train: images/train
  val: images/val
  ```

---

## `coco_exporter.py`
Output: `images/` (flat, all splits), `annotations/instances_{split}.json`.

**Critical**: use a **global** `img_id` counter across all splits (not reset per split). If each split resets to 1, `COCOLoader` merges all JSONs into one dict keyed by `image_id`, causing the first image of each later split to overwrite the first image of prior splits.

Similarly global `ann_id`. Increment before appending (`ann_id += 1` then use it).

COCO JSON structure per split:
```json
{"info": {"description": "Exported by Dataset Viewer"},
 "categories": [{"id": 1, "name": "...", "supercategory": "none"}],
 "images": [{"id": int, "file_name": str, "width": int, "height": int}],
 "annotations": [{"id": int, "image_id": int, "category_id": int,
                  "bbox": [x,y,w,h], "area": float,
                  "segmentation": [[x1,y1,...]] or [], "iscrowd": 0}]}
```
Map class_id (0-based) → category_id (1-based). Segmentation: flatten `points` or all `polygons` to `[[x1,y1,...]]`.

---

## `voc_exporter.py`
Output: `JPEGImages/`, `Annotations/{stem}.xml`, `ImageSets/Main/{split}.txt`.

XML built with `xml.etree.ElementTree` + `ET.indent(root, space='  ')` (Python 3.9+). Prepend `<?xml version="1.0" encoding="utf-8"?>`. Fields: `<filename>`, then per annotation `<object>` with `<name>`, `<difficult>0</difficult>`, `<bndbox>` (xmin/ymin/xmax/ymax as rounded ints).

---

## `pth_exporter.py`
Output: single `dataset.pth` in the chosen folder.

- Load each image as RGB numpy array via `PIL.Image.open().convert('RGB')`.
- Convert annotations to int32 instance mask (H×W, 0=bg):
  - Bbox: fill `mask[y:y2, x:x2] = instance_id`
  - Polygon: use `PIL.ImageDraw` with a temporary `'L'` image, draw polygon with fill=1, then `mask[tmp_array > 0] = instance_id` (avoids 32-bit ImageDraw issues).
- Split label mapping: `'train'→'Train'`, `'validation'→'Validation'`, `'test'→'Test'`, others → `.title()`.
- Mask key: `'cell_masks'` if `loader.class_names[0] == 'cell'`, else `'nucleus_masks'`.
- `torch.save(dataset, path)`.

---

## `save_as_dialog.py`
`QDialog` with `QFormLayout`:
- `QComboBox` with options: `['YOLO', 'COCO', 'Pascal VOC', 'InstanSeg PTH']`
- Folder row: `QLineEdit` + `QPushButton("Browse…")` → `QFileDialog.getExistingDirectory`
- `QDialogButtonBox(Ok | Cancel)`
- `accept()` validates folder is not empty before closing.

---

## `image_viewer.py`
`QGraphicsView` subclass:

- Dark background `QColor(45,45,45)`, no frame, antialiasing + smooth pixmap transform.
- Drag mode: `ScrollHandDrag`. Zoom anchor: `AnchorUnderMouse`.
- Zoom: ×1.25 per step, range 2%–3200%. Emit `zoom_changed(float)` signal.
- Mouse wheel → zoom in/out. Double-click → fit to window.
- **Annotation rendering** (called after loading image):
  - Colors: `matplotlib tab20` colormap cycled by annotation index (not class). Fallback: hardcoded 20 hex colors.
  - **Bbox**: `addRect` with cosmetic `QPen(color, 2)`.
  - **Polygon**: `addPolygon` with cosmetic pen + semi-transparent fill (alpha=45). Multi-part via `polygons` key, single via `points`.
  - **Mask (RLE)**: decode column-major RLE to `(H,W)` bool mask via numpy, build RGBA array (mask pixels get color at alpha=80), create `QImage(data, w, h, w*4, Format_RGBA8888).copy()` (`.copy()` needed to own the data buffer).
  - **Labels**: `addSimpleText` with `ItemIgnoresTransformations` flag (stays readable at all zoom levels), positioned at annotation top-left, z-value 2.
- Placeholder text shown when no image is loaded: `"Open a dataset folder to get started\nFile → Open Folder   or   Ctrl+O"` in grey.

---

## `file_browser.py`
`QWidget` (placed in a `QDockWidget`).

- `QTreeWidget` (header hidden, alternating row colors, single selection) + small `QLabel` info bar.
- `image_selected = pyqtSignal(str, list)` — emits `(image_path, annotations)`.
- `load_dataset(loader)`: if splits exist, create non-selectable bold header items + child image items; else flat list. Store `img_info` dict in `Qt.UserRole`.
- Selection via `itemClicked` and `currentItemChanged` (for keyboard nav).
- `select_offset(delta)`: move selection by delta among selectable items using `QTreeWidgetItemIterator`.

---

## `main_window.py`
`QMainWindow`, initial size `1440×900`, `self._loader = None`.

**Menu bar:**
- **File**: `Open Folder… (Ctrl+O)`, `Open File… (Ctrl+Shift+O)`, `Save As… (Ctrl+Shift+S)`, separator, `Exit (Ctrl+Q)`
- **View**: `File Browser (Ctrl+B)`, separator, `Zoom In/Out`, `Reset Zoom 1:1 (Ctrl+0)`, `Fit to Window (Ctrl+F)`, separator, `Show Annotations (A, checkable)`
- **Navigate**: `Previous Image (←)`, `Next Image (→)`

**Toolbar**: `Open… | Save As… | — | Zoom In | Zoom Out | 1:1 | Fit | — | Ann (checkable) | — | ◀ Prev | Next ▶`

**Statusbar**: left=dataset label, center=image name + annotation count, permanent right=zoom%.

**`open_folder()`**: `QFileDialog.getExistingDirectory` → `detect_format` → `_load_dataset`. On failure warn with supported formats and hint about `Open File…` for PTH.

**`open_file()`**: `QFileDialog.getOpenFileName` filtered to `"InstanSeg PTH (*.pth)"` → `_load_dataset`.

**`_load_dataset(path)`**: shared helper — calls `detect_format`, stores `self._loader`, calls `browser.load_dataset`, updates window title and status bar.

**`save_as()`**: guard `self._loader`, show `SaveAsDialog`, instantiate the matching exporter, run with `QProgressDialog` (window-modal, min duration 0). Progress callback: `setValue(done*100//total)`, `processEvents()`, return `not wasCanceled()`. Show success message on completion.

---

## Key design invariants
- All annotation coordinates are **pixel-absolute** inside the app; only exporters/loaders do normalization.
- Bounding box pen is **cosmetic** (`pen.setCosmetic(True)`) → constant 2px screen width at all zoom levels.
- Label text uses `ItemIgnoresTransformations` → constant screen size at all zoom levels.
- COCO exporter uses a **global image ID counter across splits** to prevent ID collisions when `COCOLoader` merges multiple JSON files.
- PTH mask key selection uses `'key' in item` dict check, never `item.get(key) or ...` — numpy arrays raise `ValueError` on boolean evaluation.
- PTH loader extracts images to a temp dir on construction; cleanup via both `atexit` and `__del__`.
