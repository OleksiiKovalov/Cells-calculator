# Training Studio

A desktop application for **training InstanSeg models**. Point it at a prepared
`.pth` dataset and **Train**, **Evaluate**, and **Export** to TorchScript — from a
polished PySide6 GUI, no notebooks.

> **Dataset preparation & conversion live in the sibling [Dataset Viewer](../dataset-viewer)
> app.** Open/convert your COCO / YOLO / VOC data there, use *Prepare & Export →
> InstanSeg PTH* (preprocess, split, resize/contrast) to produce a `.pth`, then
> train it here. This app does not build datasets.

The GUI is launched with `python src/main.py` (or `run.bat`). The CLI pipeline
lives in `src/runner.py`; each step is also a standalone script in `src/`.

---

## How it works

1. **Open a dataset** — *File ▸ Open Dataset (.pth)* loads a prepared dataset and
   shows its Train / Validation / Test counts.
2. **Train** — `instanseg_training(...)` fine-tunes a model on the `.pth`
   (target Cells/Nuclei, epochs, early-stopping, pixel size, optional resume).
3. **Evaluate** — InstanSeg's `test.py` scores a trained model on a chosen split
   and can save prediction overlays.
4. **Export** — `export_to_torchscript` writes a deployable `.pt` for the sibling
   analysis apps (spheroid / morphology / cells-calculator).

---

## Requirements

```bash
pip install -r requirements.txt
```

Installs the always-on stack (numpy, OpenCV, scikit-image, Pillow for the image
preview; **torch** for reading the `.pth`; **PySide6**) **and** the mainline
InstanSeg backend (`instanseg-torch` + `monai`). A CUDA GPU is strongly
recommended for real training.

### Training backends: mainline vs. fork

Train / Evaluate / Export run on **either** InstanSeg backend, auto-detected:

| | Mainline `instanseg-torch` | Cryobiology fork |
|---|---|---|
| Install | `pip install instanseg-torch monai` (default) | `pip install git+https://github.com/sonyalytv/instanseg_cryobiology.git` + `monai stardist edt` |
| Early stopping (`max_no_improvement`) | not available — trains full `num_epochs`, saving the best checkpoint | supported |

`train.py` introspects the installed `instanseg_training` and passes fork-only
arguments only when supported. If **no** backend is importable, the step stops
with a clear install message. On Windows you can instead run `install.bat`
(creates the `cells-calculator-training` conda env) and then `run.bat`.

---

## Directory layout

```
data/
├── datasets/       prepared <name>.pth (made in Dataset Viewer)   <- input
├── models/         <model_folder>/...   trained weights           <- Train output
├── test_results/   metrics + prediction overlays                  <- Evaluate output
└── exported/       <...>.pt   TorchScript model                   <- Export output
```

---

## Quick start

```bash
# 1. Prepare a .pth in Dataset Viewer, e.g.:
#    (in apps/dataset-viewer)  python src/convert.py --src path/to/coco --out out --to pth
# 2. Train / evaluate / export it here:
python src/runner.py --dataset data/datasets/my_dataset.pth --steps train
python src/runner.py --dataset data/datasets/my_dataset.pth --steps train,evaluate,export
```

GUI: `python src/main.py` → *File ▸ Open Dataset (.pth)* → set options in the
Train card → **Train**.

---

## Pipeline steps

`--steps` selects a subset (comma-separated); each is also runnable standalone.

| Step | Script | Purpose |
|---|---|---|
| `train` | `src/train.py` | `instanseg_training` fine-tuning |
| `evaluate` | `src/evaluate.py` | Score a model on a split |
| `export` | `src/export_model.py` | TorchScript export |

```
train ──> evaluate
   └────> export
```

---

## Desktop GUI

An MDI workspace with three sub-windows (`View ▸ Tile/Cascade`):

| Sub-window | Contents |
|---|---|
| **Image Viewer** | Interactive 2D canvas — wheel zoom, drag pan. *File ▸ Open Image / Open Mask* preview any file (masks colourised per instance). |
| **Results** | The opened dataset's split counts; run summaries. |
| **Log** | Live streaming output of the running step (child process). |

A docked **Control Panel** holds the Train / Evaluate / Export cards. Each runs in
a child process; progress streams to the Log and the UI stays responsive. Outputs
go under the working directory (*File ▸ Set Working Dir…*).

---

## Configuration — `config.json`

Training / evaluation / export defaults are configurable without editing code:
precedence built-in defaults **<** `config.json` **<** CLI flags / GUI fields.
Pass `--config other.json` to any script.

---

## Provenance

Ported from the InstanSeg training notebook (kept under `docs/notebooks/`); each
wrapper carries `Source:` comments back to the notebook cell it came from. The
dataset-creation cells now live in the Dataset Viewer app.

---

## Tests

```bash
python -m pytest
```

Verifies each step's `--help`, config loading, the image-preview helpers, and the
backend-kwarg gating / no-backend message — no InstanSeg backend or GPU required.
