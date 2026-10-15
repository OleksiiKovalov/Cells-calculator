"""
Training Studio — PySide6 desktop front-end for InstanSeg model training.

Trains, evaluates and exports InstanSeg models from a prepared ``.pth`` dataset.
Dataset preparation and format conversion live in the sibling **Dataset Viewer**
app, which exports the ``.pth`` this app consumes — nothing here builds datasets.

Features:
  * File > Open Dataset (.pth)     -> load a prepared dataset, show its splits
  * File > Open Image / Open Mask  -> preview any image or (colourised) mask
  * Train    -> instanseg_training on the .pth (needs an InstanSeg backend)
  * Evaluate -> score a trained model on a split
  * Export   -> TorchScript .pt for the sibling analysis apps
  * Interactive 2D viewer (wheel zoom, drag pan) + MDI workspace + streaming Log

Each heavy step runs in a CHILD PROCESS via QProcess, streaming progress to the
Log so the UI stays responsive — the pattern is lifted from the sibling
morphology app's Segment action.

Run:  python src/main.py
The design language (Fusion base, group cards, docked control panel, MDI
sub-windows) mirrors the sibling morphology/spheroid apps; the accent is amber
to give this app its own identity.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

try:
    from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QRectF
    from PySide6.QtGui import (
        QPixmap, QImage, QFont, QAction, QPalette, QColor, QPainter,
    )
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QMdiArea, QMdiSubWindow, QDockWidget, QWidget,
        QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, QComboBox,
        QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QPlainTextEdit,
        QTabWidget, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
        QScrollArea, QGroupBox, QStyle, QGraphicsView, QGraphicsScene,
        QGraphicsPixmapItem,
    )
except ImportError:
    sys.exit("PySide6 is required: pip install PySide6")

import numpy as np

import datalib
from config import load_config

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Light theme — clean neutral surfaces with an amber accent (this app's colour).
# ---------------------------------------------------------------------------
ACCENT = "#d97706"
ACCENT_HOVER = "#b45309"
ACCENT_PRESSED = "#92400e"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
    color: #1f2933;
}}
QMainWindow, QDialog {{ background: #f4f6f8; }}

QMdiArea {{ background: #e9edf1; }}

/* --- Group cards --------------------------------------------------------- */
QGroupBox {{
    background: #ffffff;
    border: 1px solid #dfe3e8;
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 6px;
    color: {ACCENT};
    background: transparent;
    text-transform: uppercase;
    font-size: 8.5pt;
    letter-spacing: 1px;
}}

/* --- Inputs -------------------------------------------------------------- */
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QAbstractSpinBox {{
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    min-height: 18px;
}}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QAbstractSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{
    background: #f2f4f6;
    color: #9aa5b1;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    outline: none;
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid #cbd2d9; border-radius: 4px; background: #ffffff;
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

/* --- Buttons ------------------------------------------------------------- */
QPushButton {{
    background: #ffffff;
    border: 1px solid #cbd2d9;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}}
QPushButton:hover {{ background: #f0f2f4; border-color: #b6bec7; }}
QPushButton:pressed {{ background: #e4e7eb; }}
QPushButton:disabled {{ background: #f2f4f6; color: #9aa5b1; border-color: #e2e6ea; }}

QPushButton[class="primary"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #ffffff;
    padding: 7px 16px;
}}
QPushButton[class="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[class="primary"]:pressed {{ background: {ACCENT_PRESSED}; border-color: {ACCENT_PRESSED}; }}
QPushButton[class="primary"]:disabled {{ background: #f0cfa0; border-color: #f0cfa0; color: #fffaf3; }}

/* --- Tabs ---------------------------------------------------------------- */
QTabWidget::pane {{
    border: 1px solid #dfe3e8;
    border-radius: 8px;
    background: #ffffff;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: #52606d;
    padding: 7px 16px;
    margin-right: 2px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover:!selected {{ color: #1f2933; }}

/* --- Tables & text ------------------------------------------------------- */
QTableWidget {{
    background: #ffffff;
    border: 1px solid #dfe3e8;
    border-radius: 8px;
    gridline-color: #eceff1;
    selection-background-color: #fbe6c8;
    selection-color: #7c2d12;
}}
QHeaderView::section {{
    background: #f4f6f8;
    color: #52606d;
    padding: 6px 8px;
    border: none;
    border-right: 1px solid #eceff1;
    border-bottom: 1px solid #dfe3e8;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: #f4f6f8; border: none; }}
QPlainTextEdit {{
    background: #ffffff;
    border: 1px solid #dfe3e8;
    border-radius: 8px;
    padding: 6px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 9pt;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}

/* --- Dock / menu / toolbar / status ------------------------------------- */
QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; font-weight: 600; }}
QDockWidget::title {{ background: #eef1f4; padding: 7px 10px; border-bottom: 1px solid #dfe3e8; }}
QMenuBar {{ background: #ffffff; border-bottom: 1px solid #dfe3e8; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
QMenuBar::item:selected {{ background: #eef1f4; border-radius: 4px; }}
QMenu {{ background: #ffffff; border: 1px solid #dfe3e8; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT}; color: #ffffff; }}
QToolBar {{ background: #ffffff; border-bottom: 1px solid #dfe3e8; padding: 4px 6px; spacing: 4px; }}
QToolBar QToolButton {{ padding: 5px 10px; border-radius: 6px; font-weight: 600; }}
QToolBar QToolButton:hover {{ background: #eef1f4; }}
QToolBar QToolButton:pressed {{ background: #e4e7eb; }}
QStatusBar {{ background: #ffffff; border-top: 1px solid #dfe3e8; color: #52606d; }}
QStatusBar::item {{ border: none; }}

/* --- Scrollbars ---------------------------------------------------------- */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c3ccd4; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #a7b1bb; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #c3ccd4; border-radius: 5px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: #a7b1bb; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QScrollArea {{ border: none; background: transparent; }}
"""


# ---------------------------------------------------------------------------
# Interactive 2D image viewer (zoom with the wheel, pan by dragging).
# Source: morphology/src/gui.py -> ImageViewer.
# ---------------------------------------------------------------------------
class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#e9edf1"))
        self._img_ref = None       # keep the ndarray buffer alive for QImage
        self._has_image = False

    def set_image(self, rgb):
        img = np.ascontiguousarray(rgb)
        self._img_ref = img
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._item.setPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        if not self._has_image:
            self.fitInView(self._item, Qt.KeepAspectRatio)
        self._has_image = True

    def wheelEvent(self, event):
        if not self._has_image:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def fit(self):
        if self._has_image:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def reset_zoom(self):
        if self._has_image:
            self.resetTransform()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Studio")
        self.resize(1320, 860)

        self.cfg = load_config()
        self.work_dir = str(SCRIPT_DIR.parent)   # app root, regardless of launch CWD

        # ---- state ----
        self.proc = None                 # single QProcess for the running step
        self._active_button = None       # button to re-enable when proc finishes
        self._on_success = None          # callback(code) after a successful step

        # ---- MDI workspace ----
        self.mdi = QMdiArea()
        self.mdi.setViewMode(QMdiArea.SubWindowView)
        self.setCentralWidget(self.mdi)

        self._build_viewer_subwindow()
        self._build_results_subwindow()
        self._build_log_subwindow()
        self.mdi.tileSubWindows()

        self._build_control_dock()
        self._build_menus_and_toolbar()

        self.statusBar().showMessage(
            "Open a prepared dataset (.pth) — File ▸ Open Dataset — then Train / Evaluate / Export.")

    # ----- sub-windows -----
    def _build_viewer_subwindow(self):
        self.viewer = ImageViewer()
        sub = QMdiSubWindow()
        sub.setWidget(self.viewer)
        sub.setWindowTitle("Image Viewer")
        self.mdi.addSubWindow(sub)
        self.viewer_sub = sub

    def _build_results_subwindow(self):
        tabs = QTabWidget()

        self.summary_view = QPlainTextEdit()
        self.summary_view.setReadOnly(True)
        tabs.addTab(self.summary_view, "Summary")

        self.table = QTableWidget()
        tabs.addTab(self.table, "Dataset")

        sub = QMdiSubWindow()
        sub.setWidget(tabs)
        sub.setWindowTitle("Results")
        self.mdi.addSubWindow(sub)
        self.results_sub = sub

    def _build_log_subwindow(self):
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        sub = QMdiSubWindow()
        sub.setWidget(self.log_view)
        sub.setWindowTitle("Log")
        self.mdi.addSubWindow(sub)
        self.log_sub = sub

    # ----- control panel dock -----
    @staticmethod
    def _card(title):
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)
        form.setContentsMargins(4, 6, 4, 4)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        return box, form

    @staticmethod
    def _dspin(lo, hi, step, value, decimals=2):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        sp.setValue(float(value))
        return sp

    def _dir_row(self, initial, placeholder):
        """A read/write path field + '...' browse button (for directories)."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(initial)
        edit.setPlaceholderText(placeholder)
        btn = QPushButton("...")
        btn.setFixedWidth(34)
        btn.clicked.connect(lambda: self._browse_into(edit, directory=True))
        lay.addWidget(edit, stretch=1)
        lay.addWidget(btn)
        return row, edit

    def _file_row(self, initial, placeholder, filt):
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(initial)
        edit.setPlaceholderText(placeholder)
        btn = QPushButton("...")
        btn.setFixedWidth(34)
        btn.clicked.connect(lambda: self._browse_into(edit, directory=False, filt=filt))
        lay.addWidget(edit, stretch=1)
        lay.addWidget(btn)
        return row, edit

    def _browse_into(self, edit, directory, filt="All files (*)"):
        start = edit.text().strip() or self.work_dir
        if directory:
            d = QFileDialog.getExistingDirectory(self, "Select directory", start)
            if d:
                edit.setText(d)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", start, filt)
            if path:
                edit.setText(path)

    def _build_control_dock(self):
        dock = QDockWidget("Control Panel", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        d = str(Path(self.work_dir) / "data")
        self._build_train_card(outer, d)
        self._build_evaluate_card(outer, d)
        self._build_export_card(outer, d)
        outer.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(panel)

        dock.setWidget(scroll)
        dock.setMinimumWidth(330)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _primary_button(self, text, slot):
        btn = QPushButton(text)
        btn.setProperty("class", "primary")
        btn.setMinimumHeight(32)
        btn.clicked.connect(slot)
        return btn

    # --- Train card ----------------------------------------------------------
    def _build_train_card(self, outer, data_dir):
        card, form = self._card("Train")
        t = self.cfg["train"]

        row, self.tr_dataset = self._file_row(
            str(Path(data_dir) / "datasets"), "dataset .pth", "Dataset (*.pth);;All files (*)")
        form.addRow("Dataset .pth", row)
        row, self.tr_models = self._dir_row(str(Path(data_dir) / "models"), "models output dir")
        form.addRow("Models dir", row)

        self.tr_folder = QLineEdit(t["model_folder"])
        form.addRow("Model folder", self.tr_folder)
        self.tr_source = QLineEdit(t["source_dataset"])
        form.addRow("Source dataset", self.tr_source)

        self.tr_target = QComboBox()
        self.tr_target.addItems(["C", "N"])
        self.tr_target.setCurrentText(t["target_segmentation"])
        form.addRow("Target (C/N)", self.tr_target)

        self.tr_pixel = self._dspin(0.05, 10.0, 0.05, t["requested_pixel_size"])
        form.addRow("Pixel size", self.tr_pixel)
        self.tr_epochs = QSpinBox()
        self.tr_epochs.setRange(1, 100000)
        self.tr_epochs.setValue(int(t["num_epochs"]))
        form.addRow("Epochs", self.tr_epochs)
        self.tr_noimp = QSpinBox()
        self.tr_noimp.setRange(1, 1000)
        self.tr_noimp.setValue(int(t["max_no_improvement"]))
        form.addRow("Max no-improve", self.tr_noimp)
        self.tr_hot = QSpinBox()
        self.tr_hot.setRange(0, 1000)
        self.tr_hot.setValue(int(t["hotstart_training"]))
        form.addRow("Hotstart", self.tr_hot)

        row, self.tr_resume = self._file_row(
            t["resume_weights"], "(optional) model_weights_best.pth", "Weights (*.pth);;All files (*)")
        form.addRow("Resume from", row)

        self.train_btn = self._primary_button("Train", self.run_train)
        form.addRow(self.train_btn)
        outer.addWidget(card)

    # --- Evaluate card -------------------------------------------------------
    def _build_evaluate_card(self, outer, data_dir):
        card, form = self._card("Evaluate")
        e = self.cfg["evaluate"]

        row, self.ev_dsdir = self._dir_row(str(Path(data_dir) / "datasets"), "dir containing the .pth")
        form.addRow("Dataset dir", row)
        self.ev_data = QLineEdit("dataset.pth")
        form.addRow("Dataset file", self.ev_data)
        row, self.ev_model = self._dir_row(str(Path(data_dir) / "models"), "trained model dir (with version)")
        form.addRow("Model path", row)
        row, self.ev_out = self._dir_row(str(Path(data_dir) / "test_results"), "results output dir")
        form.addRow("Output dir", row)

        self.ev_set = QComboBox()
        self.ev_set.addItems(["Test", "Validation", "Train"])
        self.ev_set.setCurrentText(e["set"])
        form.addRow("Split", self.ev_set)
        self.ev_target = QComboBox()
        self.ev_target.addItems(["C", "N"])
        self.ev_target.setCurrentText(e["target_segmentation"])
        form.addRow("Target (C/N)", self.ev_target)
        self.ev_saveims = QCheckBox("Save prediction overlays")
        self.ev_saveims.setChecked(bool(e["save_images"]))
        form.addRow(self.ev_saveims)

        self.evaluate_btn = self._primary_button("Evaluate", self.run_evaluate)
        form.addRow(self.evaluate_btn)
        outer.addWidget(card)

    # --- Export card ---------------------------------------------------------
    def _build_export_card(self, outer, data_dir):
        card, form = self._card("Export TorchScript")
        x = self.cfg["export"]

        row, self.ex_model = self._dir_row(
            str(Path(data_dir) / "models"), "model dir EXCLUDING version subfolder")
        form.addRow("Model path", row)
        self.ex_version = QLineEdit(str(x["version"]))
        form.addRow("Version", self.ex_version)
        row, self.ex_out = self._dir_row(str(Path(data_dir) / "exported"), "output dir for .pt")
        form.addRow("Output dir", row)

        self.export_btn = self._primary_button("Export TorchScript", self.run_export)
        form.addRow(self.export_btn)
        outer.addWidget(card)

    # ----- menus / toolbar -----
    def _build_menus_and_toolbar(self):
        menubar = self.menuBar()
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        style = self.style()

        file_menu = menubar.addMenu("&File")
        act_open_img = QAction(style.standardIcon(QStyle.SP_DirOpenIcon), "&Open Image...", self)
        act_open_img.triggered.connect(self.open_image)
        file_menu.addAction(act_open_img)
        toolbar.addAction(act_open_img)
        act_open_mask = QAction(style.standardIcon(QStyle.SP_FileDialogContentsView), "Open &Mask...", self)
        act_open_mask.triggered.connect(self.open_mask)
        file_menu.addAction(act_open_mask)
        toolbar.addAction(act_open_mask)
        act_open_pth = QAction(style.standardIcon(QStyle.SP_FileDialogDetailedView), "Open &Dataset (.pth)...", self)
        act_open_pth.triggered.connect(self.open_pth)
        file_menu.addAction(act_open_pth)
        toolbar.addAction(act_open_pth)
        act_workdir = QAction("Set &Working Dir...", self)
        act_workdir.triggered.connect(self.set_work_dir)
        file_menu.addAction(act_workdir)
        file_menu.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        toolbar.addSeparator()

        pipe_menu = menubar.addMenu("&Pipeline")
        for label, slot, icon in (
            ("&Train", self.run_train, QStyle.SP_MediaPlay),
            ("&Evaluate", self.run_evaluate, QStyle.SP_DialogApplyButton),
            ("E&xport TorchScript", self.run_export, QStyle.SP_DialogYesButton),
        ):
            act = QAction(style.standardIcon(icon), label, self)
            act.triggered.connect(slot)
            pipe_menu.addAction(act)
            toolbar.addAction(act)

        view_menu = menubar.addMenu("&View")
        act_fit = QAction("&Fit", self)
        act_fit.triggered.connect(self.viewer.fit)
        view_menu.addAction(act_fit)
        act_reset = QAction("&1:1", self)
        act_reset.triggered.connect(self.viewer.reset_zoom)
        view_menu.addAction(act_reset)
        view_menu.addSeparator()
        act_tile = QAction("&Tile Windows", self)
        act_tile.triggered.connect(self.mdi.tileSubWindows)
        view_menu.addAction(act_tile)
        act_cascade = QAction("&Cascade Windows", self)
        act_cascade.triggered.connect(self.mdi.cascadeSubWindows)
        view_menu.addAction(act_cascade)

        help_menu = menubar.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    # ----- helpers -----
    def log(self, text):
        self.log_view.appendPlainText(str(text).rstrip())

    def _about(self):
        QMessageBox.about(
            self, "About Training Studio",
            "Training Studio\n\nPySide6 front-end for InstanSeg model training: "
            "Train / Evaluate / Export on a prepared .pth dataset.\n\n"
            "Prepare and convert datasets in the Dataset Viewer app (it exports "
            "the .pth this app consumes). Training requires an InstanSeg backend "
            "(mainline instanseg-torch or the cryobiology fork) and a PyTorch "
            "install.")

    def set_work_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Set working (output) directory", self.work_dir)
        if d:
            self.work_dir = d
            self.log(f"Working dir set to {d}")

    def _all_run_buttons(self):
        return [self.train_btn, self.evaluate_btn, self.export_btn]

    # ----- Open a prepared dataset (.pth) -----
    def open_pth(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open dataset (.pth)", self.work_dir, "Dataset (*.pth);;All files (*)")
        if not path:
            return
        try:
            import torch
            data = torch.load(path, weights_only=False)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.tr_dataset.setText(path)          # prefill the Train card
        self._show_pth_summary(path, data)
        self.log(f"Opened dataset {Path(path).name}.")
        self.statusBar().showMessage("Dataset loaded. Set model options and Train.")

    def _show_pth_summary(self, path, data):
        counts = {k: len(v) for k, v in data.items()} if isinstance(data, dict) else {}
        total = sum(counts.values())
        ordered = [k for k in ("Train", "Validation", "Test") if k in counts]
        ordered += [k for k in counts if k not in ("Train", "Validation", "Test")]

        lines = ["========== DATASET ==========", f"  File : {path}"]
        lines += [f"    {k:<11}: {counts[k]}" for k in ordered]
        lines.append(f"  Total: {total}")
        self.summary_view.setPlainText("\n".join(lines))

        rows = [(k, counts[k]) for k in ordered] + [("Total", total)]
        self.table.clear()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["split", "items"])
        self.table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.table.setItem(r, 1, QTableWidgetItem(str(v)))
        self.table.resizeColumnsToContents()
        self._raise(self.results_sub)

    # ----- File actions: previews -----
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", self.work_dir,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)")
        if not path:
            return
        try:
            rgb = datalib.load_image_rgb(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.viewer.set_image(rgb)
        self._raise(self.viewer_sub)
        self.log(f"Opened image {Path(path).name} ({rgb.shape[1]}x{rgb.shape[0]}).")

    def open_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open mask", self.work_dir,
            "Masks (*.png *.tif *.tiff *.bmp);;All files (*)")
        if not path:
            return
        try:
            rgb = datalib.load_mask_preview(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.viewer.set_image(rgb)
        self._raise(self.viewer_sub)
        n = int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]) - 1
        self.log(f"Opened mask {Path(path).name} (~{max(n, 0)} distinct instances).")

    def _raise(self, sub):
        sub.showNormal()
        sub.raise_()

    # ----- generic child-process runner (streams to Log) -----
    def _start(self, script, args, button, busy_msg, on_success=None):
        """Launch ``script`` with ``args`` as a child process; stream to Log.

        Source: morphology/src/gui.py -> run_segment()/_on_proc_output().
        Only one step runs at a time; all run buttons are disabled meanwhile.
        """
        if self.proc is not None:
            QMessageBox.information(self, "Busy", "A pipeline step is already running.")
            return

        full = [str(SCRIPT_DIR / script)] + [str(a) for a in args]
        self.log("=" * 60)
        self.log(f"Running: {Path(sys.executable).name} -u {' '.join(full)}")
        self._active_button = button
        self._on_success = on_success
        for b in self._all_run_buttons():
            b.setEnabled(False)
        self.statusBar().showMessage(busy_msg)
        self._raise(self.log_sub)

        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(self.work_dir)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.start(sys.executable, ["-u"] + full)

    def _on_proc_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        # tqdm progress bars use carriage returns; treat \r as a line break too.
        for line in data.replace("\r", "\n").splitlines():
            if line.strip():
                self.log(line)

    def _on_proc_finished(self, code, _status):
        on_success = self._on_success
        self.proc = None
        self._on_success = None
        for b in self._all_run_buttons():
            b.setEnabled(True)
        if code != 0:
            self.log(f"Step FAILED (exit {code}).")
            self.statusBar().showMessage(f"Step failed (exit {code}). See the Log.")
            return
        self.log("Step complete.")
        self.statusBar().showMessage("Step complete.")
        if on_success:
            try:
                on_success()
            except Exception as e:  # noqa: BLE001
                self.log(f"(post-step display error: {e})")

    # ----- Pipeline actions -----
    def run_train(self):
        dataset = self.tr_dataset.text().strip()
        if not dataset or not Path(dataset).is_file():
            QMessageBox.warning(self, "No dataset", "Choose a dataset .pth file to train on.")
            return
        args = ["--dataset", dataset,
                "--models-dir", self.tr_models.text().strip(),
                "--output-dir", self.tr_models.text().strip(),
                "--model-folder", self.tr_folder.text().strip(),
                "--source-dataset", self.tr_source.text().strip(),
                "--target", self.tr_target.currentText(),
                "--pixel-size", self.tr_pixel.value(),
                "--epochs", self.tr_epochs.value(),
                "--max-no-improvement", self.tr_noimp.value(),
                "--hotstart", self.tr_hot.value()]
        resume = self.tr_resume.text().strip()
        if resume:
            args += ["--resume-weights", resume]
        self._start("train.py", args, self.train_btn,
                    "Training (needs the InstanSeg fork + GPU)...")

    def run_evaluate(self):
        ds_dir = self.ev_dsdir.text().strip()
        data = self.ev_data.text().strip()
        model = self.ev_model.text().strip()
        if not ds_dir or not data or not model:
            QMessageBox.warning(self, "Missing input",
                                "Set the dataset dir, dataset file and model path.")
            return
        args = ["--dataset-dir", ds_dir, "--data", data, "--model-path", model,
                "--out", self.ev_out.text().strip(),
                "--set", self.ev_set.currentText(),
                "--target", self.ev_target.currentText()]
        args += ["--save-images"] if self.ev_saveims.isChecked() else ["--no-save-images"]
        self._start("evaluate.py", args, self.evaluate_btn,
                    "Evaluating (needs the InstanSeg fork)...")

    def run_export(self):
        model = self.ex_model.text().strip()
        if not model:
            QMessageBox.warning(self, "No model", "Set the model path (excluding the version subfolder).")
            return
        args = ["--model-path", model, "--version", self.ex_version.text().strip() or "1",
                "--out", self.ex_out.text().strip()]
        self._start("export_model.py", args, self.export_btn,
                    "Exporting to TorchScript (needs the InstanSeg fork)...")



def apply_theme(app):
    """Apply the light amber theme: Fusion base, font, palette, stylesheet."""
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    pal = app.palette()
    pal.setColor(QPalette.Window, QColor("#f4f6f8"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f4f6f8"))
    pal.setColor(QPalette.Text, QColor("#1f2933"))
    pal.setColor(QPalette.WindowText, QColor("#1f2933"))
    pal.setColor(QPalette.Button, QColor("#ffffff"))
    pal.setColor(QPalette.ButtonText, QColor("#1f2933"))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Active, QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.Inactive, QPalette.Highlight, QColor("#c3ccd4"))
    pal.setColor(QPalette.Inactive, QPalette.HighlightedText, QColor("#3e4c59"))
    app.setPalette(pal)

    app.setStyleSheet(STYLESHEET)


def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
