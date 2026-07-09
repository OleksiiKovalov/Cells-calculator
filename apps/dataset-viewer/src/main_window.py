import os
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox,
    QToolBar, QDockWidget, QLabel, QSizePolicy,
    QProgressDialog, QDialog, QApplication,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeySequence, QAction

from widgets.image_viewer import ImageViewer
from widgets.file_browser import FileBrowser
from datasets.format_detector import detect_format


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Viewer")
        self.resize(1440, 900)
        self._loader = None

        self._init_viewer()
        self._init_browser_dock()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()

    # ------------------------------------------------------------------
    # Widget setup
    # ------------------------------------------------------------------
    def _init_viewer(self):
        self.viewer = ImageViewer()
        self.setCentralWidget(self.viewer)

    def _init_browser_dock(self):
        self.dock = QDockWidget("Dataset Browser", self)
        self.dock.setObjectName("dataset_browser")
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.dock.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
        self.browser = FileBrowser()
        self.dock.setWidget(self.browser)
        self.dock.setMinimumWidth(240)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

        self.browser.image_selected.connect(self._on_image_selected)

    def _init_menu(self):
        mb = self.menuBar()

        # File
        file_m = mb.addMenu("&File")

        open_act = QAction("&Open Folder…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.setStatusTip("Open a dataset folder (auto-detects YOLO / COCO / VOC / InstanSeg PTH)")
        open_act.triggered.connect(self.open_folder)
        file_m.addAction(open_act)

        open_file_act = QAction("Open &File…", self)
        open_file_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_file_act.setStatusTip("Open a dataset file (e.g. InstanSeg .pth)")
        open_file_act.triggered.connect(self.open_file)
        file_m.addAction(open_file_act)

        save_as_act = QAction("Save &As…", self)
        save_as_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_act.setStatusTip("Export the loaded dataset to a different format")
        save_as_act.triggered.connect(self.save_as)
        file_m.addAction(save_as_act)

        file_m.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut(QKeySequence("Ctrl+Q"))
        exit_act.triggered.connect(self.close)
        file_m.addAction(exit_act)

        # View
        view_m = mb.addMenu("&View")

        toggle_browser = self.dock.toggleViewAction()
        toggle_browser.setText("File &Browser")
        toggle_browser.setShortcut(QKeySequence("Ctrl+B"))
        view_m.addAction(toggle_browser)

        view_m.addSeparator()

        zoom_in = QAction("Zoom &In", self)
        zoom_in.setShortcut(QKeySequence.ZoomIn)
        zoom_in.triggered.connect(self.viewer.zoom_in)
        view_m.addAction(zoom_in)

        zoom_out = QAction("Zoom &Out", self)
        zoom_out.setShortcut(QKeySequence.ZoomOut)
        zoom_out.triggered.connect(self.viewer.zoom_out)
        view_m.addAction(zoom_out)

        reset = QAction("Reset Zoom  (1:1)", self)
        reset.setShortcut(QKeySequence("Ctrl+0"))
        reset.triggered.connect(self.viewer.reset_zoom)
        view_m.addAction(reset)

        fit = QAction("&Fit to Window", self)
        fit.setShortcut(QKeySequence("Ctrl+F"))
        fit.triggered.connect(self.viewer.fit_to_window)
        view_m.addAction(fit)

        view_m.addSeparator()

        self._ann_action = QAction("Show &Annotations", self)
        self._ann_action.setShortcut(QKeySequence("A"))
        self._ann_action.setCheckable(True)
        self._ann_action.setChecked(True)
        self._ann_action.toggled.connect(self._set_annotations)
        view_m.addAction(self._ann_action)

        # Navigate
        nav_m = mb.addMenu("&Navigate")

        prev_act = QAction("&Previous Image", self)
        prev_act.setShortcut(QKeySequence("Left"))
        prev_act.triggered.connect(lambda: self.browser.select_offset(-1))
        nav_m.addAction(prev_act)

        next_act = QAction("&Next Image", self)
        next_act.setShortcut(QKeySequence("Right"))
        next_act.triggered.connect(lambda: self.browser.select_offset(+1))
        nav_m.addAction(next_act)

    def _init_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setObjectName("main_toolbar")
        tb.setIconSize(QSize(16, 16))
        tb.setMovable(False)
        self.addToolBar(tb)

        open_act = QAction("Open…", self)
        open_act.setToolTip("Open dataset folder  (Ctrl+O)")
        open_act.triggered.connect(self.open_folder)
        tb.addAction(open_act)

        save_as_act = QAction("Save As…", self)
        save_as_act.setToolTip("Export dataset to another format  (Ctrl+Shift+S)")
        save_as_act.triggered.connect(self.save_as)
        tb.addAction(save_as_act)

        tb.addSeparator()

        zi = QAction("Zoom In", self)
        zi.setToolTip("Zoom in  (+)")
        zi.triggered.connect(self.viewer.zoom_in)
        tb.addAction(zi)

        zo = QAction("Zoom Out", self)
        zo.setToolTip("Zoom out  (-)")
        zo.triggered.connect(self.viewer.zoom_out)
        tb.addAction(zo)

        r = QAction("1:1", self)
        r.setToolTip("Reset to 100%  (Ctrl+0)")
        r.triggered.connect(self.viewer.reset_zoom)
        tb.addAction(r)

        fit = QAction("Fit", self)
        fit.setToolTip("Fit image to window  (Ctrl+F)")
        fit.triggered.connect(self.viewer.fit_to_window)
        tb.addAction(fit)

        tb.addSeparator()

        ann = QAction("Ann", self)
        ann.setToolTip("Toggle annotation overlay  (A)")
        ann.setCheckable(True)
        ann.setChecked(True)
        ann.toggled.connect(self._set_annotations)
        tb.addAction(ann)
        self._tb_ann_action = ann

        tb.addSeparator()

        prev_a = QAction("◀ Prev", self)
        prev_a.setToolTip("Previous image  (←)")
        prev_a.triggered.connect(lambda: self.browser.select_offset(-1))
        tb.addAction(prev_a)

        next_a = QAction("Next ▶", self)
        next_a.setToolTip("Next image  (→)")
        next_a.triggered.connect(lambda: self.browser.select_offset(+1))
        tb.addAction(next_a)

    def _init_statusbar(self):
        sb = self.statusBar()

        self._lbl_dataset = QLabel("No dataset loaded")
        sb.addWidget(self._lbl_dataset, 1)

        self._lbl_image = QLabel()
        sb.addWidget(self._lbl_image)

        self._lbl_zoom = QLabel("100%")
        sb.addPermanentWidget(self._lbl_zoom)

        self.viewer.zoom_changed.connect(
            lambda z: self._lbl_zoom.setText(f"{z:.0f}%")
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Open Dataset Folder", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self._open_path(folder)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset File", "",
            "InstanSeg PTH (*.pth);;All Files (*)"
        )
        if path:
            self._open_path(path)

    def _open_path(self, path: str):
        """Detect, load and display a dataset from a folder or file path."""
        loader, fmt = detect_format(path)
        if loader is None:
            QMessageBox.warning(
                self, "Unknown Dataset Format",
                f"Could not detect a supported dataset at:\n{path}\n\n"
                "Supported formats: YOLO, COCO JSON, Pascal VOC, InstanSeg PTH\n"
                "For a .pth file use File → Open File…"
            )
            return
        self._loader = loader
        self.browser.load_dataset(loader)
        self.dock.show()
        name = os.path.basename(os.path.normpath(path))
        self.setWindowTitle(f"Dataset Viewer — {name}  [{fmt}]")
        self._lbl_dataset.setText(f"{fmt}  ·  {path}")
        self._lbl_image.clear()

    def save_as(self):
        if not self._loader:
            QMessageBox.information(self, "No Dataset", "Open a dataset first.")
            return

        from dialogs.save_as_dialog import SaveAsDialog
        from datasets.yolo_exporter import YOLOExporter
        from datasets.coco_exporter import COCOExporter
        from datasets.voc_exporter import VOCExporter
        from datasets.pth_exporter import PTHExporter

        dialog = SaveAsDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        fmt = dialog.selected_format()
        dest = dialog.selected_folder()

        exporter = {
            'YOLO': YOLOExporter,
            'COCO': COCOExporter,
            'Pascal VOC': VOCExporter,
            'InstanSeg PTH': PTHExporter,
        }[fmt]()

        progress = QProgressDialog(f"Exporting to {fmt}…", "Cancel", 0, 100, self)
        progress.setWindowTitle("Saving Dataset")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def on_progress(done: int, total: int) -> bool:
            progress.setValue(int(done * 100 / total) if total else 100)
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            exporter.export(self._loader, dest, on_progress)
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Export Failed", str(e))
            return

        progress.close()
        if not progress.wasCanceled():
            QMessageBox.information(self, "Export Complete", f"Dataset saved to:\n{dest}")

    def _on_image_selected(self, path: str, annotations: list):
        self.viewer.load_image(path, annotations)
        name = os.path.basename(path)
        n = len(annotations)
        ann_txt = f"{n} annotation{'s' if n != 1 else ''}"
        from PySide6.QtGui import QImageReader
        size = QImageReader(path).size()
        dim_txt = f"{size.width()}×{size.height()}" if size.isValid() else ""
        parts = [name, dim_txt, ann_txt]
        self._lbl_image.setText("  " + "  ·  ".join(p for p in parts if p) + "  ")

    def _set_annotations(self, checked: bool):
        """Toggle annotation overlay, keeping the menu and toolbar items in sync."""
        self.viewer.set_annotations_visible(checked)
        for act in (self._ann_action, self._tb_ann_action):
            if act.isChecked() != checked:
                act.blockSignals(True)
                act.setChecked(checked)
                act.blockSignals(False)
