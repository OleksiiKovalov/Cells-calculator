import json
import math
import os

from ui.ImageViewer import ImageViewer

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QAction, QCursor, QFontMetrics, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QStatusBar, QStyle,
    QToolBar, QToolTip, QVBoxLayout, QWidget,
)
from superqt import QRangeSlider

from ui.InfoPanel import InfoPanel
from ui.OptionsPanel import OptionsPanel
from ui.FileBrowserPanel import FileBrowserPanel
from ui.InferenceWorker import InferenceWorker
from ui.ProgressPanel import ProgressPanel
from ui.ToolbarDropDown import ToolbarDropDown
from ui.errorhandling import app_logger, log_event_emitter
from model.Model import Model
from model.utils import (
    filter_detections,
    filter_segmentation_detections,
    get_segmentation_detections_range,
    morphology_to_micrometers,
    plot_predictions,
    read_img,
)

# Spatial grid for detection hit-test acceleration.
# Hard limits — the auto-sizing algorithm stays within these bounds.
DETECTION_GRID_MAX_COLS: int = 20
DETECTION_GRID_MAX_ROWS: int = 20
# Target number of masks per bucket.  Lower = faster per-frame lookup,
# higher = faster cache build and less memory when masks are large.
DETECTION_GRID_TARGET_PER_BUCKET: int = 10

# JSON file where UI option values are persisted between sessions.
SETTINGS_FILE: str = "ui_settings.json"

def _compute_grid_dims(img_w: int, img_h: int, n_masks: int) -> tuple:
    """Return (cols, rows) for the spatial detection grid.

    Algorithm
    ---------
    We want each bucket to hold ~DETECTION_GRID_TARGET_PER_BUCKET masks on
    average (a good balance between lookup cost and build cost).

        total_buckets = ceil(n_masks / target)

    Distribute tiles proportional to the image aspect ratio so tiles stay
    as square as possible (square tiles minimise the number of tiles a mask
    bbox can straddle):

        cols = round(sqrt(total_buckets * W/H))
        rows = round(total_buckets / cols)

    Both axes are clamped to [1, MAX].
    """
    if n_masks <= 0:
        return 1, 1
    target = max(1, DETECTION_GRID_TARGET_PER_BUCKET)
    total = max(1, math.ceil(n_masks / target))
    aspect = img_w / max(img_h, 1)
    cols = max(1, round(math.sqrt(total * aspect)))
    rows = max(1, round(total / cols))
    cols = min(cols, DETECTION_GRID_MAX_COLS)
    rows = min(rows, DETECTION_GRID_MAX_ROWS)
    return cols, rows


class MainWindow(QMainWindow):
    """Main application window: image viewer, inference controls and floating panels."""

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(self):
        """Build the window, its widgets and panels, then load saved settings."""
        super().__init__()
        self.setWindowTitle("Cells Calculator")
        self.resize(1000, 700)

        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)

        self._create_members()

        self._create_actions()
        self._create_menu()
        self._create_toolbar()
        self._create_statusbar()
        self._create_info_panel()
        self._create_console_panel()
        self._create_progress_panel()
        self._create_options_panel()
        self._create_tools_dropdown()
        self._create_file_browser_panel()
        self.viewer.measure_distance.connect(self._on_measure_distance)
        self.viewer.region_selected.connect(self._on_region_selected)
        self.viewer.mouse_image_pos.connect(self._on_mouse_image_pos)
        log_event_emitter.log_line_added.connect(self.write_console)
        self._load_settings()
        self._update_actions_enabled()

    def _create_members(self):
        """Initialize instance attributes to their default empty/None state."""
        self.current_model = None
        self.prediction_image = None
        self.original_image = None
        self.detections = None
        self._detection_cache = None  # list of {bbox, coords, tooltip} built after inference
        self.inference_duration = 0
        self._last_inference_duration = 0.0
        self._inference_worker = None
        self._image_name = ""
        self._image_dims = (0, 0, 0)  # (width, height, channels)
        self._image_path = ""
        self._file_browser_panel = None
        self._options_panel_anchor = None

    # =========================================================================
    # UI construction — actions, menus, toolbar, statusbar
    # =========================================================================

    def _create_actions(self):
        """Create the QAction objects for open, exit, zoom, fit, reset and about."""
        style = self.style()

        self.act_open = QAction(
            style.standardIcon(QStyle.SP_DialogOpenButton), "&Open...", self
        )
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.setStatusTip("Open an image file")
        self.act_open.triggered.connect(self.open_image)

        self.act_exit = QAction("E&xit", self)
        self.act_exit.setShortcut(QKeySequence.Quit)
        self.act_exit.setStatusTip("Exit the application")
        self.act_exit.triggered.connect(self.close)

        self.act_zoom_in = QAction(
            style.standardIcon(QStyle.SP_ArrowUp), "Zoom &In", self
        )
        self.act_zoom_in.setShortcuts([QKeySequence.ZoomIn, QKeySequence("Ctrl+=")])
        self.act_zoom_in.setStatusTip("Zoom in")
        self.act_zoom_in.triggered.connect(self._on_zoom_in)

        self.act_zoom_out = QAction(
            style.standardIcon(QStyle.SP_ArrowDown), "Zoom &Out", self
        )
        self.act_zoom_out.setShortcut(QKeySequence.ZoomOut)
        self.act_zoom_out.setStatusTip("Zoom out")
        self.act_zoom_out.triggered.connect(self._on_zoom_out)

        self.act_fit = QAction(
            style.standardIcon(QStyle.SP_FileDialogContentsView), "&Fit to Window", self
        )
        self.act_fit.setShortcut("Ctrl+0")
        self.act_fit.setStatusTip("Fit image to window")
        self.act_fit.triggered.connect(self._on_fit)

        self.act_reset = QAction(
            style.standardIcon(QStyle.SP_BrowserReload), "&Reset (1:1)", self
        )
        self.act_reset.setShortcut("Ctrl+1")
        self.act_reset.setStatusTip("Reset zoom to actual size (1:1)")
        self.act_reset.triggered.connect(self._on_reset)

        self.act_about = QAction("&About", self)
        self.act_about.triggered.connect(self.show_about)

    def _create_menu(self):
        """Populate the menu bar with the File, View and Help menus."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.act_open)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.act_zoom_in)
        view_menu.addAction(self.act_zoom_out)
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_reset)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.act_about)

    def _create_toolbar(self):
        """Build the main toolbar with its actions, model selector and control buttons."""
        toolbar = QToolBar("Main Toolbar", self)
        self._toolbar = toolbar
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        toolbar.addAction(self.act_open)
        toolbar.addSeparator()
        self.btnFileBrowser = QPushButton("Files \U0001f4c1", self)
        self.btnFileBrowser.setObjectName("btnFileBrowser")
        self.btnFileBrowser.setToolTip("Show file browser")
        self.btnFileBrowser.clicked.connect(self._toggle_file_browser)
        toolbar.addWidget(self.btnFileBrowser)
        toolbar.addSeparator()
        toolbar.addAction(self.act_zoom_in)
        toolbar.addAction(self.act_zoom_out)
        toolbar.addAction(self.act_fit)
        toolbar.addAction(self.act_reset)
        toolbar.addSeparator()

        self.cbModel = QComboBox(self)
        self.cbModel.setObjectName("cbModel")
        # ~40 characters wide
        char_width = QFontMetrics(self.cbModel.font()).averageCharWidth()
        self.cbModel.setFixedWidth(char_width * 40)
        toolbar.addWidget(self.cbModel)

        self.btnCalculate = QPushButton("Calculate", self)
        self.btnCalculate.setObjectName("btnCalculate")
        self.btnCalculate.clicked.connect(self._on_calculate_clicked)
        toolbar.addWidget(self.btnCalculate)
        toolbar.addSeparator()

        self.cbShowOriginal = QCheckBox("Show original", self)
        self.cbShowOriginal.setObjectName("cbShowOriginal")
        self.cbShowOriginal.stateChanged.connect(self._on_show_original_changed)
        toolbar.addWidget(self.cbShowOriginal)
        toolbar.addSeparator()

        self.cellSizeSlider = QRangeSlider(Qt.Horizontal)
        self.cellSizeSlider.setObjectName("cellSizeSlider")
        self.cellSizeSlider.setMinimum(0)
        self.cellSizeSlider.setMaximum(100)
        self.cellSizeSlider.setValue((0, 100))
        self.cellSizeSlider.setFixedWidth(150)
        self.cellSizeSlider.valueChanged.connect(self._on_cell_size_changed)
        toolbar.addWidget(self.cellSizeSlider)
        toolbar.addSeparator()

        self.btnFilter = QPushButton("Filter", self)
        self.btnFilter.setObjectName("btnFilter")
        self.btnFilter.clicked.connect(self._on_filter_clicked)
        toolbar.addWidget(self.btnFilter)
        toolbar.addSeparator()

        self.btnInfo = QPushButton("ℹ Info", self)
        self.btnInfo.setObjectName("btnInfo")
        self.btnInfo.setToolTip("Show info panel")
        self.btnInfo.clicked.connect(lambda: self._info_panel.show_and_raise())
        toolbar.addWidget(self.btnInfo)
        toolbar.addSeparator()

        self.btnConsole = QPushButton("▤ Console", self)
        self.btnConsole.setObjectName("btnConsole")
        self.btnConsole.setToolTip("Show console panel")
        self.btnConsole.clicked.connect(lambda: self._console_panel.show_and_raise())
        toolbar.addWidget(self.btnConsole)
        toolbar.addSeparator()

        self.btnMeasureOptions = QPushButton("\u2699 Options", self)
        self.btnMeasureOptions.setObjectName("btnMeasureOptions")
        self.btnMeasureOptions.setToolTip("Show options panel")
        self.btnMeasureOptions.clicked.connect(self._toggle_measure_options)
        toolbar.addWidget(self.btnMeasureOptions)

        self.btnTools = QPushButton("\U0001f9f0 Tools \u25be", self)
        self.btnTools.setObjectName("btnTools")
        self.btnTools.setToolTip("Tools")
        self.btnTools.clicked.connect(self._toggle_tools_dropdown)
        toolbar.addWidget(self.btnTools)



    def _create_statusbar(self):
        """Create the status bar with permanent image-info and zoom labels."""
        self._statusbar = QStatusBar(self)
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")
        self._image_info_label = QLabel()
        self._image_info_label.setMinimumWidth(200)
        self._statusbar.addPermanentWidget(self._image_info_label)
        self._zoom_label = QLabel("Zoom: —")
        self._zoom_label.setMinimumWidth(90)
        self._statusbar.addPermanentWidget(self._zoom_label)
        self.viewer.zoom_changed.connect(self._on_zoom_changed)

    # =========================================================================
    # Window events
    # =========================================================================

    def showEvent(self, event):
        """Position the floating panels and capture their anchors on first show."""
        super().showEvent(event)
        # Viewer is now sized — set default panel positions once
        vw, vh = self.viewer.width(), self.viewer.height()
        iw, ih = self._info_panel.width(), self._info_panel.height()
        cw, ch = self._console_panel.width(), self._console_panel.height()
        fw, fh = self._file_browser_panel.width(), self._file_browser_panel.height()
        ow, oh = self._options_panel.width(), self._options_panel.height()
        self._info_panel.move(max(0, vw - iw - 10), 10)
        self._console_panel.move(max(0, vw - cw - 10), max(0, vh - ch - 10))
        self._file_browser_panel.move(10, max(0, vh // 2 - fh // 2))
        self._options_panel.move(max(0, vw // 2 - ow // 2), 10)
        self._info_panel_anchor = self._make_anchor(self._info_panel)
        self._console_panel_anchor = self._make_anchor(self._console_panel)
        self._file_browser_panel_anchor = self._make_anchor(self._file_browser_panel)
        self._options_panel_anchor = self._make_anchor(self._options_panel)

    def resizeEvent(self, event):
        """Re-clamp the floating panels to their anchors when the window resizes."""
        super().resizeEvent(event)
        self._clamp_panel(getattr(self, '_info_panel', None),
                          getattr(self, '_info_panel_anchor', None))
        self._clamp_panel(getattr(self, '_console_panel', None),
                          getattr(self, '_console_panel_anchor', None))
        self._clamp_panel(getattr(self, '_file_browser_panel', None),
                          getattr(self, '_file_browser_panel_anchor', None))
        self._clamp_panel(getattr(self, '_options_panel', None),
                          getattr(self, '_options_panel_anchor', None))

    def closeEvent(self, event):
        """Persist UI settings before the window closes."""
        self._save_settings()
        super().closeEvent(event)

    # =========================================================================
    # Image — open, load, view
    # =========================================================================

    def load_image(self, path: str) -> bool:
        """Load an image from disk, display it and update window state.

        Clears any previous detections/prediction, reads the image, fits it to
        the window, refreshes the status label and points the file browser at
        the image's directory.

        Args:
            path: Filesystem path of the image to load.

        Returns:
            True if the image loaded successfully, False otherwise.
        """
        self.original_image = None
        self.detections = None
        self._detection_cache = None
        self.prediction_image = None
        try:
            image: np.ndarray = read_img(path)
        except Exception:
            app_logger().exception("Failed to read image: %s", path)
            return False
        if image is None or image.size == 0:
            return False
        self.original_image = image
        self.viewer.set_image(image)
        self.act_fit.trigger()
        name = os.path.basename(path)
        h, w = image.shape[:2]
        ch = image.shape[2] if image.ndim == 3 else 1
        self._image_name = name
        self._image_path = path
        self._image_dims = (w, h, ch)
        self._image_info_label.setText(f"{name}  {w} × {h}  ({ch}ch)")
        # Update file browser directory
        if self._file_browser_panel:
            dir_path = os.path.dirname(path)
            self._file_browser_panel.set_directory(dir_path)
        return True

    def open_image(self):
        """Prompt for an image file and load it, reporting success or failure."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp *.lsm);;All Files (*)",
        )
        if not path:
            return
        if self.load_image(path):
            self._statusbar.showMessage(f"Loaded: {path}")
            self.setWindowTitle(f"Cells Calculator - {path}")
        else:
            QMessageBox.warning(self, "Open Image", f"Cannot load image:\n{path}")
            self._statusbar.showMessage("Failed to load image")
        self._update_actions_enabled()

    def _set_current_image(self, show_original: bool = True):
        """Show either the original or the prediction image in the viewer.

        Args:
            show_original: If True show the original image, otherwise the
                prediction image. Syncs the 'Show original' checkbox.
        """
        try:
            self.cbShowOriginal.blockSignals(True)
            self.cbShowOriginal.setChecked(show_original)
            if self.cbShowOriginal.isChecked():
                self.viewer.set_image(self.original_image, True)
            else:
                self.viewer.set_image(self.prediction_image, True)
        finally:
            self.cbShowOriginal.blockSignals(False)

    def _on_show_original_changed(self, state: int):
        """Switch the viewer image when the 'Show original' checkbox toggles."""
        self._set_current_image(state == Qt.CheckState.Checked)

    # =========================================================================
    # Zoom
    # =========================================================================

    def _on_zoom_in(self):
        """Zoom the viewer in."""
        self.viewer.zoom_in()

    def _on_zoom_out(self):
        """Zoom the viewer out."""
        self.viewer.zoom_out()

    def _on_fit(self):
        """Fit the image to the viewer window."""
        self.viewer.fit_to_window()

    def _on_reset(self):
        """Reset the viewer to actual size (1:1)."""
        self.viewer.reset_view()

    def _on_zoom_changed(self, scale: float):
        """Update the status-bar zoom label for the given scale factor."""
        self._zoom_label.setText(f"Zoom: {scale * 100:.0f}%")

    # ---- panel positioning helpers --------------------------------------

    def _clamp_panel(self, panel, anchor):
        """Clamp a floating panel within its parent, respecting stored border anchor."""
        if panel is None:
            return
        parent = panel.parent()
        if parent is None:
            return
        pr = parent.rect()
        pw, ph = pr.width(), pr.height()
        iw, ih = panel.width(), panel.height()
        if anchor is None:
            x = max(0, min(panel.x(), pw - iw))
            y = max(0, min(panel.y(), ph - ih))
        else:
            x = (pw - iw - anchor['dist_h']) if anchor['h'] == 'right' else anchor['dist_h']
            y = (ph - ih - anchor['dist_v']) if anchor['v'] == 'bottom' else anchor['dist_v']
            x = max(0, min(x, pw - iw))
            y = max(0, min(y, ph - ih))
        panel.move(x, y)

    def _make_anchor(self, panel):
        """Compute and return an anchor dict for a floating panel."""
        parent = panel.parent()
        if parent is None:
            return None
        pr = parent.rect()
        pw, ph = pr.width(), pr.height()
        x, y = panel.x(), panel.y()
        iw, ih = panel.width(), panel.height()
        dist_left, dist_right = x, pw - x - iw
        dist_top, dist_bottom = y, ph - y - ih
        return {
            'h': 'right' if dist_right <= dist_left else 'left',
            'dist_h': min(dist_left, dist_right),
            'v': 'bottom' if dist_bottom <= dist_top else 'top',
            'dist_v': min(dist_top, dist_bottom),
        }

    # =========================================================================
    # Detection / segmentation
    # =========================================================================

    def _register_segmenters(self, models):
        """Store the available segmentation models and populate the model combo box."""
        self.loaded_models = models
        self.cbModel.addItems(self.loaded_models.keys())
        self.cbModel.setCurrentIndex(0)


    def _on_filter_clicked(self):
        """Re-apply the cell-size filter and refresh the prediction image and stats."""
        if self.prediction_image is None or self.detections is None:
            return
        self._refresh_prediction_image()
        self.show_detection_stats()

    def _on_calculate_clicked(self):
        """Start inference when the Calculate button is clicked."""
        self._start_inference()

    def _on_cell_size_changed(self, value: tuple):
        """Show the current cell-size slider range in the status bar."""
        self._statusbar.showMessage(f"Cell size: {value[0]} – {value[1]}")

    # =========================================================================
    # Inference (threaded)
    # =========================================================================

    def _start_inference(self):
        """Run model inference on the current image in a background worker.

        Lazily (re)constructs the selected model, disables the toolbar, shows
        the progress panel and starts an InferenceWorker whose signals drive
        the result/error/cancel handlers. No-op if a worker is already running.
        """
        if self._inference_worker is not None and self._inference_worker.isRunning():
            return
        if self.current_model is None or self.current_model.model_name != self.cbModel.currentText():
            self.current_model = None
            self.current_model_name = self.cbModel.currentText()
            segmenter_model = self.loaded_models[self.current_model_name]
            self.current_model = Model(
                path=segmenter_model['path'],
                model_type=segmenter_model['model_type'],
                model_data=segmenter_model,
                model_name=self.current_model_name
            )
        self.btnCalculate.setEnabled(False)
        self._toolbar.setEnabled(False)
        self._progress_panel.start(self._last_inference_duration)
        self._progress_panel.show()
        self._progress_panel.raise_()
        self._inference_worker = InferenceWorker(self.current_model, self.original_image)
        self._inference_worker.status_changed.connect(self._on_inference_status)
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_worker.cancelled.connect(self._on_inference_cancelled)
        self._inference_worker.start()
        self._statusbar.showMessage("Inference started…")

    # =========================================================================
    # Floating panels — Info, Console, Progress
    # =========================================================================

    def _create_info_panel(self):
        """Create the floating Info panel and wire up its anchor tracking."""
        self._info_panel = InfoPanel(self.viewer)
        self._info_panel.set_title("Info")
        self._info_panel.hide()
        self._info_panel.panel_moved.connect(
            lambda: setattr(self, '_info_panel_anchor', self._make_anchor(self._info_panel))
        )
        self._info_panel_anchor = None
        self._info_panel.set_max_lines(1000)

    def write_info(self, text: str):
        """Append a line to the info panel, showing it if hidden."""
        self._info_panel.write(text)
        if not self._info_panel.isVisible():
            self._info_panel.show_and_raise()

    # =========================================================================
    # Options panel
    # =========================================================================

    def _create_options_panel(self):
        """Build the floating Options panel (ruler, region and display checkboxes)."""
        self._options_panel = OptionsPanel(self.viewer)
        self._options_panel.hide()
        self._options_panel.panel_moved.connect(
            lambda: setattr(self, '_options_panel_anchor', self._make_anchor(self._options_panel))
        )

        self.chkClearRuler = QCheckBox("Clear ruler after measurement")
        self.chkClearRuler.setChecked(False)
        self.chkClearRuler.toggled.connect(self._on_clear_ruler_toggled)
        self._options_panel.add_widget(self.chkClearRuler)

        self.chkClearRegion = QCheckBox("Clear region after selection")
        self.chkClearRegion.setChecked(False)
        self.chkClearRegion.toggled.connect(self._on_clear_region_toggled)
        self._options_panel.add_widget(self.chkClearRegion)

        self.chkDrawLabels = QCheckBox("Draw labels on masks")
        self.chkDrawLabels.setChecked(False)
        self.chkDrawLabels.toggled.connect(lambda _: self._refresh_prediction_image() if self.prediction_image is not None else None)
        self._options_panel.add_widget(self.chkDrawLabels)

        self.chkWrapInfo = QCheckBox("Wrap Info window")
        self.chkWrapInfo.setChecked(False)
        self.chkWrapInfo.toggled.connect(lambda checked: self._info_panel.set_wrap(checked))
        self._options_panel.add_widget(self.chkWrapInfo)

        self.chkWrapConsole = QCheckBox("Wrap Console window")
        self.chkWrapConsole.setChecked(False)
        self.chkWrapConsole.toggled.connect(lambda checked: self._console_panel.set_wrap(checked))
        self._options_panel.add_widget(self.chkWrapConsole)

        self.chkDetectionTooltip = QCheckBox("Show detection tooltip")
        self.chkDetectionTooltip.setChecked(False)
        self._options_panel.add_widget(self.chkDetectionTooltip)

        self.chkFillPolygons = QCheckBox("Fill polygons")
        self.chkFillPolygons.setChecked(False)
        self.chkFillPolygons.toggled.connect(lambda _: self._refresh_prediction_image() if self.prediction_image is not None else None)
        self._options_panel.add_widget(self.chkFillPolygons)

        # µm/mm scale row: [0.] [spinbox] [µm/mm]
        _scale_row = QWidget()
        _scale_layout = QHBoxLayout(_scale_row)
        _scale_layout.setContentsMargins(0, 0, 0, 0)
        _scale_layout.setSpacing(4)
        _scale_layout.addWidget(QLabel("Scale"))
        self.spnUmPerMm = QDoubleSpinBox()
        self.spnUmPerMm.setObjectName("spnUmPerMm")
        self.spnUmPerMm.setRange(0.0, 1000.0)
        self.spnUmPerMm.setDecimals(6)
        self.spnUmPerMm.setSingleStep(0.001)
        self.spnUmPerMm.setValue(0.325)
        self.spnUmPerMm.setToolTip(
            "Pixel size in micrometers per pixel; used to report morphology in \u00b5m. "
            "Set to 0 to show relative values only. Re-run Calculate/Filter to refresh."
        )
        _scale_layout.addWidget(self.spnUmPerMm)
        _scale_layout.addWidget(QLabel("\u00b5m/px"))
        _scale_layout.addStretch()
        self._options_panel.add_widget(_scale_row)

    def _toggle_measure_options(self):
        """Show or hide the floating Options panel."""
        if self._options_panel.isVisible():
            self._options_panel.hide()
        else:
            self._options_panel.show_and_raise()

    # =========================================================================
    # Settings  — persist Options dropdown values
    # =========================================================================

    def _collect_settings(self) -> dict:
        """Return a dict of all persisted UI option values."""
        return {
            # checkboxes
            'chkClearRuler':        self.chkClearRuler.isChecked(),
            'chkClearRegion':       self.chkClearRegion.isChecked(),
            'chkDrawLabels':        self.chkDrawLabels.isChecked(),
            'chkWrapInfo':          self.chkWrapInfo.isChecked(),
            'chkWrapConsole':       self.chkWrapConsole.isChecked(),
            'chkDetectionTooltip':  self.chkDetectionTooltip.isChecked(),
            'chkFillPolygons':       self.chkFillPolygons.isChecked(),
            # spinboxes
            'spnUmPerMm':           self.spnUmPerMm.value(),
            # comboboxes
            'cbModel':              self.cbModel.currentText(),
        }

    def _apply_settings(self, s: dict):
        """Apply a settings dict to the UI controls (silently ignores unknown keys)."""
        def _set_chk(attr, key):
            """Set checkbox `attr` from settings key `key` if both exist."""
            w = getattr(self, attr, None)
            if w is not None and key in s:
                w.setChecked(bool(s[key]))

        _set_chk('chkClearRuler',       'chkClearRuler')
        _set_chk('chkClearRegion',      'chkClearRegion')
        _set_chk('chkDrawLabels',       'chkDrawLabels')
        _set_chk('chkWrapInfo',         'chkWrapInfo')
        _set_chk('chkWrapConsole',      'chkWrapConsole')
        _set_chk('chkDetectionTooltip', 'chkDetectionTooltip')
        _set_chk('chkFillPolygons',       'chkFillPolygons')

        w = getattr(self, 'spnUmPerMm', None)
        if w is not None and 'spnUmPerMm' in s:
            try:
                w.setValue(float(s['spnUmPerMm']))
            except (TypeError, ValueError):
                pass

        if 'cbModel' in s:
            idx = self.cbModel.findText(s['cbModel'])
            if idx >= 0:
                self.cbModel.setCurrentIndex(idx)

    def _save_settings(self):
        """Persist UI options to SETTINGS_FILE."""
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._collect_settings(), f, indent=2)
        except Exception as e:
            print(f"[settings] save failed: {e}")

    def _load_settings(self):
        """Load UI options from SETTINGS_FILE and apply them to controls."""
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                s = json.load(f)
            self._apply_settings(s)
        except FileNotFoundError:
            pass  # first run — use defaults
        except Exception as e:
            print(f"[settings] load failed: {e}")

    # =========================================================================
    # Tools dropdown
    # =========================================================================

    def _create_tools_dropdown(self):
        """Build the Tools dropdown."""
        self._tools_dropdown = ToolbarDropDown()

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        btn_print = QPushButton("Print detection")
        btn_print.clicked.connect(lambda: (self._tools_dropdown.hide(), self._print_detection()))
        layout.addWidget(btn_print)

        self._tools_dropdown.set_content(content)

    def _toggle_tools_dropdown(self):
        """Pop up the Tools dropdown below the Tools button."""
        self._tools_dropdown.popup_below(self.btnTools)

    def _toggle_file_browser(self):
        """Prompt for a folder and open the file browser panel showing it."""
        if self._file_browser_panel is None:
            self._create_file_browser_panel()
        
        # Open folder dialog
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            self._image_path if self._image_path else "",
        )
        if folder_path:
            self._file_browser_panel.set_directory(folder_path)
            self._file_browser_panel.show_and_raise()

    def _print_detection(self):
        """Called when 'Print detection' is selected from the Tools dropdown."""
        if self.detections is None:
            self.write_info("No detections available.")
            return
        df = self.detections
        cols = df.columns.tolist()
        has_id = 'id_label' in cols
        has_area = 'area' in cols
        has_diam = 'diameter' in cols
        has_vol = 'volume' in cols
        header_parts = []
        if has_id:     header_parts.append(f"{'ID':<6}")
        if has_area:   header_parts.append(f"{'Area':>12}")
        if has_diam:   header_parts.append(f"{'Diameter':>12}")
        if has_vol:    header_parts.append(f"{'Volume':>12}")
        self.write_info("  ".join(header_parts))
        self.write_info("-" * (len("  ".join(header_parts)) + 2))
        for _, row in df.iterrows():
            parts = []
            if has_id:   parts.append(f"{str(row['id_label']):<6}")
            if has_area: parts.append(f"{row['area'] * 1000:>12.2f}")
            if has_diam: parts.append(f"{row['diameter'] * 1000:>12.2f}")
            if has_vol:  parts.append(f"{row['volume'] * 1000:>12.2f}")
            self.write_info("  ".join(parts))

    def _build_detection_cache(self):
        """Pre-compute bounding boxes, denormalized coords and tooltip strings for all masks,
        then assign each mask into a spatial grid (sized by _compute_grid_dims, bounded by
        DETECTION_GRID_MAX_COLS x DETECTION_GRID_MAX_ROWS) so _on_mouse_image_pos only checks
        the small subset in the cursor's tile.
        """
        if self.detections is None or self.original_image is None:
            self._detection_cache = None
            return
        df = self.detections
        masks = df['mask'].tolist()
        if not masks:
            self._detection_cache = None
            return
        img_h, img_w = self.original_image.shape[:2]
        # Contract: mask coordinates are normalized [0, 1]; multiply by image size to get pixel space.
        scale = np.array([img_w, img_h], dtype=np.float32)

        # --- build flat entry list ---
        entries = []
        for _, row in df.iterrows():
            coords = np.asarray(row['mask'], dtype=np.float32).reshape(-1, 2)
            if coords.shape[0] < 3:
                continue
            coords = coords * scale
            x1, y1 = coords.min(axis=0).tolist()
            x2, y2 = coords.max(axis=0).tolist()
            parts = []
            if 'id_label' in row.index: parts.append(f"ID: {row['id_label']}")
            if 'area'     in row.index: parts.append(f"Area: {row['area'] * 1000:.2f}")
            if 'diameter' in row.index: parts.append(f"Diameter: {row['diameter'] * 1000:.2f}")
            if 'volume'   in row.index: parts.append(f"Volume: {row['volume'] * 1000:.2f}")
            entries.append({
                'bbox':    (x1, y1, x2, y2),
                'coords':  coords,
                'tooltip': "\n".join(parts),
            })

        # --- assign entries to spatial buckets ---
        cols, rows = _compute_grid_dims(img_w, img_h, len(entries))
        tile_w = img_w / cols
        tile_h = img_h / rows
        grid = [[[] for _ in range(cols)] for _ in range(rows)]
        for entry in entries:
            x1, y1, x2, y2 = entry['bbox']
            col_min = max(0, int(x1 / tile_w))
            col_max = min(cols - 1, int(x2 / tile_w))
            row_min = max(0, int(y1 / tile_h))
            row_max = min(rows - 1, int(y2 / tile_h))
            for r in range(row_min, row_max + 1):
                for c in range(col_min, col_max + 1):
                    grid[r][c].append(entry)

        self._detection_cache = {
            'grid':   grid,
            'tile_w': tile_w,
            'tile_h': tile_h,
            'img_w':  img_w,
            'img_h':  img_h,
            'cols':   cols,
            'rows':   rows,
        }

    def _on_mouse_image_pos(self, pos: QPointF):
        """Show a tooltip with detection info when the cursor is over a mask.

        Uses a spatial grid for O(1) bucket lookup; only masks whose bounding
        box overlaps the cursor's tile are tested with pointPolygonTest.
        """
        if not self.chkDetectionTooltip.isChecked():
            return
        if self._detection_cache is None:
            QToolTip.hideText()
            return
        cache = self._detection_cache
        px, py = pos.x(), pos.y()
        # Discard positions outside the image
        if not (0 <= px < cache['img_w'] and 0 <= py < cache['img_h']):
            QToolTip.hideText()
            return
        # Determine which grid tile the cursor falls in
        col = min(cache['cols'] - 1, int(px / cache['tile_w']))
        row = min(cache['rows'] - 1, int(py / cache['tile_h']))
        for entry in cache['grid'][row][col]:
            x1, y1, x2, y2 = entry['bbox']
            if not (x1 <= px <= x2 and y1 <= py <= y2):
                continue
            if cv2.pointPolygonTest(entry['coords'], (float(px), float(py)), False) >= 0:
                QToolTip.showText(QCursor.pos(), entry['tooltip'], self.viewer)
                return
        QToolTip.hideText()

    def _on_measure_distance(self, dist: float):
        """Report a measured ruler distance (in pixels) to the info panel."""
        self.write_info(f"Distance: {dist:.1f} px")

    def _on_clear_ruler_toggled(self, checked: bool):
        """Toggle whether the viewer clears the ruler after each measurement."""
        self.viewer.auto_clear_measure = checked

    # =========================================================================
    # Region selection  (Shift + click)
    # =========================================================================

    def _on_region_selected(self, rect):
        """Called when the user finishes a rubber-band region selection.

        Args:
            rect: QRectF with the selected region in image pixel coordinates.
                  Coordinates are zoom- and pan-aware (actual image coordinates
                  regardless of the current view transform).
        """
        x, y, w, h = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
        self.write_info(f"Region selected: x={x}, y={y}, w={w}, h={h}")

    def _on_clear_region_toggled(self, checked: bool):
        """Toggle whether the viewer clears the region after each selection."""
        self.viewer.auto_clear_selection = checked

    def _on_file_selected(self, file_path):
        """Load the image chosen in the file browser, reporting success or failure."""
        if self.load_image(file_path):
            self._statusbar.showMessage(f"Loaded: {file_path}")
            self.setWindowTitle(f"Cells Calculator - {file_path}")
        else:
            QMessageBox.warning(self, "Open Image", f"Cannot load image:\n{file_path}")
            self._statusbar.showMessage("Failed to load image")
        self._update_actions_enabled()

    def _create_console_panel(self):
        """Create the floating Console panel and wire up its anchor tracking."""
        self._console_panel = InfoPanel(self.viewer)
        self._console_panel.set_title("Console")
        self._console_panel.hide()
        self._console_panel.panel_moved.connect(
            lambda: setattr(self, '_console_panel_anchor', self._make_anchor(self._console_panel))
        )
        self._console_panel_anchor = None
        self._console_panel.set_max_lines(1000)

    def write_console(self, text: str):
        """Append a line to the console panel and make it visible."""
        self._console_panel.write(text)
        if not self._console_panel.isVisible():
            self._console_panel.show_and_raise()

    def _create_progress_panel(self):
        """Create the centered floating progress panel for inference."""
        self._progress_panel = ProgressPanel(self.viewer)
        self._progress_panel.hide()
        self._progress_panel.move(
            max(0, (self.viewer.width() - self._progress_panel.width()) // 2),
            max(0, (self.viewer.height() - self._progress_panel.height()) // 2),
        )
        self._progress_panel.cancel_requested.connect(self._on_inference_cancel)

    def _create_file_browser_panel(self):
        """Create the floating File Browser panel and set its initial directory."""
        self._file_browser_panel = FileBrowserPanel(self.viewer)
        self._file_browser_panel.set_title("File Browser")
        self._file_browser_panel.hide()
        self._file_browser_panel.panel_moved.connect(
            lambda: setattr(self, '_file_browser_panel_anchor', self._make_anchor(self._file_browser_panel))
        )
        self._file_browser_panel.file_selected.connect(self._on_file_selected)
        self._file_browser_panel_anchor = None
        # Set initial directory
        if self._image_path:
            dir_path = os.path.dirname(self._image_path)
            self._file_browser_panel.set_directory(dir_path)
        else:
            self._file_browser_panel.set_directory(os.getcwd())

    def _on_inference_status(self, text: str):
        """Relay an inference status message to the progress panel and status bar."""
        self._progress_panel.set_status(text)
        self._statusbar.showMessage(text)

    def _on_inference_finished(self, detections):
        """Handle successful inference: store detections, build cache and refresh UI."""
        elapsed = self._progress_panel.elapsed_seconds()
        self._last_inference_duration = elapsed
        self.inference_duration = elapsed
        self.detections = detections
        self._build_detection_cache()
        min_value, max_value = get_segmentation_detections_range(detections, size_metric="area")
        self.min_value = min_value
        self.max_value = max_value
        self.show_detection_stats()
        self._refresh_prediction_image()
        self._statusbar.showMessage(f"Model processed image in {elapsed:.2f} seconds")
        self._finish_inference_ui()
        self._update_actions_enabled()

    def _on_inference_error(self, error: str):
        """Report an inference error in a dialog and reset the inference UI."""
        QMessageBox.critical(self, "Inference Error", error)
        self._statusbar.showMessage(f"Inference failed: {error}")
        self._finish_inference_ui()

    def _on_inference_cancel(self):
        """Request cancellation of the running inference worker."""
        if self._inference_worker:
            self._inference_worker.cancel()  # non-blocking; cancelled signal drives cleanup

    def _on_inference_cancelled(self):
        """Handle a cancelled inference run by resetting the inference UI."""
        self._statusbar.showMessage("Inference cancelled")
        self._finish_inference_ui()

    def _finish_inference_ui(self):
        """Tear down the progress panel and re-enable the toolbar after inference."""
        self._progress_panel.stop()
        self._progress_panel.hide()
        self._toolbar.setEnabled(True)
        self.btnCalculate.setEnabled(True)
        self._inference_worker = None

    def show_detection_stats(self):
        """Write detection summary statistics to the info panel.

        Reports image name, dimensions, model, duration and object count, plus
        average diameter/area/volume. When a µm/px scale is set, also reports
        the averages converted to micrometers.
        """
        spheroid_df = self.detections
        w, h, ch = self._image_dims
        if spheroid_df is None or spheroid_df.empty:
            self.write_info(f"**************************************")
            self.write_info(f"Image           : {self._image_name}")
            self.write_info(f"Dimensions      : {w} × {h}  ({ch}ch)")
            self.write_info(f"Model           : {self.current_model.model_name}")
            self.write_info(f"Duration        : {self.inference_duration:.2f} seconds")
            self.write_info(f"Objects detected: 0")
            return
        diameter_norm = spheroid_df["diameter"].mean()
        area_norm = spheroid_df["area"].mean()
        volume_norm = spheroid_df["volume"].mean()
        avg_diameter = diameter_norm * 1000
        avg_area = area_norm * 1000
        avg_volume = volume_norm * 1000
        num_cells = spheroid_df.shape[0]
        self.write_info(f"**************************************")
        self.write_info(f"Image           : {self._image_name}")
        self.write_info(f"Dimensions      : {w} × {h}  ({ch}ch)")
        self.write_info(f"Model           : {self.current_model.model_name}")
        self.write_info(f"Duration        : {self.inference_duration:.2f} seconds")
        self.write_info(f"Objects detected: {num_cells}")
        self.write_info(f"Average diameter: {avg_diameter:.2f}")
        self.write_info(f"Average area    : {avg_area:.2f}")
        self.write_info(f"Average volume  : {avg_volume:.2f}")
        um_per_px = self.spnUmPerMm.value()
        if um_per_px > 0:
            d_um, a_um2, v_um3 = morphology_to_micrometers(
                diameter_norm, area_norm, volume_norm, w, h, um_per_px
            )
            self.write_info(f"Scale           : {um_per_px:g} µm/px")
            self.write_info(f"Avg diameter    : {d_um:.2f} µm")
            self.write_info(f"Avg area        : {a_um2:.2f} µm²")
            self.write_info(f"Avg volume      : {v_um3:.2f} µm³")

    def _refresh_prediction_image(self):
        """Rebuild the prediction overlay from the current slider/option settings.

        Maps the cell-size slider range onto the detection size range, filters
        detections accordingly, redraws the masks (respecting fill/label
        options) and displays the result in the viewer.
        """
        min_value = (self.max_value - self.min_value) * self.cellSizeSlider.value()[0] / 100 + self.min_value
        max_value = (self.max_value - self.min_value) * self.cellSizeSlider.value()[1] / 100 + self.min_value
        filtered_detections = self._get_filtered_detections(min_value, max_value)
        self.prediction_image = plot_predictions(
            self.original_image.copy(),
            filtered_detections["mask"].tolist(),
            color_ids=filtered_detections['id_label'].tolist() if 'id_label' in filtered_detections else None,
            filled=self.chkFillPolygons.isChecked(),
            outline_thickness=2,
            draw_labels=self.chkDrawLabels.isChecked(),
        )
        self._set_current_image(False)

    def _get_filtered_detections(self, min_value, max_value):
        """Return detections whose size falls within the given range.

        Uses area-based segmentation filtering when the detections carry
        'area'/'mask' columns, otherwise falls back to generic size filtering.

        Args:
            min_value: Lower size bound (inclusive).
            max_value: Upper size bound (inclusive).

        Returns:
            The filtered detections, or None if there are no detections.
        """
        if self.detections is None:
            return
        if hasattr(self.detections, "columns") and all(
            c in self.detections.columns for c in ["area", "mask"]
        ):
            return filter_segmentation_detections(
                self.detections,
                min_size=min_value,
                max_size=max_value,
                size_metric="area",
            )
        else:
            return filter_detections(self.detections, min_size=min_value, max_size=max_value)

    # =========================================================================
    # Misc
    # =========================================================================

    def show_about(self):
        """Display the About dialog."""
        QMessageBox.about(
            self,
            "About",
            "Cells Calculator 4.0\n\n"
            "Cell & spheroid instance segmentation and morphology analysis.",
        )

    def _update_actions_enabled(self):
        """Enable or disable actions and controls based on current image/state."""
        enabled = self.viewer.has_image()
        for act in (self.act_zoom_in, self.act_zoom_out, self.act_fit, self.act_reset):
            act.setEnabled(enabled)
        self.btnCalculate.setEnabled(enabled and self.cbModel.count() > 0 and self.original_image is not None)
        self.btnFilter.setEnabled(enabled and self.detections is not None)
        self.cbShowOriginal.setEnabled(enabled and self.prediction_image is not None)
        self.cellSizeSlider.setEnabled(enabled and self.detections is not None)
        
