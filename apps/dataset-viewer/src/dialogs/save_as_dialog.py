from PySide6.QtWidgets import (
    QDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit,
    QPushButton, QDialogButtonBox, QFileDialog, QMessageBox, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLabel,
)


class SaveAsDialog(QDialog):
    """Prepare & Export: pick a target format, optional preprocessing and split,
    and (for PTH) file name / modality. Applies to every target format."""

    FORMATS = ['YOLO', 'COCO', 'Pascal VOC', 'InstanSeg PTH']

    def __init__(self, parent=None, start_dir: str = ""):
        super().__init__(parent)
        self._start_dir = start_dir
        self.setWindowTitle("Prepare & Export Dataset")
        self.setMinimumWidth(500)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        form = QFormLayout()
        form.setSpacing(8)
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(self.FORMATS)
        self._fmt_combo.currentTextChanged.connect(self._on_format_changed)
        form.addRow("Target format:", self._fmt_combo)

        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select output folder…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(browse_btn)
        form.addRow("Output folder:", folder_row)
        outer.addLayout(form)

        outer.addWidget(self._build_preprocess_group())
        outer.addWidget(self._build_split_group())
        outer.addWidget(self._build_pth_group())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._on_format_changed(self._fmt_combo.currentText())

    # ------------------------------------------------------------------
    # Preprocess (applies to every format)
    # ------------------------------------------------------------------
    def _build_preprocess_group(self) -> QGroupBox:
        box = QGroupBox("Preprocess (applied to every format)")
        form = QFormLayout(box)
        form.setSpacing(8)

        self._pp_standardize = QCheckBox("Standardize images to RGB / uint8")
        self._pp_standardize.setChecked(True)
        form.addRow(self._pp_standardize)

        self._pp_resize = QCheckBox("Resize toward")
        self._pp_resize_target = QSpinBox()
        self._pp_resize_target.setRange(16, 8192)
        self._pp_resize_target.setValue(512)
        self._pp_resample = QComboBox()
        self._pp_resample.addItems(["LANCZOS", "NEAREST"])
        resize_row = QHBoxLayout()
        resize_row.addWidget(self._pp_resize)
        resize_row.addWidget(self._pp_resize_target)
        resize_row.addWidget(QLabel("px"))
        resize_row.addWidget(self._pp_resample)
        resize_row.addStretch(1)
        form.addRow("Resize:", resize_row)

        self._pp_contrast = QCheckBox("Enhance contrast ×")
        self._pp_contrast_factor = QDoubleSpinBox()
        self._pp_contrast_factor.setRange(0.1, 10.0)
        self._pp_contrast_factor.setSingleStep(0.1)
        self._pp_contrast_factor.setValue(2.0)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self._pp_contrast)
        contrast_row.addWidget(self._pp_contrast_factor)
        contrast_row.addStretch(1)
        form.addRow("Contrast:", contrast_row)
        return box

    # ------------------------------------------------------------------
    # Split (applies to every format)
    # ------------------------------------------------------------------
    def _build_split_group(self) -> QGroupBox:
        box = QGroupBox("Split")
        form = QFormLayout(box)
        form.setSpacing(8)

        self._split = QComboBox()
        self._split.addItem("Keep source splits", "keep")
        self._split.addItem("Re-split by ratio", "ratio")
        self._split.currentIndexChanged.connect(self._on_split_changed)
        form.addRow("Mode:", self._split)

        ratio_row = QHBoxLayout()
        self._train = self._ratio_spin(0.70)
        self._val = self._ratio_spin(0.10)
        self._test = self._ratio_spin(0.20)
        for lbl, sp in (("train", self._train), ("val", self._val), ("test", self._test)):
            ratio_row.addWidget(QLabel(lbl))
            ratio_row.addWidget(sp)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(42)
        ratio_row.addWidget(QLabel("seed"))
        ratio_row.addWidget(self._seed)
        self._ratio_widgets = [self._train, self._val, self._test, self._seed]
        form.addRow("Ratios:", ratio_row)

        self._on_split_changed()
        return box

    @staticmethod
    def _ratio_spin(value: float) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(0.0, 1.0)
        sp.setSingleStep(0.05)
        sp.setDecimals(2)
        sp.setValue(value)
        return sp

    # ------------------------------------------------------------------
    # InstanSeg PTH-only options
    # ------------------------------------------------------------------
    def _build_pth_group(self) -> QGroupBox:
        box = QGroupBox("InstanSeg PTH options")
        form = QFormLayout(box)
        form.setSpacing(8)
        self._pth_name = QLineEdit("dataset.pth")
        form.addRow("File name:", self._pth_name)
        self._pth_modality = QComboBox()
        self._pth_modality.addItems(["Brightfield", "Fluorescence"])
        form.addRow("Modality:", self._pth_modality)
        self._pth_group = box
        return box

    # ------------------------------------------------------------------
    def _on_format_changed(self, fmt: str):
        self._pth_group.setVisible(fmt == "InstanSeg PTH")
        self.adjustSize()

    def _on_split_changed(self, *_):
        by_ratio = self._split.currentData() == "ratio"
        for w in self._ratio_widgets:
            w.setEnabled(by_ratio)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self._folder_edit.text().strip() or self._start_dir)
        if folder:
            self._folder_edit.setText(folder)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def selected_format(self) -> str:
        return self._fmt_combo.currentText()

    def selected_folder(self) -> str:
        return self._folder_edit.text().strip()

    def prepare_spec(self) -> dict:
        """Spec for PreparedLoader reflecting the preprocess + split fields."""
        return {
            "standardize": self._pp_standardize.isChecked(),
            "resize": self._pp_resize.isChecked(),
            "resize_target": self._pp_resize_target.value(),
            "resample": self._pp_resample.currentText(),
            "contrast": self._pp_contrast.isChecked(),
            "contrast_factor": self._pp_contrast_factor.value(),
            "split_mode": self._split.currentData(),
            "ratios": (self._train.value(), self._val.value(), self._test.value()),
            "seed": self._seed.value(),
        }

    def pth_options(self) -> dict:
        """Constructor kwargs for PTHExporter (file name + modality)."""
        return {
            "filename": self._pth_name.text().strip() or "dataset.pth",
            "modality": self._pth_modality.currentText(),
        }

    def accept(self):
        if not self.selected_folder():
            QMessageBox.warning(self, "No Folder Selected", "Please select an output folder.")
            return
        super().accept()
