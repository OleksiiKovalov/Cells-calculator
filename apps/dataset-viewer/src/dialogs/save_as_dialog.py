from PySide6.QtWidgets import (
    QDialog, QFormLayout, QHBoxLayout, QComboBox, QLineEdit,
    QPushButton, QDialogButtonBox, QFileDialog, QMessageBox,
)


class SaveAsDialog(QDialog):
    FORMATS = ['YOLO', 'COCO', 'Pascal VOC', 'InstanSeg PTH']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Dataset As")
        self.setFixedWidth(460)

        layout = QFormLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(self.FORMATS)
        layout.addRow("Target format:", self._fmt_combo)

        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select output folder…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(browse_btn)
        layout.addRow("Output folder:", folder_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._folder_edit.setText(folder)

    def selected_format(self) -> str:
        return self._fmt_combo.currentText()

    def selected_folder(self) -> str:
        return self._folder_edit.text().strip()

    def accept(self):
        if not self.selected_folder():
            QMessageBox.warning(self, "No Folder Selected", "Please select an output folder.")
            return
        super().accept()
