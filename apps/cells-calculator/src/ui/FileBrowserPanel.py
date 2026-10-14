"""Floating resizable file browser panel that stays within its parent widget."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from ui.FloatingPanel import FloatingResizablePanel


class FileBrowserPanel(FloatingResizablePanel):
    """
    Floating resizable file browser panel.

    Public API
    ----------
    set_directory(path) – set the directory to list files from
    file_selected(path) – signal emitted when a file is double-clicked
    """

    file_selected = Signal(str)  # emits file path

    def __init__(self, parent: QWidget | None = None):
        """Initialize the panel with a file list and preview area."""
        self._current_directory: Path | None = None
        super().__init__(parent, title="File Browser",
                         min_size=(200, 300), initial_size=(200, 400))

    def _build_content(self, root: QVBoxLayout):
        """Build the file list and preview area."""
        self._file_list = QListWidget(self)
        self._file_list.setObjectName("FileBrowserList")
        self._file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._file_list.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._file_list, 1)

        self._preview_label = QLabel(self)
        self._preview_label.setObjectName("FileBrowserPreview")
        self._preview_label.setFixedHeight(150)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setText("Select a file to preview")
        root.addWidget(self._preview_label)

    def _content_stylesheet(self) -> str:
        """File-list and preview styling."""
        return """
            #FileBrowserList {
                background-color: rgba(40, 40, 40, 200);
                color: #d4d4d4;
                border: none;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 10pt;
            }
            QListWidget::item:selected {
                background-color: #666;
                color: #fff;
            }
            #FileBrowserPreview {
                background-color: rgba(50, 50, 50, 200);
                color: #aaa;
                border: 1px solid #555;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }
        """

    # ── public API ───────────────────────────────────────────────────────

    def set_directory(self, path: str):
        """Set the directory to list image files from."""
        directory = Path(path)
        self._current_directory = directory
        self._file_list.clear()
        if not directory.exists() or not directory.is_dir():
            return
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.webp', '.lsm'}
        for file_path in sorted(directory.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                size = file_path.stat().st_size
                size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
                item_text = f"{file_path.name} ({size_str})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, str(file_path))
                self._file_list.addItem(item)

    def _on_item_double_clicked(self, item):
        """Emit ``file_selected`` with the double-clicked item's file path."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        self.file_selected.emit(file_path)

    def _on_selection_changed(self):
        """Load a preview for the newly selected file, or clear it if none."""
        current_item = self._file_list.currentItem()
        if current_item:
            file_path = current_item.data(Qt.ItemDataRole.UserRole)
            self._load_preview(file_path)
            self._file_list.scrollToItem(current_item, QListWidget.EnsureVisible)
        else:
            self._preview_label.setPixmap(QPixmap())
            self._preview_label.setText("Select a file to preview")

    def _load_preview(self, file_path: str):
        """Load a thumbnail preview of the image."""
        try:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled = pixmap.scaledToHeight(140, Qt.TransformationMode.SmoothTransformation)
                self._preview_label.setPixmap(scaled)
            else:
                self._preview_label.setText("Cannot load preview")
        except Exception:
            self._preview_label.setText("Error loading preview")

    def show_and_raise(self):
        """Show the panel, bring it to the front, and focus the file list."""
        super().show_and_raise()
        self._file_list.setFocus()
