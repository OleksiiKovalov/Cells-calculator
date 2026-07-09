import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QTreeWidgetItemIterator, QLabel, QAbstractItemView,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class FileBrowser(QWidget):
    """Floating/dockable panel showing dataset files grouped by split."""

    image_selected = Signal(str, list)   # (image_path, annotations)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loader = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.tree)

        self.info = QLabel("No dataset loaded")
        self.info.setAlignment(Qt.AlignCenter)
        small = QFont()
        small.setPointSize(8)
        self.info.setFont(small)
        layout.addWidget(self.info)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def load_dataset(self, loader):
        self._loader = loader
        self.tree.clear()

        splits = loader.get_splits()
        if splits:
            total = 0
            for split in splits:
                images = loader.get_images(split=split)
                if not images:
                    continue
                header = QTreeWidgetItem(self.tree)
                header.setText(0, f"{split}   ({len(images)} images)")
                bold = QFont()
                bold.setBold(True)
                header.setFont(0, bold)
                header.setFlags(header.flags() & ~Qt.ItemIsSelectable)
                for img in images:
                    child = QTreeWidgetItem(header)
                    child.setText(0, os.path.basename(img['path']))
                    child.setData(0, Qt.ItemDataRole.UserRole, img)
                    child.setToolTip(0, img['path'])
                header.setExpanded(True)
                total += len(images)
            self.info.setText(f"{total} images · {len(splits)} splits")
        else:
            images = loader.get_images()
            for img in images:
                item = QTreeWidgetItem(self.tree)
                item.setText(0, os.path.basename(img['path']))
                item.setData(0, Qt.ItemDataRole.UserRole, img)
                item.setToolTip(0, img['path'])
            self.info.setText(f"{len(images)} images")

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _on_clicked(self, item: QTreeWidgetItem, _col: int = 0):
        self._emit_image(item)

    def _on_current_changed(self, current: QTreeWidgetItem, _prev):
        """Fires on keyboard navigation so arrow keys browse images."""
        if current is not None:
            self._emit_image(current)

    def _emit_image(self, item: QTreeWidgetItem):
        img_info = item.data(0, Qt.ItemDataRole.UserRole)
        if not img_info:
            return
        path = img_info['path']
        if not os.path.isfile(path):
            return
        anns = self._loader.get_annotations(path) if self._loader else []
        self.image_selected.emit(path, anns)

    # ------------------------------------------------------------------
    # Navigation helpers (called from main window shortcuts)
    # ------------------------------------------------------------------
    def select_offset(self, delta: int):
        """Move selection by delta items (skip non-selectable headers)."""
        items = self._selectable_items()
        if not items:
            return
        cur = self.tree.currentItem()
        try:
            idx = items.index(cur)
        except ValueError:
            idx = -1
        new_idx = max(0, min(len(items) - 1, idx + delta))
        self.tree.setCurrentItem(items[new_idx])

    def _selectable_items(self) -> list:
        result = []
        it = QTreeWidgetItemIterator(self.tree, QTreeWidgetItemIterator.IteratorFlag.Selectable)
        while it.value():
            result.append(it.value())
            it += 1
        return result
