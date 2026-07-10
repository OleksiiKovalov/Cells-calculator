"""Floating resizable file browser panel that stays within its parent widget."""

from pathlib import Path
from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QFrame,
    QVBoxLayout, QWidget,
)

# ── resize constants ──────────────────────────────────────────────────────────
_RESIZE_MARGIN = 6   # px from edge that triggers resize cursor/drag

_EDGE_LEFT   = 1
_EDGE_RIGHT  = 2
_EDGE_TOP    = 4
_EDGE_BOTTOM = 8


def _edge_at(pos, w, h, margin=_RESIZE_MARGIN):
    """Return a bitmask of which panel edges *pos* is within the resize margin."""
    x, y = pos.x(), pos.y()
    edge = 0
    if x <= margin:      edge |= _EDGE_LEFT
    if x >= w - margin:  edge |= _EDGE_RIGHT
    if y <= margin:      edge |= _EDGE_TOP
    if y >= h - margin:  edge |= _EDGE_BOTTOM
    return edge


def _cursor_for_edge(edge):
    """Return the resize cursor shape appropriate for the given edge bitmask."""
    h = edge & (_EDGE_LEFT | _EDGE_RIGHT)
    v = edge & (_EDGE_TOP  | _EDGE_BOTTOM)
    if h and v:
        nw_se = (h == _EDGE_LEFT  and v == _EDGE_TOP) or \
                (h == _EDGE_RIGHT and v == _EDGE_BOTTOM)
        return Qt.SizeBDiagCursor if nw_se else Qt.SizeFDiagCursor
    if h:
        return Qt.SizeHorCursor
    if v:
        return Qt.SizeVerCursor
    return Qt.ArrowCursor


class FileBrowserPanel(QFrame):
    """
    Floating resizable file browser panel.

    Floats as a child widget over its parent; stays within parent bounds.
    Drag by the title bar.  Resize from any edge or corner.

    Public API
    ----------
    set_directory(path) – set the directory to list files from
    file_selected(path) – signal emitted when file is double-clicked
    """

    panel_moved = Signal()
    file_selected = Signal(str)  # emits file path

    def __init__(self, parent: QWidget | None = None):
        """Initialize the panel, its drag/resize state, and build the UI."""
        super().__init__(parent)
        self.setObjectName("FileBrowserPanel")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumSize(200, 300)
        self.resize(200, 400)

        # drag state
        self._drag_active = False
        self._drag_offset = QPoint()

        # resize state
        self._resize_edge = 0
        self._resize_start_global = QPoint()
        self._resize_start_geom = (0, 0, 0, 0)  # x, y, w, h

        self._current_directory: Path | None = None

        self._build_ui()
        self._apply_style()
        self._install_mouse_filter()

    # ── construction ─────────────────────────────────────────────────────

    def _install_mouse_filter(self):
        """Install this panel as event filter on itself and all children."""
        self.setMouseTracking(True)
        self.installEventFilter(self)
        for child in self.findChildren(QWidget):
            child.setMouseTracking(True)
            child.installEventFilter(self)

    def _build_ui(self):
        """Build the title bar, file list, and preview area."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        self._title_bar = QWidget(self)
        self._title_bar.setObjectName("FileBrowserPanelTitleBar")
        self._title_bar.setFixedHeight(26)
        tbl = QHBoxLayout(self._title_bar)
        tbl.setContentsMargins(8, 0, 4, 0)
        tbl.setSpacing(4)

        self._title_label = QLabel("File Browser", self._title_bar)
        self._title_label.setObjectName("FileBrowserPanelTitle")
        f = QFont()
        f.setBold(True)
        f.setPointSize(9)
        self._title_label.setFont(f)

        self._btn_close = QPushButton("✕", self._title_bar)
        self._btn_close.setObjectName("FileBrowserPanelClose")
        self._btn_close.setFixedSize(20, 20)
        self._btn_close.setFlat(True)
        self._btn_close.clicked.connect(self.hide)

        tbl.addWidget(self._title_label)
        tbl.addStretch()
        tbl.addWidget(self._btn_close)
        root.addWidget(self._title_bar)

        # File list
        self._file_list = QListWidget(self)
        self._file_list.setObjectName("FileBrowserList")
        self._file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._file_list.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._file_list, 1)

        # Preview area
        self._preview_label = QLabel(self)
        self._preview_label.setObjectName("FileBrowserPreview")
        self._preview_label.setFixedHeight(150)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setText("Select a file to preview")
        root.addWidget(self._preview_label)

    def _apply_style(self):
        """Apply the panel's stylesheet."""
        self.setStyleSheet("""
            FileBrowserPanel {
                background-color: rgba(28, 28, 28, 215);
                border: 1px solid #555;
                border-radius: 5px;
            }
            #FileBrowserPanelTitleBar {
                background-color: #3a3a3a;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            #FileBrowserPanelTitle {
                color: #e0e0e0;
            }
            #FileBrowserPanelClose {
                color: #aaa;
                background: transparent;
                border: none;
                font-size: 11px;
            }
            #FileBrowserPanelClose:hover {
                color: #fff;
                background-color: #c0392b;
                border-radius: 3px;
            }
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
        """)

    # ── event filter: drag + all-edge resize ─────────────────────────────

    def eventFilter(self, obj, event):
        """Handle title-bar dragging and edge resizing for the panel and its children."""
        etype = event.type()

        if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = self.mapFromGlobal(obj.mapToGlobal(event.pos()))
            edge = _edge_at(pos, self.width(), self.height())
            if edge:
                self._resize_edge = edge
                self._resize_start_global = event.globalPos()
                self._resize_start_geom = (self.x(), self.y(), self.width(), self.height())
                self.setCursor(_cursor_for_edge(edge))
                return True
            # Title bar drag (exclude close button)
            close_in_panel = self._btn_close.rect().translated(
                self.mapFromGlobal(self._btn_close.mapToGlobal(QPoint(0, 0))))
            if self._title_bar.geometry().contains(pos) and \
                    not close_in_panel.contains(pos):
                self._drag_active = True
                self._drag_offset = event.globalPos() - self.mapToGlobal(QPoint(0, 0))
                return True

        elif etype == QEvent.MouseMove:
            pos = self.mapFromGlobal(obj.mapToGlobal(event.pos()))
            buttons = event.buttons()
            if self._resize_edge and (buttons & Qt.LeftButton):
                self._do_resize(event.globalPos())
                return True
            if self._drag_active and (buttons & Qt.LeftButton):
                new_global = event.globalPos() - self._drag_offset
                if self.parent():
                    pr = self.parent().rect()
                    local = self.parent().mapFromGlobal(new_global)
                    x = max(0, min(local.x(), pr.width()  - self.width()))
                    y = max(0, min(local.y(), pr.height() - self.height()))
                    self.move(x, y)
                else:
                    self.move(new_global)
                return True
            # Idle — show resize cursor near edges
            if not (buttons & Qt.LeftButton):
                edge = _edge_at(pos, self.width(), self.height())
                self.setCursor(_cursor_for_edge(edge))

        elif etype == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._resize_edge:
                self._resize_edge = 0
                pos = self.mapFromGlobal(obj.mapToGlobal(event.pos()))
                self.setCursor(_cursor_for_edge(_edge_at(pos, self.width(), self.height())))
                return True
            if self._drag_active:
                self._drag_active = False
                self.panel_moved.emit()
                return True

        elif etype == QEvent.Leave and obj is self:
            if not self._resize_edge and not self._drag_active:
                self.setCursor(Qt.ArrowCursor)

        return super().eventFilter(obj, event)

    def _do_resize(self, global_pos):
        """Resize the panel for the active edge drag, clamped to the parent bounds."""
        delta = global_pos - self._resize_start_global
        dx, dy = delta.x(), delta.y()
        x0, y0, w0, h0 = self._resize_start_geom
        x, y, w, h = x0, y0, w0, h0

        if self._resize_edge & _EDGE_LEFT:
            new_w = w0 - dx
            if new_w >= self.minimumWidth():
                x, w = x0 + dx, new_w
        if self._resize_edge & _EDGE_RIGHT:
            w = max(self.minimumWidth(), w0 + dx)
        if self._resize_edge & _EDGE_TOP:
            new_h = h0 - dy
            if new_h >= self.minimumHeight():
                y, h = y0 + dy, new_h
        if self._resize_edge & _EDGE_BOTTOM:
            h = max(self.minimumHeight(), h0 + dy)

        if self.parent():
            pr = self.parent().rect()
            if x < 0:
                w = max(self.minimumWidth(), w + x);  x = 0
            if y < 0:
                h = max(self.minimumHeight(), h + y); y = 0
            if x + w > pr.width():
                w = max(self.minimumWidth(), pr.width() - x)
            if y + h > pr.height():
                h = max(self.minimumHeight(), pr.height() - y)

        self.setGeometry(x, y, w, h)

    # ── public API ───────────────────────────────────────────────────────

    def set_title(self, title: str):
        """Change the panel title."""
        self._title_label.setText(title)

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
                item.setData(Qt.UserRole, str(file_path))
                self._file_list.addItem(item)

    def _on_item_double_clicked(self, item):
        """Emit ``file_selected`` with the double-clicked item's file path."""
        file_path = item.data(Qt.UserRole)
        self.file_selected.emit(file_path)

    def _on_selection_changed(self):
        """Load a preview for the newly selected file, or clear it if none."""
        current_item = self._file_list.currentItem()
        if current_item:
            file_path = current_item.data(Qt.UserRole)
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
                # Scale to fit the preview area
                scaled = pixmap.scaledToHeight(140, Qt.SmoothTransformation)
                self._preview_label.setPixmap(scaled)
            else:
                self._preview_label.setText("Cannot load preview")
        except Exception:
            self._preview_label.setText("Error loading preview")

    def show_and_raise(self):
        """Show the panel and bring it to the front."""
        self.show()
        self.raise_()
        self._file_list.setFocus()