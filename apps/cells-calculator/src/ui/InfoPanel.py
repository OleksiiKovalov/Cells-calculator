"""Floating resizable info panel that stays within its parent widget."""

from PyQt5.QtCore import QEvent, QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QFrame, QTextEdit,
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


class InfoPanel(QFrame):
    """
    Floating resizable info panel.

    Floats as a child widget over its parent; stays within parent bounds.
    Drag by the title bar.  Resize from any edge or corner.

    Public API
    ----------
    write(text)       – append a line of text
    set_text(text)    – replace all text
    clear()           – clear all text
    set_title(title)  – change the panel title
    set_wrap(enabled) – toggle line wrapping
    show_and_raise()  – show and bring to front
    """

    panel_moved = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        """Initialize the panel, its drag/resize state, and build the UI."""
        super().__init__(parent)
        self.setObjectName("InfoPanel")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumSize(220, 120)
        self.resize(340, 240)

        # drag state
        self._drag_active = False
        self._drag_offset = QPoint()

        # resize state
        self._resize_edge = 0
        self._resize_start_global = QPoint()
        self._resize_start_geom = (0, 0, 0, 0)  # x, y, w, h

        # line limit (0 = unlimited)
        self._max_lines = 0

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
        """Build the title bar and read-only text area."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        self._title_bar = QWidget(self)
        self._title_bar.setObjectName("InfoPanelTitleBar")
        self._title_bar.setFixedHeight(26)
        tbl = QHBoxLayout(self._title_bar)
        tbl.setContentsMargins(8, 0, 4, 0)
        tbl.setSpacing(4)

        self._title_label = QLabel("Info", self._title_bar)
        self._title_label.setObjectName("InfoPanelTitle")
        f = QFont()
        f.setBold(True)
        f.setPointSize(9)
        self._title_label.setFont(f)

        self._btn_close = QPushButton("✕", self._title_bar)
        self._btn_close.setObjectName("InfoPanelClose")
        self._btn_close.setFixedSize(20, 20)
        self._btn_close.setFlat(True)
        self._btn_close.clicked.connect(self.hide)

        tbl.addWidget(self._title_label)
        tbl.addStretch()
        tbl.addWidget(self._btn_close)
        root.addWidget(self._title_bar)

        # Text area
        self._text_edit = QTextEdit(self)
        self._text_edit.setObjectName("InfoPanelText")
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QTextEdit.NoWrap)
        root.addWidget(self._text_edit, 1)

    def _apply_style(self):
        """Apply the panel's stylesheet."""
        self.setStyleSheet("""
            InfoPanel {
                background-color: rgba(28, 28, 28, 215);
                border: 1px solid #555;
                border-radius: 5px;
            }
            #InfoPanelTitleBar {
                background-color: #3a3a3a;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            #InfoPanelTitle {
                color: #e0e0e0;
            }
            #InfoPanelClose {
                color: #aaa;
                background: transparent;
                border: none;
                font-size: 11px;
            }
            #InfoPanelClose:hover {
                color: #fff;
                background-color: #c0392b;
                border-radius: 3px;
            }
            #InfoPanelText {
                background-color: transparent;
                color: #d4d4d4;
                border: none;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 10pt;
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

    def write(self, text: str):
        """Append a line of text to the panel, trimming oldest lines if needed."""
        self._text_edit.append(text)
        if self._max_lines > 0:
            doc = self._text_edit.document()
            while doc.blockCount() > self._max_lines:
                cursor = self._text_edit.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.select(cursor.BlockUnderCursor)
                cursor.removeSelectedText()
                # also remove the trailing newline that was part of that block
                cursor.deleteChar()

    def set_text(self, text: str):
        """Replace all text in the panel."""
        self._text_edit.setPlainText(text)

    def clear(self):
        """Clear all text."""
        self._text_edit.clear()

    def set_wrap(self, enabled: bool):
        """Enable or disable line wrapping in the text area."""
        mode = QTextEdit.WidgetWidth if enabled else QTextEdit.NoWrap
        self._text_edit.setLineWrapMode(mode)

    def set_max_lines(self, max_lines: int):
        """Set the maximum number of lines kept in the panel (0 = unlimited)."""
        self._max_lines = max(0, max_lines)
        if self._max_lines > 0:
            doc = self._text_edit.document()
            while doc.blockCount() > self._max_lines:
                cursor = self._text_edit.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.select(cursor.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()

    def show_and_raise(self):
        """Show the panel and bring it to the front."""
        self.show()
        self.raise_()
