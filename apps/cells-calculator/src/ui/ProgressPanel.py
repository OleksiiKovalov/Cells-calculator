"""Floating resizable progress panel for long-running operations."""

from PySide6.QtCore import QElapsedTimer, QPoint, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)


class _ResizeGrip(QWidget):
    """Bottom-right corner grip that resizes its owning panel by dragging."""

    def __init__(self, panel):
        """Initialize the grip for *panel* and reset its drag state."""
        super().__init__(panel)
        self._panel = panel
        self.setFixedSize(14, 14)
        self.setCursor(Qt.SizeFDiagCursor)
        self._dragging = False
        self._start_global = QPoint()
        self._start_size = QSize()

    def mousePressEvent(self, event):
        """Begin a resize drag, recording the panel's starting size and cursor position."""
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._start_global = event.globalPos()
            self._start_size = self._panel.size()
            event.accept()

    def mouseMoveEvent(self, event):
        """Resize the panel to follow the cursor, clamped to its minimum size and parent bounds."""
        if self._dragging and event.buttons() & Qt.LeftButton:
            delta = event.globalPos() - self._start_global
            new_w = max(self._panel.minimumWidth(), self._start_size.width() + delta.x())
            new_h = max(self._panel.minimumHeight(), self._start_size.height() + delta.y())
            if self._panel.parent():
                pr = self._panel.parent().rect()
                new_w = min(new_w, pr.width() - self._panel.x())
                new_h = min(new_h, pr.height() - self._panel.y())
            self._panel.resize(new_w, new_h)
            event.accept()

    def mouseReleaseEvent(self, event):
        """End the resize drag."""
        self._dragging = False

    def paintEvent(self, event):
        """Paint the diagonal grip lines."""
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#888"), 1))
        w, h = self.width() - 2, self.height() - 2
        for i in range(2, w + 1, 4):
            painter.drawLine(i, h, w, i)


class ProgressPanel(QFrame):
    """
    Floating resizable progress panel.

    Drag by the title bar; resize via the bottom-right grip.
    Not a dialog — stays as a child widget over the viewer.
    """

    cancel_requested = Signal()
    panel_moved = Signal()

    def __init__(self, parent=None):
        """Initialize the panel, its drag state and elapsed timers, and build the UI."""
        super().__init__(parent)
        self.setObjectName("ProgressPanel")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumSize(260, 190)
        self.resize(320, 210)

        self._drag_active = False
        self._drag_offset = QPoint()

        self._elapsed_timer = QElapsedTimer()
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(100)
        self._tick_timer.timeout.connect(self._update_elapsed)

        self._build_ui()
        self._apply_style()

    # ── construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        """Build the title bar, status/elapsed/expected labels, cancel button, and resize grip."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        self._title_bar = QWidget(self)
        self._title_bar.setObjectName("ProgressTitleBar")
        self._title_bar.setFixedHeight(26)
        tbl = QHBoxLayout(self._title_bar)
        tbl.setContentsMargins(8, 0, 6, 0)
        tbl.setSpacing(4)

        self._title_label = QLabel("Processing…", self._title_bar)
        self._title_label.setObjectName("ProgressTitle")
        f = QFont()
        f.setBold(True)
        f.setPointSize(9)
        self._title_label.setFont(f)
        tbl.addWidget(self._title_label)
        tbl.addStretch()
        root.addWidget(self._title_bar)

        # Body
        body = QWidget(self)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(10, 8, 10, 4)
        bl.setSpacing(6)

        self._status_label = QLabel("Starting…", body)
        self._status_label.setObjectName("ProgressStatus")
        self._status_label.setWordWrap(True)
        bl.addWidget(self._status_label)

        self._elapsed_label = QLabel("Elapsed:  0.0s", body)
        self._elapsed_label.setObjectName("ProgressElapsed")
        bl.addWidget(self._elapsed_label)

        self._expected_label = QLabel("Expected: unknown", body)
        self._expected_label.setObjectName("ProgressExpected")
        bl.addWidget(self._expected_label)

        bl.addStretch()

        self._btn_cancel = QPushButton("Cancel", body)
        self._btn_cancel.setObjectName("ProgressCancel")
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        bl.addWidget(self._btn_cancel)

        root.addWidget(body, 1)

        # Bottom resize grip
        bottom = QWidget(self)
        btl = QHBoxLayout(bottom)
        btl.setContentsMargins(0, 0, 0, 0)
        btl.addStretch()
        self._grip = _ResizeGrip(self)
        btl.addWidget(self._grip)
        root.addWidget(bottom)

    def _apply_style(self):
        """Apply the panel's stylesheet."""
        self.setStyleSheet("""
            ProgressPanel {
                background-color: rgba(28, 28, 28, 220);
                border: 1px solid #555;
                border-radius: 5px;
            }
            #ProgressTitleBar {
                background-color: #2c4a6e;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            #ProgressTitle {
                color: #e0e8f0;
            }
            #ProgressStatus {
                color: #c0d0e0;
                font-size: 10pt;
            }
            #ProgressElapsed {
                color: #a0c0a0;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
            #ProgressExpected {
                color: #909090;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
            #ProgressCancel {
                background-color: #6e2c2c;
                color: #f0d0d0;
                border: none;
                border-radius: 3px;
                padding: 4px 12px;
                font-size: 10pt;
            }
            #ProgressCancel:hover { background-color: #c0392b; color: white; }
            #ProgressCancel:disabled { background-color: #444; color: #777; }
        """)

    # ── drag ──────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        """Begin a title-bar drag if the press lands on the title bar."""
        if event.button() == Qt.LeftButton and \
                self._title_bar.geometry().contains(event.pos()):
            self._drag_active = True
            self._drag_offset = event.globalPos() - self.mapToGlobal(QPoint(0, 0))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Move the panel to follow a title-bar drag, clamped to the parent bounds."""
        if self._drag_active and event.buttons() & Qt.LeftButton:
            new_global = event.globalPos() - self._drag_offset
            if self.parent():
                pr = self.parent().rect()
                new_local = self.parent().mapFromGlobal(new_global)
                x = max(0, min(new_local.x(), pr.width() - self.width()))
                y = max(0, min(new_local.y(), pr.height() - self.height()))
                self.move(x, y)
            else:
                self.move(new_global)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End a title-bar drag and emit ``panel_moved``."""
        if self._drag_active:
            self._drag_active = False
            self.panel_moved.emit()
        super().mouseReleaseEvent(event)

    # ── public API ────────────────────────────────────────────────────────

    def start(self, expected_seconds: float = 0.0):
        """Start the elapsed timer and show expected duration."""
        self._elapsed_timer.start()
        self._tick_timer.start()
        self._btn_cancel.setEnabled(True)
        self._btn_cancel.setText("Cancel")
        self._elapsed_label.setText("Elapsed:  0.0s")
        self.set_status("Running…")
        self.set_expected_duration(expected_seconds)

    def stop(self):
        """Stop the elapsed timer."""
        self._tick_timer.stop()

    def set_title(self, title: str):
        """Change the panel title."""
        self._title_label.setText(title)

    def set_status(self, text: str):
        """Set the status line text."""
        self._status_label.setText(text)

    def set_expected_duration(self, seconds: float):
        """Show the expected duration, or 'unknown' when *seconds* is not positive."""
        if seconds > 0:
            self._expected_label.setText(f"Expected: ~{seconds:.1f}s")
        else:
            self._expected_label.setText("Expected: unknown")

    def elapsed_seconds(self) -> float:
        """Return seconds elapsed since the timer was started."""
        return self._elapsed_timer.elapsed() / 1000.0

    # ── internals ─────────────────────────────────────────────────────────

    def _update_elapsed(self):
        """Refresh the elapsed-time label (called on each timer tick)."""
        self._elapsed_label.setText(f"Elapsed:  {self.elapsed_seconds():.1f}s")

    def _on_cancel_clicked(self):
        """Disable the cancel button, show a cancelling state, and emit ``cancel_requested``."""
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setText("Cancelling…")
        self.set_status("Cancelling — waiting for inference to finish…")
        self.cancel_requested.emit()
