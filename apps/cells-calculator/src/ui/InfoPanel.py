"""Floating resizable info panel that stays within its parent widget."""

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget

from ui.FloatingPanel import FloatingResizablePanel


class InfoPanel(FloatingResizablePanel):
    """
    Floating resizable info panel.

    Public API
    ----------
    write(text)       – append a line of text
    set_text(text)    – replace all text
    clear()           – clear all text
    set_wrap(enabled) – toggle line wrapping
    set_max_lines(n)  – cap the number of retained lines (0 = unlimited)
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the panel with a read-only text area."""
        self._max_lines = 0  # 0 = unlimited
        super().__init__(parent, title="Info",
                         min_size=(220, 120), initial_size=(340, 240))

    def _build_content(self, root: QVBoxLayout):
        """Build the read-only text area."""
        self._text_edit = QTextEdit(self)
        self._text_edit.setObjectName("InfoPanelText")
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QTextEdit.NoWrap)
        root.addWidget(self._text_edit, 1)

    def _content_stylesheet(self) -> str:
        """Text-area styling."""
        return """
            #InfoPanelText {
                background-color: transparent;
                color: #d4d4d4;
                border: none;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 10pt;
            }
        """

    # ── public API ───────────────────────────────────────────────────────

    def write(self, text: str):
        """Append a line of text to the panel, trimming oldest lines if needed."""
        self._text_edit.append(text)
        self._trim_to_max_lines()

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
        self._trim_to_max_lines()

    def _trim_to_max_lines(self):
        """Drop the oldest lines until the document fits within ``_max_lines``."""
        if self._max_lines <= 0:
            return
        doc = self._text_edit.document()
        while doc.blockCount() > self._max_lines:
            cursor = self._text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            # also remove the trailing newline that was part of that block
            cursor.deleteChar()
