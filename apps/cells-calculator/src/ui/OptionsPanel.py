"""Floating resizable options panel that stays within its parent widget."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget

from ui.FloatingPanel import FloatingResizablePanel


class OptionsPanel(FloatingResizablePanel):
    """
    Floating resizable options panel.

    Public API
    ----------
    add_widget(widget)  – add any QWidget to the panel body
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the panel with a scrollable content area."""
        super().__init__(parent, title="Options",
                         min_size=(200, 100), initial_size=(260, 280))

    def _build_content(self, root: QVBoxLayout):
        """Build the scrollable content area."""
        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("OptionsPanelScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._content_widget = QWidget()
        self._content_widget.setObjectName("OptionsPanelContent")
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(6)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content_widget)
        root.addWidget(self._scroll, 1)

    def _content_stylesheet(self) -> str:
        """Scroll-area and control styling."""
        return """
            #OptionsPanelScroll {
                background: transparent;
            }
            #OptionsPanelContent {
                background: transparent;
            }
            QLabel {
                color: #dddddd;
            }
            QCheckBox {
                color: #dddddd;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
        """

    # ── public API ───────────────────────────────────────────────────────

    def add_widget(self, widget: QWidget) -> None:
        """Append a widget to the panel body (above the trailing stretch)."""
        count = self._content_layout.count()
        self._content_layout.insertWidget(count - 1, widget)
        # Extend mouse tracking / event filter to the new widget.
        self._track_child(widget)
