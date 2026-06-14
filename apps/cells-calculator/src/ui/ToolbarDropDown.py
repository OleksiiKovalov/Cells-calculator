from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QWidget


class ToolbarDropDown(QFrame):
    """Reusable popup panel that drops down from a toolbar button.

    Usage:
        dropdown = ToolbarDropDown()
        dropdown.set_content(some_widget)

        # in button clicked handler:
        dropdown.popup_below(button)

    The panel closes automatically when the user clicks outside it
    (Qt.Popup window flag behaviour).
    """

    def __init__(self):
        """Build the frameless popup frame with its layout and dark theme styling."""
        # No parent — Qt.Popup is a top-level window; parent would pin geometry
        super().__init__(None, Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("ToolbarDropDown")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)

        self.setStyleSheet("""
            QFrame#ToolbarDropDown {
                background: #2d2d2d;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
            }
            QLabel  { color: #dddddd; }
            QCheckBox { color: #dddddd; spacing: 6px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
            QComboBox {
                color: #dddddd;
                background: #3c3c3c;
                border: 1px solid #5a5a5a;
                border-radius: 3px;
                padding: 2px 6px;
            }
            QSpinBox, QDoubleSpinBox {
                color: #dddddd;
                background: #3c3c3c;
                border: 1px solid #5a5a5a;
                border-radius: 3px;
            }
        """)

    def set_content(self, widget: QWidget) -> None:
        """Replace the dropdown content with *widget*."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self._layout.addWidget(widget)
        self.adjustSize()

    def popup_below(self, button: QWidget) -> None:
        """Show the dropdown immediately below *button*."""
        self.adjustSize()
        global_pos = button.mapToGlobal(QPoint(0, button.height()))
        self.move(global_pos)
        self.show()
        self.raise_()
