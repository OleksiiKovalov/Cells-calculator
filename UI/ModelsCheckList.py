"""
Dialog for selecting models from a checklist.

Provides a PyQt5 dialog for user selection of ground truth and available
models using checkboxes and combo boxes. Allows configuration of which
models to enable for processing.

Key components:
- ModelsCheckListDialog: Dialog with model selection widgets
"""

# Third-party imports
from typing import List, Tuple, Optional
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QListWidgetItem,
    QDialogButtonBox, QComboBox, QLabel
)

class ModelsCheckListDialog(QDialog):
    """
    Dialog for selecting models from a checklist.
    """

    def __init__(self, items: List[str], checked_indices: Optional[List[int]] = None, parent=None) -> None:
        """
        Initialize the dialog.

        Args:
            items: List of model names.
            checked_indices: List of indices that should be checked.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Select Models")
        self.setModal(True)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("ground truth model"))
        self.ground_model = QComboBox()
        layout.addWidget(self.ground_model)

        layout.addWidget(QLabel("available models"))
        self.list_widget = QListWidget()
        
        for i, text in enumerate(items):
            item = QListWidgetItem(text)
            item.setCheckState(
                Qt.CheckState.Checked if checked_indices and i in checked_indices else Qt.CheckState.Unchecked
            )
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.list_widget.addItem(item)
            self.ground_model.addItem(text)
        layout.addWidget(self.list_widget)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_checked_items(self) -> List[Tuple[int, str]]:
        """
        Get the list of checked items.

        Returns:
            List of tuples (index, text) for checked items.
        """
        checked_items = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                checked_items.append((i, item.text()))
        return checked_items

    def get_ground_truth_model(self) -> str:
        """
        Get the selected ground truth model.

        Returns:
            str: The current text of the ground truth model combo box.
        """
        return self.ground_model.currentText()

    def execute(self, default_ground_model: Optional[str] = None) -> bool:
        """
        Execute the dialog.

        Args:
            default_ground_model: Default model to select.

        Returns:
            bool: True if accepted.
        """
        if default_ground_model is not None:
            self.ground_model.setCurrentText(default_ground_model)
        return self.exec_() == QDialog.Accepted