from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QListWidget, QListWidgetItem,QDialogButtonBox)
from PyQt5.QtCore import Qt

class ModelsCheckListDialog(QDialog):
    def __init__(self, items, checked_indices=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Models")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Create the list with checkable items
        self.list_widget = QListWidget()
        for i, text in enumerate(items):
            item = QListWidgetItem(text)
            item.setCheckState(2 if checked_indices and i in checked_indices else 0)  # 2 = Qt.Checked, 0 = Qt.Unchecked
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_checked_items(self):
        return [
            (i, self.list_widget.item(i).text())
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == 2
        ]
    def Execute(self):
        return self.exec_() == QDialog.Accepted