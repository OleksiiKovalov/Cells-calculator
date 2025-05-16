from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QListWidget, QListWidgetItem,QDialogButtonBox,QComboBox, QLabel)
from PyQt5.QtCore import Qt

class ModelsCheckListDialog(QDialog):
    def __init__(self, items, checked_indices=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Models")
        self.setModal(True)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(text="ground truth model"))
        self.ground_model = QComboBox()
        layout.addWidget(self.ground_model)
        
        
        layout.addWidget(QLabel(text="available models"))
        self.list_widget = QListWidget()
        
        for i, text in enumerate(items):
            item = QListWidgetItem(text)
            item.setCheckState(2 if checked_indices and i in checked_indices else 0)  # 2 = Qt.Checked, 0 = Qt.Unchecked
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            self.list_widget.addItem(item)
            self.ground_model.addItem(text)
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
    def getGroundTruthModel(self):
        return self.ground_model.currentText()
    
    def Execute(self,defaultgroundmodel = None):
        if defaultgroundmodel is not None:
            self.ground_model.setCurrentText(defaultgroundmodel)
        return self.exec_() == QDialog.Accepted