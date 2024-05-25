import sys
from PyQt5.QtWidgets import QMessageBox,QGraphicsPixmapItem,QApplication, QMainWindow, QAction, QGraphicsView, QGraphicsScene, QVBoxLayout, QWidget, QFileDialog, QGraphicsTextItem, QComboBox, QLabel, QHBoxLayout,QPushButton
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
import numpy as np
import os
import tifffile

def has_duplicates(lst):
        seen = set()
        for item in lst:
            if item in seen:
                return True
            seen.add(item)
        return False
class DialogWindow(QMainWindow):
    def show_warning_dialog(self, text):
    # Создаем диалоговое окно предупреждения
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(text)
        #msgBox.setInformativeText("Choose method and file.")
        msgBox.setWindowTitle("Warning")
        msgBox.adjustSize()
        msgBox.exec_()
        if self.first_warning and self.call_back:
            print(self.call_back)
            self.first_warning = 0
            self.show_warning_dialog("If you have problem to choose Channel press Cancel\n\n It will calculate with standart parameters")
    def __init__(self,  parametrs : dict, lsm_path : str, parent=None,call_back = None  ):
        
        super().__init__(parent)
        self.parent_ = parent
        self.warning_count = 0
        self.first_warning = 1

        self.call_back = call_back
        self.setStyleSheet('''
            QWidget {
                font-size: 32px;
            }
        ''')
        self.lsm_list = None
        self.parametrs = parametrs
        self.number = 0
        if isinstance(lsm_path, list):
            self.lsm_path = lsm_path [0]
            self.lsm_list = lsm_path
        else:
            self.lsm_path = lsm_path
        self.num_channels = 0
        self.setWindowTitle("Dialog Window")
        #self.setMinimumSize()
        self.initUI()
    def closeEvent(self, event):
        #добавить логику для call_back функции
        
        pass

    def initUI(self):
        
        self.parent_width = self.parent().width()
        self.parent_height = self.parent().height()
        self.setFixedSize(self.parent_width * 0.75, self.parent_height * 0.75)
        #self.setFixedSize(parent_width * 0.75, parent_height * 0.75)  #self.setFixedSize
        #self.resize(parent_width * 0.75, parent_height * 0.75)
        self.setWindowTitle(f'Settings - {os.path.basename( self.lsm_path)}')
        #self.center()
        self.makeScen()
        
    def makeScen(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        self.scene = QGraphicsScene()
        
        self.view = QGraphicsView(self.scene)
        #self.view.setFixedWidth(int(self.width()*0.75))
        self.view.setFixedWidth(int(self.parent_width*0.75*0.75))
        self.view.setFixedHeight(int(self.parent_height*0.70))
        #self.view.setFixedSize(int(self.width()*0.75),int(self.height()*0.75))
        right_layout = QVBoxLayout()
        self.combo_box_dict = None
        try:
            self.add_images()
            
        except:
            if self.warning_count == 0:
                self.show_warning_dialog("Error during layout images\n\nUse Next button")
                self.warning_count = 1
            if not self.call_back and self.warning_count == 0 :
                self.show_warning_dialog("Error during layout images")
                self.warning_count = 1
            self.num_channels = 5
            self.scene.clear()
        options = list(self.parametrs.keys())
        for option in options:
            
            self.num_channels = max (self.num_channels, self.parametrs[option]+1 )

        
        self.combo_box_dict = self.parametrs.copy()
        for option in options:
            label = QLabel(option)
            #label.setContentsMargins(0, 0, 0, 0)
            combo_box = QComboBox()
            
            #combo_box.setContentsMargins(0, 0, 0, 0)
            combo_box.addItems([f'Channel {i+1}'for i in range(self.num_channels)])
            combo_box.setCurrentText(f"Channel {self.parametrs[option]+1}")
            
           
            self.combo_box_dict[option] = combo_box
            label_combo_layout = QHBoxLayout()
            #label_combo_layout.setContentsMargins(0, 0, 0, 0)
            label_combo_layout.addWidget(label)
            label_combo_layout.addWidget(combo_box)
    
            right_layout.addLayout(label_combo_layout)
        
        buttons_layout = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_action)

        choose_button = QPushButton("Choose")
        choose_button.clicked.connect(self.choose_function)

        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(choose_button)
        
        right_layout.addLayout(buttons_layout)
        
        buttons_next_layout = QHBoxLayout()

        next_button = QPushButton("Next")
        if not self.lsm_list:
            next_button.setEnabled(False)
        next_button.clicked.connect(self.next_action)
        previous_button = QPushButton("Previous")
        
        buttons_next_layout.addWidget(next_button)
        #buttons_next_layout.addWidget(previous_button)
        
        right_layout.addLayout(buttons_next_layout)
        main_layout.addWidget(self.view)
        main_layout.addLayout(right_layout)
       #main_layout.addWidget(spacer)
        central_widget.setLayout(main_layout)
    def cancel_action(self):
        
        if self.call_back:
            self.call_back()
        self.close()
    def choose_function(self):
        #check_parametrs(self.combo_box_dict)
        self.parent_.draw_bounding = 0
        options_list = []
        for option,combo_box in self.combo_box_dict.items():
            options_list.append(int (combo_box.currentText()[len("Channel "):]) -1)
        if has_duplicates(options_list):
            self.show_warning_dialog("Channels can not be dublicated")
            return
        for option,combo_box in self.combo_box_dict.items():
            self.parametrs[option] = int (combo_box.currentText()[len("Channel "):]) -1

        #if okay
        if self.call_back:
            self.call_back()
        self.close()
    def next_action(self):
        self.number += 1
        if self.number == len(self.lsm_list):
            self.number = 0
        self.lsm_path = self.lsm_list[self.number]
        try:
            self.add_images()
        except:
            if self.warning_count == 0:
                self.warning_count = 1
                self.show_warning_dialog("Error during layout images\n\nUse Next button")
          
            
            self.scene.clear()
        self.setWindowTitle(f'Settings - {os.path.basename( self.lsm_path)}')
        
    def add_images(self):
        self.scene.clear()
        with tifffile.TiffFile(self.lsm_path) as tif:
            lsm = tif.pages[0].asarray()
        self.num_channels = lsm.shape[0] #* 100
        if self.combo_box_dict:
            options = list(self.parametrs.keys())
            for option in options:
                current_amount = self.combo_box_dict[option].count()
                if len(self.combo_box_dict[option])< self.num_channels:
                    self.combo_box_dict[option].addItems([f'Channel {i}'for i in range(current_amount+1,self.num_channels+1)])
        # Добавление QPixmap и текста в сцену
        image_width = int(self.parent_width * 0.75 * 0.75 / 2)  # Ширина изображения на экране
        image_height = int(self.parent_height * 0.75 / 2)  # Высота изображения на экране
        
        for i in range(self.num_channels):
            pixmap = QPixmap.fromImage(QImage(lsm[i], lsm.shape[1], lsm.shape[2], QImage.Format_Grayscale8))
            pixmap_item = QGraphicsPixmapItem(pixmap)
            current_image_height = (i // 2) * (image_height + 40)
            pixmap_item.setPos((i % 2) * image_width, current_image_height )
            pixmap_item.setScale(min(image_width / pixmap.width(), image_height / pixmap.height()))
            self.scene.addItem(pixmap_item)
            
            # Добавляем текст "Channel X" под каждым изображением
            text_item = QGraphicsTextItem(f"Channel {i+1}")
            text_item.setPos((i % 2) * (image_width + 20) + image_width /4, current_image_height + image_height    )
            self.scene.addItem(text_item)
          
        
    def center(self):
        parent_rect = self.parent().frameGeometry()
        dialog_rect = self.frameGeometry()
        dialog_rect.moveCenter(parent_rect.center())
        self.move(dialog_rect.topLeft())
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DialogWindow()
    window.showMaximized()
    sys.exit(app.exec_())