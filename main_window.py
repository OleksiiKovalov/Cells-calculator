import sys
from PyQt5.QtWidgets import QAbstractItemView,QSizePolicy,QGraphicsProxyWidget, QGraphicsRectItem,QHeaderView, QMessageBox,QTableWidget, QTableWidgetItem, QPushButton,QGraphicsView, QApplication, QMainWindow, QAction, QGraphicsView, QGraphicsScene, QVBoxLayout, QWidget, QFileDialog, QGraphicsTextItem, QComboBox, QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont,QColor, QPen
from PyQt5.QtCore import Qt
import numpy as np
import tifffile
from cahnal_setting import DialogWindow
from calculate_functions import metod1, metod2, metod3
from model.Model import Model as model1
from table import calculate_table
import os
class MainWindow(QMainWindow):
    
    parametrs = {'Cell' : 0,
                 'Nuclei': 1
                 
                 }
    
    metods = {
        'Model 1': model1().calculate,
        'Model 2': metod2,
        'Model 3': metod3
       
    }
    lsm_path = None
    lsm_filesList = None
    folder_path = None
    def __init__(self):
        
        super().__init__()
        
        desktop = QApplication.desktop()
        screen_geometry = desktop.availableGeometry()

        self.setFixedSize(screen_geometry.width(), desktop.availableGeometry().height()-self.menuBar().height())

        
        self.setWindowTitle("Cells Calculator")
        #self.setMinimumSize
        
        self.initUI()

    def initUI(self):
        #self.setGeometry(100, 100, 800, 600)
        self.df = None
        # Create a menu action to open LSM file
        self.open_lsm_action = QAction("Open LSM File", self)
        self.open_lsm_action.triggered.connect(self.open_lsm)

        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        self.settings_action = QAction("Settings", self)
        self.settings_action.setEnabled(False)
        self.settings_action.triggered.connect(self.open_dialogWindow)

        self.save_as_action = QAction("Save As", self)
        self.save_as_action.setEnabled(False)
        self.save_as_action.triggered.connect(self.save_as)
        # Добавление действия в меню
        
        # Create a menu bar
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        file_menu.addAction(self.open_lsm_action)
        file_menu.addAction(self.open_folder_action) 
        file_menu.addAction(self.save_as_action)
        settings_menu = menubar.addMenu("Settings")
        settings_menu.addAction(self.settings_action)
        self.init_mainScen()
    def init_mainView(self):
  
        self.main_view.setFixedWidth(int(self.width() * 0.75))
        #self.main_view.setFixedHeight(int(self.height()*0.95))
        #self.main_view.setMaximumHeight(int(self.height()))
        
    def save_as(self):
        if  self.df is None:
            self.show_warning_dialog("Nothing to Save")
            self.save_as_action.setEnabled(False)
            return
        try:
            ptions = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Excel Files (*.xlsx);;CSV Files (*.csv)")
            if file_name:
                if file_name.endswith('.xlsx'):
                    self.df.to_excel(file_name, index=False)
                elif file_name.endswith('.csv'):
                    self.df.to_csv(file_name, index=False)
        except:
            self.show_warning_dialog("Error during saving table")
            self.save_as_action.setEnabled(False)
            return
    def init_rightLayout(self):
        self.combo_box = QComboBox()
        label = QLabel("Choose model:")
        self.right_scene = QGraphicsScene()
      
        self.right_view = QGraphicsView(self.right_scene)
        self.right_view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        self.right_button = QPushButton("Calculate")
        self.right_button.setEnabled(False)
        self.save_button =  QComboBox()
        self.save_button.addItems(['Save_as....','excel','csv'])

        self.save_button.setFont(QFont("Arial", 32))
        self.combo_box.setFont(QFont("Arial", 24)) 
        self.combo_box.addItems(['All_metod'])
        self.combo_box.addItems([key for key in self.metods])
        self.combo_box.currentTextChanged.connect(self.selection_changed)
        self.combo_box.setCurrentIndex(-1)
        label.setFont(QFont("Arial", 32))
        self.right_button.setFont(QFont("Arial", 32))
        self.right_button.clicked.connect(self.calculate_button)
        
        
        

        self.right_layout.addWidget(label)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.combo_box)
        self.right_layout.addSpacing(20)
      
        self.right_layout.addWidget(self.right_view)
        self.right_layout.addSpacing(20)
        self.right_layout.addWidget(self.right_button)
        self.right_layout.addSpacing(20)
        #self.right_layout.addWidget(self.save_button)
        
        
        #self.right_layout.addSpacing(20)
        
        #max_width = self.right_view.maximumWidth()
        #max_height = self.right_view.maximumHeight()

# Установка размера сцены равным максимальной ширине и высоте
        #self.right_scene.setSceneRect(0, 0, max_width, max_height)
        
    def init_mainScen(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()

        self.main_scen = QGraphicsScene()
        self.main_view = QGraphicsView(self.main_scen)
        self.main_view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.init_mainView()

        self.right_layout = QVBoxLayout()
        self.init_rightLayout()
        
        self.main_layout.addWidget(self.main_view)
        self.main_layout.addLayout(self.right_layout)
        self.central_widget.setLayout(self.main_layout)
        self.main_scen.setSceneRect(0, 0, self.main_view.width()-10, self.main_view.height())
        
       
        
        
        
    def show_warning_dialog(self, text):
    # Создаем диалоговое окно предупреждения
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(text)
        #msgBox.setInformativeText("Choose method and file.")
        msgBox.setWindowTitle("Warning")
        msgBox.adjustSize()
        msgBox.exec_()
    
    def calculate_button(self):
        metod = self.combo_box.currentText()
        if metod == "" or self.lsm_path is None:
            self.show_warning_dialog("Warning\n\nChoose method and file.")
            print("choose metod and fille")
            return 0
           
        
        
            
        #self.right_view.setFixedSize(451, 661)
        if metod != "All_metod":
            try:
                result = self.metods[metod](img_path = self.lsm_path, cell_channel=self.parametrs['Cell'], nuclei_channel=self.parametrs['Nuclei'])
            except:
               
                self.show_warning_dialog("Error during calculation \n\nChoose another model or change channels settings")
                result = None
            if not result:
              
                return 0
            label_cells = QGraphicsTextItem(f'Cells: {result["Cells"]}')
            label_cells.setFont(QFont('Arial',24))
            label_cells.setPos(0, 0)
            label_nuclei = QGraphicsTextItem(f'Nuclei: {result["Nuclei"]}')
            label_nuclei.setFont(QFont('Arial',24))
            label_nuclei.setPos(0, 100)
            label_alive = QGraphicsTextItem(f'Alive: {result["%"]}%')
            label_alive.setFont(QFont('Arial',24))
            label_alive.setPos(0, 200)
            
            self.right_scene.clear()
            self.right_scene.addItem(label_cells)

            self.right_scene.addItem(label_nuclei)

            self.right_scene.addItem(label_alive)
        else:
            table = QTableWidget()
            view_width = self.right_view.viewport().width()
            view_height = self.right_view.viewport().height()
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            columns = ['Cells', 'Nuclei', 'Alive']
            table.setRowCount(len(self.metods))
            table.setColumnCount(4)

            table.setHorizontalHeaderLabels(['Model'] + columns)
            row = 0
            for metod_name, metod in self.metods.items():
                colum_number = 0
                table.setItem(row, colum_number, QTableWidgetItem(metod_name))
                colum_number +=1
                if metod_name == "All_metod":
                    continue
                try:
                    result = metod(img_path = self.lsm_path, cell_channel=self.parametrs['Cell'], nuclei_channel=self.parametrs['Nuclei'])
                except:
                    
                    result = None
                if result:
                    row_toAdd = []
                    for colum in columns:
                        if colum == "Alive":
                            row_toAdd = f"{result['%']}%"
                        else:
                            row_toAdd = f"{result[colum]}"
                        table.setItem(row, colum_number,QTableWidgetItem(row_toAdd) )
                        colum_number+=1
                else:
                    row_toAdd = "-"
                
                
                row+=1
           
            # Устанавливаем размеры таблицы
            table.setMinimumSize(view_width, view_height)
            #table.setMaximumSize(view_width, view_height)
            #table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
            #table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.resizeRowsToContents()
            table.resizeColumnsToContents()
            self.right_scene.clear()
            self.right_scene.addWidget(table)
        #self.right_view.fitInView(self.right_scene.sceneRect())#, Qt.KeepAspectRatio)
        #self.right_view.adjustSize()
       
        
        
    def selection_changed(self, text):
        print(text)
        
    def open_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", "")
        if self.folder_path:
            self.lsm_filesList = [os.path.join(self.folder_path, file) for file in os.listdir(self.folder_path) if file.endswith('.lsm')]

            if self.lsm_filesList:
                self.main_scen.clear()
                self.lsm_path = None
                self.settings_action.setEnabled(True)
                self.save_as_action.setEnabled(True)
                self.right_button.setEnabled(False)
                self.open_dialogWindow()

                
                
                
                
                self.setWindowTitle(f"Cells Calculator - {os.path.basename(self.folder_path)}/")   
            else:
                self.folder_path = None
                self.show_warning_dialog("No LSM files found in the selected folder")
                print("No LSM files found in the selected folder.")
                #self.setWindowTitle(f"Cells Calculator")
            
    def create_table(self):
       
        if not self.lsm_filesList:
            return
        self.main_scen.clear()
        try:
            table = QTableWidget()
            #self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
           
            try:
                df = calculate_table(metod_dict=self.metods,files_name=self.lsm_filesList,parametrs=self.parametrs)
            except:
                self.settings_action.setEnabled(False)
                self.save_as_action.setEnabled(False)
                self.lsm_filesList = None
                self.show_warning_dialog("Error during calculation.")
                return

            self.df = df
            view_width = self.main_view.viewport().width()
            view_height = self.main_view.viewport().height()
            cell_width = table.horizontalHeader().defaultSectionSize()
            cell_height = table.verticalHeader().defaultSectionSize()
            #rows_to_add = int(view_height / cell_height) - len(df.index()) - 1
            #columns_to_add = int(view_width / cell_width) - len(df.columns())

        # Устанавливаем количество строк и столбцов в таблице
        
            #table.setRowCount(df.index() + rows_to_add)
            #table.setColumnCount(columns + columns_to_add)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)

            #for row in range(rows):
            #    for col in range(columns):
            #        item = QTableWidgetItem()
            #        table.setItem(row, col, item)
            #
            table.setRowCount(df.shape[0])
            table.setColumnCount(df.shape[1])

            # Устанавливаем заголовки столбцов
            table.setHorizontalHeaderLabels(df.columns)

            # Добавляем данные из DataFrame в QTableWidget
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    item = QTableWidgetItem(str(df.iloc[i, j]))
                    table.setItem(i, j, item)
            # Устанавливаем размеры таблицы
            table.setMinimumSize(view_width, view_height)
            #table.setMaximumSize(view_width, view_height)
            table.resizeRowsToContents()
            table.resizeColumnsToContents()
            

            #proxy.setWidget(self.table)

            # Добавляем QGraphicsProxyWidget в main_scen
        
            self.main_scen.addWidget(table)
            print(self.main_scen.height())
            print(self.main_view.viewport().height())
            # Устанавливаем политику изменения размеров для QGraphicsProxyWidget
            #self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except:
            self.settings_action.setEnabled(False)
            self.save_as_action.setEnabled(False)
            self.lsm_filesList = None
            self.df = None
            self.show_warning_dialog("Error during creating table")
    
    def open_lsm(self):
        
        #запомнить путь последней папки  вместо ""
        lsm_path, _ = QFileDialog.getOpenFileName(self, "Open .LSM File", "", "LSM Files (*.lsm)")
        
        
        if not lsm_path:
            return 0
        self.lsm_path = lsm_path
        self.main_scen.clear()
        self.settings_action.setEnabled(True)
        self.save_as_action.setEnabled(False)
        self.right_button.setEnabled(True)
        self.lsm_filesList = None
        try:
            with tifffile.TiffFile(self.lsm_path) as tif:
                lsm_file = tif.pages[0].asarray()
            if lsm_file.shape[0] < max([value+1 for key, value in self.parametrs.items()]):
                self.parametrs = {'Cell' : 0,
                                        'Nuclei': 1
                                    }
            if lsm_file.shape[0] <=1:
                self.show_warning_dialog("File is wrong\n\nAmount of Channel less than 2")
                return
            image = QImage(lsm_file[1], lsm_file.shape[1],  lsm_file.shape[2] , QImage.Format_Grayscale8)

            # Создаем QPixmap из QImage
            pixmap = QPixmap.fromImage(image)
            
            # Определяем размеры видового окна
            view_width = self.main_view.viewport().width()
            view_height = self.main_view.viewport().height()
            
            # Определяем соотношение сторон изображения
            pixmap_aspect_ratio = pixmap.width() / pixmap.height()
            
            # Определяем размеры пиксмапа, чтобы он занимал хотя бы одну часть экрана
            if view_width / view_height > pixmap_aspect_ratio:
                pixmap_width = view_height * pixmap_aspect_ratio
                pixmap_height = view_height
            else:
                pixmap_width = view_width
                pixmap_height = view_width / pixmap_aspect_ratio
            
            # Изменяем размеры пиксмапа
            pixmap = pixmap.scaled(pixmap_width, pixmap_height, aspectRatioMode=Qt.KeepAspectRatio)
            
            # Создаем пиксмап изображения с измененными размерами
            pixmap_item = self.main_scen.addPixmap(pixmap)
            
            # Вычисляем позицию пиксмапа, чтобы он был по центру видового окна
            x_pos = (view_width - pixmap.width()) / 2
            y_pos = (view_height - pixmap.height()) / 2
            
            # Устанавливаем позицию пиксмапа
            pixmap_item.setPos(x_pos, y_pos)
                #self.main_view.fitInView(pixmap_item)#, Qt.KeepAspectRatio)
            self.setWindowTitle(f"Cells Calculator - {os.path.basename(lsm_path)}")
        except  Exception as e:
            print(e)
            self.show_warning_dialog("Error during opening file.")
            self.lsm_path = None
            self.main_scen.clear()
            self.settings_action.setEnabled(False)
            self.save_as_action.setEnabled(False)
            self.right_button.setEnabled(False)
            self.lsm_filesList = None
            return 0
            
    def open_dialogWindow(self):
        #запомнить путь последней папки  вместо ""
        #self.lsm_path, _ = QFileDialog.getOpenFileName(self, "Open .LSM File", "", "LSM Files (*.lsm)")
        try:
            if self.lsm_filesList:
                lsm_path = self.lsm_filesList
                call_back = self.create_table
            else:
                call_back = None
                lsm_path = self.lsm_path
            if lsm_path:
                dialog = DialogWindow(parent=self, lsm_path=lsm_path, parametrs = self.parametrs, call_back = call_back)
                dialog.setWindowModality(Qt.ApplicationModal)
                #dialog.setFixedSize(1080, 720)
                #dialog.move(100,100)
                dialog.show()
                dialog.center()
        except:
            self.show_warning_dialog("Error during opening channels settings")
       
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        
        print(QApplication.desktop().availableGeometry())
        window = MainWindow()
        window.showMaximized()
        print(window.right_view.size())
        #window.open_lsm_action.trigger()
        
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Critical Error", str(e), QMessageBox.Ok)
        sys.exit(1)