from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMenu, QAction, QLabel

class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Button with Dropdown Menu and Single Selection")

        # Создаем кнопку
        self.button = QPushButton("Выберите вариант", self)
        
        # Лейбл, который не изменяет текст
        self.label = QLabel("Выберите вариант из выпадающего списка:", self)

        # Подключаем действие для показа контекстного меню
        self.button.setMenu(self.create_menu())

        # Лейаут
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def create_menu(self):
        menu = QMenu(self)

        # Добавляем действия в контекстное меню с флажками
        self.action1 = QAction("Option 1", self)
        self.action2 = QAction("Option 2", self)
        self.action3 = QAction("Option 3", self)

        # Устанавливаем возможность выбора
        self.action1.setCheckable(True)
        self.action2.setCheckable(True)
        self.action3.setCheckable(True)

        # Устанавливаем начальное состояние, где все пункты не выбраны
        self.action1.setChecked(False)
        self.action2.setChecked(False)
        self.action3.setChecked(False)

        # Привязываем действия к функциям
        self.action1.triggered.connect(lambda: self.on_option_selected(self.action1))
        self.action2.triggered.connect(lambda: self.on_option_selected(self.action2))
        self.action3.triggered.connect(lambda: self.on_option_selected(self.action3))

        # Добавляем действия в меню
        menu.addAction(self.action1)
        menu.addAction(self.action2)
        menu.addAction(self.action3)

        return menu

    def on_option_selected(self, selected_action):
        # Снимаем галочки с всех действий
        self.action1.setChecked(False)
        self.action2.setChecked(False)
        self.action3.setChecked(False)

        # Устанавливаем галочку только для выбранного действия
        selected_action.setChecked(True)

        # Обновляем текст в лейбле с выбранной опцией
        self.label.setText(f"Вы выбрали: {selected_action.text()}")

if __name__ == '__main__':
    app = QApplication([])
    ex = Example()
    ex.show()
    app.exec_()
