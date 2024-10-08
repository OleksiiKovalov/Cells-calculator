import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIntValidator

class Slider(QWidget):
    def __init__(self, dic, key):
        self.dic = dic
        self.key = key
        self.min = dic['min']
        self.max = dic['max']
        super().__init__()
        self.initUI()

    def initUI(self):
        # Переменная для хранения значения


        # Основной вертикальный layout
        main_layout = QVBoxLayout()

        # Первая строка с надписью "Value"
        label_layout = QHBoxLayout()
        
        value_label = QLabel(self.key.capitalize())
        font = QFont()
        font.setPointSize(16)  # Устанавливаем размер шрифта 16 пунктов
        value_label.setFont(font)
        label_layout.addWidget(value_label)
        main_layout.addLayout(label_layout)

        # Вторая строка: QLineEdit и QSlider
        slider_layout = QHBoxLayout()

        # Создаем QLineEdit для отображения значения
        self.value_input = QLineEdit(f"{self.dic[self.key]}")
        self.value_input.setFixedWidth(50)
        self.value_input.setAlignment(Qt.AlignCenter)

        # Устанавливаем валидатор, чтобы разрешать ввод только цифр от 0 до 10
        self.value_input.setValidator(QIntValidator(self.dic['min'], self.dic['max']))

        slider_layout.addWidget(self.value_input)

        # Создаем QSlider
        self.value_slider = QSlider(Qt.Horizontal)
        self.value_slider.setMinimum(self.dic['min'])
        self.value_slider.setMaximum(self.dic['max'])
        self.value_slider.setValue(self.dic[self.key])
        slider_layout.addWidget(self.value_slider)

        # Добавляем слайдер и поле ввода в основной layout
        main_layout.addLayout(slider_layout)

        # Соединяем сигналы и слоты
        self.value_slider.valueChanged.connect(self.update_value_from_slider)
        self.value_input.textChanged.connect(self.update_value_from_input)

        # Настройки окна
        self.setLayout(main_layout)
        self.setWindowTitle('Slider and Input Example')
        self.show()

    def update_value_from_slider(self):
        # Обновляем переменную и QLineEdit при изменении значения слайдера
        value = self.value_slider.value()
        # Проверяем значение на корректность, если вне диапазона — устанавливаем границы
        if self.key == 'min' and value > self.dic['max']:
            value = self.dic['max']
        elif self.key == 'max' and value < self.dic['min']:
            value = self.dic['min']
        elif value < self.min:
            value = self.min
        elif value > self.max:
            value = self.max 
        self.value_slider.setValue(value)
        self.dic[self.key] = value
        self.value_input.setText(str(self.dic[self.key]))

    def update_value_from_input(self):
        # Обновляем переменную и слайдер при изменении значения в QLineEdit
        try:
            value = int(self.value_input.text())
            # Проверяем значение на корректность, если вне диапазона — устанавливаем границы
            if self.key == 'min' and value > self.dic['max']:
                value = self.dic['max']
            elif self.key == 'max' and value < self.dic['min']:
                value = self.dic['min']
            elif value < self.min:
                value = self.min
            elif value > self.max:
                value = self.max

            self.dic[self.key] = value
            self.value_slider.setValue(value)
            self.value_input.setText(str(value))
        except ValueError:
            # Игнорируем нечисловые значения
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Slider()
    sys.exit(app.exec_())
