import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout,\
     QHBoxLayout, QLabel, QSlider, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class CustomSlider(QSlider):
    """
    Class for filtering sliders in the right layout menu.
    """
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Определяем новое значение в зависимости от позиции клика
            new_value = self.minimum() + (self.maximum() - self.minimum()) * event.pos().x() / self.width()
            self.setValue(int(new_value))  # Устанавливаем новое значение
            event.accept()  # Отмечаем событие как обработанное
        super().mousePressEvent(event)

class Slider(QWidget):
    """
    Class for filtering sliders in the right layout menu.
    """
    def __init__(self, object_size : dict, default_object_size : dict, key : str): 
        self.object_size = object_size
        self.key = key
        self.round_parametr_slider = object_size['round_parametr_slider']
        self.round_parametr_value_input = object_size['round_parametr_value_input']
        self.default_object_size = default_object_size
        super().__init__()
        self.initUI()

    def change_default(self, min_size, max_size):
        #TODO: some validation for min and max
        self.default_object_size['min_size'] = min_size - min_size / 100
        self.default_object_size['max_size'] = max_size + max_size / 100
        self.value_slider.setMinimum(int (self.default_object_size['min_size'] * self.round_parametr_slider))
        self.value_slider.setMaximum(int (self.default_object_size['max_size'] * self.round_parametr_slider))
        self.set_default()

    def set_default(self):
        value = self.default_object_size[self.key]
        self.object_size[self.key] = value
        self.value_input.setText(str(f'{value * self.round_parametr_value_input:.2f}'))
        self.value_slider.setValue(int (value * self.round_parametr_slider))

    def initUI(self):
        # Переменная для хранения значения
        # Основной вертикальный layout
        main_layout = QVBoxLayout()

        # Первая строка с надписью "Value"
        label_layout = QHBoxLayout()

        value_label = QLabel(self.key.capitalize().replace('_size', ''))
        font = QFont()
        font.setPointSize(16)  # Устанавливаем размер шрифта 16 пунктов
        value_label.setFont(font)
        label_layout.addWidget(value_label)
        main_layout.addLayout(label_layout)

        # Вторая строка: QLineEdit и QSlider
        slider_layout = QHBoxLayout()

        # Создаем QLineEdit для отображения значения
        self.value_input = QLineEdit(f"")
        self.value_input.setFixedWidth(50)
        self.value_input.setAlignment(Qt.AlignCenter)

        # Устанавливаем валидатор, чтобы разрешать ввод только цифр от 0 до 10
        #self.value_input.setValidator(QDoubleValidator(self.object_size['min_size'], self.object_size['max_size'], 3))

        slider_layout.addWidget(self.value_input)

        # Создаем QSlider
        self.value_slider = CustomSlider(Qt.Horizontal)
        self.value_slider.setMinimum(int (self.object_size['min_size']) * self.round_parametr_slider)
        self.value_slider.setMaximum(int (self.object_size['max_size'] * self.round_parametr_slider))
        self.value_slider.setValue(int (self.object_size[self.key] * self.round_parametr_slider) )
        slider_layout.addWidget(self.value_slider)

        # Добавляем слайдер и поле ввода в основной layout
        main_layout.addLayout(slider_layout)

        # Соединяем сигналы и слоты
        self.value_slider.valueChanged.connect(self.update_value_from_slider)
        self.value_input.returnPressed.connect(self.update_value_from_input)

        # Настройки окна
        self.setLayout(main_layout)
        self.setWindowTitle('Slider and Input Example')
        self.show()

    def update_value_from_slider(self):
        # Обновляем переменную и QLineEdit при изменении значения слайдера
        min_size = self.default_object_size['min_size']
        max_size = self.default_object_size['max_size']

        value = self.value_slider.value() / self.round_parametr_slider
        # Проверяем значение на корректность, если вне диапазона — устанавливаем границы
        if self.key == 'min_size' and value > self.object_size['max_size']:
            value = self.object_size['max_size']
        elif self.key == 'max_size' and value < self.object_size['min_size']:
            value = self.object_size['min_size']
        elif value < min_size:
            value = min_size
        elif value > max_size:
            value = max_size 
        self.value_slider.setValue(int(value * self.round_parametr_slider))
        self.object_size[self.key] = value
        self.value_input.setText(str(f'{value * self.round_parametr_value_input:.2f}'))

    def update_value_from_input(self):
        # Обновляем переменную и слайдер при изменении значения в QLineEdit
        min_size = self.default_object_size['min_size']
        max_size = self.default_object_size['max_size']
        try:
            value = float(self.value_input.text()) / self.round_parametr_value_input
            # Проверяем значение на корректность, если вне диапазона — устанавливаем границы
            if self.key == 'min_size' and value >= self.object_size['max_size']:
                value = self.object_size['max_size']
            elif self.key == 'max_size' and value <= self.object_size['min_size']:
                value = self.object_size['min_size']
            elif value < min_size:
                value = min_size
            elif value > max_size:
                value = max_size

            self.object_size[self.key] = value
            self.value_slider.setValue(int(value * self.round_parametr_slider))
            self.value_input.setText(str(f'{value * self.round_parametr_value_input:.2f}'))
        except ValueError:

            self.value_input.setText(str(f'{self.object_size[self.key] * self.round_parametr_value_input:.2f}'))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Slider()
    sys.exit(app.exec_())
