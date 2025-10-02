# -*- coding: utf-8 -*-

# Third-party imports
from PyQt5 import QtCore, QtGui, QtWidgets


class RangeSlider(QtWidgets.QWidget):
    """
    Горизонтальний слайдер з двома ручками: low (мін) і high (макс).

    Події:
      - valueChanged(int low, int high)
      - lowValueChanged(int)
      - highValueChanged(int)

    API:
      - setRange(minimum, maximum)
      - setValues(low, high)
      - setLow(value), setHigh(value)
      - setSingleStep(step)
      - setMinimumRange(delta)     # мін. інтервал між low і high у значеннях
      - values(), minimum(), maximum(), minimumRange()

    Клавіатура (коли є фокус та вибрана ручка):
      - ← / → : -/+ step
      - PgUp / PgDn : +/- 10 * step
      - Home / End : до мінімуму / максимуму (з урахуванням minRange)
      - Tab : переключити фокус ручки (low <-> high)
      - Клік по ручці також ставить фокус саме на неї
    """
    valueChanged = QtCore.pyqtSignal(int, int)
    lowValueChanged = QtCore.pyqtSignal(int)
    highValueChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, minimum=0, maximum=100, low=25, high=75, step=1):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._min = int(minimum)
        self._max = int(maximum)
        self._step = max(1, int(step))
        self._min_range = self._step  # не даємо ручкам накладатися за замовчуванням

        self._low = int(low)
        self._high = int(high)
        self._ensure_order_and_gap()

        # Візуальні параметри
        self._groove_h = 6
        self._handle_r = 9
        self._margin = self._handle_r + 2

        # Стан
        self._active = None       # "low" | "high" | None (перетягування мишею)
        self._focus = "high"      # яка ручка отримає реакцію на клавіатуру
        self._press_offset = 0

        self.setMinimumHeight(2 * (self._handle_r + 4))

    # ---------- API ----------
    def setRange(self, minimum, maximum):
        minimum, maximum = int(minimum), int(maximum)
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        self._min, self._max = minimum, maximum
        self._low = max(self._min, min(self._low, self._max))
        self._high = max(self._min, min(self._high, self._max))
        self._ensure_order_and_gap()
        self.update()
        self.valueChanged.emit(self._low, self._high)

    def setValues(self, low, high):
        low, high = int(low), int(high)
        self._low, self._high = low, high
        self._ensure_order_and_gap()
        self.update()
        self.valueChanged.emit(self._low, self._high)

    def setLow(self, v):
        v = self._snap(int(v))
        v = max(self._min, v)
        v = min(v, self._high - self._min_range)
        if v != self._low:
            self._low = v
            self.update()
            self.lowValueChanged.emit(self._low)
            self.valueChanged.emit(self._low, self._high)

    def setHigh(self, v):
        v = self._snap(int(v))
        v = min(self._max, v)
        v = max(v, self._low + self._min_range)
        if v != self._high:
            self._high = v
            self.update()
            self.highValueChanged.emit(self._high)
            self.valueChanged.emit(self._low, self._high)

    def setSingleStep(self, step):
        self._step = max(1, int(step))
        # Якщо step збільшився, мін. інтервал теж має лишатися валідним
        if self._min_range < self._step:
            self._min_range = self._step
            self._ensure_order_and_gap()
        self.update()

    def setMinimumRange(self, delta):
        """Встановити мінімальний інтервал між low і high у 'значеннях'."""
        self._min_range = max(0, int(delta))
        # не даємо інтервалу бути меншим за step, якщо хочеш дозволити торкатися — передай 0
        if self._min_range and self._min_range < self._step:
            self._min_range = self._step
        self._ensure_order_and_gap()
        self.update()

    def minimumRange(self):
        return self._min_range

    def values(self):
        return self._low, self._high

    def minimum(self):
        return self._min

    def maximum(self):
        return self._max

    # ---------- Внутрішні допоміжні ----------
    def _ensure_order_and_gap(self):
        # впорядкувати
        if self._low > self._high:
            self._low, self._high = self._high, self._low
        # забезпечити мін. інтервал
        gap = self._high - self._low
        if gap < self._min_range:
            # намагаємося посунути high вгору, якщо є місце, інакше low вниз
            needed = self._min_range - gap
            if self._high + needed <= self._max:
                self._high += needed
            elif self._low - needed >= self._min:
                self._low -= needed
            else:
                # Затискаємо в найближчі можливі межі
                self._high = min(self._max, max(self._high, self._low + self._min_range))
                self._low = max(self._min, min(self._low, self._high - self._min_range))

    def _available_w(self):
        return max(1, self.width() - 2 * self._margin)

    def _value_to_pos(self, v):
        if self._max == self._min:
            return self._margin
        ratio = (v - self._min) / float(self._max - self._min)
        return int(round(self._margin + ratio * self._available_w()))

    def _pos_to_value(self, x):
        x = max(self._margin, min(self.width() - self._margin, x))
        ratio = (x - self._margin) / float(self._available_w())
        v = self._min + ratio * (self._max - self._min)
        return self._snap(int(round(v)))

    def _clamp_to_range(self, v):
        return max(self._min, min(self._max, v))

    def _snap(self, v):
        # округлення до кроку відносно мінімуму
        delta = v - self._min
        steps = int(round(delta / float(self._step)))
        return self._min + steps * self._step

    # ---------- Малювання ----------
    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        h = self.height()
        cy = h // 2
        groove_rect = QtCore.QRect(self._margin, cy - self._groove_h // 2,
                                   self._available_w(), self._groove_h)

        pal = self.palette()
        groove_brush = pal.mid().color()
        fill_brush = pal.highlight().color()
        handle_brush = pal.button().color()
        focus_pen = QtGui.QPen(pal.highlight().color().darker(120), 2)
        handle_pen = QtGui.QPen(pal.dark().color(), 1)

        # Groove
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(groove_brush)
        r = groove_rect
        p.drawRoundedRect(r, 3, 3)

        # Selected range
        x1 = self._value_to_pos(self._low)
        x2 = self._value_to_pos(self._high)
        sel = QtCore.QRect(min(x1, x2), r.y(), abs(x2 - x1), r.height())
        p.setBrush(fill_brush)
        p.drawRoundedRect(sel, 3, 3)

        # Order: спочатку неактивна, потім активна/у фокусі — зверху
        order = [("low", x1), ("high", x2)]
        if self._focus == "low":
            order = [("high", x2), ("low", x1)]
        elif self._focus == "high":
            order = [("low", x1), ("high", x2)]
        if self._active == "low":
            order = [("high", x2), ("low", x1)]
        elif self._active == "high":
            order = [("low", x1), ("high", x2)]

        # Handles
        for role, xpos in order:
            p.setBrush(handle_brush)
            p.setPen(handle_pen)
            p.drawEllipse(QtCore.QPoint(xpos, cy), self._handle_r, self._handle_r)
            if role == self._focus:
                p.setPen(focus_pen)
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawEllipse(QtCore.QPoint(xpos, cy), self._handle_r + 2, self._handle_r + 2)

    # ---------- Миша ----------
    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return super().mousePressEvent(ev)
        x = ev.x()
        lx = self._value_to_pos(self._low)
        hx = self._value_to_pos(self._high)

        # Вибрати найближчу ручку
        if abs(x - lx) <= abs(x - hx):
            self._active = "low"
            self._focus = "low"
            self._press_offset = x - lx
        else:
            self._active = "high"
            self._focus = "high"
            self._press_offset = x - hx
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.update()  # щоби перемалювати фокусну рамку

    def mouseMoveEvent(self, ev):
        if not self._active:
            return super().mouseMoveEvent(ev)
        x = ev.x() - self._press_offset
        val = self._pos_to_value(x)
        if self._active == "low":
            self.setLow(min(val, self._high - self._min_range))
        else:
            self.setHigh(max(val, self._low + self._min_range))

    def mouseReleaseEvent(self, ev):
        if self._active:
            self.unsetCursor()
        self._active = None
        return super().mouseReleaseEvent(ev)

    # ---------- Клавіатура ----------
    def keyPressEvent(self, ev):
        key = ev.key()
        mod = ev.modifiers()
        step = self._step
        big = self._step * 10

        def move_low(delta):
            self.setLow(self._low + delta)

        def move_high(delta):
            self.setHigh(self._high + delta)

        target = self._focus or "high"

        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                   QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown,
                   QtCore.Qt.Key_Home, QtCore.Qt.Key_End,
                   QtCore.Qt.Key_Tab):
            if key == QtCore.Qt.Key_Tab:
                # Перемкнути фокус ручки
                self._focus = "low" if self._focus == "high" else "high"
                self.update()
                return

            delta = 0
            if key == QtCore.Qt.Key_Left:
                delta = -step
            elif key == QtCore.Qt.Key_Right:
                delta = +step
            elif key == QtCore.Qt.Key_PageDown:
                delta = -big
            elif key == QtCore.Qt.Key_PageUp:
                delta = +big
            elif key == QtCore.Qt.Key_Home:
                if target == "low":
                    self.setLow(self._min)
                else:
                    self.setHigh(self._low + self._min_range)
                return
            elif key == QtCore.Qt.Key_End:
                if target == "high":
                    self.setHigh(self._max)
                else:
                    self.setLow(self._high - self._min_range)
                return

            if delta != 0:
                if target == "low":
                    move_low(delta)
                else:
                    move_high(delta)
            return
        super().keyPressEvent(ev)

    def focusInEvent(self, e):
        self.update()
        super().focusInEvent(e)

    def focusOutEvent(self, e):
        self.update()
        super().focusOutEvent(e)

    # ---------- Розміри ----------
    def sizeHint(self):
        return QtCore.QSize(260, max(2 * (self._handle_r + 6), 30))

    def minimumSizeHint(self):
        return self.sizeHint()
