import math
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView,
)


class ImageViewer(QGraphicsView):
    """Graphics view that displays an image with zoom/pan support."""

    zoom_changed = Signal(float)       # emitted with scale as fraction (1.0 == 100%)
    measure_distance = Signal(float)   # emitted with distance in image pixels
    region_selected = Signal(QRectF)   # emitted with selected rect in image coords
    mouse_image_pos = Signal(QPointF)  # emitted with cursor position in image coords

    ZOOM_IN_FACTOR = 1.25
    ZOOM_OUT_FACTOR = 1 / 1.25
    MIN_SCALE = 0.05
    MAX_SCALE = 40.0

    def __init__(self, parent=None):
        """Set up the scene, pixmap item, view options and interaction state."""
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(self.palette().dark())
        self.setFocusPolicy(Qt.StrongFocus)  # needed to receive key events
        self.setCursor(Qt.ArrowCursor)

        self._has_image = False
        self._image: np.ndarray | None = None
        self._fit_active: bool = False  # True when last zoom was fit-to-window

        # ---- measurement state ------------------------------------------
        self._meas_start: QPointF | None = None
        self._meas_line: QGraphicsLineItem | None = None
        self._meas_dot_start: QGraphicsEllipseItem | None = None
        self._meas_dot_end: QGraphicsEllipseItem | None = None

        # ---- pan state --------------------------------------------------
        self._pan_origin: QPoint | None = None

        # ---- options ----------------------------------------------------
        self.auto_clear_measure: bool = False    # clear overlay after each measurement
        self.auto_clear_selection: bool = False  # clear region rect after each selection

        # ---- region selection state -------------------------------------
        self._sel_start: QPointF | None = None
        self._sel_rect_item: QGraphicsRectItem | None = None

    # ---- public API -----------------------------------------------------
    def set_image(self, image: np.ndarray, keep_view: bool = False) -> None:
        """Accept a numpy array (H x W x 3/4 uint8 or H x W uint8) and display it.

        Args:
            image: numpy array to display.
            keep_view: if True and an image is already shown with the same dimensions,
                       the current zoom level, pan position and transform are preserved.
                       Pass False (default) to reset to 1:1 and center the image.
        """
        pixmap = self._ndarray_to_pixmap(image)
        same_size = (
            keep_view
            and self._has_image
            and self._pixmap_item.pixmap().size() == pixmap.size()
        )
        self._image = image
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._has_image = True
        self._clear_measure()
        self._clear_selection()
        if not same_size:
            self.reset_view()

    def get_image(self) -> np.ndarray | None:
        """Return the currently displayed image as a numpy array."""
        return self._image

    @staticmethod
    def _ndarray_to_pixmap(image: np.ndarray) -> QPixmap:
        """Convert a numpy image array to a QPixmap.

        Args:
            image: H x W (grayscale), H x W x 3 (RGB) or H x W x 4 (RGBA)
                uint8 array.

        Returns:
            A QPixmap built from the array's pixel data.
        """
        image = np.ascontiguousarray(image)
        if image.ndim == 2:
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        elif image.shape[2] == 4:
            h, w = image.shape[:2]
            qimg = QImage(image.data, w, h, 4 * w, QImage.Format_RGBA8888)
        else:
            h, w = image.shape[:2]
            qimg = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def has_image(self) -> bool:
        """Return True if an image is currently loaded."""
        return self._has_image

    def zoom_in(self):
        """Zoom in by one step."""
        self._scale_by(self.ZOOM_IN_FACTOR)

    def zoom_out(self):
        """Zoom out by one step."""
        self._scale_by(self.ZOOM_OUT_FACTOR)

    def reset_view(self):
        """Reset zoom to 1:1 (actual pixels)."""
        if not self._has_image:
            return
        self._fit_active = False
        self.resetTransform()
        self.zoom_changed.emit(self.current_scale())

    def fit_to_window(self):
        """Scale the image to fit the viewport, preserving aspect ratio."""
        if not self._has_image:
            return
        self._fit_active = True
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self.zoom_changed.emit(self.current_scale())

    def current_scale(self) -> float:
        """Return the current zoom scale as a fraction (1.0 == 100%)."""
        return self.transform().m11()

    # ---- internal -------------------------------------------------------
    def _scale_by(self, factor: float):
        """Multiply the current zoom by ``factor``, clamped to the scale limits."""
        if not self._has_image:
            return
        new_scale = self.current_scale() * factor
        if new_scale < self.MIN_SCALE or new_scale > self.MAX_SCALE:
            return
        self._fit_active = False
        self.scale(factor, factor)
        self.zoom_changed.emit(self.current_scale())

    # ---- events ---------------------------------------------------------
    def resizeEvent(self, event):
        """Re-fit the image to the viewport when fit-to-window mode is active."""
        super().resizeEvent(event)
        # Re-fit when the widget has a real size and fit mode is active
        if self._fit_active and self._has_image and self.width() > 0 and self.height() > 0:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self.zoom_changed.emit(self.current_scale())

    def enterEvent(self, event):
        """Grab keyboard focus when the cursor enters the view."""
        self.setFocus()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Cancel any in-progress measurement/selection and reset the cursor."""
        if self._meas_start is not None:
            self._clear_measure()
            self.setCursor(Qt.ArrowCursor)
        if self._sel_start is not None:
            self._clear_selection()
            self.setCursor(Qt.ArrowCursor)
        elif self._has_image:
            self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Dispatch left clicks to region selection (Shift), measurement (Ctrl) or panning."""
        if event.button() == Qt.LeftButton and self._has_image:
            mods = event.modifiers()
            if mods & Qt.ShiftModifier and not (mods & Qt.ControlModifier):
                if self._sel_start is None:
                    # First click — start region selection
                    self._clear_selection()
                    self._sel_start = self.mapToScene(event.pos())
                    self._sel_rect_item = self._scene.addRect(
                        QRectF(self._sel_start, self._sel_start),
                        self._sel_pen(),
                        QColor(100, 180, 255, 40)
                    )
                else:
                    # Second click — finalise region selection
                    end = self.mapToScene(event.pos())
                    rect = QRectF(self._sel_start, end).normalized()
                    img_rect = self._scene.sceneRect()
                    rect = rect.intersected(img_rect)
                    if not self.auto_clear_selection:
                        self._sel_start = None  # keep rect visible
                    else:
                        self._clear_selection()
                    self.region_selected.emit(rect)
                self.setCursor(Qt.CrossCursor)
                event.accept()
                return
            elif self._sel_start is not None and not (mods & Qt.ControlModifier):
                # Finalise region selection
                end = self.mapToScene(event.pos())
                rect = QRectF(self._sel_start, end).normalized()
                # Clamp to image bounds
                img_rect = self._scene.sceneRect()
                rect = rect.intersected(img_rect)
                if not self.auto_clear_selection:
                    self._sel_start = None  # keep rect visible, just stop dragging
                else:
                    self._clear_selection()
                self.region_selected.emit(rect)
                event.accept()
                return
            elif mods & Qt.ControlModifier:
                if self._meas_start is None:
                    # First click — start measurement
                    self._clear_measure()
                    self._meas_start = self.mapToScene(event.pos())
                    self._meas_line = self._scene.addLine(
                        self._meas_start.x(), self._meas_start.y(),
                        self._meas_start.x(), self._meas_start.y(),
                        self._meas_pen()
                    )
                    self._meas_dot_start = self._add_dot(self._meas_start)
                else:
                    # Second click — finalise measurement
                    end = self.mapToScene(event.pos())
                    dx = end.x() - self._meas_start.x()
                    dy = end.y() - self._meas_start.y()
                    dist = math.hypot(dx, dy)
                    self._meas_dot_end = self._add_dot(end)
                    self._meas_start = None
                    self.setCursor(Qt.CrossCursor)
                    self.measure_distance.emit(dist)
                    if self.auto_clear_measure:
                        self._clear_measure()
                event.accept()
                return
            else:
                # Start panning (only if not in any selection/measure mode)
                if self._sel_start is None:
                    self._pan_origin = event.pos()
                    self.setCursor(Qt.ClosedHandCursor)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """End panning on left-button release and restore the modifier-aware cursor."""
        if event.button() == Qt.LeftButton and self._pan_origin is not None:
            self._pan_origin = None
            mods = event.modifiers()
            if mods & Qt.ControlModifier:
                self.setCursor(Qt.CrossCursor)
            elif mods & Qt.ShiftModifier:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """Report the cursor position and update panning, selection rect, measure line or cursor."""
        if self._has_image:
            self.mouse_image_pos.emit(self.mapToScene(event.pos()))
        if self._pan_origin is not None:
            delta = event.pos() - self._pan_origin
            self._pan_origin = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        mods = event.modifiers()
        # Update rubber-band rect
        if self._sel_start is not None:
            if not (mods & Qt.ShiftModifier):
                self._clear_selection()
                self.setCursor(Qt.ArrowCursor)
            elif self._sel_rect_item is not None:
                cur = self.mapToScene(event.pos())
                self._sel_rect_item.setRect(QRectF(self._sel_start, cur).normalized())
            super().mouseMoveEvent(event)
            return
        # Update cursor based on modifier state
        if self._has_image:
            if mods & Qt.ControlModifier:
                self.setCursor(Qt.CrossCursor)
            elif mods & Qt.ShiftModifier:
                self.setCursor(Qt.CrossCursor)
            elif self._meas_start is None:
                self.setCursor(Qt.ArrowCursor)
        if self._meas_start is not None:
            # Cancel if Ctrl was released between events
            if not (mods & Qt.ControlModifier):
                self._clear_measure()
                self.setCursor(Qt.ArrowCursor)
                super().mouseMoveEvent(event)
                return
            if self._meas_line is not None:
                cur = self.mapToScene(event.pos())
                self._meas_line.setLine(
                    self._meas_start.x(), self._meas_start.y(),
                    cur.x(), cur.y()
                )
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        """Update the cursor for Ctrl/Shift and cancel measurement/selection on Escape."""
        if self._has_image:
            if event.key() == Qt.Key_Control:
                self.setCursor(Qt.CrossCursor)
            elif event.key() == Qt.Key_Shift:
                self.setCursor(Qt.CrossCursor)
        if event.key() == Qt.Key_Escape:
            if self._meas_start is not None:
                self._clear_measure()
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
            if self._sel_start is not None:
                self._clear_selection()
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Cancel any in-progress measurement/selection when Ctrl/Shift is released."""
        if event.key() == Qt.Key_Control:
            if self._meas_start is not None:
                self._clear_measure()
            self.setCursor(Qt.ArrowCursor)
        elif event.key() == Qt.Key_Shift:
            if self._sel_start is not None:
                self._clear_selection()
            self.setCursor(Qt.ArrowCursor)
        super().keyReleaseEvent(event)

    def wheelEvent(self, event):
        """Handle Ctrl+wheel as zoom; otherwise scroll normally."""
        if not self._has_image:
            return
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    # ---- measurement helpers --------------------------------------------
    def _meas_pen(self) -> QPen:
        """Return the cosmetic dashed pen used to draw the measurement line."""
        pen = QPen(QColor(255, 220, 0))
        pen.setCosmetic(True)
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        return pen

    # ---- selection helpers ----------------------------------------------
    def _sel_pen(self) -> QPen:
        """Return the cosmetic dashed pen used to draw the selection rectangle."""
        pen = QPen(QColor(100, 180, 255))
        pen.setCosmetic(True)
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        return pen

    def _clear_selection(self):
        """Remove the selection rectangle and reset selection state."""
        if self._sel_rect_item is not None:
            self._scene.removeItem(self._sel_rect_item)
            self._sel_rect_item = None
        self._sel_start = None

    def _add_dot(self, pt: QPointF) -> QGraphicsEllipseItem:
        """Add and return a measurement endpoint dot centered at ``pt``."""
        r = 4
        item = QGraphicsEllipseItem(pt.x() - r, pt.y() - r, 2 * r, 2 * r)
        pen = QPen(QColor(255, 220, 0))
        pen.setCosmetic(True)
        pen.setWidth(2)
        item.setPen(pen)
        item.setBrush(QColor(255, 220, 0, 140))
        self._scene.addItem(item)
        return item

    def _clear_measure(self):
        """Remove the measurement line and endpoint dots and reset measurement state."""
        for item in (self._meas_line, self._meas_dot_start, self._meas_dot_end):
            if item is not None:
                self._scene.removeItem(item)
        self._meas_line = None
        self._meas_dot_start = None
        self._meas_dot_end = None
        self._meas_start = None