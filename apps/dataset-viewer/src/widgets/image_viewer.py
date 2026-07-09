from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem,
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPixmap, QPen, QColor, QBrush, QPainter, QFont, QPolygonF, QImage


def _build_colors() -> list[QColor]:
    try:
        import matplotlib
        cmap = matplotlib.colormaps['tab20']   # modern, non-deprecated API
        return [QColor(*(int(ch * 255) for ch in cmap(i)[:3])) for i in range(cmap.N)]
    except Exception:
        # Hardcoded tab20 fallback (exact matplotlib values)
        return [QColor(h) for h in (
            '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
            '#2ca02c', '#98df8a', '#d62728', '#ff9896',
            '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
            '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
            '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
        )]


_COLORS = _build_colors()


class ImageViewer(QGraphicsView):
    zoom_changed = Signal(float)   # emits zoom percent

    _ZOOM_IN  = 1.25
    _ZOOM_OUT = 1.0 / 1.25
    _MIN_PCT  = 2.0
    _MAX_PCT  = 3200.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._ann_items: list = []
        self._ann_visible = True
        self._setup_view()
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_view(self):
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(45, 45, 45)))
        self.setFrameShape(QGraphicsView.NoFrame)

    def _show_placeholder(self):
        self._scene.clear()
        self._pixmap_item = None
        self._ann_items = []
        msg = self._scene.addText(
            "Open a dataset folder to get started\n"
            "File → Open Folder   or   Ctrl+O"
        )
        msg.setDefaultTextColor(QColor(110, 110, 110))
        font = QFont("Segoe UI", 13)
        msg.setFont(font)
        self._scene.setSceneRect(msg.boundingRect().adjusted(-40, -20, 40, 20))

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def load_image(self, image_path: str, annotations: list | None = None):
        self._scene.clear()
        self._pixmap_item = None
        self._ann_items = []

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self._show_placeholder()
            return

        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        if annotations:
            self._draw_annotations(annotations)

        self.fit_to_window()

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    def _draw_annotations(self, annotations: list):
        for i, ann in enumerate(annotations):
            cls      = ann.get('class_id', 0)
            color    = _COLORS[i % len(_COLORS)]
            label    = ann.get('label', str(cls))
            ann_type = ann.get('type', 'bbox')
            lx, ly   = ann['x'], ann['y']   # label anchor (bbox top-left)

            pen = QPen(color, 2)
            pen.setCosmetic(True)

            if ann_type == 'polygon':
                fill = QColor(color)
                fill.setAlpha(45)
                brush = QBrush(fill)
                # 'polygons' (COCO multi-part) takes priority over 'points' (YOLO single)
                parts = ann.get('polygons') or (
                    [ann['points']] if ann.get('points') else []
                )
                for pts in parts:
                    poly = QPolygonF([QPointF(px, py) for px, py in pts])
                    item = self._scene.addPolygon(poly, pen, brush)
                    item.setZValue(1)
                    self._ann_items.append(item)

            elif ann_type == 'mask':
                img = self._rle_to_qimage(
                    ann['rle_counts'], ann['rle_size'], color
                )
                if img is not None:
                    pm_item = self._scene.addPixmap(QPixmap.fromImage(img))
                    pm_item.setZValue(1)
                    self._ann_items.append(pm_item)

            else:  # bbox
                x, y, w, h = ann['x'], ann['y'], ann['w'], ann['h']
                item = self._scene.addRect(x, y, w, h, pen, QBrush(Qt.BrushStyle.NoBrush))
                item.setZValue(1)
                self._ann_items.append(item)

            # Label — fixed screen size regardless of zoom level
            if label:
                txt = self._scene.addSimpleText(label)
                txt.setBrush(QBrush(color))
                font = QFont("Segoe UI", 9)
                font.setBold(True)
                txt.setFont(font)
                txt.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                txt.setPos(lx, ly)
                txt.setZValue(2)
                self._ann_items.append(txt)

        self._apply_ann_visibility()

    @staticmethod
    def _rle_to_qimage(counts: list, size: list, color: QColor) -> QImage | None:
        """Decode uncompressed COCO RLE and return a colored RGBA QImage."""
        try:
            import numpy as np
            h, w = size
            flat = np.zeros(h * w, dtype=np.uint8)
            idx, val = 0, 0
            for cnt in counts:
                end = min(idx + cnt, h * w)
                flat[idx:end] = val
                idx += cnt
                val ^= 1
            # Column-major (Fortran order) → row-major (h, w)
            mask = flat.reshape(h, w, order='F')
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            m = mask == 1
            rgba[m, 0] = color.red()
            rgba[m, 1] = color.green()
            rgba[m, 2] = color.blue()
            rgba[m, 3] = 80
            data = rgba.tobytes()
            # QImage does not copy data — call .copy() to own it
            return QImage(data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        except Exception:
            return None

    def _apply_ann_visibility(self):
        for item in self._ann_items:
            item.setVisible(self._ann_visible)

    def toggle_annotations(self):
        self._ann_visible = not self._ann_visible
        self._apply_ann_visibility()

    def set_annotations_visible(self, visible: bool):
        self._ann_visible = visible
        self._apply_ann_visibility()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    def _zoom_pct(self) -> float:
        return self.transform().m11() * 100.0

    def zoom_in(self):
        if self._zoom_pct() * self._ZOOM_IN <= self._MAX_PCT:
            self.scale(self._ZOOM_IN, self._ZOOM_IN)
            self.zoom_changed.emit(self._zoom_pct())

    def zoom_out(self):
        if self._zoom_pct() * self._ZOOM_OUT >= self._MIN_PCT:
            self.scale(self._ZOOM_OUT, self._ZOOM_OUT)
            self.zoom_changed.emit(self._zoom_pct())

    def reset_zoom(self):
        self.resetTransform()
        self.zoom_changed.emit(100.0)

    def fit_to_window(self):
        if self._pixmap_item is None:
            return
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_changed.emit(self._zoom_pct())

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.fit_to_window()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
