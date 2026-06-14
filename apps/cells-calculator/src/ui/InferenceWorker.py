"""QThread worker for running model inference off the main thread."""

import ctypes
import threading

from PyQt5.QtCore import QThread, pyqtSignal


class InferenceWorker(QThread):
    """Background thread that runs model inference and reports results via signals."""

    status_changed = pyqtSignal(str)   # safe to connect to status bar
    finished = pyqtSignal(object)      # emits detections DataFrame
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, model, image, parent=None):
        """Store the model and image to run inference on when the thread starts."""
        super().__init__(parent)
        self._model = model
        self._image = image
        self._python_thread_id = None

    def cancel(self):
        """
        Inject SystemExit into the worker thread at the next Python bytecode
        boundary. Non-blocking — cleanup is driven by the `cancelled` signal.
        """
        if self._python_thread_id is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self._python_thread_id),
                ctypes.py_object(SystemExit),
            )

    def run(self):
        """Run inference, emitting ``finished``, ``cancelled`` or ``error`` accordingly."""
        self._python_thread_id = threading.current_thread().ident
        try:
            self.status_changed.emit("Running inference…")
            result = self._model.inference(self._image)
            self.finished.emit(result)
        except SystemExit:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))
