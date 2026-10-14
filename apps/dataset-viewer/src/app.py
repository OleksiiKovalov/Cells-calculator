import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def main():
    # High-DPI scaling is always on in Qt6; no attribute needed.
    app = QApplication(sys.argv)
    app.setApplicationName("Dataset Viewer")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
