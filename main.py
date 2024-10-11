from PyQt5.QtWidgets import QApplication, QMessageBox

import sys
from UI.main_window import MainWindow
import traceback
print('111s ')
if __name__ == '__main__':
    # Create a QApplication instance
    app = QApplication(sys.argv)
    try:
        print('111s ')
        # Attempt to create and show the main window
        window = MainWindow()
        window.showMaximized()
        # Start the application event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        traceback.print_exc()
        # If an exception occurs, display a critical error message and exit the application
        QMessageBox.critical(None, "Critical Error", str(e), QMessageBox.Ok)
        sys.exit(1)
