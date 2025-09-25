"""
Main entry point for the Cells Calculator application.

This module serves as the application launcher, handling:
- Application startup and initialization
- Splash screen display with progress updates
- MainWindow instantiation with progress callbacks
- Error handling during application startup
- Application icon and window management

The actual MainWindow class is now located in MainWindow.py
The SplashScreen class is located in splashscreen.py

Usage:
    python main.py
"""
import sys
print("Importing modules...")
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from UI.splashscreen import SplashScreen,init_splash,close_splash,update_splash, globalsplash

if __name__ == '__main__':
    from UI.errorhandling import app_logger
    # Create a QApplication instance
    app_logger().info("Starting application...")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("ui/Cells-calculator-v3-icon2.png"))    
    
    # Create and show splash screen
    app_logger().info("Creating splash screen...")
    init_splash()
    app_logger().info("Loading modules...")
    import UI.imports  # Ensure all UI imports are done before MainWindow

    try:
        # Create progress callback function
        def progress_callback(value, message):
            update_splash(value, message)
        
        # Attempt to create the main window with progress callback
        app_logger().info("Initializing main window...")
        from UI.MainWindow import MainWindow
        window = MainWindow(progress_callback=progress_callback)
        
        # Show main window and close splash screen
        def show_main_window():
            #globalsplash.finish(window)
            close_splash()
            window.showMaximized()
        
        # Brief delay to ensure splash screen is visible and initialization complete
        QTimer.singleShot(200, show_main_window)
        
        # Start the application event loop
        app_logger().info("Entering application event loop...")
        sys.exit(app.exec_())
        app_logger().info("Exiting application event loop...")
    except Exception as e:
        traceback.print_exc()
        # Show error on splash screen briefly before showing dialog
        globalsplash.show_error(str(e))
        QTimer.singleShot(2000, close_splash)  # Close splash after 2 seconds
        
        # If an exception occurs, display a critical error message and exit the application
        QMessageBox.critical(None, "Critical Error", f"Failed to initialize or run application:\n\n{str(e)}", QMessageBox.Ok)
        sys.exit(1)
