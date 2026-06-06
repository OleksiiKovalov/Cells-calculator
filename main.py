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
from collections import OrderedDict
import json
import sys
from UI.app_globals import FILENAME_MODEL_CONFIG, get_global, register_model, set_global
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon
from UI.splashscreen import splash_manager

if __name__ == '__main__':
    from UI.errorhandling import app_logger
    # Create a QApplication instance
    app_logger().info("Starting application...")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("ui/Cells-calculator-v3-icon2.png"))    
    
    # Create and show splash screen
    app_logger().info("Creating splash screen...")
    splash_manager.init_splash()

    app_logger().info("Loading modules...")

    known_models = {
        "cellcounter": 'model.CellCounter.CellCounter',
        "cellpose": 'model.CellposeSegmenter.CellposeSegmenter',
        "yolo": 'model.YOLOSegmenter.YoloSegmenter',
        "instanseg": 'model.InstanSegSegmenter.InstansegSegmenter',
        "stardist": 'model.StardistSegmenter.StardistSegmenter'
    }

    with open(FILENAME_MODEL_CONFIG, 'r') as f:
        models = json.load(f, object_pairs_hook=OrderedDict)
        models = OrderedDict((k, v) for k, v in models.items() if 'enabled' not in v or v.get('enabled','true').lower() == 'true')
        set_global('loaded_models', models)

    #register all models from config to make them available in UI/processing    
    for model_name, model_data in get_global('loaded_models').items():
        model_type = model_data.get('model_type')
        register_model(model_type, known_models.get(model_type), model_data.get('preload', False))

    import UI.imports  # Ensure all UI imports are done before MainWindow

    try:
        # Create progress callback function
        def progress_callback(value, message):
            splash_manager.update_splash(value, message)
        
        # Attempt to create the main window with progress callback
        app_logger().info("Initializing main window...")
        from UI.MainWindow import MainWindow
        window = MainWindow(progress_callback=progress_callback)
        set_global('main_window_ref', window)
        
        # Show main window and close splash screen
        def show_main_window():
            #globalsplash.finish(window)
            splash_manager.close_splash()
            window.showMaximized()
        
        # Brief delay to ensure splash screen is visible and initialization complete
        QTimer.singleShot(200, show_main_window)
        
        # Start the application event loop
        app_logger().info("Entering application event loop...")
        sys.exit(app.exec_())
        app_logger().info("Exiting application event loop...")
    except Exception as e:
        traceback.print_exc()
        # Close splash screen immediately to show error dialog on top
        splash_manager.close_splash()
        
        # Create error dialog with WindowStaysOnTopHint to ensure it appears in front
        error_box = QMessageBox()
        error_box.setWindowFlags(error_box.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Critical Error")
        error_box.setText(f"Failed to initialize or run application:\n\n{str(e)}")
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()
        
        sys.exit(1)
