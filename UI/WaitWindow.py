"""
Wait Window for long-running processes in Cells Calculator.

This module provides a wait window that displays:
- Animated "Wait" indicator
- Duration counter showing elapsed time
- Information label for current operation
- Last duration label showing expected completion time
- Cancel button for cancellable operations

Features:
- Non-blocking operation using QTimer and threading
- Keeps UI responsive during long operations
- Customizable animation styles
- Progress tracking with time estimates
- Cancel functionality with callbacks
- Persistent last duration storage
"""

# Standard library imports
import time
from typing import Callable, Optional

# Third-party imports
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QApplication, QSizePolicy
)

# Local application imports
from UI.settings_manager import get_setting, set_setting

# Constants
ANIMATION_INTERVAL_MS = 500
DURATION_UPDATE_INTERVAL_MS = 100
EVENT_PROCESS_INTERVAL_MS = 50
THREAD_WAIT_TIMEOUT_MS = 1000

# Stylesheets
WAIT_WINDOW_STYLE = """
    QDialog {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 10px;
    }
    QLabel {
        color: #495057;
    }
    QPushButton {
        background-color: #6c757d;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #5a6268;
    }
    QPushButton:pressed {
        background-color: #545b62;
    }
    QFrame {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        background-color: white;
        margin: 5px;
        padding: 10px;
    }
"""

ANIMATED_LABEL_STYLE = """
    QLabel {
        color: #2c3e50;
        font-size: 16px;
        font-weight: bold;
        padding: 10px;
    }
"""

INFO_LABEL_STYLE = "font-size: 12px; margin: 5px;"
DURATION_LABEL_STYLE = "font-size: 14px; font-weight: bold; color: #007bff;"
LAST_DURATION_LABEL_STYLE = "font-size: 12px; color: #6c757d;"


class WorkerThread(QThread):
    """
    Worker thread for running long processes without blocking the UI.
    """
    
    # Signals for thread communication
    progress_update = pyqtSignal(str)  # For updating info text
    finished = pyqtSignal(object)     # For completion with result
    error_occurred = pyqtSignal(str)  # For error handling
    
    def __init__(self, target_function: Callable, *args, **kwargs):
        super().__init__()
        self.target_function = target_function
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.cancelled = False
    
    def run(self):
        """Run the target function in this thread"""
        try:
            # Add progress callback to kwargs if the function supports it
            if 'progress_callback' not in self.kwargs:
                self.kwargs['progress_callback'] = self.emit_progress
            
            self.result = self.target_function(*self.args, **self.kwargs)
            if not self.cancelled:
                self.finished.emit(self.result)
        except KeyboardInterrupt:
            # Handle user cancellation specifically
            if not self.cancelled:
                self.error_occurred.emit("Operation cancelled by user")
        except Exception as e:
            # Log and emit other exceptions
            if not self.cancelled:
                self.error_occurred.emit(f"Unexpected error: {str(e)}")
    
    def emit_progress(self, message: str):
        """Emit progress update signal"""
        if not self.cancelled:
            self.progress_update.emit(message)
    
    def cancel(self):
        """Cancel the operation"""
        self.cancelled = True
        self.quit()
        self.wait(THREAD_WAIT_TIMEOUT_MS)  # Wait up to 1 second for thread to finish


class AnimatedWaitLabel(QLabel):
    """Custom label with animated wait indicator"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation_frame = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        
        # Animation styles
        self.dots_style = ["Wait", "Wait.", "Wait..", "Wait..."]
        self.spinner_style = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_style = self.dots_style
        
        # Styling
        self.setStyleSheet(ANIMATED_LABEL_STYLE)
    
    def update_animation(self):
        """Update animation frame"""
        self.animation_frame = (self.animation_frame + 1) % len(self.current_style)
        self.setText(self.current_style[self.animation_frame])
    
    def set_animation_style(self, style: str):
        """Set animation style: 'dots' or 'spinner'"""
        if style == "spinner":
            self.current_style = self.spinner_style
        else:
            self.current_style = self.dots_style
        self.animation_frame = 0
    
    def start_animation(self):
        """Start the animation"""
        self.animation_timer.start(ANIMATION_INTERVAL_MS)
    
    def stop_animation(self):
        """Stop the animation"""
        self.animation_timer.stop()
        self.setText("Wait")


class WaitWindow(QDialog):
    """
    Wait window for long-running processes with time tracking and cancel functionality.
    Keeps UI responsive by processing events during long operations.
    """
    
    # Signal emitted when cancel button is clicked
    cancel_requested = pyqtSignal()
    # Signal emitted when process completes
    process_completed = pyqtSignal(object)
    # Signal emitted when process fails
    process_failed = pyqtSignal(str)
    
    def __init__(self, title: str = "Processing", info_text: str = "Processing, please wait...", 
                 cancellable: bool = True, parent=None):
        super().__init__(parent)
        
        # Window properties
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint | Qt.Tool)
        self.setModal(True)
        
        # Time tracking
        self.start_time: Optional[float] = None
        self.last_duration_key = f"last_duration_{title.lower().replace(' ', '_')}"
        self.cancel_callback: Optional[Callable] = None
        
        # Worker thread for non-blocking operations
        self.worker_thread: Optional[WorkerThread] = None
        
        # Timer for duration counter
        self.duration_timer = QTimer()
        self.duration_timer.timeout.connect(self.update_duration)
        
        # Timer for keeping UI responsive during blocking operations
        self.event_timer = QTimer()
        self.event_timer.timeout.connect(self.process_events)
        
        # Setup UI
        self.setup_ui(info_text, cancellable)
        
        # Load last duration
        self.load_last_duration()
        
        # Styling
        self.setStyleSheet(WAIT_WINDOW_STYLE)
    
    def process_events(self):
        """Process Qt events to keep UI responsive"""
        QApplication.processEvents()
    
    def run_threaded_process(self, target_function, *args, **kwargs):
        """
        Run a function in a separate thread to avoid blocking the UI.
        
        Args:
            target_function: Function to run in background
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        """
        if self.worker_thread and self.worker_thread.isRunning():
            return  # Already running a process
        
        # Create and setup worker thread
        self.worker_thread = WorkerThread(target_function, *args, **kwargs)
        self.worker_thread.progress_update.connect(self.set_info_text)
        self.worker_thread.finished.connect(self._on_process_completed)
        self.worker_thread.error_occurred.connect(self._on_process_failed)
        
        # Start the process
        self.start_wait()
        self.worker_thread.start()
    
    def run_blocking_process_responsive(self, target_function: Callable, *args, **kwargs):
        """
        Run a blocking function while keeping the UI responsive.
        This method processes Qt events periodically during execution.
        
        Args:
            target_function: Function to run (blocking)
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Result of the target function
        """
        self.start_wait()
        
        # Start event processing timer to keep UI responsive
        self.event_timer.start(EVENT_PROCESS_INTERVAL_MS)  # Process events every 50ms
        
        try:
            # Add progress callback if supported
            def progress_callback(message: str):
                self.set_info_text(message)
                QApplication.processEvents()  # Process events immediately on progress
            
            if 'progress_callback' not in kwargs:
                kwargs['progress_callback'] = progress_callback
            
            # Run the function
            result = target_function(*args, **kwargs)
            
            self.process_completed.emit(result)
            return result
            
        except Exception as e:
            self.process_failed.emit(str(e))
            raise
        finally:
            self.event_timer.stop()
            self.stop_wait()
    
    def _on_process_completed(self, result):
        """Handle threaded process completion"""
        self.stop_wait()
        self.process_completed.emit(result)
    
    def _on_process_failed(self, error_message):
        """Handle threaded process failure"""
        self.stop_wait()
        self.process_failed.emit(error_message)
    
    def setup_ui(self, info_text: str, cancellable: bool):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Main content frame
        content_frame = QFrame()
        content_layout = QVBoxLayout(content_frame)
        content_layout.setSpacing(10)
        
        # Wait animation label
        self.wait_label = AnimatedWaitLabel()
        self.wait_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.wait_label)
        
        # Info label
        self.info_label = QLabel(info_text)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(INFO_LABEL_STYLE)
        content_layout.addWidget(self.info_label)
        
        # Time information frame
        time_frame = QFrame()
        time_layout = QVBoxLayout(time_frame)
        time_layout.setSpacing(5)
        
        # Duration counter
        self.duration_label = QLabel("Elapsed: 00:00")
        self.duration_label.setAlignment(Qt.AlignCenter)
        self.duration_label.setStyleSheet(DURATION_LABEL_STYLE)
        self.duration_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.duration_label.setMinimumWidth(220)
        time_layout.addWidget(self.duration_label)
        
        # Last duration label
        self.last_duration_label = QLabel("Estimated: Unknown")
        self.last_duration_label.setAlignment(Qt.AlignCenter)
        self.last_duration_label.setStyleSheet(LAST_DURATION_LABEL_STYLE)
        self.last_duration_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.last_duration_label.setMinimumWidth(220)
        time_layout.addWidget(self.last_duration_label)
        
        content_layout.addWidget(time_frame)
        layout.addWidget(content_frame)
        
        # Cancel button (if cancellable)
        if cancellable:
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.on_cancel_clicked)
            self.cancel_button.setFixedSize(80, 30)
            button_layout.addWidget(self.cancel_button)
            
            button_layout.addStretch()
            layout.addLayout(button_layout)
    
    def set_info_text(self, text: str):
        """Update the information text"""
        self.info_label.setText(text)
    
    def set_cancel_callback(self, callback: Callable):
        """Set callback function to be called when cancel is clicked"""
        self.cancel_callback = callback
    
    def on_cancel_clicked(self):
        """Handle cancel button click"""
        self.cancel_requested.emit()
        
        # Cancel worker thread if running
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
        
        # Call custom cancel callback
        if self.cancel_callback:
            self.cancel_callback()
        
        self.close()
    
    def start_wait(self):
        """Start the wait process"""
        self.start_time = time.time()
        self.wait_label.start_animation()
        self.duration_timer.start(100)  # Update every 100ms for smooth counter
        self.show()
    
    def stop_wait(self, save_duration: bool = True):
        """Stop the wait process"""
        if self.start_time and save_duration:
            elapsed = time.time() - self.start_time
            self.save_last_duration(elapsed)
        
        self.wait_label.stop_animation()
        self.duration_timer.stop()
        self.close()
    
    def update_duration(self):
        """Update the duration counter display"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.duration_label.setText(f"Elapsed: {minutes:02d}:{seconds:02d}")
            self.duration_label.adjustSize()
    
    def load_last_duration(self):
        """Load last duration from settings and display estimated time"""
        last_duration = get_setting(self.last_duration_key, None)
        if last_duration:
            minutes = int(last_duration // 60)
            seconds = int(last_duration % 60)
            self.last_duration_label.setText(f"Estimated: ~{minutes:02d}:{seconds:02d}")
        else:
            self.last_duration_label.setText("Estimated: Unknown")
    
    def save_last_duration(self, duration: float):
        """Save duration to settings for future reference"""
        set_setting(self.last_duration_key, duration)
    
    def set_animation_style(self, style: str):
        """Set animation style: 'dots' or 'spinner'"""
        self.wait_label.set_animation_style(style)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Cancel worker thread if running
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
        
        self.event_timer.stop()
        self.stop_wait(save_duration=False)
        super().closeEvent(event)


# Convenience functions for common usage patterns

def show_wait_window(title="Processing", info_text="Processing, please wait...", 
                    cancellable=True, parent=None, animation_style="dots"):
    """
    Show a wait window and return the instance for control.
    
    Args:
        title: Window title
        info_text: Information text to display
        cancellable: Whether to show cancel button
        parent: Parent widget
        animation_style: 'dots' or 'spinner'
    
    Returns:
        WaitWindow instance
    """
    wait_window = WaitWindow(title, info_text, cancellable, parent)
    wait_window.set_animation_style(animation_style)
    wait_window.start_wait()
    return wait_window


def run_with_wait_window(target_function, *args, title="Processing", 
                        info_text="Processing, please wait...", 
                        cancellable=True, parent=None, threaded=True, **kwargs):
    """
    Run a function with a wait window, handling UI responsiveness automatically.
    
    Args:
        target_function: Function to execute
        *args: Arguments for the function
        title: Window title
        info_text: Information text
        cancellable: Whether operation can be cancelled
        parent: Parent widget
        threaded: If True, run in separate thread; if False, use event processing
        **kwargs: Keyword arguments for the function
    
    Returns:
        Result of the function (only for non-threaded mode)
    
    Usage:
        # Threaded mode (recommended for long operations)
        wait = run_with_wait_window(my_long_function, arg1, arg2, threaded=True)
        wait.process_completed.connect(lambda result: print(f"Done: {result}"))
        wait.process_failed.connect(lambda error: print(f"Error: {error}"))
        
        # Non-threaded mode (for functions that need main thread)
        result = run_with_wait_window(my_function, arg1, arg2, threaded=False)
    """
    wait_window = WaitWindow(title, info_text, cancellable, parent)
    
    if threaded:
        # Run in separate thread
        wait_window.run_threaded_process(target_function, *args, **kwargs)
        return wait_window
    else:
        # Run in main thread but keep UI responsive
        return wait_window.run_blocking_process_responsive(target_function, *args, **kwargs)


def create_wait_context(title="Processing", info_text="Processing, please wait...", 
                       cancellable=True, parent=None, threaded=False):
    """
    Create a context manager for wait window with responsive UI.
    
    Args:
        title: Window title
        info_text: Information text
        cancellable: Whether operation can be cancelled
        parent: Parent widget
        threaded: If True, requires manual thread management
    
    Usage:
        # For blocking operations (keeps UI responsive)
        with create_wait_context("Long Process", "Calculating...") as wait:
            time.sleep(5)  # Long operation
            wait.set_info_text("Almost done...")
            time.sleep(2)
        
        # For operations that support progress callbacks
        def my_process(progress_callback=None):
            if progress_callback:
                progress_callback("Step 1...")
            time.sleep(2)
            if progress_callback:
                progress_callback("Step 2...")
            time.sleep(2)
            return "Result"
        
        with create_wait_context("Processing") as wait:
            result = my_process(progress_callback=wait.set_info_text)
    """
    class ResponsiveWaitContext:
        def __init__(self, title: str, info_text: str, cancellable: bool, parent, threaded: bool):
            self.wait_window = WaitWindow(title, info_text, cancellable, parent)
            self.threaded = threaded
        
        def __enter__(self):
            if not self.threaded:
                self.wait_window.start_wait()
                # Start event processing for responsiveness
                self.wait_window.event_timer.start(EVENT_PROCESS_INTERVAL_MS)
            return self.wait_window
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.threaded:
                self.wait_window.event_timer.stop()
            # Always stop wait, even if exception occurred
            self.wait_window.stop_wait()
    
    return ResponsiveWaitContext(title, info_text, cancellable, parent, threaded)