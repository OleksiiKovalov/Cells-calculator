"""
Splash screen module for Cells Calculator application.

This module provides a custom splash screen with progress bar for application initialization.
The splash screen displays the application logo, title, and a progress bar to show loading progress
during the potentially long initialization of the main window and its components.

Features:
- Custom styled splash screen with gradient background
- Application logo/icon display (with fallback)
- Progress bar with custom styling
- Status message display
- Error display capability

Usage:
    from splashscreen import SplashScreen
    
    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    splash.update_progress(50, "Loading...")
"""

from typing import Optional
import logging

# Third-party imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor, QPen, QLinearGradient
from PyQt5.QtWidgets import QSplashScreen, QProgressBar, QLabel, QApplication

# Local application imports
from UI.errorhandling import app_logger

# Constants
WINDOW_WIDTH = 450
WINDOW_HEIGHT = 320
ICON_SIZE = 80
PROGRESS_BAR_WIDTH = 350
PROGRESS_BAR_HEIGHT = 25
STATUS_LABEL_HEIGHT = 25

# Colors
TITLE_COLOR = QColor(25, 25, 112)  # Midnight blue
SUBTITLE_COLOR = QColor(70, 70, 70)
ICON_BG_COLOR = QColor(70, 130, 180)  # Steel blue
ICON_BORDER_COLOR = QColor(25, 25, 112)  # Midnight blue
GRADIENT_START = Qt.white
GRADIENT_END = QColor(240, 248, 255)  # Light blue

# Styles
PROGRESS_BAR_STYLE = """
    QProgressBar {
        border: 2px solid #4682B4;
        border-radius: 12px;
        background-color: #F0F8FF;
        text-align: center;
        font-weight: bold;
        color: #191970;
    }
    QProgressBar::chunk {
        background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #87CEEB, stop: 1 #4682B4);
        border-radius: 10px;
        margin: 1px;
    }
"""

STATUS_LABEL_STYLE = """
    color: #191970; 
    font-size: 13px; 
    font-weight: bold;
    background: transparent;
"""

ERROR_STATUS_STYLE = """
    color: #8B0000; 
    font-size: 13px; 
    font-weight: bold;
    background: transparent;
"""

ERROR_PROGRESS_STYLE = """
    QProgressBar {
        border: 2px solid #8B0000;
        border-radius: 12px;
        background-color: #FFE4E1;
        text-align: center;
        font-weight: bold;
        color: #8B0000;
    }
    QProgressBar::chunk {
        background-color: #CD5C5C;
        border-radius: 10px;
        margin: 1px;
    }
"""

class SplashPixmapPainter:
    """Helper class for painting the splash screen pixmap."""
    
    __slots__ = ('logger',)
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def create_pixmap(self) -> QPixmap:
        """Create the splash screen pixmap with background and content."""
        pixmap = QPixmap(WINDOW_WIDTH, WINDOW_HEIGHT)
        pixmap.fill(Qt.white)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        try:
            self._draw_background(painter)
            self._draw_icon(painter)
            self._draw_text(painter)
        finally:
            painter.end()
        
        return pixmap
    
    def _draw_background(self, painter: QPainter) -> None:
        """Draw the gradient background."""
        gradient = QLinearGradient(0, 0, 0, WINDOW_HEIGHT)
        gradient.setColorAt(0, GRADIENT_START)
        gradient.setColorAt(1, GRADIENT_END)
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    def _draw_icon(self, painter: QPainter) -> None:
        """Draw the application icon or fallback."""
        try:
            icon_pixmap = QPixmap("UI/Cells-calculator-v3-icon2.png")
            if not icon_pixmap.isNull():
                icon_pixmap = icon_pixmap.scaled(
                    ICON_SIZE, ICON_SIZE, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                icon_x = (WINDOW_WIDTH - icon_pixmap.width()) // 2
                painter.drawPixmap(icon_x, 40, icon_pixmap)
                return
        except (FileNotFoundError, OSError) as e:
            self.logger.warning(f"Could not load icon: {e}")
        
        # Fallback: draw a simple colored circle as logo
        painter.setBrush(ICON_BG_COLOR)
        painter.setPen(QPen(ICON_BORDER_COLOR, 3))
        painter.drawEllipse(185, 40, ICON_SIZE, ICON_SIZE)
        
        # Draw "CC" text in the circle
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 24, QFont.Bold))
        painter.drawText(185, 40, ICON_SIZE, ICON_SIZE, Qt.AlignCenter, "CC")
    
    def _draw_text(self, painter: QPainter) -> None:
        """Draw the title and subtitle text."""
        # Title
        painter.setPen(TITLE_COLOR)
        title_font = QFont("Arial", 20, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(0, 140, WINDOW_WIDTH, 40, Qt.AlignCenter, "Cells Calculator v3")
        
        # Subtitle
        painter.setPen(SUBTITLE_COLOR)
        subtitle_font = QFont("Arial", 12)
        painter.setFont(subtitle_font)
        painter.drawText(0, 175, WINDOW_WIDTH, 25, Qt.AlignCenter, "Advanced Cell Analysis Tool")


class SplashScreen(QSplashScreen):
    """Custom splash screen with progress bar for application initialization"""
    
    __slots__ = ('max_progress', '_current_progress', 'logger', 'progress_bar', 'status_label')
    
    def __init__(self, max_progress: int = 100):
        """
        Initialize the splash screen.
        
        Args:
            max_progress: Maximum value for the progress bar
        """
        self.max_progress = max_progress
        self._current_progress = 0
        
        self.logger = app_logger()
        painter = SplashPixmapPainter(self.logger)
        splash_pixmap = painter.create_pixmap()
        
        super().__init__(splash_pixmap, Qt.WindowStaysOnTopHint)
        
        # Setup UI components
        self._setup_progress_bar()
        self._setup_status_label()
    
    def _setup_progress_bar(self) -> None:
        """Setup the progress bar widget."""
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.max_progress)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedSize(PROGRESS_BAR_WIDTH, PROGRESS_BAR_HEIGHT)
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        
        self.progress_bar.setParent(self)
        self.progress_bar.move(50, 260)
        self.progress_bar.show()
    
    def _setup_status_label(self) -> None:
        """Setup the status label widget."""
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(STATUS_LABEL_STYLE)
        
        self.status_label.setParent(self)
        self.status_label.move(50, 230)
        self.status_label.resize(PROGRESS_BAR_WIDTH, STATUS_LABEL_HEIGHT)
        self.status_label.show()
    
    def update_progress(self, value: int, message: str = "") -> None:
        """
        Update progress bar and status message.
        
        Args:
            value: New progress value
            message: Status message to display
        """
        clamped_value = max(0, min(self.max_progress, value))
        self._current_progress = clamped_value
        self.progress_bar.setValue(clamped_value)
        if message:
            self.status_label.setText(message)
            self.logger.info(f"SplashScreen:{clamped_value}: {message}")
        QApplication.processEvents()
    
    def show_error(self, error_message: str) -> None:
        """
        Show error message on splash screen.
        
        Args:
            error_message: Error message to display
        """
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet(ERROR_STATUS_STYLE)
        self.progress_bar.setStyleSheet(ERROR_PROGRESS_STYLE)
        QApplication.processEvents()


class SplashScreenManager:
    """Manager class for splash screen to avoid global variables."""
    
    __slots__ = ('_splash', '_progress')
    
    def __init__(self):
        self._splash: Optional[SplashScreen] = None
        self._progress = 0
    
    def init_splash(self, max_progress: int = 100) -> None:
        """Initialize and show the splash screen."""
        if self._splash is None:
            self._progress = 0
            self._splash = SplashScreen(max_progress)
            self._splash.show()
            self._splash.update_progress(5, "Starting application...")
    
    def close_splash(self) -> None:
        """Close the splash screen."""
        if self._splash is not None:
            self._splash.close()
            self._splash = None
    
    def update_splash(self, value: int, message: str = "") -> None:
        """Update splash screen progress."""
        if self._splash is not None:
            self._progress = value
            self._splash.update_progress(value, message)
    
    def show_error(self, message: str) -> None:
        """Show error on splash screen."""
        if self._splash is not None:
            self._splash.show_error(message)


# Global manager instance (better than raw globals)
splash_manager = SplashScreenManager()

# Backward compatibility functions
def init_splash() -> None:
    splash_manager.init_splash()

def close_splash() -> None:
    splash_manager.close_splash()

def update_splash(value: int, message: str = "") -> None:
    splash_manager.update_splash(value, message)

def show_splash_error(message: str) -> None:
    splash_manager.show_error(message)        
        