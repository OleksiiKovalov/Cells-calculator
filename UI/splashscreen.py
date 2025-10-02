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

# Third-party imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor, QPen, QLinearGradient
from PyQt5.QtWidgets import QSplashScreen, QProgressBar, QLabel, QApplication

# Local application imports
from UI.errorhandling import app_logger


class SplashScreen(QSplashScreen):
    """Custom splash screen with progress bar for application initialization"""
    
    def __init__(self,maxprogress=100):
        # Create a splash screen pixmap
        splash_pixmap = QPixmap(450, 320)
        splash_pixmap.fill(Qt.white)
        
        # Draw application title and logo on splash screen
        painter = QPainter(splash_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background gradient
        gradient = QLinearGradient(0, 0, 0, 320)
        gradient.setColorAt(0, Qt.white)
        gradient.setColorAt(1, QColor(240, 248, 255))  # Light blue
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, 450, 320)
        
        # Try to load application icon
        try:
            icon_pixmap = QPixmap("UI/Cells-calculator-v3-icon2.png")
            if not icon_pixmap.isNull():
                # Scale icon to appropriate size
                icon_pixmap = icon_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # Draw icon centered horizontally, positioned in upper part
                icon_x = (450 - icon_pixmap.width()) // 2
                painter.drawPixmap(icon_x, 40, icon_pixmap)
            else:
                raise FileNotFoundError("Icon file not found")
        except:
            # Fallback: draw a simple colored circle as logo
            painter.setBrush(QColor(70, 130, 180))  # Steel blue
            painter.setPen(QPen(QColor(25, 25, 112), 3))  # Midnight blue border
            painter.drawEllipse(185, 40, 80, 80)
            
            # Draw "CC" text in the circle
            painter.setPen(Qt.white)
            painter.setFont(QFont("Arial", 24, QFont.Bold))
            painter.drawText(185, 40, 80, 80, Qt.AlignCenter, "CC")
        
        # Set font for title
        painter.setPen(QColor(25, 25, 112))  # Midnight blue
        title_font = QFont("Arial", 20, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(0, 140, 450, 40, Qt.AlignCenter, "Cells Calculator v3")
        
        # Set font for subtitle
        painter.setPen(QColor(70, 70, 70))
        subtitle_font = QFont("Arial", 12)
        painter.setFont(subtitle_font)
        painter.drawText(0, 175, 450, 25, Qt.AlignCenter, "Advanced Cell Analysis Tool")
        
        painter.end()
        
        super().__init__(splash_pixmap, Qt.WindowStaysOnTopHint)
        
        # Create progress bar with custom style
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, maxprogress)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedSize(350, 25)
        self.progress_bar.setStyleSheet("""
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
        """)
        
        # Create status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            color: #191970; 
            font-size: 13px; 
            font-weight: bold;
            background: transparent;
        """)
        
        # Position widgets on splash screen
        self.progress_bar.setParent(self)
        self.progress_bar.move(50, 260)
        self.progress_bar.show()
        
        self.status_label.setParent(self)
        self.status_label.move(50, 230)
        self.status_label.resize(350, 25)
        self.status_label.show()
        
    def update_progress(self, value, message=""):
        """Update progress bar and status message"""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
            app_logger().info(f"SplashScreen:{value}: {message}")
        QApplication.processEvents()
    
    def show_error(self, error_message):
        """Show error message on splash screen"""
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("""
            color: #8B0000; 
            font-size: 13px; 
            font-weight: bold;
            background: transparent;
        """)
        self.progress_bar.setStyleSheet("""
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
        """)
        QApplication.processEvents()

globalsplash = None  # Global reference to splash screen instance
globalsplashprogress = 0

def init_splash():
    global globalsplash
    if globalsplash is None:
        globalspalshprogress = 0
        globalsplash = SplashScreen(33)
        globalsplash.show()
        globalsplash.update_progress(5, "Starting application...")

def close_splash():
    global globalsplash
    if globalsplash is not None:
        globalsplash.close()
        globalsplash = None

def update_splash(value, message=""):
    global globalsplash
    global globalsplashprogress
    globalsplashprogress +=  1
    value = globalsplashprogress
    if globalsplash is not None:
        globalsplash.update_progress(value, message)
        
def show_splash_error(message):
    global globalsplash
    if globalsplash is not None:
        globalsplash.show_error(message)        
        