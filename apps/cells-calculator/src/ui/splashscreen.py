"""Splash screen — dark theme matching the main application visual style."""

# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont, QPainter, QColor, QPen
from pathlib import Path
from PySide6.QtWidgets import QSplashScreen, QProgressBar, QLabel, QApplication

# Local application imports
from ui.errorhandling import app_logger

# ── colour palette (mirrors InfoPanel / app dark theme) ──────────────────────
_BG          = QColor(0x2d, 0x2d, 0x2d)       # main background
_PANEL       = QColor(0x38, 0x38, 0x38)       # inner panel card
_TITLEBAR    = QColor(0x48, 0x48, 0x48)       # title-bar strip
_BORDER      = QColor(0x68, 0x68, 0x68)       # border / separator
_ACCENT      = QColor(0x4a, 0x9f, 0xd4)       # blue accent
_TEXT_TITLE  = QColor(0xe0, 0xe0, 0xe0)       # bright text
_TEXT_SUB    = QColor(0xb0, 0xb0, 0xb0)       # dimmer text
_TEXT_STATUS = QColor(0xd4, 0xd4, 0xd4)       # status / console text
_ERROR       = QColor(0xc0, 0x39, 0x2b)       # error red

_W, _H = 480, 290


class SplashScreen(QSplashScreen):
    """Dark-themed splash screen matching the main application style."""

    def __init__(self, maxprogress=100):
        """Render the splash artwork and create the progress bar and status label."""
        pixmap = QPixmap(_W, _H)
        pixmap.fill(_BG)

        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)

        # ── outer border ────────────────────────────────────────────────────
        p.setPen(QPen(_BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, _W - 2, _H - 2, 6, 6)

        # ── title-bar strip ──────────────────────────────────────────────────
        p.setPen(Qt.NoPen)
        p.setBrush(_TITLEBAR)
        p.drawRoundedRect(1, 1, _W - 2, 36, 6, 6)
        p.fillRect(1, 20, _W - 2, 17, _TITLEBAR)   # square off bottom of strip

        # title text in strip
        p.setPen(_TEXT_TITLE)
        f = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(f)
        p.drawText(0, 1, _W, 36, Qt.AlignCenter, "Cells Calculator  ·  v4.0")

        # ── icon / logo ──────────────────────────────────────────────────────
        icon_y = 54
        icon_loaded = False
        try:
            icon_px = QPixmap(str(Path(__file__).resolve().parent.parent / "resources" / "Cells-calculator-v3-icon2.png"))
            if not icon_px.isNull():
                icon_px = icon_px.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                p.drawPixmap((_W - icon_px.width()) // 2, icon_y, icon_px)
                icon_loaded = True
        except Exception:
            pass

        if not icon_loaded:
            # Fallback: filled circle with initials
            cx, cy, r = _W // 2, icon_y + 32, 32
            p.setBrush(_ACCENT)
            p.setPen(QPen(_BORDER, 1))
            p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
            p.setPen(Qt.white)
            p.setFont(QFont("Segoe UI", 20, QFont.Bold))
            p.drawText(cx - r, cy - r, r * 2, r * 2, Qt.AlignCenter, "CC")

        # ── app title ────────────────────────────────────────────────────────
        p.setPen(_TEXT_TITLE)
        p.setFont(QFont("Segoe UI", 17, QFont.Bold))
        p.drawText(0, 128, _W, 32, Qt.AlignCenter, "Cells Calculator")

        # ── subtitle ─────────────────────────────────────────────────────────
        p.setPen(_TEXT_SUB)
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(0, 160, _W, 22, Qt.AlignCenter, "Advanced Cell Analysis Tool")

        # ── separator line above progress area ───────────────────────────────
        p.setPen(QPen(_BORDER, 1))
        p.drawLine(20, 192, _W - 20, 192)

        p.end()

        super().__init__(pixmap, Qt.WindowStaysOnTopHint)

        # ── progress bar ─────────────────────────────────────────────────────
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, maxprogress)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedSize(_W - 40, 6)
        self.progress_bar.move(20, 248)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #484848;
            }
            QProgressBar::chunk {
                background-color: #4a9fd4;
                border-radius: 3px;
            }
        """)
        self.progress_bar.show()

        # ── status label ─────────────────────────────────────────────────────
        self.status_label = QLabel("Initializing…", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedSize(_W - 40, 22)
        self.status_label.move(20, 220)
        self.status_label.setStyleSheet(
            "color: #d4d4d4; font-family: 'Segoe UI'; font-size: 9pt; background: transparent;"
        )
        self.status_label.show()

    def update_progress(self, value, message=""):
        """Update progress bar and status message."""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
            app_logger().info(f"SplashScreen:{value}: {message}")
        QApplication.processEvents()

    def show_error(self, error_message):
        """Show error state on the splash screen."""
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet(
            "color: #e74c3c; font-family: 'Segoe UI'; font-size: 9pt; "
            "font-weight: bold; background: transparent;"
        )
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #3a1010;
            }
            QProgressBar::chunk {
                background-color: #c0392b;
                border-radius: 3px;
            }
        """)
        QApplication.processEvents()

globalsplash = None  # Global reference to splash screen instance
globalsplashprogress = 0

def init_splash():
    """Create and show the global splash screen if one is not already active."""
    global globalsplash, globalsplashprogress
    if globalsplash is None:
        globalsplashprogress = 0
        globalsplash = SplashScreen(33)
        globalsplash.show()
        globalsplash.update_progress(0, "Starting application...")

def close_splash():
    """Close and discard the global splash screen if one is active."""
    global globalsplash
    if globalsplash is not None:
        globalsplash.close()
        globalsplash = None

def update_splash(value, message=""):
    """Advance the global splash progress by one step and update its message."""
    global globalsplash
    global globalsplashprogress
    globalsplashprogress +=  1
    value = globalsplashprogress
    if globalsplash is not None:
        globalsplash.update_progress(value, message)
        
def show_splash_error(message):
    """Display an error message on the global splash screen if one is active."""
    global globalsplash
    if globalsplash is not None:
        globalsplash.show_error(message)
        