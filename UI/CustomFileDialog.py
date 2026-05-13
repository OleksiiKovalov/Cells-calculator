"""
Custom file dialog widget with advanced file selection features.

Provides a customizable file selection dialog with table-based file
listing, directory navigation, file preview, and filtering capabilities.
Includes FileTableModel for managing file data and CustomFileDialog
for user interaction.

Key components:
- FileTableModel: Abstract table model for dynamic file data display
- CustomFileDialog: Main dialog for file selection with preview
"""

# Standard library imports
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any, Union

# Third-party imports
from PyQt5.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QDir, QFileInfo, 
    pyqtSignal, QEvent, QDateTime
)
from PyQt5.QtGui import QBrush, QColor, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTreeView, QHeaderView, QLabel, QComboBox, QLineEdit, 
    QMessageBox, QFileIconProvider, QMenu, QAction,
    QScrollArea, QSplitter, QFrame
)

class FileTableModel(QAbstractTableModel):
    """Custom table model for file listing with configurable columns"""
    
    def __init__(self, parent: Optional[Any] = None):
        """
        Initialize the file table model.
        
        Args:
            parent: Parent object.
        """
        super().__init__(parent)
        self.files: list[QFileInfo] = []
        self.current_dir = QDir.home()
        self.columns: list[Dict[str, Any]] = []
        self.icon_provider = QFileIconProvider()
    
    def set_columns(self, columns: List[Dict[str, Any]]) -> None:
        """
        Set the columns configuration.
        columns: list of dicts with keys: 'name', 'width', 'data_func'
        Example:
        [
            {'name': 'Name', 'width': 200, 'data_func': lambda info: info.fileName()},
            {'name': 'Size', 'width': 100, 'data_func': lambda info: self.format_size(info.size())},
            {'name': 'Process Time', 'width': 120, 'data_func': lambda info: self.get_process_time(info)}
        ]
        """
        self.beginResetModel()
        self.columns = columns
        self.endResetModel()
    
    def set_color_rule(self, color_rule: Callable[[QFileInfo], Optional[QColor]]) -> None:
        """
        Set a function to determine text color for each file.
        
        Args:
            color_rule: Function that takes QFileInfo and returns QColor or None
        """
        self.color_rule = color_rule
        self.dataChanged.emit(self.createIndex(0, 0), 
                             self.createIndex(self.rowCount()-1, self.columnCount()-1))
    
    def set_background_color_rule(self, background_color_rule: Callable[[QFileInfo], Optional[QColor]]) -> None:
        """
        Set a function to determine background color for each file.
        
        Args:
            background_color_rule: Function that takes QFileInfo and returns QColor or None
        """
        self.background_color_rule = background_color_rule
        self.dataChanged.emit(self.createIndex(0, 0), 
                             self.createIndex(self.rowCount()-1, self.columnCount()-1))
    
    def set_alternating_colors(self, enabled: bool) -> None:
        """
        Enable or disable alternating row colors (zebra stripes).
        
        Args:
            enabled: True to enable alternating colors, False to disable
        """
        self.alternating_colors = enabled
        self.dataChanged.emit(self.createIndex(0, 0), 
                             self.createIndex(self.rowCount()-1, self.columnCount()-1))
    
    def get_current_file_filters(self) -> QDir.Filters:
        """
        Get current file filters based on the dialog's filter settings.
        This method is called by the file table model to get current filtering.
        """
        # Get the parent dialog if available
        parent_dialog = self.parent()
        if parent_dialog is not None and hasattr(parent_dialog, 'get_current_file_extensions'):
            extensions = parent_dialog.get_current_file_extensions()
            if extensions:
                # Apply custom filtering - we'll filter in refresh_files method
                return QDir.NoFilter
        
        # Default: no additional filtering, show all files
        return QDir.NoFilter
    
    def set_directory(self, path: Union[str, Path]) -> None:
        """Set the current directory and refresh file list"""
        self.beginResetModel()
        self.current_dir = QDir(str(path))
        self.refresh_files()
        self.endResetModel()
    
    def refresh_files(self) -> None:
        """Refresh the file list from current directory"""
        self.files = []
        
        # Add parent directory entry if not at root
        if not self.current_dir.isRoot():
            parent_info = QFileInfo(self.current_dir.absolutePath() + "/..")
            self.files.append(parent_info)
        
        # Add directories first
        for entry in self.current_dir.entryInfoList(QDir.Dirs | QDir.NoDotAndDotDot, QDir.Name):
            self.files.append(entry)
        
        # Add files with optional filtering
        #TODO: fix filtering
        #parent_dialog = self.parent()
        #allowed_extensions = parent_dialog.get_current_file_extensions()
        # all_files = self.current_dir.entryInfoList(allowed_extensions, QDir.Files, QDir.Name)
        #all_files = [f for f in  self.current_dir.entryInfoList(QDir.Files, QDir.Name) if f.suffix().lower() in allowed_extensions]
        all_files = self.current_dir.entryInfoList(QDir.Files, QDir.Name)
        self.files.extend(all_files)
        pass
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.files)
    
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.columns)
    
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.columns):
                return self.columns[section]['name']
        return None
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self.files):
            return None
        
        file_info = self.files[index.row()]
        column_config = self.columns[index.column()]
        
        if role == Qt.ItemDataRole.DisplayRole:
            try:
                return column_config['data_func'](file_info)
            except:
                return ""
        elif role == Qt.ItemDataRole.DecorationRole and index.column() == 0:
            # Show file/folder icons in first column
            return self.icon_provider.icon(file_info)
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            # Right-align size columns, center others except name
            if 'size' in column_config['name'].lower():
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            elif index.column() == 0:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            else:
                return Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        elif role == Qt.ItemDataRole.ForegroundRole:
            # Apply custom color rules if set
            if hasattr(self, 'color_rule') and callable(self.color_rule):
                try:
                    color = self.color_rule(file_info)
                    if color:
                        return QBrush(color)
                except:
                    pass
        elif role == Qt.ItemDataRole.BackgroundRole:  
            # Apply custom background color rules if set
            if hasattr(self, 'background_color_rule') and callable(self.background_color_rule):
                try:
                    color = self.background_color_rule(file_info)
                    if color:
                        return QBrush(color)
                except:
                    pass
            # Apply alternating row colors if enabled
            elif hasattr(self, 'alternating_colors') and self.alternating_colors:
                if index.row() % 2 == 1:
                    return QBrush(QColor(245, 245, 245))  # Light gray for odd rows
        return None
    
    def get_file_info(self, index: QModelIndex) -> Optional[QFileInfo]:
        """Get QFileInfo for the given index"""
        if 0 <= index.row() < len(self.files):
            return self.files[index.row()]
        return None
    
    def format_size(self, size: int) -> str:
        """Format file size in human readable format"""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"

class CustomFileDialog(QDialog):
    """Custom file dialog with configurable columns"""
    
    fileSelected = pyqtSignal(str)
    
    def __init__(self, parent: Optional[Any] = None, caption: str = "Select File", directory: Optional[Union[str, Path]] = None):
        """
        Initialize the custom file dialog.
        
        Args:
            parent: Parent widget.
            caption: Dialog window title.
            directory: Initial directory to open.
        """
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle(caption)
        self.setMinimumSize(1000, 650)  # Increased width for preview panel
        
        # Initialize properties
        self.selected_file: Optional[str] = None
        self.file_filters = ["All files (*.*)"]
        self.current_filter_index = 0
        
        # Create the model
        self.model = FileTableModel(self)
        
        # Set up default columns
        self.setup_default_columns()
        
        # Create UI
        self.setup_ui()
        
        # Set initial directory
        if directory:
            self.set_directory(directory)
        else:
            self.set_directory(Path.home())
        
        # Set initial drive selection
        self.update_drive_selection(Path(self.model.current_dir.absolutePath()))
    
    def setup_default_columns(self) -> None:
        """Set up default columns configuration"""
        columns = [
            {
                'name': 'Name',
                'width': 250,
                'data_func': lambda info: info.fileName()
            },
            {
                'name': 'Size',
                'width': 100,
                'data_func': lambda info: self.model.format_size(info.size()) if info.isFile() else ""
            },
            {
                'name': 'Type',
                'width': 120,
                'data_func': lambda info: "Folder" if info.isDir() else info.suffix().upper() + " File" if info.suffix() else "File"
            },
            {
                'name': 'Modified',
                'width': 150,
                'data_func': lambda info: info.lastModified().toString("yyyy-MM-dd hh:mm")
            }
        ]
        self.model.set_columns(columns)
    
    def add_custom_column(self, name: str, width: int, data_function: Callable[[QFileInfo], str]) -> None:
        """
        Add a custom column to the dialog
        
        Args:
            name (str): Column header name
            width (int): Column width in pixels
            data_function (callable): Function that takes QFileInfo and returns display text
        """
        current_columns = self.model.columns.copy()
        current_columns.append({
            'name': name,
            'width': width,
            'data_func': data_function
        })
        self.model.set_columns(current_columns)
        self.setup_column_widths()
    
    def set_columns(self, columns: List[Dict[str, Any]]) -> None:
        """
        Set all columns at once
        
        Args:
            columns: list of dicts with keys: 'name', 'width', 'data_func'
        """
        self.model.set_columns(columns)
        self.setup_column_widths()
    
    def setup_ui(self) -> None:
        """Create the user interface"""
        layout = QVBoxLayout(self)
        
        # Drive selection (Windows-style)
        drive_layout = QHBoxLayout()
        drive_layout.addWidget(QLabel("Drive:"))
        self.drive_combo = QComboBox()
        self.drive_combo.setMinimumWidth(100)
        self.drive_combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.drive_combo.customContextMenuRequested.connect(self.show_drive_context_menu)
        self.populate_drives()
        self.drive_combo.currentTextChanged.connect(self.on_drive_changed)
        drive_layout.addWidget(self.drive_combo)
        
        # Add refresh button for drives
        refresh_button = QPushButton("🔄")
        refresh_button.setToolTip("Refresh drive list")
        refresh_button.setMaximumWidth(30)
        refresh_button.clicked.connect(self.refresh_drives)
        drive_layout.addWidget(refresh_button)
        
        drive_layout.addStretch()
        layout.addLayout(drive_layout)
        
        # Directory navigation
        nav_layout = QHBoxLayout()
        self.path_label = QLabel()
        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(self.go_up)
        nav_layout.addWidget(QLabel("Location:"))
        nav_layout.addWidget(self.path_label, 1)
        nav_layout.addWidget(self.up_button)
        layout.addLayout(nav_layout)
        
        # Create main horizontal splitter for file view and preview
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: File view
        left_widget = QFrame()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setSelectionBehavior(QTreeView.SelectRows)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.doubleClicked.connect(self.on_double_click)
        self.tree_view.clicked.connect(self.on_click)
        selection_model = self.tree_view.selectionModel()
        if selection_model is not None:
            selection_model.currentChanged.connect(self.current_file_changed)
        
        
        # Always show selection even when tree view is not focused
       
        # Set style to always show selection
        self.tree_view.setStyleSheet("""
            QTreeView::item:selected:!active {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
            QTreeView::item:selected:active {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
        """)
        
        # Alternative approach: Set focus policy to maintain visual selection
        self.tree_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Install event filter to handle keyboard navigation
        self.tree_view.installEventFilter(self)
        
        left_layout.addWidget(self.tree_view)
        
        # Right side: Image preview
        right_widget = QFrame()
        right_widget.setMaximumWidth(300)
        right_widget.setMinimumWidth(200)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)
        
        # Preview label
        preview_title = QLabel("Preview")
        preview_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        right_layout.addWidget(preview_title)
        
        # Image preview area
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setMinimumHeight(200)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 1px solid #cccccc;
                background-color: #f9f9f9;
                margin: 2px;
            }
        """)
        self.preview_label.setText("No preview available")
        self.preview_label.setMinimumSize(180, 180)
        
        self.preview_scroll.setWidget(self.preview_label)
        right_layout.addWidget(self.preview_scroll)
        
        # Image info
        self.image_info_label = QLabel()
        self.image_info_label.setWordWrap(True)
        self.image_info_label.setStyleSheet("font-size: 9pt; color: #666666;")
        right_layout.addWidget(self.image_info_label)
        
        right_layout.addStretch()
        
        # Add both sides to splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        
        # Set splitter proportions (70% file view, 30% preview)
        main_splitter.setSizes([500, 300])
        
        layout.addWidget(main_splitter, 1)
        
        # File name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("File name:"))
        self.filename_edit = QLineEdit()
        name_layout.addWidget(self.filename_edit)
        layout.addLayout(name_layout)
        
        # File type filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Files of type:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.file_filters)
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.accept_selection)
        self.open_button.setDefault(True)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        # Set up column widths
        self.setup_column_widths()
    
    def setup_column_widths(self) -> None:
        """Configure column widths based on column configuration"""
        header = self.tree_view.header()
        if header is not None:
            for i, column in enumerate(self.model.columns):
                if i < len(self.model.columns) - 1:
                    header.setSectionResizeMode(i, QHeaderView.Interactive)
                    self.tree_view.setColumnWidth(i, column['width'])
                else:
                    # Last column stretches
                    header.setSectionResizeMode(i, QHeaderView.Stretch)
    
    def set_directory(self, path: Union[str, Path]) -> None:
        """Set the current directory"""
        self.model.set_directory(path)
        self.path_label.setText(str(path))
        self.filename_edit.clear()
        self.update_drive_selection(path)
    
    def populate_drives(self) -> None:
        """Populate the drive combobox with available drives"""
        self.drive_combo.clear()
        
        # Get all available drives
        drives = QDir.drives()
        for drive in drives:
            drive_path = drive.absolutePath()
            
            # Format drive display (e.g., "C:\ (System)")
            display_text = drive_path
            
            # Try to get volume label on Windows
            try:
                if sys.platform == "win32":
                    try:
                        import win32api
                        volume_name = win32api.GetVolumeInformation(drive_path)[0]
                        if volume_name:
                            display_text = f"{drive_path} ({volume_name})"
                    except ImportError:
                        # win32api not available, use basic format
                        if len(drive_path) >= 2 and drive_path[1] == ':':
                            display_text = f"{drive_path} (Local Disk)"
                    except Exception:
                        # Any other error, just use the drive path
                        pass
            except:
                pass
            
            self.drive_combo.addItem(display_text, drive_path)
    
    def update_drive_selection(self, path: Union[str, Path]) -> None:
        """Update drive combo selection based on current path"""
        #TODO: fix drive selection - now it does not work properly, not selecting correct drive
        current_drive = Path(str(path)).anchor
        for i in range(self.drive_combo.count()):
            if self.drive_combo.itemData(i) == current_drive:
                self.drive_combo.blockSignals(True)
                self.drive_combo.setCurrentIndex(i)
                self.drive_combo.blockSignals(False)
                break
    
    def on_drive_changed(self, text: str) -> None:
        """Handle drive selection change"""
        if text:
            # Get the drive path from the combo data
            current_index = self.drive_combo.currentIndex()
            if current_index >= 0:
                drive_path = self.drive_combo.itemData(current_index)
                if drive_path and drive_path != Path(self.model.current_dir.absolutePath()).anchor:
                    self.set_directory(drive_path)
    
    def show_drive_context_menu(self, position) -> None:
        """Show context menu for drive combobox"""
        menu = QMenu(self)
        refresh_action = QAction("Refresh Drive List", self)
        refresh_action.triggered.connect(self.refresh_drives)
        menu.addAction(refresh_action)
        
        # Show menu at the requested position
        global_pos = self.drive_combo.mapToGlobal(position)
        menu.exec_(global_pos)
    
    def set_drive_by_index(self, index: int) -> bool:
        """
        Set the current drive by combobox index.
        
        Args:
            index: Index of the drive in the combobox
            
        Returns:
            bool: True if successful, False if index is invalid
        """
        if 0 <= index < self.drive_combo.count():
            self.drive_combo.setCurrentIndex(index)
            drive_path = self.drive_combo.itemData(index)
            if drive_path:
                self.set_directory(drive_path)
                return True
        return False
    
    def set_drive_by_path(self, drive_path: str) -> bool:
        """
        Set the current drive by drive path (e.g., "C:\", "D:\").
        
        Args:
            drive_path: Path of the drive to select
            
        Returns:
            bool: True if successful, False if drive not found
        """
        for i in range(self.drive_combo.count()):
            if self.drive_combo.itemData(i) == drive_path:
                self.drive_combo.setCurrentIndex(i)
                self.set_directory(drive_path)
                return True
        return False
    
    def refresh_drives(self) -> None:
        """Refresh the list of available drives"""
        current_selection = self.drive_combo.currentData()
        self.populate_drives()
        
        # Restore selection if possible
        for i in range(self.drive_combo.count()):
            if self.drive_combo.itemData(i) == current_selection:
                self.drive_combo.setCurrentIndex(i)
                break
    
    def get_current_drive(self) -> Optional[str]:
        """
        Get the currently selected drive path.
        
        Returns:
            str: Current drive path (e.g., "C:\") or None if no drive selected
        """
        current_index = self.drive_combo.currentIndex()
        if current_index >= 0:
            return self.drive_combo.itemData(current_index)
        return None
    
    def get_available_drives(self) -> List[str]:
        """
        Get a list of all available drive paths.
        
        Returns:
            List[str]: List of drive paths (e.g., ["C:\", "D:\", "E:\"])
        """
        drives = []
        for i in range(self.drive_combo.count()):
            drive_path = self.drive_combo.itemData(i)
            if drive_path:
                drives.append(drive_path)
        return drives
    
    def go_up(self) -> None:
        """Navigate to parent directory"""
        current_path = Path(self.model.current_dir.absolutePath())
        if current_path.parent != current_path:
            self.set_directory(current_path.parent)
    
    def current_file_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        self.on_click( current)

    def on_click(self, index: QModelIndex) -> None:
        """Handle single click on item"""
        file_info = self.model.get_file_info(index)
        self.update_current_file_preview()
       
        # Ensure the selection remains visible
        self.ensure_selection_visible()
    
    def on_double_click(self, index: QModelIndex) -> None:
        """Handle double click on item"""
        file_info = self.model.get_file_info(index)
        if file_info:
            if file_info.isDir():
                # Navigate into directory
                if file_info.fileName() == "..":
                    self.go_up()
                else:
                    self.set_directory(file_info.absoluteFilePath())
            else:
                # Select file and close dialog
                self.filename_edit.setText(file_info.fileName())
                self.accept_selection()
    
    def accept_selection(self) -> None:
        """Accept the current selection"""
        filename = self.filename_edit.text().strip()
        if not filename:
            QMessageBox.warning(self, "Warning", "Please select a file.")
            return
        
        file_path = Path(self.model.current_dir.absolutePath()) / filename
        if not file_path.exists():
            QMessageBox.warning(self, "Warning", f"File '{filename}' does not exist.")
            return
        
        self.selected_file = str(file_path)
        self.fileSelected.emit(self.selected_file)
        self.accept()
    
    def get_selected_file(self) -> Optional[str]:
        """Get the selected file path"""
        return self.selected_file
    
    def showEvent(self, event) -> None:
        """Override showEvent to set focus to the tree view when dialog is shown"""
        super().showEvent(event)
        # Set focus to the tree view to make it active
        self.tree_view.setFocus()
        
        # Optionally, select the first item if no selection exists
        selection_model = self.tree_view.selectionModel()
        if selection_model is not None and not selection_model.hasSelection() and self.model.rowCount() > 0:
            first_index = self.model.index(0, 0)
            self.tree_view.setCurrentIndex(first_index)
            self.on_click(first_index)
    
    def activate_file_list(self) -> None:
        """
        Explicitly activate the file list by setting focus and optionally selecting first item.
        Call this method after creating the dialog to ensure the file list is active.
        """
        self.tree_view.setFocus()
        
        # Select the first item if no selection exists
        selection_model = self.tree_view.selectionModel()
        if selection_model is not None and not selection_model.hasSelection() and self.model.rowCount() > 0:
            first_index = self.model.index(0, 0)
            self.tree_view.setCurrentIndex(first_index)
            self.on_click(first_index)
            
        # Force repaint to ensure selection is visible
        self.tree_view.update()
    
    def ensure_selection_visible(self) -> None:
        """
        Ensure that the current selection is always visible in the tree view,
        even when the tree view doesn't have focus.
        """
        current_index = self.tree_view.currentIndex()
        if current_index.isValid():
            # Scroll to the current selection to make it visible
            self.tree_view.scrollTo(current_index, QTreeView.EnsureVisible)
            # Force update to refresh the visual state
            self.tree_view.update()
    
    def is_image_file(self, file_info: QFileInfo) -> bool:
        """Check if the file is a supported image format"""
        if not file_info.isFile():
            return False
        
        image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp', 'ico'}
        extension = file_info.suffix().lower()
        return extension in image_extensions
    
    def update_current_file_preview(self) -> None:
        file_info = self.model.get_file_info(self.tree_view.currentIndex()) 
        if file_info and file_info.isFile():
            self.filename_edit.setText(file_info.fileName())
            # Update image preview
            self.update_image_preview(file_info)
        else:
            # Clear preview for directories
            self.clear_image_preview()

        pass

    def update_image_preview(self, file_info: QFileInfo) -> None:
        """Update the image preview for the selected file"""
        if not self.is_image_file(file_info):
            self.clear_image_preview()
            return
        
        try:
            # Load the image
            image_path = file_info.absoluteFilePath()
            pixmap = QPixmap(image_path)
            
            if pixmap.isNull():
                self.clear_image_preview("Failed to load image")
                return
            
            # Scale image to fit preview area while maintaining aspect ratio
            preview_size = 280  # Maximum size for preview
            scaled_pixmap = pixmap.scaled(
                preview_size, preview_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Update preview label
            self.preview_label.setPixmap(scaled_pixmap)
            self.preview_label.setMinimumSize(scaled_pixmap.size())
            
            # Update image info
            file_size = self.model.format_size(file_info.size())
            dimensions = f"{pixmap.width()} × {pixmap.height()}"
            info_text = f"Size: {file_size}\nDimensions: {dimensions}\nFormat: {file_info.suffix().upper()}"
            self.image_info_label.setText(info_text)
            
        except Exception as e:
            self.clear_image_preview(f"Error loading image: {str(e)}")
    
    def clear_image_preview(self, message: str = "No preview available") -> None:
        """Clear the image preview and show a message"""
        self.preview_label.clear()
        self.preview_label.setText(message)
        self.preview_label.setMinimumSize(180, 180)
        self.image_info_label.clear()
    
    def set_file_filters(self, filters: List[str]) -> None:
        """Set file type filters"""
        self.file_filters = filters
        self.filter_combo.clear()
        self.filter_combo.addItems(filters)
        # Refresh the file list to apply new filters
        self.model.refresh_files()
    
    def on_filter_changed(self, filter_text: str) -> None:
        """Handle file type filter change"""
        # Update current filter index
        self.current_filter_index = self.filter_combo.currentIndex()
        # Refresh the file list to apply the new filter
        self.model.refresh_files()
    
    def get_current_file_extensions(self) -> List[str]:
        """
        Extract file extensions from the current filter.
        Returns a list of extensions (without dots) or empty list for "All files".
        """
        if self.current_filter_index >= len(self.file_filters):
            return []
        
        current_filter = self.file_filters[self.current_filter_index]
        
        # Extract extensions from filter string like "Images (*.png *.jpg *.jpeg)"
        extensions = []
        
        # Find all *.ext patterns
        matches = re.findall(r'\*\.([a-zA-Z0-9]+)', current_filter)
        for match in matches:
            extensions.append(match.lower())
        
        return extensions
    
    def set_color_rule(self, color_rule: Callable[[QFileInfo], Optional[QColor]]) -> None:
        """
        Set a function to determine text color for each file row.
        
        Args:
            color_rule: Function that takes QFileInfo and returns QColor or None
            
        Example:
            def my_color_rule(file_info):
                if file_info.suffix().lower() == '.jpg':
                    return QColor(255, 0, 0)  # Red for JPG files
                return None
        """
        self.model.set_color_rule(color_rule)
    
    def set_background_color_rule(self, background_color_rule: Callable[[QFileInfo], Optional[QColor]]) -> None:
        """
        Set a function to determine background color for each file row.
        
        Args:
            background_color_rule: Function that takes QFileInfo and returns QColor or None
            
        Example:
            def my_bg_rule(file_info):
                if file_info.size() > 10*1024*1024:  # Files > 10MB
                    return QColor(255, 255, 200)  # Light yellow background
                return None
        """
        self.model.set_background_color_rule(background_color_rule)
    
    def set_alternating_row_colors(self, enabled: bool) -> None:
        """
        Enable or disable alternating row colors (zebra stripes).
        
        Args:
            enabled: True to enable alternating colors, False to disable
        """
        self.model.set_alternating_colors(enabled)
    
    def openAt(self, path: Union[str, Path], selectNext:bool = False) -> None:
        """
        Open dialog at the specified path. If path is a file, navigate to its 
        directory and select the file in the list. Also ensures the correct
        drive is selected in the drive combobox.
        
        Args:
            path: Path to file or folder to open at
            selectNext: If True, select the next file after the specified file
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            path_obj = Path.home()
            QMessageBox.information(self,"Open file/folder",f"Path '{path}' does not exist. Defaulting to home directory.")            
            return
        
        # Determine the target directory
        if path_obj.is_file():
            target_directory = path_obj.parent
            filename = path_obj.name
        else:
            target_directory = path_obj
            filename = None
        
        # Ensure the drive is properly selected first
        self.update_drive_selection(target_directory)
        
        # Set the directory (this will also call update_drive_selection again, but it's safe)
        self.set_directory(target_directory)
        
        # Handle file-specific operations
        if filename:
            # Set the filename in the text field
            self.filename_edit.setText(filename)
            # Find and select the file in the tree view
            self._select_file_in_view(filename, selectNext)
       
        # Always activate the file list after opening
        self.activate_file_list()
    
    def eventFilter(self, obj, event):
        """Handle keyboard events for the tree view"""
        if obj == self.tree_view and event.type() == QEvent.KeyPress:
            key_event = event
            if key_event.key() == Qt.Key_Return or key_event.key() == Qt.Key_Enter:
                # Handle Enter key press
                current_index = self.tree_view.currentIndex()
                if current_index.isValid():
                    file_info = self.model.get_file_info(current_index)
                    if file_info:
                        if file_info.isDir():
                            # Navigate into directory (same as double-click on folder)
                            if file_info.fileName() == "..":
                                self.go_up()
                            else:
                                self.set_directory(file_info.absoluteFilePath())
                            return True  # Event handled
                        else:
                            # Select file and close dialog (same as double-click on file)
                            self.filename_edit.setText(file_info.fileName())
                            self.accept_selection()
                            return True  # Event handled
            elif key_event.key() == Qt.Key_Backspace:
                # Handle Backspace key to go up one directory level
                self.go_up()
                return True  # Event handled
            elif key_event.key() == Qt.Key_PageUp and key_event.modifiers() == Qt.ControlModifier:
                self.go_up()
                return True
        # Pass event to parent class
        return super().eventFilter(obj, event)
    
    def _select_file_in_view(self, filename: str, selectNext: bool = False) -> None:
        """
        Helper method to select a specific file in the tree view.
        
        Args:
            filename: Name of the file to select
        """
        # Search for the file in the current model
        totalRows = self.model.rowCount()
        for row in range(totalRows):
            index = self.model.index(row, 0)  # First column contains filenames
            file_info = self.model.get_file_info(index)
            
            if file_info and file_info.fileName() == filename:
                if selectNext:
                    # Try to select the next file if possible
                    if row + 1 < totalRows:
                        index = self.model.index(row + 1, 0)    
                # Select the row using the selection model
                self.tree_view.setCurrentIndex(index)
                # Scroll to make it visible
                self.tree_view.scrollTo(index, QTreeView.PositionAtCenter)
                self.on_click(index)
                break

# Example usage and custom column functions
def get_process_time(file_info: QFileInfo) -> str:
    """Example function for process time column"""
    if file_info.isDir():
        return ""
    
    size = file_info.size()
    if size < 1024 * 1024:  # < 1MB
        return "~1s"
    elif size < 10 * 1024 * 1024:  # < 10MB
        return "~5s"
    elif size < 100 * 1024 * 1024:  # < 100MB
        return "~30s"
    else:
        return "~2m"

def get_file_priority(file_info: QFileInfo) -> str:
    """Example function for priority column"""
    if file_info.isDir():
        return ""
    
    name = file_info.fileName().lower()
    if name.endswith(('.jpg', '.jpeg', '.png')):
        return "High"
    elif name.endswith(('.txt', '.doc', '.pdf')):
        return "Medium"
    else:
        return "Low"

def get_color_code(file_info: QFileInfo) -> str:
    """Example function for color coding files"""
    if file_info.isDir():
        return "Folder"
    
    name = file_info.fileName().lower()
    if name.endswith(('.jpg', '.jpeg')):
        return "Red"
    elif name.endswith('.png'):
        return "Blue"
    elif name.endswith(('.tif', '.tiff')):
        return "Green"
    else:
        return "White"

# Example color rule functions
def example_text_color_rule(file_info: QFileInfo) -> Optional[QColor]:
    """Example function to set text color based on file type"""
    if file_info.isDir():
        return QColor(100, 100, 100)  # Gray for directories
    
    name = file_info.fileName().lower()
    if name.endswith(('.jpg', '.jpeg')):
        return QColor(255, 100, 0)  # Orange for JPEG
    elif name.endswith('.png'):
        return QColor(0, 0, 200)  # Blue for PNG
    elif name.endswith(('.tif', '.tiff')):
        return QColor(0, 150, 0)  # Green for TIFF
    elif file_info.size() > 10 * 1024 * 1024:  # Files > 10MB
        return QColor(200, 0, 0)  # Red for large files
    
    return None  # Use default color

def example_background_color_rule(file_info: QFileInfo) -> Optional[QColor]:
    """Example function to set background color based on file properties"""
    if file_info.isDir():
        return None
    
    # Highlight very large files with light red background
    if file_info.size() > 100 * 1024 * 1024:  # Files > 100MB
        return QColor(255, 230, 230)
    
    # Highlight recently modified files with light green background
    if file_info.lastModified().daysTo(QDateTime.currentDateTime()) < 1:
        return QColor(230, 255, 230)
    
    return None  # Use default background
