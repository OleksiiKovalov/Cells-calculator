"""
Dynamic Import System for Cells Calculator
This module provides a cleaner, more maintainable approach to importing dependencies
with progress tracking and conditional loading based on configuration.
"""

import importlib
import sys
from typing import Dict, List, Optional, Tuple, Any
from UI.app_globals import get_registered_model
from UI.splashscreen import update_splash

class ImportManager:
    """Manages dynamic imports with progress tracking and error handling."""
    
    def __init__(self):
        self.imported_modules = {}
        self.failed_imports = []
        self.progress = 25  # Starting progress from splash screen
        
    def import_module_group(self, group_name: str, modules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Import a group of modules with progress tracking.
        
        Args:
            group_name: Name of the module group for progress display
            modules: List of module configurations
            
        Returns:
            Dictionary mapping module names to imported objects
        """
        update_splash(self.progress, f"Loading {group_name}...")
        self.progress += 1
        
        imported = {}
        
        for module_config in modules:
            result = self._import_single_module(module_config)
            if result:
                imported.update(result)
                
        return imported
    
    def _import_single_module(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Import a single module based on configuration.
        
        Args:
            config: Module configuration dictionary
            
        Returns:
            Dictionary with imported objects or None if failed
        """
        module_name = config['module']
        import_items = config.get('items', [])
        alias = config.get('alias')
        optional = config.get('optional', False)
        condition = config.get('condition')
        
        # Check condition if specified
        if condition and not self._evaluate_condition(condition):
            return None
            
        try:
            if import_items:
                # Import specific items from module
                module = importlib.import_module(module_name)
                result = {}
                
                for item in import_items:
                    if item == '*':
                        # Handle wildcard import
                        if hasattr(module, '__all__'):
                            for item_name in module.__all__:
                                result[item_name] = getattr(module, item_name)
                        else:
                            # Import all public attributes
                            for item_name in dir(module):
                                if not item_name.startswith('_'):
                                    result[item_name] = getattr(module, item_name)
                    elif isinstance(item, dict):
                        item_name = item['name']
                        item_alias = item.get('alias', item_name)
                        result[item_alias] = getattr(module, item_name)
                    else:
                        result[item] = getattr(module, item)
                        
                return result
            else:
                # Import entire module
                module = importlib.import_module(module_name)
                key = alias if alias else module_name.split('.')[-1]
                return {key: module}
                
        except ImportError as e:
            if not optional:
                self.failed_imports.append((module_name, str(e)))
            return None
    
    def _evaluate_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate import condition."""
        if condition['type'] == 'model_registered':
            model_name = condition['model']
            preload = condition.get('preload', True)
            r = get_registered_model(model_name)
            return r is not None and r.get('preload') == preload
        return True

# Import configuration
IMPORT_CONFIG = {
    'system': [
        {'module': 'os'},
        {'module': 'sys'},
        {'module': 'shutil'},
        {'module': 'tempfile'},
        {'module': 'threading'},
        {'module': 'traceback'},
        {'module': 'io'},
        {'module': 'glob'},
        {'module': 'logging'},
        {'module': 'math'},
        {'module': 'string'},
        {'module': 'time'},
        {'module': 're'},
        {'module': 'datetime', 'items': ['datetime']},
    ],
    
    'path_handling': [
        {'module': 'pathlib', 'items': ['Path', 'PureWindowsPath', 'PurePosixPath']},
        {'module': 'contextlib', 'items': ['redirect_stdout', 'redirect_stderr']},
    ],
    
    'data_structures': [
        {'module': 'collections', 'items': ['OrderedDict']},
        {'module': 'json'},
        {'module': 'typing', 'items': ['Optional', 'List', 'Tuple', 'Dict', 'Any', 'Union', 'Callable']},
        {'module': 'pyparsing', 'items': ['Optional'], 'alias': 'pyparsing_optional'},
    ],
    
    'windows_optional': [
        {'module': 'win32api', 'optional': True},
        {'module': 'winsound', 'optional': True},
    ],
    
    'scientific_computing': [
        {'module': 'numpy', 'alias': 'np'},
        {'module': 'pandas', 'alias': 'pd'},
        {'module': 'cv2'},
        {'module': 'tifffile'},
        {'module': 'tiffile'},
        {'module': 'skimage.io', 'items': ['imread', 'imsave']},
        {'module': 'skimage.color', 'items': ['rgb2gray', 'gray2rgb']},
        {'module': 'skimage.measure', 'items': ['regionprops']},
        {'module': 'skimage.transform', 'items': ['resize']},
        {'module': 'scipy.ndimage', 'items': ['find_objects']},
        {'module': 'matplotlib.pyplot', 'alias': 'plt'},
        {'module': 'matplotlib.colors', 'alias': 'mcolors'},
    ],
    
    'machine_learning': [
        {'module': 'torch'},
        {'module': 'sklearn.cluster', 'items': ['DBSCAN'], 'optional': True},
        {'module': 'onnxruntime', 'optional': True},
    ],
    
    'yolo_models': [
        {
            'module': 'ultralytics', 
            'items': ['YOLO'],
            'condition': {'type': 'model_registered', 'model': 'yolo', 'preload': True}
        },
        {
            'module': 'ultralytics.engine.results', 
            'items': ['Results', 'Masks'],
            'condition': {'type': 'model_registered', 'model': 'yolo', 'preload': True}
        },
        {
            'module': 'model.sahi.utils.cv', 
            'items': ['read_image'],
            'condition': {'type': 'model_registered', 'model': 'yolo', 'preload': True}
        },
        {
            'module': 'model.sahi.predict', 
            'items': ['get_sliced_prediction'],
            'condition': {'type': 'model_registered', 'model': 'yolo', 'preload': True}
        },
        {
            'module': 'model.sahi.auto_model', 
            'items': ['AutoDetectionModel'],
            'condition': {'type': 'model_registered', 'model': 'yolo', 'preload': True}
        },
    ],
    
    'specialized_ai': [
        {
            'module': 'cellpose.models',
            'alias': 'cp_models',
            'condition': {'type': 'model_registered', 'model': 'cellpose', 'preload': True}
        },
        {
            'module': 'instanseg',
            'items': ['InstanSeg'],
            'condition': {'type': 'model_registered', 'model': 'instanseg', 'preload': True}
        },
        {
            'module': 'instanseg.utils.utils',
            'items': ['labels_to_features', 'export_to_torchscript'],
            'condition': {'type': 'model_registered', 'model': 'instanseg', 'preload': True}
        },
        {
            'module': 'tensorflow',
            'alias': 'tf',
            'condition': {'type': 'model_registered', 'model': 'stardist', 'preload': True}
        },
        {
            'module': 'stardist.models',
            'items': ['StarDist2D'],
            'condition': {'type': 'model_registered', 'model': 'stardist', 'preload': True}
        },
        {
            'module': 'csbdeep.utils',
            'items': ['normalize'],
            'condition': {'type': 'model_registered', 'model': 'stardist', 'preload': True}
        },
    ],
    
    'optional_packages': [
        {'module': 'pycocotools.coco', 'items': ['COCO'], 'optional': True},
        {'module': 'pycocotools.cocoeval', 'items': ['COCOeval'], 'optional': True},
        {'module': 'importlib_metadata', 'optional': True},
        {'module': 'IPython', 'optional': True},
        {'module': 'fiftyone', 'alias': 'fo', 'optional': True},
        {'module': 'imantics', 'optional': True},
    ],
    
    'geometry': [
        {'module': 'shapely.geometry', 'items': ['shape']},
    ],
    
    'pyqt5_core': [
        {'module': 'PyQt5.QtCore', 'items': [
            'Qt', 'QObject', 'QTimer', 'QDir', 'QFileInfo', 
            'QAbstractTableModel', 'QModelIndex', 'QEvent', 'QDateTime',
            'pyqtSignal', 'pyqtSlot'
        ]},
        {'module': 'PyQt5.QtGui', 'items': [
            'QPixmap', 'QImage', 'QPainter', 'QFont', 'QColor', 'QPen', 
            'QLinearGradient', 'QBrush', 'QIcon', 'QKeyEvent'
        ]},
    ],
    
    'pyqt5_widgets': [
        {'module': 'PyQt5.QtWidgets', 'items': [
            'QApplication', 'QMainWindow', 'QWidget', 'QDialog', 'QMessageBox',
            'QVBoxLayout', 'QHBoxLayout', 
            'QLabel', 'QPushButton', 'QCheckBox', 'QRadioButton', 'QButtonGroup',
            'QComboBox', 'QLineEdit', 'QSlider', 'QProgressBar',
            'QTableWidget', 'QTableWidgetItem', 'QAbstractItemView',
            'QGraphicsView', 'QGraphicsScene', 'QGraphicsPixmapItem', 'QGraphicsTextItem',
            'QMenuBar', 'QAction', 'QFileDialog',
            'QListWidget', 'QListWidgetItem', 'QDialogButtonBox',
            'QSplashScreen', 'QTabWidget', 'QGroupBox', 'QScrollArea',
            'QSpinBox', 'QDoubleSpinBox', 'QTextEdit', 'QPlainTextEdit',
            'QSplitter'
        ]},
    ],
    
    'ui_components': [
        {'module': 'UI.splashscreen', 'items': ['*']},
        {'module': 'UI.errorhandling', 'items': ['*']},
        {'module': 'UI.app_globals', 'items': ['*']},
        {'module': 'UI.settings_manager', 'items': ['*']},
        {'module': 'UI.SettingsWindow', 'items': ['*']},
        {'module': 'UI.menubar', 'items': ['*']},
        {'module': 'UI.table', 'items': ['*']},
        {'module': 'UI.Slider', 'items': ['*']},
        {'module': 'UI.rangeslider', 'items': ['*']},
        {'module': 'UI.CustomFileDialog', 'items': ['*']},
        {'module': 'UI.ModelsCheckList', 'items': ['*']},
        {'module': 'UI.ImageNormalizeDialog', 'items': ['*']},
        {'module': 'UI.WaitWindow', 'items': ['*']},
    ],
    
    'ui_layout': [
        {'module': 'UI.right_layout.right_layout', 'items': ['*']},
        {'module': 'UI.right_layout.plugins.BasePlugin', 'items': ['*']},
        {'module': 'UI.right_layout.plugins.CellDetectorPlugin', 'items': [
            {'name': 'CellDetectorPlugin', 'alias': 'CellDetector_plugin'}
        ]},
        {'module': 'UI.right_layout.plugins.TrackerPlugin', 'items': [
            {'name': 'TrackerPlugin', 'alias': 'Tracker_plugin'}
        ]},
        {'module': 'UI.right_layout.plugins.SpheroidSegmenterPlugin', 'items': ['*']},
    ],
    
    'model_components': [
        {'module': 'model.BaseModel', 'items': ['BaseModel']},
        {'module': 'model.Model', 'items': ['Model']},
        {'module': 'model.utils', 'items': ['*']},
    ],
}

def load_all_imports() -> Dict[str, Any]:
    """
    Load all imports using the dynamic system.
    
    Returns:
        Dictionary containing all successfully imported modules/objects
    """
    import_manager = ImportManager()
    all_imports = {}
    
    for group_name, modules in IMPORT_CONFIG.items():
        imported = import_manager.import_module_group(group_name.replace('_', ' ').title(), modules)
        all_imports.update(imported)
    
    update_splash(import_manager.progress, "Import loading complete!")
    
    # Report any failed imports
    if import_manager.failed_imports:
        print("Warning: Some imports failed:")
        for module, error in import_manager.failed_imports:
            print(f"  {module}: {error}")
    
    return all_imports

def get_import_globals() -> Dict[str, Any]:
    """
    Get all imports as a globals dictionary for easy injection into modules.
    
    Returns:
        Dictionary suitable for updating globals()
    """
    imports = load_all_imports()
    
    # Add special cases for wildcard imports
    # Note: Wildcard imports need special handling in a real implementation
    # This is a simplified version
    
    return imports

# Convenience function for backwards compatibility
def import_all():
    """Load all imports and inject into calling module's globals."""
    import inspect
    frame = inspect.currentframe().f_back
    frame.f_globals.update(get_import_globals())