"""
Application-wide globals for Cells Calculator.

This module manages global state and variables that need to be shared
across different components of the application.
"""

from sys import modules


FILENAME_MODEL_CONFIG = "modelconfig.json"
CASH_DIRECTORY = ".cache"

IMAGE_FILE_NAME_DETECTION = ".cache/cell_tmp_img_with_detections.png"
IMAGE_FILE_NAME_INGFERENCE = ".cache/cell_tmp_img_inference.png"
IMAGE_FILE_NAME_TMP = ".cache/cell_tmp_img.png"
IMAGE_FILE_NAME_INSTANCES = ".cache/instances.jpg"
IMAGE_FILE_NAME_GRID = ".cache/image_grid_output.png"


class AppGlobals:
    """
    Centralized management for application-wide global variables.
    Implements singleton pattern to ensure single instance.
    Stores all values in a dictionary indexed by string names for flexibility.
    """
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Initialize the data dictionary with default values
        self._data = {
            # Model predictions and inference data
            'predictions': None,
            'image_inference': None,
            
            # Additional global state
            'current_model_name': None,
            'processing_active': False,
            'last_processing_time': None,
            
            # Image processing state
            'original_image_shape': None,
            'current_channels': {'Cell': 0, 'Nuclei': 1},
            
            # UI state
            'main_window_ref': None,
            'current_plugin': "Cell Processor",
        }
        
        self.registered_models = {}
        self.registered_models_libs = {}
        self._initialized = True
    
    def get(self, key: str, default=None):
        """
        Get a global value by key name.
        
        Args:
            key (str): The key name
            default: Default value if key doesn't exist
            
        Returns:
            The value associated with the key, or default if not found
        """
        return self._data.get(key, default)
    
    def set(self, key: str, value):
        """
        Set a global value by key name.
        
        Args:
            key (str): The key name
            value: The value to set
        """
        self._data[key] = value
    
    def update(self, **kwargs):
        """
        Update multiple values at once.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        self._data.update(kwargs)
    
    def has(self, key: str) -> bool:
        """
        Check if a key exists and has a non-None value.
        
        Args:
            key (str): The key name
            
        Returns:
            bool: True if key exists and value is not None
        """
        return key in self._data and self._data[key] is not None
    
    def delete(self, key: str):
        """
        Delete a key from the global data.
        
        Args:
            key (str): The key name to delete
        """
        if key in self._data:
            del self._data[key]
    
    def keys(self):
        """Get all available keys."""
        return self._data.keys()
    
    def items(self):
        """Get all key-value pairs."""
        return self._data.items()
    
    def to_dict(self) -> dict:
        """Get a copy of all data as dictionary."""
        return self._data.copy()
    
    def reset_predictions(self):
        """Reset prediction-related data."""
        self.update(
            predictions=None,
            image_inference=None,
            last_processing_time=None,
            processing_active=False
        )
    
    def reset_all(self):
        """Reset all global state to initial values."""
        self._data = {
            'detections': None,
            'image_inference': None,
            'current_model_name': None,
            'processing_active': False,
            'last_processing_time': None,
            'original_image_shape': None,
            'current_channels': {'Cell': 0, 'Nuclei': 1},
            'main_window_ref': None,
            'current_plugin': "Cell Processor",
        }
    

# Global instance - import this in other modules
app_globals = AppGlobals()

def register_model(model_type: str, model_class, preload=False):
    """Register a model instance by name."""
    app_globals.registered_models[model_type] = {'model_type':model_type, 'model_class': model_class, 'preload': preload}

def get_registered_model(name: str):
    """Retrieve a registered model instance by name."""
    return app_globals.registered_models.get(name, None)

def get_registered_models():
    """Retrieve all registered model instances."""
    return app_globals.registered_models

    # Additional convenience functions for dictionary-based access
def get_global(key: str, default=None):
    """Get any global value by key name."""
    return app_globals.get(key, default)

def set_global(key: str, value):
    """Set any global value by key name."""
    app_globals.set(key, value)

def update_globals(**kwargs):
    """Update multiple global values at once."""
    app_globals.update(**kwargs)

def has_global(key: str) -> bool:
    """Check if a global key exists and has a non-None value."""
    return app_globals.has(key)

def delete_global(key: str):
    """Delete a global key."""
    app_globals.delete(key)

def get_all_globals() -> dict:
    """Get a copy of all global data."""
    return app_globals.to_dict()

def list_global_keys():
    """Get list of all global keys."""
    return list(app_globals.keys())
