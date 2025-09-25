"""
Settings Manager for Cells Calculator application.

This module provides a robust settings management system that:
- Stores settings in JSON format
- Handles file corruption gracefully
- Survives application crashes
- Provides automatic backup and recovery
- Supports default values and validation
- Thread-safe operations

Features:
- Automatic backup creation before writing
- Corruption detection and recovery
- Default settings fallback
- Settings validation
- Safe atomic writes
- Lock-based thread safety

Usage:
    from settings_manager import SettingsManager
    
    settings = SettingsManager()
    settings.set('window_size', {'width': 1200, 'height': 800})
    window_size = settings.get('window_size', {'width': 1000, 'height': 600})
    settings.save()
"""

import json
import os
import shutil
import threading
import tempfile
import traceback
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging


class SettingsManager:
    """
    Robust settings manager with corruption recovery and thread safety.
    """
    
    def __init__(self, settings_file: str = "application_settings.json", 
                 backup_count: int = 3, auto_save: bool = True):
        """
        Initialize the settings manager.
        
        Args:
            settings_file: Path to the settings file (relative to application directory)
            backup_count: Number of backup files to maintain
            auto_save: Whether to automatically save settings on changes
        """
        # Get the application directory (where main.py is located)
        app_dir = self._get_application_directory()
        
        # If settings_file is just a filename, place it in the application directory
        if not os.path.isabs(settings_file):
            self.settings_file = app_dir / settings_file
        else:
            self.settings_file = Path(settings_file)
        
        self.backup_count = backup_count
        self.auto_save = auto_save
        self._lock = threading.RLock()
        self._settings = {}
        self._defaults = self._get_default_settings()
        
        # Set up logging (also in application directory)
        self._setup_logging(app_dir)
        
        # Load settings with recovery
        self._load_settings()
    
    def _get_application_directory(self) -> Path:
        """
        Get the application directory (where main.py is located).
        
        Returns:
            Path to the application directory
        """
        try:
            # Try to find main.py by looking in parent directories
            current_dir = Path(__file__).parent
            
            # Look for main.py in current directory and parent directories
            for _ in range(5):  # Limit search to 5 levels up
                if (current_dir / "main.py").exists():
                    return current_dir
                parent = current_dir.parent
                if parent == current_dir:  # Reached root
                    break
                current_dir = parent
            
            # Fallback: use the directory containing this file's parent (UI folder's parent)
            return Path(__file__).parent.parent
            
        except Exception as e:
            # Ultimate fallback: use current working directory
            return Path.cwd()
    
    def _setup_logging(self, app_dir: Path):
        """Set up logging for settings operations.
        
        Args:
            app_dir: Application directory where logs should be stored
        """
        # Create logs directory if it doesn't exist
        log_dir = app_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('SettingsManager')
        if not self.logger.handlers:
            log_file = log_dir / 'settings.log'
            handler = logging.FileHandler(str(log_file))
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """
        Define default settings for the application.
        Override this method to customize defaults.
        """
        return {
            # File paths
            'paths': {
                'last_opened_file': "",
                'recent_files': []
            }
            
        }
    
    def _get_backup_files(self) -> list:
        """Get list of backup files sorted by modification time (newest first)."""
        backup_pattern = f"{self.settings_file.stem}.backup.*{self.settings_file.suffix}"
        backup_files = list(self.settings_file.parent.glob(backup_pattern))
        return sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def _create_backup(self) -> bool:
        """Create a backup of the current settings file."""
        if not self.settings_file.exists():
            return True
        
        try:
            # Create backup filename with timestamp
            import time
            timestamp = int(time.time())
            backup_file = self.settings_file.parent / f"{self.settings_file.stem}.backup.{timestamp}{self.settings_file.suffix}"
            
            # Copy current file to backup
            shutil.copy2(self.settings_file, backup_file)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            self.logger.info(f"Created backup: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Remove old backup files, keeping only the specified count."""
        try:
            backup_files = self._get_backup_files()
            if len(backup_files) > self.backup_count:
                for old_backup in backup_files[self.backup_count:]:
                    old_backup.unlink()
                    self.logger.info(f"Removed old backup: {old_backup}")
        except Exception as e:
            self.logger.error(f"Error cleaning up backups: {e}")
    
    def _validate_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Validate settings structure and values.
        Override this method to add custom validation.
        """
        try:
            # Basic structure validation
            if not isinstance(settings, dict):
                return False
            
            # # Check for required sections
            # required_sections = ['window', 'ui', 'application', 'paths', 'processing']
            # for section in required_sections:
            #     if section not in settings:
            #         self.logger.warning(f"Missing required section: {section}")
            #         return False
            
            # # Validate specific settings
            # window_settings = settings.get('window', {})
            # if not isinstance(window_settings.get('width'), (int, float)) or window_settings.get('width', 0) <= 0:
            #     return False
            # if not isinstance(window_settings.get('height'), (int, float)) or window_settings.get('height', 0) <= 0:
            #     return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}")
            return False
    
    def _load_from_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load settings from a specific file."""
        try:
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            if self._validate_settings(settings):
                self.logger.info(f"Successfully loaded settings from: {file_path}")
                return settings
            else:
                self.logger.warning(f"Invalid settings structure in: {file_path}")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading settings from {file_path}: {e}")
            return None
    
    def _load_settings(self):
        """Load settings with backup recovery."""
        with self._lock:
            # Try to load from main file
            settings = self._load_from_file(self.settings_file)
            
            if settings is not None:
                self._settings = settings
                return
            
            # Main file failed, try backups
            self.logger.warning("Main settings file failed, trying backups...")
            backup_files = self._get_backup_files()
            
            for backup_file in backup_files:
                self.logger.info(f"Trying backup: {backup_file}")
                settings = self._load_from_file(backup_file)
                
                if settings is not None:
                    self._settings = settings
                    self.logger.info(f"Recovered settings from backup: {backup_file}")
                    # Save recovered settings to main file
                    self._save_to_file()
                    return
            
            # All files failed, use defaults
            self.logger.warning("All settings files failed, using defaults")
            self._settings = self._defaults.copy()
            # Save defaults to file
            self._save_to_file()
    
    def _save_to_file(self) -> bool:
        """Save settings to file using atomic write."""
        try:
            # Create backup before writing
            self._create_backup()
            
            # Use atomic write (write to temp file, then rename)
            temp_file = self.settings_file.parent / f"{self.settings_file.name}.tmp"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            if os.name == 'nt':  # Windows
                if self.settings_file.exists():
                    self.settings_file.unlink()
                temp_file.rename(self.settings_file)
            else:  # Unix-like
                temp_file.rename(self.settings_file)
            
            self.logger.info(f"Settings saved to: {self.settings_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            # Clean up temp file if it exists
            try:
                temp_file = self.settings_file.parent / f"{self.settings_file.name}.tmp"
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value using dot notation.
        
        Args:
            key: Setting key (supports dot notation like 'window.width')
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        with self._lock:
            try:
                keys = key.split('.')
                value = self._settings
                
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        # Try to get from defaults
                        default_value = self._defaults
                        for dk in keys:
                            if isinstance(default_value, dict) and dk in default_value:
                                default_value = default_value[dk]
                            else:
                                return default
                        return default_value
                
                return value
                
            except Exception as e:
                self.logger.error(f"Error getting setting '{key}': {e}")
                return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a setting value using dot notation.
        
        Args:
            key: Setting key (supports dot notation like 'window.width')
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                keys = key.split('.')
                current = self._settings
                
                # Navigate to parent dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    elif not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]
                
                # Set the value
                current[keys[-1]] = value
                
                # Auto-save if enabled
                if self.auto_save:
                    self.save()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error setting '{key}' to '{value}': {e}")
                return False
    
    def save(self) -> bool:
        """Manually save settings to file."""
        with self._lock:
            return self._save_to_file()
    
    def reset_to_defaults(self, section: Optional[str] = None) -> bool:
        """
        Reset settings to defaults.
        
        Args:
            section: Specific section to reset, or None for all settings
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                if section:
                    if section in self._defaults:
                        self._settings[section] = self._defaults[section].copy()
                else:
                    self._settings = self._defaults.copy()
                
                if self.auto_save:
                    self.save()
                
                self.logger.info(f"Reset settings to defaults" + (f" for section: {section}" if section else ""))
                return True
                
            except Exception as e:
                self.logger.error(f"Error resetting settings: {e}")
                return False
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get a copy of all settings."""
        with self._lock:
            return self._settings.copy()
    
    def import_settings(self, file_path: Union[str, Path]) -> bool:
        """
        Import settings from another file.
        
        Args:
            file_path: Path to settings file to import
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                settings = self._load_from_file(Path(file_path))
                if settings is not None:
                    self._settings = settings
                    if self.auto_save:
                        self.save()
                    self.logger.info(f"Imported settings from: {file_path}")
                    return True
                return False
                
            except Exception as e:
                self.logger.error(f"Error importing settings from {file_path}: {e}")
                return False
    
    def export_settings(self, file_path: Union[str, Path]) -> bool:
        """
        Export current settings to a file.
        
        Args:
            file_path: Path where to export settings
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                export_path = Path(file_path)
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(self._settings, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"Exported settings to: {export_path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error exporting settings to {file_path}: {e}")
                return False


# Global settings instance
_settings_instance = None


def get_settings() -> SettingsManager:
    """Get the global settings instance (singleton pattern)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SettingsManager()
    return _settings_instance


# Convenience functions
def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value (convenience function)."""
    return get_settings().get(key, default)


def set_setting(key: str, value: Any) -> bool:
    """Set a setting value (convenience function)."""
    return get_settings().set(key, value)


def save_settings() -> bool:
    """Save settings to file (convenience function)."""
    return get_settings().save()