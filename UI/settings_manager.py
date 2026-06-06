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

# Standard library imports
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Constants
DEFAULT_BACKUP_COUNT = 3
DEFAULT_AUTO_SAVE = True
MAX_SEARCH_LEVELS = 5
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class BackupManager:
    """Manages backup creation and cleanup for settings files."""

    def __init__(self, settings_file: Path, backup_count: int, logger: logging.Logger):
        self.settings_file = settings_file
        self.backup_count = backup_count
        self.logger = logger

    def create_backup(self) -> bool:
        """Create a backup of the current settings file."""
        if not self.settings_file.exists():
            return True

        try:
            timestamp = int(time.time())
            backup_file = self.settings_file.parent / f"{self.settings_file.stem}.backup.{timestamp}{self.settings_file.suffix}"
            shutil.copy2(self.settings_file, backup_file)
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

    def _get_backup_files(self) -> list:
        """Get list of backup files sorted by modification time (newest first)."""
        backup_pattern = f"{self.settings_file.stem}.backup.*{self.settings_file.suffix}"
        backup_files = list(self.settings_file.parent.glob(backup_pattern))
        return sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True)


class SettingsValidator:
    """Validates settings structure and values."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, settings: Dict[str, Any]) -> bool:
        """
        Validate settings structure and values.

        Args:
            settings: Settings dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(settings, dict):
                return False
            return True
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}")
            return False


class SettingsLoader:
    """Handles loading settings from files with recovery."""

    def __init__(self, settings_file: Path, backup_manager: BackupManager,
                 validator: SettingsValidator, logger: logging.Logger):
        self.settings_file = settings_file
        self.backup_manager = backup_manager
        self.validator = validator
        self.logger = logger

    def load_settings(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load settings with backup recovery.

        Args:
            defaults: Default settings to use if loading fails

        Returns:
            Loaded settings dictionary
        """
        # Try to load from main file
        settings = self._load_from_file(self.settings_file)
        if settings is not None:
            return settings

        # Main file failed, try backups
        self.logger.warning("Main settings file failed, trying backups...")
        backup_files = self.backup_manager._get_backup_files()

        for backup_file in backup_files:
            self.logger.info(f"Trying backup: {backup_file}")
            settings = self._load_from_file(backup_file)
            if settings is not None:
                self.logger.info(f"Recovered settings from backup: {backup_file}")
                return settings

        # All files failed, use defaults
        self.logger.warning("All settings files failed, using defaults")
        return defaults.copy()

    def _load_from_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load settings from a specific file."""
        try:
            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            if self.validator.validate(settings):
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


class SettingsSaver:
    """Handles saving settings to file with atomic writes."""

    def __init__(self, settings_file: Path, backup_manager: BackupManager, logger: logging.Logger):
        self.settings_file = settings_file
        self.backup_manager = backup_manager
        self.logger = logger

    def save(self, settings: Dict[str, Any]) -> bool:
        """
        Save settings to file using atomic write.

        Args:
            settings: Settings dictionary to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup before writing
            self.backup_manager.create_backup()

            # Use atomic write (write to temp file, then rename)
            temp_file = self.settings_file.parent / f"{self.settings_file.name}.tmp"

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

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


class SettingsManager:
    """
    Robust settings manager with corruption recovery and thread safety.
    """

    def __init__(self, settings_file: str = "application_settings.json",
                 backup_count: int = DEFAULT_BACKUP_COUNT, auto_save: bool = DEFAULT_AUTO_SAVE):
        """
        Initialize the settings manager.

        Args:
            settings_file: Path to the settings file (relative to application directory)
            backup_count: Number of backup files to maintain
            auto_save: Whether to automatically save settings on changes
        """
        app_dir = self._get_application_directory()
        self.settings_file = app_dir / settings_file if not os.path.isabs(settings_file) else Path(settings_file)

        self.auto_save = auto_save
        self._lock = threading.RLock()
        self._defaults = self._get_default_settings()

        # Set up logging
        self._setup_logging(app_dir)
        self.logger = logging.getLogger('SettingsManager')

        # Initialize components
        self.backup_manager = BackupManager(self.settings_file, backup_count, self.logger)
        self.validator = SettingsValidator(self.logger)
        self.loader = SettingsLoader(self.settings_file, self.backup_manager, self.validator, self.logger)
        self.saver = SettingsSaver(self.settings_file, self.backup_manager, self.logger)

        # Load settings
        self._settings = self.loader.load_settings(self._defaults)
        # Save loaded settings to ensure file exists
        self.saver.save(self._settings)

    @staticmethod
    def _get_application_directory() -> Path:
        """Get the application directory (where main.py is located)."""
        try:
            current_dir = Path(__file__).parent
            for _ in range(MAX_SEARCH_LEVELS):
                if (current_dir / "main.py").exists():
                    return current_dir
                parent = current_dir.parent
                if parent == current_dir:
                    break
                current_dir = parent
            return Path(__file__).parent.parent
        except Exception:
            return Path.cwd()

    def _setup_logging(self, app_dir: Path):
        """Set up logging for settings operations."""
        log_dir = app_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger('SettingsManager')
        if not logger.handlers:
            log_file = log_dir / 'settings.log'
            handler = logging.FileHandler(str(log_file))
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    @staticmethod
    def _get_default_settings() -> Dict[str, Any]:
        """Define default settings for the application."""
        return {
            'paths': {
                'last_opened_file': "",
                'recent_files': []
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value using dot notation.

        Args:
            key: Setting key (supports dot notation like 'window.width').
            default: Default value if key not found.

        Returns:
            Setting value or default.
        """
        with self._lock:
            try:
                keys = key.split('.')
                value = self._settings
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return self._get_from_defaults(keys, default)
                return value
            except Exception as e:
                self.logger.error(f"Error getting setting '{key}': {e}")
                return default

    def _get_from_defaults(self, keys: list, default: Any) -> Any:
        """Get value from defaults using key path."""
        try:
            value = self._defaults
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except Exception:
            return default

    def set(self, key: str, value: Any) -> bool:
        """
        Set a setting value using dot notation.

        Args:
            key: Setting key (supports dot notation like 'window.width').
            value: Value to set.

        Returns:
            True if successful, False otherwise.
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
            return self.saver.save(self._settings)

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
                        return False
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
                settings = self.loader._load_from_file(Path(file_path))
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