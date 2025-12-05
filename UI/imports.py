"""
Dynamic Import System for Cells Calculator
This module handles all imports with progress tracking and conditional loading.
"""

# Import the dynamic import function
from UI.dynamic_imports import get_import_globals

# Execute all imports dynamically
globals().update(get_import_globals())

# Note: This file now uses a dynamic import system that:
# - Provides better maintainability through configuration-driven approach
# - Handles optional dependencies gracefully
# - Shows progress during loading
# - Supports conditional loading based on model configuration
# - Uses class-based architecture for better extensibility
# - Provides comprehensive error tracking and reporting
