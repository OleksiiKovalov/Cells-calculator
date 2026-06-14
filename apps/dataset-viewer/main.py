"""Launcher for Dataset Viewer.

Run it from this folder:

    python main.py

It puts ``src/`` on the import path and starts the app, so the application runs
as a self-contained, shippable folder.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from app import main

if __name__ == "__main__":
    main()
