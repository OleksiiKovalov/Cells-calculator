"""Launcher for the Training Studio GUI.

Lives inside ``src/`` (alongside ``gui.py``), so it imports the GUI directly.
Run from the app root:

    python src/main.py

The CLI pipeline is ``src/runner.py``; each step is also a standalone script in
``src/`` (train / evaluate / export_model). Prepare datasets in Dataset Viewer.
"""

from gui import main

if __name__ == "__main__":
    main()
