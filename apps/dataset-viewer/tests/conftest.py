"""Shared pytest fixtures for Dataset Viewer.

A QApplication is created once for the whole session (off-screen) because the
loaders/exporters use QImageReader/QPixmap, which need a Qt application.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "datasets")


@pytest.fixture(scope="session", autouse=True)
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def datasets_dir():
    return DATASETS


def sample(name: str) -> str:
    """Absolute path to a bundled sample dataset."""
    return os.path.join(DATASETS, name)
