"""Tests for the cp1252-safe console redirection in UI.errorhandling.

Regression for the bug where printing non-cp1252 characters (the UI's µm/×/✓ or
Cyrillic) crashed the app via the stdout redirector on a Windows console.
"""
import io
import logging
import sys


def test_loggerwriter_survives_cp1252_unencodable_chars():
    from ui.errorhandling import LoggerWriter, app_logger
    lw = LoggerWriter(app_logger(), logging.INFO)
    real = sys.__stdout__
    buf = io.BytesIO()
    cp = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
    sys.__stdout__ = cp
    try:
        lw.write("area 253.50 µm²; vol µm³; Ініц; check ✓\n")  # must NOT raise
        cp.flush()
    finally:
        sys.__stdout__ = real
    data = buf.getvalue()
    assert b"area 253.50" in data  # cp1252-representable text preserved


def test_loggerwriter_survives_missing_console():
    from ui.errorhandling import LoggerWriter, app_logger
    lw = LoggerWriter(app_logger(), logging.INFO)
    real = sys.__stdout__
    sys.__stdout__ = None  # GUI / pythonw has no console
    try:
        lw.write("hello µm²\n")  # must NOT raise
        lw.flush()
    finally:
        sys.__stdout__ = real
