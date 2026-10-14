@echo off
REM ============================================================================
REM  Remove runtime and build artifacts from this app folder.
REM  Leaves source, config, requirements and data untouched.
REM ============================================================================
cd /d "%~dp0"
echo Cleaning runtime/build artifacts in "%cd%" ...

REM Python bytecode + tool caches (recursive, any sub-folder)
for /d /r %%d in (__pycache__ .pytest_cache .mypy_cache .hypothesis .ruff_cache) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc *.pyo >nul 2>nul

REM Coverage
del /q .coverage coverage.xml >nul 2>nul
if exist htmlcov rd /s /q htmlcov

REM Build / packaging output
for %%d in (build dist) do @if exist "%%d" rd /s /q "%%d"
for /d %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"

REM Runtime output (logs, caches, *_output dirs)
if exist logs rd /s /q logs
if exist .cache rd /s /q .cache
for /d %%d in (*_output) do @if exist "%%d" rd /s /q "%%d"

echo Done.
