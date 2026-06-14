@echo off
REM Launch Dataset Viewer in the flagship Cells Calculator conda environment.
cd /d "%~dp0"

REM Find conda on PATH, otherwise the per-user Miniconda install.bat created.
set "CONDA_EXE=conda"
where conda >nul 2>nul || set "CONDA_EXE=%USERPROFILE%\Miniconda3\Scripts\conda.exe"

call "%CONDA_EXE%" run --name cells-calculator --no-capture-output python main.py
