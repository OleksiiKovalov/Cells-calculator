@echo off
REM Launch Cells Calculator in its conda environment (created by install.bat).
cd /d "%~dp0"

set ENV_NAME=cells-calculator-v4

set CONDA_HOME=%USERPROFILE%\Miniconda3

REM Find conda on PATH, otherwise the per-user Miniconda install.bat created.
set "CONDA_EXE="
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if not defined CONDA_EXE if exist "%CONDA_HOME%\Scripts\conda.exe" set "CONDA_EXE=%CONDA_HOME%\Scripts\conda.exe"

call "%CONDA_EXE%" run --name "%ENV_NAME%" --no-capture-output python main.py
