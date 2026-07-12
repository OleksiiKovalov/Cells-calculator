@echo off
REM ============================================================================
REM  Training Studio - one-time setup.
REM  1) Finds conda; if missing, installs Miniconda from redistributables\.
REM  2) Creates a conda environment and installs the app's dependencies.
REM ============================================================================
cd /d "%~dp0"

set ENV_NAME=cells-calculator-training
set PY_VERSION=3.13
set CONDA_HOME=%USERPROFILE%\Miniconda3
set INSTALLER=redistributables\Miniconda3-latest-Windows-x86_64.exe

REM --- locate conda: on PATH, then our own install, then bundled installer ----
set "CONDA_EXE="
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if not defined CONDA_EXE if exist "%CONDA_HOME%\Scripts\conda.exe" set "CONDA_EXE=%CONDA_HOME%\Scripts\conda.exe"

if not defined CONDA_EXE (
    if not exist "%INSTALLER%" (
        echo [ERROR] conda not found and installer missing:
        echo         %INSTALLER%
        echo         Run download-conda.bat first, or install Miniconda manually.
        pause & exit /b 1
    )
    echo Installing Miniconda to "%CONDA_HOME%" ...
    start /wait "" "%INSTALLER%" /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=%CONDA_HOME%
    set "CONDA_EXE=%CONDA_HOME%\Scripts\conda.exe"
)

echo Creating conda env "%ENV_NAME%" with Python %PY_VERSION% and pip ...
REM conda-forge avoids the Anaconda channel Terms-of-Service prompt. `pip` is
REM listed explicitly because a bare `python` env may not include it. If the env
REM already exists, fall back to installing pip into it (makes re-runs safe).
call "%CONDA_EXE%" create --yes --name %ENV_NAME% --channel conda-forge python=%PY_VERSION% pip || call "%CONDA_EXE%" install --yes --name %ENV_NAME% --channel conda-forge python=%PY_VERSION% pip || (
    echo [ERROR] Failed to create/prepare the environment. & pause & exit /b 1
)

echo Installing dependencies from requirements.txt ...
REM --no-capture-output streams conda's child output live (otherwise it's
REM buffered and looks frozen); pip -v shows what it's resolving/downloading.
call "%CONDA_EXE%" run --no-capture-output --name %ENV_NAME% python -m pip install --upgrade pip
call "%CONDA_EXE%" run --no-capture-output --name %ENV_NAME% python -m pip install -v -r requirements.txt || (
    echo [ERROR] Failed to install dependencies. & pause & exit /b 1
)

echo.
echo Done. Start the app with:  run.bat
pause
