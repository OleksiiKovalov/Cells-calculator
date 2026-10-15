@echo off
REM ============================================================================
REM  Cells Calculator - uninstall.
REM  1) Removes the conda environment created by install.bat.
REM  2) If no other conda environments remain, offers to uninstall Miniconda too.
REM ============================================================================
cd /d "%~dp0"

set ENV_NAME=cells-calculator-v4
set CONDA_HOME=%USERPROFILE%\Miniconda3

REM --- locate conda: on PATH, then our own install ----------------------------
set "CONDA_EXE="
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if not defined CONDA_EXE if exist "%CONDA_HOME%\Scripts\conda.exe" set "CONDA_EXE=%CONDA_HOME%\Scripts\conda.exe"

if not defined CONDA_EXE (
    echo Conda not found - nothing to uninstall.
    pause & exit /b 0
)

echo Removing conda env "%ENV_NAME%" ...
call "%CONDA_EXE%" env remove --yes --name %ENV_NAME%

REM --- check whether any environments remain besides base ---------------------
set "OTHER_ENVS="
for /f "tokens=1" %%E in ('call "%CONDA_EXE%" env list ^| findstr /v "^#"') do (
    if not "%%E"=="base" set "OTHER_ENVS=1"
)

if defined OTHER_ENVS (
    echo Other conda environments still exist - leaving Miniconda installed.
    pause & exit /b 0
)

echo No other conda environments found.
choice /M "Uninstall Miniconda entirely"
if errorlevel 2 (
    echo Skipping Miniconda uninstall.
    pause & exit /b 0
)

if exist "%CONDA_HOME%\Uninstall-Miniconda3.exe" (
    echo Uninstalling Miniconda from "%CONDA_HOME%" ...
    start /wait "" "%CONDA_HOME%\Uninstall-Miniconda3.exe" /S _?=%CONDA_HOME%
) else (
    echo Removing "%CONDA_HOME%" ...
    rmdir /s /q "%CONDA_HOME%"
)

echo.
echo Done.
pause
