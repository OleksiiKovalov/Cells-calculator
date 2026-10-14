@echo off
REM Run the Cells Calculator test suite in its conda environment (created by install.bat).
REM Any extra arguments are passed straight through to pytest, e.g.:
REM     run-test.bat tests/unit -k morphology
cd /d "%~dp0"

set CONDA_HOME=%USERPROFILE%\Miniconda3

REM Find conda on PATH, otherwise the per-user Miniconda install.bat created.
set "CONDA_EXE="
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if not defined CONDA_EXE if exist "%CONDA_HOME%\Scripts\conda.exe" set "CONDA_EXE=%CONDA_HOME%\Scripts\conda.exe"

if not defined CONDA_EXE (
    echo [ERROR] conda not found. Run install.bat first.
    pause & exit /b 1
)

REM Ensure the test tooling (pytest, hypothesis, pillow) is present.
call "%CONDA_EXE%" run --no-capture-output --name cells-calculator-v4 python -m pip install -q -r requirements-dev.txt || (
    echo [ERROR] Failed to install test dependencies. & pause & exit /b 1
)

REM Verbose live output so a long run visibly progresses instead of looking hung:
REM   -v            : print each test id as it runs (see the current test)
REM   -s            : don't capture stdout — model-loading prints / inference logs
REM                   stream live, so there is constant on-screen activity
REM   --durations   : list the slowest tests at the end
REM Force unbuffered Python so lines appear immediately (not in blocks).
set PYTHONUNBUFFERED=1
call "%CONDA_EXE%" run --no-capture-output --name cells-calculator-v4 python -m pytest -v -s --durations=20 --color=yes %*
pause
