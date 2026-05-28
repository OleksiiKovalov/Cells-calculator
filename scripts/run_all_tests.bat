@echo off
cd /d "%~dp0"
call ./condaactivate.bat
if "%conda_root%"=="" goto no_conda_found

echo.
echo Using conda environment: %condaname%

set FAILURE=0

echo Running unit tests...
call conda run -n "%condaname%" python -m pytest -v ../tests
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% NEQ 0 (
    echo.
    echo Unit tests FAILED.
    set FAILURE=1
) else (
    echo.
    echo Unit tests passed.
)

    echo.
    echo Running smoke tests...
    call conda run -n "%condaname%" python -m pytest -v ../tests_local/test_smoke.py
    set EXITCODE=%ERRORLEVEL%
    if %EXITCODE% NEQ 0 (
        echo.
        echo Smoke tests FAILED.
        set FAILURE=1
    ) else (
        echo.
        echo Smoke tests passed.
    )

echo.
echo Running golden regression tests...
call conda run -n "%condaname%" python -m pytest -v ../tests_local/test_image_golden_regressions.py
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% NEQ 0 (
    echo.
    echo Golden regression tests FAILED.
    set FAILURE=1
) else (
    echo.
    echo Golden regression tests passed.
)

echo.
if "%FAILURE%"=="0" (
    echo All available tests completed successfully.
    pause
    goto exit
) else (
    echo One or more test suites failed.
    pause
    goto exit_with_error
)

:no_conda_found
echo NO ANACONDA/MINICONDA FOUND!
echo PLEASE INSTALL ANACONDA/MINICONDA FIRST!
pause
goto exit_with_error

:exit_with_error
echo.
echo Press any key to close.
pause
exit /b 1

:exit
echo.
echo Press any key to close.
pause
exit /b 0