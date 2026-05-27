@echo off
cd /d "%~dp0"
call ./condaactivate.bat
if "%conda_root%"=="" goto no_conda_found

echo.
echo Using conda environment: %condaname%

echo Running tests in the repository...
call conda run -n "%condaname%" python -m pytest -v ../tests
if errorlevel 1 (
    echo.
    echo One or more tests failed.
    goto exit_with_error
)

echo.
echo All available tests completed successfully.
pause
goto exit

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


