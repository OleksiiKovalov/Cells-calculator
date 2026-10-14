@echo off
cd /d "%~dp0"
call ./scripts/condaactivate.bat
if "%conda_root%"=="" goto no_conda_found
rem ==============================================================================
conda run -n %condaname% python main.py
goto exit

:no_conda_found
echo NO ANACONDA/MINICONDA FOUND!
echo PLEASE INSTALL ANACONDA/MINICONDA FIRST!
pause
goto exit

:exit
