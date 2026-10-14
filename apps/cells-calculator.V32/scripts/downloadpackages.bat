@echo off
rem ==============================================================================
call condaactivate.bat
if "%conda_root%"=="" goto no_conda_found
rem ==============================================================================
call conda activate %condaname%

pip download pip setuptools wheel -d ../offline_packages
pip download -r ../requirements.txt -d ../offline_packages
goto exit

:no_conda_found
echo NO ANACONDA/MINICONDA FOUND!
echo PLEASE INSTALL ANACONDA/MINICONDA FIRST!
pause
goto exit

:exit


