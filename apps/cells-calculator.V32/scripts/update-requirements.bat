@echo off
cd /d "%~dp0"
rem pip install --no-index --find-links=offline_packages -r requirements.txt 
call condaactivate.bat
if "%conda_root%"=="" goto no_conda_found

call conda list --export > ..\requirements.txt
echo requirements.txt updated
goto exit

:no_conda_found
echo NO ANACONDA/MINICONDA FOUND!
echo PLEASE INSTALL ANACONDA/MINICONDA FIRST!
pause
goto exit

:exit
