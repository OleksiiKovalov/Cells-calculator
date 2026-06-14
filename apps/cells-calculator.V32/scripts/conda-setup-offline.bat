@echo off
cd /d "%~dp0"
rem pip install --no-index --find-links=offline_packages -r requirements.txt 
call condaactivate.bat
if "%conda_root%"=="" goto no_conda_found
rem ==============================================================================
echo found CONDA at %CONDA_ROOT%
echo calling %CONDA_ROOT%\Scripts\activate.bat
CALL "%CONDA_ROOT%\Scripts\activate.bat"

rem offline conda setup
call conda create --yes -n %condaname% python=3.11
echo conda create completed
conda create --yes --name %condaname% --offline --override-channels --channel file:./../packages python=3.11

call conda activate %condaname%
echo conda activate completed

if exist ./../packages/*.* goto OfflineInstall
echo Online PIP Install
python -m pip install --upgrade pip
goto :PipInstallCompleted
:OfflineInstall
echo Local PIP Install
pip install --no-index --find-links=./../packages -r ../requirements.txt
goto :PipInstallCompleted

:PipInstallCompleted
echo pip upgrade completed
call pip install -r ../requirements.txt
echo pip install completed
echo CONDA INSTALLED
goto exit

:no_conda_found
echo NO ANACONDA/MINICONDA FOUND!
echo PLEASE INSTALL ANACONDA/MINICONDA FIRST!
pause
goto exit

:exit
