@echo off
cd /d "%~dp0"
rem pip install --no-index --find-links=offline_packages -r requirements.txt 
call condaactivate.bat
if "%conda_root%"=="" goto no_conda_found
rem ==============================================================================
echo found CONDA at %CONDA_ROOT%
echo calling %CONDA_ROOT%\Scripts\activate.bat
CALL "%CONDA_ROOT%\Scripts\activate.bat"
rem online conda setup
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
call conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
echo updating conda
call conda activate base
call conda update conda --yes -q
echo Conda Update Complete
call conda create --yes -n %condaname% python=3.11 -q
echo conda create completed
call conda activate %condaname%
echo conda activate completed

echo Online PIP Install
python -m pip install --upgrade pip -q
echo pip upgrade completed
goto :PipInstallCompleted

:PipInstallCompleted
echo pip installing requirements
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
