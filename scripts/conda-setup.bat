@echo off
rem pip install --no-index --find-links=offline_packages -r requirements.txt 
call condaactivate.bat
if "%conda_root%"=="" goto no_conda_found
rem ==============================================================================
echo found CONDA at %CONDA_ROOT%
echo calling %CONDA_ROOT%\Scripts\activate.bat
CALL "%CONDA_ROOT%\Scripts\activate.bat"
echo updating conda
call conda activate base
call conda update conda --yes

rem conda-pack -n %condaname%  -o %condaname%.tar.gz
call conda create --yes -n %condaname% python=3.11
echo conda create completed
call conda activate %condaname%
echo conda activate completed
python -m pip install --upgrade pip
echo pip upgrade completed
call pip install -r ../requirements.txt
echo pip install completed
echo CONDA INSTALLED
pause
goto exit

:no_conda_found
echo NO ANACONDA/MINICONDA FOUND!
echo PLEASE INSTALL ANACONDA/MINICONDA FIRST!
pause
goto exit

:exit
