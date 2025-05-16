@echo off
rem ==============================================================================
rem Conda env name
rem ==============================================================================
set condaname=cellscounterv3
rem ==============================================================================
if exist "%LOCALAPPDATA%\Anaconda3" set CONDA_ROOT=%LOCALAPPDATA%\Anaconda3
if exist "%LOCALAPPDATA%\Miniconda3" set CONDA_ROOT=%LOCALAPPDATA%\Miniconda3
if exist "C:\ProgramData\miniconda3" set CONDA_ROOT=C:\ProgramData\miniconda3
if exist "C:\ProgramData\anaconda3" set CONDA_ROOT=C:\ProgramData\anaconda3
if exist "d:\Anaconda3" set CONDA_ROOT=d:\Anaconda3
if exist "d:\Miniconda3" set CONDA_ROOT=d:\Miniconda3
if exist "c:\Anaconda3" set CONDA_ROOT=c:\Anaconda3
if exist "c:\Miniconda3" set CONDA_ROOT=c:\Miniconda3
if "%conda_root%"=="" goto no_conda_found
rem ==============================================================================
CALL "%CONDA_ROOT%\Scripts\activate.bat"
