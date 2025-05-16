@echo off
cd /d "%~dp0"
if exist "%LOCALAPPDATA%\Anaconda3" goto condasetup
if exist "%LOCALAPPDATA%\Miniconda3" goto condasetup
if exist "C:\ProgramData\miniconda3" goto condasetup
if exist "C:\ProgramData\anaconda3" goto condasetup
if exist "d:\Anaconda3" goto condasetup
if exist "d:\Miniconda3" goto condasetup
if exist "c:\Anaconda3" goto condasetup
if exist "c:\Miniconda3" goto condasetup
call installminiconda-allusers.bat
:condasetup
call conda-setup-offline.bat

