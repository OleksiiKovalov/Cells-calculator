@echo off
cd /d "%~dp0"
if exist "%USERPROFILE%\Anaconda3\Uninstall-Anaconda3.exe" "%USERPROFILE%\Anaconda3\Uninstall-Anaconda3.exe" /s
if exist "C:\ProgramData\Anaconda3\Uninstall-Anaconda3.exe" "C:\ProgramData\Anaconda3\Uninstall-Anaconda3.exe" /s
IF EXIST "%USERPROFILE%\Miniconda3\Uninstall-minicona3.exe" "%USERPROFILE%\Miniconda3\Uninstall-minicona3.exe" /s
if exist "C:\ProgramData\miniconda3\Uninstall-minicona3.exe" "C:\ProgramData\miniconda3\Uninstall-minicona3.exe" /s


rmdir /S /Q "%USERPROFILE%\.conda"
rmdir /S /Q "%USERPROFILE%\.anaconda_backup"
