@echo off
cd /d "%~dp0"
if exist Miniconda3-latest-Windows-x86_64.exe Miniconda3-latest-Windows-x86_64.exe /InstallationType=AllUsers /AddToPath=0 /RegisterPython=0 /S 
if exist "../packages/Miniconda3-latest-Windows-x86_64.exe" "../packages/Miniconda3-latest-Windows-x86_64.exe" /InstallationType=AllUsers /AddToPath=0 /RegisterPython=0 /S 