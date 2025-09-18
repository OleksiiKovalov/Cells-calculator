@echo off
cd /d "%~dp0"
pushd scripts
call installminiconda-allusers.bat
call conda-setup-online.bat
popd
pause
