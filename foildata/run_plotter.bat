@echo off
set CONDA_PATH=C:\Users\zyx20\anaconda3
set ENV_NAME=myml
set PROJECT_DIR=%~dp0

title Visualize starter

echo [1/3] entering directory...
cd /d "%PROJECT_DIR%"

echo [2/3] activating myml: %ENV_NAME%...
call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%

echo [3/3] starting...
python plot_airfoil.py
exit
