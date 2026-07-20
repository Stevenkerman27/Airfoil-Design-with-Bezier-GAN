@echo off
set PYTHON_EXE=C:\Users\zyx20\anaconda3\envs\myml\python.exe
set PROJECT_DIR=%~dp0

title Visualize starter

echo [1/2] entering directory...
cd /d "%PROJECT_DIR%"

echo [2/2] starting with myml...
"%PYTHON_EXE%" -u plot_airfoil.py
echo.
echo plot_airfoil.py exited with code %ERRORLEVEL%.
pause
