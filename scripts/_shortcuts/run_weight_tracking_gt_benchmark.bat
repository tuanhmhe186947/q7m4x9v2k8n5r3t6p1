@echo off
setlocal
set "PROJECT_ROOT=%~dp0\..\.."
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"
python "%PROJECT_ROOT%\scripts\benchmarks\benchmark_tracking_weights.py" %*
exit /b %ERRORLEVEL%
