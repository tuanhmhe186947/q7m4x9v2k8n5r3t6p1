@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"

python "%PROJECT_ROOT%\scripts\run_weight_tracking_gt_benchmark.py" %*
exit /b %ERRORLEVEL%
