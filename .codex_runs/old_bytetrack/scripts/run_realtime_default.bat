@echo off
:: Navigate to project root directory (one level up from scripts/)
cd /d "%~dp0\.."

echo [*] Running Realtime Tracking (100 frames)...
.venv\Scripts\python.exe scripts\run_tracking.py -v Pigs281119_000085_30fps --max-frames 100 --mode realtime

echo.
echo [OK] Execution finished.
pause
