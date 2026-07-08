@echo off
:: Navigate to project root directory (two levels up from scripts\_legacy)
cd /d "%~dp0\..\.."

echo [*] Running Realtime Tracking (100 frames)...
.venv\Scripts\python.exe scripts\track_videos.py -v Pigs281119_000085_30fps --max-frames 100 --mode realtime

echo.
echo [OK] Execution finished.
pause
