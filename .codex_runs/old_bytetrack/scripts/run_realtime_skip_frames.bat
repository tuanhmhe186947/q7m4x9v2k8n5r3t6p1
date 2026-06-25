@echo off
:: Navigate to project root directory (one level up from scripts/)
cd /d "%~dp0\.."

echo [*] Running Realtime Tracking with Skip Frames (detect_every_n_frames=2, 100 frames)...
.venv\Scripts\python.exe scripts\run_tracking.py -v Pigs281119_000085_30fps --max-frames 100 --mode realtime --detect-every-n-frames 2

echo.
echo [OK] Execution finished.
pause
