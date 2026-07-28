@echo off
cd /d "%~dp0\..\.."
echo [*] Running Realtime Tracking (100 frames)...
.venv\Scripts\python.exe scripts\track_videos.py ^
  -v Pigs281119_000085_30fps --max-frames 100 ^
  --mode realtime --eval-config realtime_fast
echo.
echo [OK] Execution finished.
pause
